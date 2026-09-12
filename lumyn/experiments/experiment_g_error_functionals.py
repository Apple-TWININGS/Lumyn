"""实验 G：守恒残差的误差泛函推导与验证 —— 为什么 angular/momentum 强、energy 弱？

问题
----
实测（实验 D/E）在**归一化**残差下：
    angular / momentum 单通道更优率 88%–100%
    energy 单通道仅 56%–60%
有两种互斥解释：
  (a) **物理的**：能量守恒只要求力可由势导出，而动量/角动量守恒要求
      力成对且中心（牛顿第三定律）—— 学习模型更容易满足前者。
  (b) **标度的**：三个通道除以的尺度量级不同（energy 除以巨大的 |E|，
      angular 除以较小的 ‖L‖），同样的绝对误差得到完全不同的相对值。

本实验的判据
------------
1. 计算**绝对**漂移（不归一化），看分离度是否仍以 angular/momentum 为主。
   若在绝对口径下 energy 反而最好 → 结论 (b) 成立，(a) 是错的。
2. 用 Noether 泛函**预测**漂移并对比实测：
       ΔP = ∫ Σ_i m_i δa_i dt
       ΔL = ∫ Σ_i m_i (r_i × δa_i) dt
       ΔE = ∫ Σ_i m_i (v_i · δa_i) dt
   其中 δa 是「学习到的加速度 − 真实加速度」。
   若预测与实测吻合，则「守恒残差 = 误差在对称性生成元上的投影」这一推导成立，
   可直接写入论文的数学部分。
"""
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from lumyn.experiments.experiment_a1_critic_quality import (   # noqa: E402
    prior_ic, accel, rollout_verlet,
)
from lumyn.experiments.experiment_a2_neural_generator import train_force_model  # noqa: E402
from lumyn.experiments.experiment_d_channels_and_generators import (   # noqa: E402
    channels, rollout_net, N, MASS, DT, SOFT, G, F_TRAIN,
)

STEPS = 40


def true_force_series(pos_t, mass):
    """真实成对力在整条轨迹上的取值。"""
    return np.stack([accel(pos_t[t], mass) for t in range(len(pos_t))])


def nn_force_series(pos_t, vel_t, net, mass):
    """学习到的力场在整条轨迹上的取值。"""
    with torch.no_grad():
        p = torch.tensor(pos_t, dtype=torch.float32)
        v = torch.tensor(vel_t, dtype=torch.float32)
        m = torch.tensor(mass, dtype=torch.float32)
        return net(p, v, m).numpy().astype(np.float64)


def main():
    t0 = time.time()
    rng = np.random.default_rng(1)
    Ptr, Vtr = [], []
    for _ in range(200):
        p, v = prior_ic(rng.uniform(*F_TRAIN), rng)
        pt, vt = rollout_verlet(p, v, np.ones(N) * MASS)
        Ptr.append(pt); Vtr.append(vt)
    Ptr, Vtr = np.stack(Ptr), np.stack(Vtr)
    mass = np.ones(N) * MASS
    net = train_force_model(Ptr[:120], Vtr[:120], mass, epochs=12, h=32, lr=3e-3)
    print(f"生成器训练完成（{time.time()-t0:.0f}s）")

    # ---------------- 收集样本 ----------------
    nrng = np.random.default_rng(7)
    rows = []
    for _ in range(40):
        p0, v0 = prior_ic(nrng.uniform(*F_TRAIN), nrng)
        ref_p, ref_v = rollout_verlet(p0, v0, mass)          # 真值轨迹
        gen_p, gen_v = rollout_net(p0, v0, net, 0.15, nrng)  # 生成轨迹

        a_true = true_force_series(ref_p, mass)
        a_nn = nn_force_series(gen_p, gen_v, net, mass)
        dA = a_nn - a_true                                   # 力场误差 δa

        ch_ref = channels(ref_p, ref_v)
        ch_gen = channels(gen_p, gen_v)

        # ---- 绝对漂移（不归一化）----
        def abs_drifts(pos_t, vel_t):
            m = mass
            ke = 0.5 * (m * (vel_t ** 2).sum(-1)).sum(-1)
            d = pos_t[:, None] - pos_t[:, :, None]
            r = np.sqrt((d * d).sum(-1) + SOFT ** 2)
            eye = 1.0 - np.eye(N)
            mm = m[:, None] * m[None, :]
            E = ke - (G * mm[None] / r * eye[None]).sum((1, 2)) / 2.0
            L = (m[None, :, None] * np.cross(pos_t, vel_t)).sum(1)
            P = (m[None, :, None] * vel_t).sum(1)
            return (float(E.max() - E.min()),
                    float(np.linalg.norm(L.max(0) - L.min(0))),
                    float(np.linalg.norm(P.max(0) - P.min(0))))

        abs_ref = abs_drifts(ref_p, ref_v)
        abs_gen = abs_drifts(gen_p, gen_v)

        # ---- Noether 泛函预测 ----
        # δa 是逐时刻的加速度误差。三个守恒量的时间导数分别是 δa 在
        # 平移 / 旋转 / 时间平移 生成元上的投影：
        #     dP/dt = Σ m δa        dL/dt = Σ m (r × δa)      dE/dt = Σ m (v · δa)
        # 对时间积分即得累积漂移。
        m4 = mass[None, :, None]                              # (1,N,1)
        pred_dP = float(np.linalg.norm((m4 * dA).sum(1).sum(0) * DT))
        pred_dL = float(np.linalg.norm(
            (m4 * np.cross(gen_p, dA)).sum(1).sum(0) * DT))
        # 标量：先对粒子与时间维度全部求和，避免留下 (1,) 形状的数组
        pred_dE = abs(float((m4 * (gen_v * dA).sum(-1, keepdims=True)).sum()) * DT)

        # 实测的端点差（更贴近泛函的积分定义）
        def endpoints(pos_t, vel_t):
            m = mass
            ke = 0.5 * (m * (vel_t ** 2).sum(-1)).sum(-1)
            d = pos_t[:, None] - pos_t[:, :, None]
            r = np.sqrt((d * d).sum(-1) + SOFT ** 2)
            eye = 1.0 - np.eye(N)
            mm = m[:, None] * m[None, :]
            E = ke - (G * mm[None] / r * eye[None]).sum((1, 2)) / 2.0
            L = (m[None, :, None] * np.cross(pos_t, vel_t)).sum(1)
            P = (m[None, :, None] * vel_t).sum(1)
            return np.array([E[-1] - E[0],
                             np.linalg.norm(L[-1] - L[0]),
                             np.linalg.norm(P[-1] - P[0])])

        # 实测：**生成轨迹自身**的端点漂移。
        # 早先写成 endpoints(gen) − endpoints(ref) 是错误的：那不是生成轨迹的漂移，
        # 而是「两条不同轨迹的守恒量之差」，与 ∫δa 无可比性。
        # 正确的对应关系（对生成轨迹而言）：
        #     a_nn = a_true + δa  ⇒  dP_gen/dt = Σ m δa   （因 a_true 的 Σm a_true = 0）
        meas = endpoints(gen_p, gen_v)
        rows.append(dict(ch_ref=ch_ref, ch_gen=ch_gen,
                         abs_ref=abs_ref, abs_gen=abs_gen,
                         pred=np.array([pred_dE, pred_dL, pred_dP]),
                         meas=meas))

    # ---------------- 1. 绝对 vs 归一化口径 ----------------
    print()
    print("=" * 82)
    print("1. 绝对漂移口径（不归一化）—— 分离度是否仍以 angular/momentum 为主？")
    print("=" * 82)
    names = ("energy", "angular", "momentum")
    print(f"  {'通道':<10}{'物理(绝对,中位)':>18}{'生成(绝对,中位)':>18}"
          f"{'生成/物理':>12}{'归一化口径的比值':>18}")
    for i, nm in enumerate(names):
        a = np.median([r["abs_ref"][i] for r in rows])
        b = np.median([r["abs_gen"][i] for r in rows])
        n_ratio = np.median([r["ch_gen"][nm] / (r["ch_ref"][nm] + 1e-30) for r in rows])
        print(f"  {nm:<10}{a:>18.4e}{b:>18.4e}{b/(a+1e-30):>12.3f}{n_ratio:>18.3f}")

    print()
    print(f"  {'通道':<10}{'归一化尺度(物理,中位)':>26}")
    for i, nm in enumerate(names):
        s = np.median([r["abs_ref"][i] / (r["ch_ref"][nm] + 1e-30) for r in rows])
        print(f"  {nm:<10}{s:>26.4e}")
    print("  → 若各通道尺度相差多个数量级，则「energy 弱」可能源于尺度而非物理。")

    # ---------------- 2. Noether 泛函预测 vs 实测 ----------------
    print()
    print("=" * 82)
    print("2. Noether 泛函预测 vs 实测端点差")
    print("   预测: ΔP = ∫Σm δa dt,  ΔL = ∫Σm(r×δa)dt,  ΔE = ∫Σm(v·δa)dt")
    print("=" * 82)
    pred = np.array([r["pred"] for r in rows])
    meas = np.array([r["meas"] for r in rows])
    print(f"  {'通道':<10}{'预测(中位)':>16}{'实测(中位)':>16}"
          f"{'预测/实测':>12}{'对数相关':>12}")
    for i, nm in enumerate(names):
        pm = np.median(np.abs(pred[:, i]))
        mm = np.median(np.abs(meas[:, i]))
        mask = (np.abs(meas[:, i]) > 0) & (np.abs(pred[:, i]) > 0)
        if mask.sum() > 3:
            lr = np.corrcoef(np.log(np.abs(pred[mask, i])),
                             np.log(np.abs(meas[mask, i])))[0, 1]
        else:
            lr = float("nan")
        print(f"  {nm:<10}{pm:>16.4e}{mm:>16.4e}{pm/(mm+1e-30):>12.3f}{lr:>12.3f}")

    print()
    print("  说明：预测与实测若同量级且对数相关高，则「守恒残差 = 力场误差在")
    print("        对称性生成元上的累积投影」这一推导成立，可写入论文。")

    # ---------------- 3. 噪声地板解释：能量弱是因为积分器误差，不是因为归一化 ----------------
    print()
    print("=" * 82)
    print("3. 噪声地板检验：缩小 dt 是否降低 energy 通道的地板并提升其判别力？")
    print("=" * 82)
    print("  预测：angular/momentum 的地板已是机器精度、不会改善；")
    print("        而 energy 的地板由积分器截断误差决定，应随 dt 下降。")
    print(f"  {'dt':<10}{'energy 地板':>16}{'angular 地板':>16}{'momentum 地板':>16}"
          f"{'energy 分离度':>16}")
    for k in (1, 2, 4):
        p_, v_, g_, gv_ = [], [], [], []
        rr = np.random.default_rng(7)
        for _ in range(25):
            q0, w0 = prior_ic(rr.uniform(*F_TRAIN), rr)
            # 真值：dt/k
            pp, vv = q0.copy(), w0.copy()
            ps, vs = [pp.copy()], [vv.copy()]
            for _ in range(STEPS * k):
                vv = vv + accel(pp, mass) * (DT / k)
                pp = pp + vv * (DT / k)
                ps.append(pp.copy()); vs.append(vv.copy())
            p_.append(np.stack(ps)); v_.append(np.stack(vs))
            gg = rollout_net(q0, w0, net, 0.15, rr)
            g_.append(gg[0]); gv_.append(gg[1])
        floors, seps = [], []
        for i, nm in enumerate(("energy", "angular", "momentum")):
            ref_c = np.median([channels(p_[j], v_[j])[nm] for j in range(25)])
            gen_c = np.median([channels(g_[j], gv_[j])[nm] for j in range(25)])
            floors.append(ref_c); seps.append(gen_c / (ref_c + 1e-300))
        print(f"  {DT/k:<10.5f}{floors[0]:>16.4e}{floors[1]:>16.4e}{floors[2]:>16.4e}"
              f"{seps[0]:>16.3f}")

    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
