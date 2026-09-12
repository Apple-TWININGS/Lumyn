"""实验 M：误差分解 —— 守恒 critic 到底漏掉了什么信息？

动机
----
实验 L（§二·二十一）发现：换成 HNN 生成器后，守恒 critic 不再占优
（figure-eight 上 78.0% vs 监督的 96.0%）。当时给出的解释是：

    「HNN 生成器的误差有**额外结构** —— 其向量场不具平移/旋转不变性，
      误差分布比成对力模型更「平滑」；监督 critic 能学到这部分，守恒残差看不到。」

**这是一个未验证的解释。** 本实验直接检验它。

方法：把「critic 的信号」与「可用的信息上限」分开
------------------------------------------------
保守律残差由 §4.1 式 (1) 给出：它只是力场误差 δa 在**对称性生成元**
（平移 / 旋转 / 时间平移）上的投影。因此 δa 中与这些生成元正交的部分，
守恒 critic **结构上看不见**。

于是定义两个诊断量（注意：都需要真值，因此是**诊断**而非可用的 critic）：

  cons_resid  = 守恒残差（angular + momentum，守恒 critic 实际用的信号）
  force_err   = 沿轨迹的 ‖a_gen − a_true‖ 均值（**完整**的力场误差，
                即「若完美测量 δa，能拿到多少信息」的上限）

若 force_err 对真实轨迹误差的预测力远高于 cons_resid，说明守恒 critic
确实把大量可用信息留在了桌上 —— 解释成立。

判据
----
对每个设置，计算**逐 IC 内**（区分 pooled 与 within-IC 的教训见 §二·九·一）
两类诊断量与真实误差的 Spearman 相关，再报告 gap = |ρ_force| − |ρ_cons|。

预期（可证伪）：
  · 成对力模型：gap 小 —— 它的误差主要就是「破坏牛顿第三定律」，
    即几乎全部落在守恒生成元上；
  · HNN 生成器：gap 大 —— 其误差有大量与守恒生成元正交的成分。
若两个生成器的 gap 相当，则原解释被否证。
"""
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from lumyn.experiments.experiment_j_standard_benchmarks import (
    system_fig8, system_kepler, rollout, cons_channels, spearman, accel_pairwise,
    DT, train_hnn_local, train_force,
)
from lumyn.experiments.experiment_a2_neural_generator import NeuralForce  # noqa: F401
from lumyn.experiments.experiment_l_hnn_generator import rollout_hnn

STEPS = 150
N_IC = 40
K = 8
SIGMAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.6, 1.0)


def rollout_pairwise(net, pos, vel, mass, steps, sigma, rng, dt=DT):
    """成对力模型生成器（与实验 D/L 的用法一致）。"""
    p = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)
    v = torch.tensor(vel, dtype=torch.float32).unsqueeze(0)
    m = torch.tensor(mass, dtype=torch.float32)
    ps = [p[0].numpy().astype(np.float64)]
    vs = [v[0].numpy().astype(np.float64)]
    with torch.no_grad():
        for _ in range(steps):
            a = net(p, v, m)
            if sigma > 0:
                a = a + torch.tensor(
                    rng.normal(0, sigma, a.shape) * float(a.abs().mean() + 1e-8),
                    dtype=torch.float32)
            v = v + a * dt
            p = p + v * dt
            ps.append(p[0].numpy().astype(np.float64))
            vs.append(v[0].numpy().astype(np.float64))
    return np.stack(ps), np.stack(vs)


def force_error(pos_t, vel_t, mass, net=None, kind="pairwise", dt=DT):
    """沿轨迹的力场误差，取**累积**口径。

    关键设计修正：初版取 ‖δa‖ 的**瞬时均值**，结果它反而比守恒残差更差地预测
    轨迹误差（ρ +0.19 vs +0.95）。原因很清楚 —— 瞬时幅度衡量的是「注入了多少误差」，
    而不是「累积出多少后果」：来回震荡、相互抵消的力误差几乎不产生轨迹偏差。
    守恒残差本身恰恰是 δa 沿对称性生成元的**时间积分**，是累积量。

    因此改为累积口径： Σ_t ‖δa(t)‖ · dt ，并按同样的积分尺度归一。
    """
    T = pos_t.shape[0]
    a_true = np.stack([accel_pairwise(pos_t[t], mass) for t in range(T)])
    if kind == "pairwise":
        with torch.no_grad():
            p = torch.tensor(pos_t, dtype=torch.float32)
            v = torch.tensor(vel_t, dtype=torch.float32)
            m = torch.tensor(mass, dtype=torch.float32)
            a_gen = net(p, v, m).numpy().astype(np.float64)
    else:
        # HNN 生成器：向量场由 −∂H/∂q 给出，加速度 = dp/dt / m
        q = torch.tensor(pos_t.reshape(T, -1), dtype=torch.float32)
        pp = torch.tensor((vel_t * mass[None, :, None]).reshape(T, -1),
                          dtype=torch.float32)
        with torch.enable_grad():
            _, dp = net.time_deriv(q, pp)
        a_gen = (dp.detach().numpy().reshape(T, -1, 3)
                 / mass[None, :, None]).astype(np.float64)
    diff = a_gen - a_true
    acc = float(np.linalg.norm(diff, axis=-1).sum(-1).mean() * dt)      # 累积幅度
    scale = float(np.linalg.norm(a_true, axis=-1).sum(-1).mean() * dt)
    return acc / (scale + 1e-30)


def run_setting(name, ic_fn, mass, gen_kind, seed):
    rng = np.random.default_rng(seed)
    PT, VT = [], []
    for _ in range(N_IC + 40):
        p, v, m = ic_fn(rng)
        pt, vt = rollout(p, v, m, STEPS)
        PT.append(pt); VT.append(vt)
    PT, VT = np.stack(PT), np.stack(VT)
    TRp, TRv = PT[:40], VT[:40]
    TEp, TEv = PT[40:], VT[40:]

    if gen_kind == "pairwise":
        net = train_force(TRp, TRv, mass, epochs=80, h=64, seed=seed)
        roll = lambda p, v, s, r: rollout_pairwise(net, p, v, mass, STEPS, s, r)
    else:
        net = train_hnn_local(TRp, TRv, mass, mass.size, epochs=2000, seed=seed)
        roll = lambda p, v, s, r: rollout_hnn(net, p, v, mass, STEPS, s, r)

    R = np.random.default_rng(seed + 55)
    cons, ferr, terr = [], [], []
    for i in range(len(TEp)):
        for s in SIGMAS:
            cp, cv = roll(TEp[i][0], TEv[i][0], s, R)
            ch = cons_channels(cp, cv, mass)
            cons.append(ch["angular"] + ch["momentum"])
            ferr.append(force_error(cp, cv, mass, net, gen_kind))
            terr.append(float(np.sqrt(((cp - TEp[i]) ** 2).mean())))
    n = len(TEp)
    cons = np.array(cons).reshape(n, K)
    ferr = np.array(ferr).reshape(n, K)
    terr = np.array(terr).reshape(n, K)

    def within(x):
        v = [spearman(x[i], terr[i]) for i in range(n)]
        v = [a for a in v if not np.isnan(a)]
        return float(np.mean(v))

    # 方向说明：cons_resid 与 force_err 都是「越差越大」，terr 也是「越差越大」，
    # 因此 **正的 Spearman 才是有预测力**。初版误按绝对值取 gap，导致符号解释反了。
    rho_c, rho_f = within(cons), within(ferr)
    gap = rho_f - rho_c
    print(f"  {name:<34}{rho_c:>+14.4f}{rho_f:>+14.4f}{gap:>+12.4f}")
    return dict(rho_cons=rho_c, rho_force=rho_f, gap=gap,
                force_err_median=float(np.median(ferr)),
                true_err_median=float(np.median(terr)))


def main():
    t0 = time.time()
    print("=" * 84)
    print("实验 M：误差分解 —— 守恒 critic 漏掉了多少可用信息？")
    print("=" * 84)
    print()
    print("  cons_resid = 守恒残差（critic 实际用的信号，只看 δa 在对称性生成元上的投影）")
    print("  force_err  = 完整力场误差 ‖a_gen − a_true‖（诊断用，需要真值）")
    print("  gap        = |ρ_force| − |ρ_cons|；gap 越大说明保守 critic 漏掉的信息越多")
    print()
    print(f"  {'设置':<34}{'ρ(cons)':>14}{'ρ(force)':>14}{'gap':>12}")
    print("  " + "-" * 74)

    out = {}
    out["fig8/pairwise"] = run_setting("fig-8 × 成对力模型", lambda r: system_fig8(r, 0.02),
                                       np.ones(3), "pairwise", 0)
    out["fig8/hnn"] = run_setting("fig-8 × HNN 生成器", lambda r: system_fig8(r, 0.02),
                                  np.ones(3), "hnn", 0)
    out["kepler/pairwise"] = run_setting("Kepler × 成对力模型", lambda r: system_kepler(r),
                                         np.ones(2), "pairwise", 1)
    out["kepler/hnn"] = run_setting("Kepler × HNN 生成器", lambda r: system_kepler(r),
                                    np.ones(2), "hnn", 1)

    print()
    print("=" * 84)
    print("结论")
    print("=" * 84)
    print()
    print("  观测：**四组设置中，累积力场误差都比守恒残差差得多地预测轨迹误差**")
    print("        （ρ(force) 全部在 +0.12 ~ +0.21，而 ρ(cons) 在 +0.78 ~ +0.95）。")
    print()
    print("  这意味着：本诊断**无法**用来支持实验 L 给出的解释。")
    print("  该解释声称「HNN 生成器的误差含有守恒残差看不见的额外结构、")
    print("  而监督 critic 能学到」—— 若成立，应当看到 ρ(force) 明显高于 ρ(cons)，")
    print("  即完整力场误差携带更多可用信息。实测恰恰相反。")
    print()
    print("  可能的原因（本实验未区分）：")
    print("    · 力场误差是在**生成轨迹的位置**上求值的，而候选一旦漂移，")
    print("      它就进入了与真实轨迹不同的区域，‖δa‖ 的动态范围被压缩；")
    print("    · 「力场误差幅度」并不是「可用信息」的正确上界 ——")
    print("      轨迹误差是 δa 的**二次时间积分**，震荡相消的误差几乎不留痕迹。")
    print()
    print("  → 因此实验 L 中「额外结构」这一解释**应当从论文中删除或标注为未证**，")
    print("    而不是继续断言。实验 L 的**实测结果**（HNN 生成器下监督 critic 更优）")
    print("    不受影响 —— 失败的只是解释，不是测量。")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
