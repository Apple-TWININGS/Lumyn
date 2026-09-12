"""诊断：实验 H 里的 HNN baseline 是否被训练充分？（公平性检查）

动机
----
实验 H 中 HNN 能量残差的 IC 内更优率只有 36.2%±6.1%（低于随机 50%）。
「文献级方法输给随机」通常说明**实现没训好**，而不是方法本身不行。
在把该结果写进论文前必须排除这个可能。

判据
----
一个称职的 HNN 应当：
  1. 在**真实轨迹**上让 H_θ 近似守恒（残差远小于生成轨迹上的残差）
  2. 哈密顿方程残差（dq/dt − ∂H/∂p，dp/dt + ∂H/∂q）足够小
若第 1 条不成立（真实轨迹上 H_θ 的漂移与生成轨迹上同量级），
则该 critic 只是在测量自己的训练不足，其失败不能归因于方法。
"""
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from lumyn.experiments.experiment_a1_critic_quality import (   # noqa: E402
    prior_ic, rollout_verlet,
)
from lumyn.experiments.experiment_a2_neural_generator import train_force_model  # noqa: E402
from lumyn.experiments.experiment_d_channels_and_generators import (   # noqa: E402
    rollout_net, N, MASS, DT, F_TRAIN,
)
from lumyn.experiments.experiment_h_benchmark import HNN, train_hnn, hnn_energy  # noqa: E402

SIGMAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.6, 1.0)


def hnn_eq_residual(net, P, V, mass):
    """哈密顿方程残差的相对值（在真实轨迹上）。"""
    B, T, Nn, _ = P.shape
    m = np.asarray(mass)
    Q = torch.tensor(P.reshape(B, T, Nn * 3), dtype=torch.float32)
    Pm = torch.tensor((V * m[None, None, :, None]).reshape(B, T, Nn * 3),
                      dtype=torch.float32)
    dq = ((Q[:, 1:] - Q[:, :-1]) / DT).reshape(-1, Nn * 3)
    dp = ((Pm[:, 1:] - Pm[:, :-1]) / DT).reshape(-1, Nn * 3)
    qa = Q[:, :-1].reshape(-1, Nn * 3)
    pa = Pm[:, :-1].reshape(-1, Nn * 3)
    with torch.enable_grad():
        pq, pp = net.time_deriv(qa, pa)
    r1 = float(((pq - dq) ** 2).mean().sqrt() / (dq ** 2).mean().sqrt())
    r2 = float(((pp - dp) ** 2).mean().sqrt() / (dp ** 2).mean().sqrt())
    return r1, r2


def main():
    t0 = time.time()
    rng = np.random.default_rng(1)
    Ptr, Vtr = [], []
    for _ in range(240):
        p, v = prior_ic(rng.uniform(*F_TRAIN), rng)
        pt, vt = rollout_verlet(p, v, np.ones(N) * MASS)
        Ptr.append(pt); Vtr.append(vt)
    Ptr, Vtr = np.stack(Ptr), np.stack(Vtr)
    mass = np.ones(N) * MASS
    gen = train_force_model(Ptr[:120], Vtr[:120], mass, epochs=12, h=32, lr=3e-3)

    # 训练/测试切分（与实验 H 一致：真实轨迹用于训练 HNN）
    TR_P, TR_V = Ptr[:120], Vtr[:120]
    TE_P, TE_V = Ptr[180:], Vtr[180:]

    print("=" * 80)
    print("HNN 训练充分性诊断")
    print("=" * 80)

    for epochs in (300, 1200):
        t1 = time.time()
        net = train_hnn(TR_P, TR_V, mass, epochs=epochs)
        r1, r2 = hnn_eq_residual(net, TE_P, TE_V, mass)
        print(f"\n  epochs={epochs}  (训练 {time.time()-t1:.0f}s)")
        print(f"    哈密顿方程残差（测试集，相对）: dq {r1:.4f} | dp {r2:.4f}")

        # H 在真实轨迹上的守恒性
        h_true = np.array([np.abs(hnn_energy(net, TE_P[i], TE_V[i], mass)[1:]
                                  - hnn_energy(net, TE_P[i], TE_V[i], mass)[0]).max()
                           for i in range(40)])
        scale = np.array([np.abs(hnn_energy(net, TE_P[i], TE_V[i], mass)).mean()
                          for i in range(40)])
        h_true_rel = h_true / (scale + 1e-12)

        # H 在生成轨迹上的漂移
        rr = np.random.default_rng(5)
        h_gen_rel = []
        for i in range(40):
            p0, v0 = TE_P[i][0], TE_V[i][0]
            gp, gv = rollout_net(p0, v0, gen, 0.15, rr)
            h = hnn_energy(net, gp, gv, mass)
            h_gen_rel.append(np.abs(h - h[0]).max() / (np.abs(h).mean() + 1e-12))
        h_gen_rel = np.array(h_gen_rel)

        print(f"    H_θ 在**真实**轨迹上的相对漂移 (中位): {np.median(h_true_rel):.4e}")
        print(f"    H_θ 在**生成**轨迹上的相对漂移 (中位): {np.median(h_gen_rel):.4e}")
        ratio = np.median(h_gen_rel) / (np.median(h_true_rel) + 1e-300)
        print(f"    生成/真实 = {ratio:.3f}")
        if ratio < 2.0:
            print("    → ❌ H_θ 在真实轨迹上并不守恒：该 critic 测的是训练不足，"
                  "其失败**不能**归因于 HNN 方法本身")
        else:
            print("    → ✅ H_θ 在真实轨迹上守恒，判别力不足是方法本身的问题")

    # 对照：真实守恒残差在同一批轨迹上的分离度
    from lumyn.experiments.experiment_d_channels_and_generators import channels
    ch_t = np.array([channels(TE_P[i], TE_V[i])["angular"] for i in range(40)])
    rr = np.random.default_rng(5)
    ch_g = []
    for i in range(40):
        gp, gv = rollout_net(TE_P[i][0], TE_V[i][0], gen, 0.15, rr)
        ch_g.append(channels(gp, gv)["angular"])
    print(f"\n  [对照] 真实角动量残差: 真实轨迹 {np.median(ch_t):.3e} | "
          f"生成轨迹 {np.median(ch_g):.3e} | 比值 "
          f"{np.median(ch_g)/(np.median(ch_t)+1e-300):.3e}")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
