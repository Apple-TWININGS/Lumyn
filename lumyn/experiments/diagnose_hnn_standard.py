"""诊断：实验 J 标准基准上的 HNN 是否**公平**？（低维下应当可训练）

实验 H 中 HNN 在 36 维状态空间没训起来（dp 残差 0.9992，H_θ 在真实与生成轨迹上
漂移完全相同 → 判别力为零）。实验 J 换到标准低维基准后方程残差降到 0.05–0.09，
但更优率仍只有 25% / 41.7%。在报告该结果前必须再次排除「实现没训好」。

判据同 diagnose_hnn_fairness.py：
  称职的 HNN 必须让 H_θ 在**真实轨迹**上的漂移显著小于在生成轨迹上的漂移。
  若两者相当（比值 ≈ 1），则该 critic 只在测量自己的训练误差。
"""
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from lumyn.experiments.experiment_j_standard_benchmarks import (
    system_fig8, system_kepler, rollout, rollout_nn, train_force,
    train_hnn_local, hnn_energy, hnn_eq_residual, cons_channels, DT,
)

STEPS = 300


def diagnose(name, ic_fn, mass, seed, epochs_list=(600, 2000)):
    rng = np.random.default_rng(seed)
    PT, VT = [], []
    for _ in range(80):
        p, v, m = ic_fn(rng)
        pt, vt = rollout(p, v, m, STEPS)
        PT.append(pt); VT.append(vt)
    PT, VT = np.stack(PT), np.stack(VT)
    TRp, TRv = PT[:50], VT[:50]
    TEp, TEv = PT[50:], VT[50:]

    net = train_force(TRp, TRv, mass, epochs=80, h=64, seed=seed)
    print()
    print("=" * 78)
    print(f"[{name}]")
    print("=" * 78)
    # 对照：真实守恒残差的分离度
    t_ch = np.array([cons_channels(TEp[i], TEv[i], mass)["angular"] for i in range(len(TEp))])
    rr = np.random.default_rng(5)
    g_ch = []
    for i in range(len(TEp)):
        gp, gv = rollout_nn(TEp[i][0], TEv[i][0], mass, net, 0.15, rr, STEPS)
        g_ch.append(cons_channels(gp, gv, mass)["angular"])
    g_ch = np.array(g_ch)
    print(f"  [对照] 真实角动量残差: 真实 {np.median(t_ch):.3e} | "
          f"生成 {np.median(g_ch):.3e} | 比值 {np.median(g_ch)/(np.median(t_ch)+1e-300):.3e}")

    for epochs in epochs_list:
        net_h = train_hnn_local(TRp, TRv, mass, mass.size, epochs=epochs, seed=seed)
        r1, r2 = hnn_eq_residual(net_h, TEp, TEv, mass)
        h_t = np.array([np.abs(hnn_energy(net_h, TEp[i], TEv[i], mass, mass.size)).max() for i in range(len(TEp))])
        # 真实轨迹上 H_θ 的相对漂移
        def rel_drift(pos_t, vel_t):
            h = hnn_energy(net_h, pos_t, vel_t, mass, mass.size)
            return float(np.abs(h - h[0]).max() / (np.abs(h).mean() + 1e-12))
        dt_t = np.array([rel_drift(TEp[i], TEv[i]) for i in range(len(TEp))])
        rr2 = np.random.default_rng(5)
        dt_g = []
        for i in range(len(TEp)):
            gp, gv = rollout_nn(TEp[i][0], TEv[i][0], mass, net, 0.15, rr2, STEPS)
            dt_g.append(rel_drift(gp, gv))
        dt_g = np.array(dt_g)
        ratio = np.median(dt_g) / (np.median(dt_t) + 1e-300)
        print(f"\n  epochs={epochs}")
        print(f"    哈密顿方程残差（测试集）: dq {r1:.4f} | dp {r2:.4f}")
        print(f"    H_θ 相对漂移: 真实轨迹 {np.median(dt_t):.4e} | "
              f"生成轨迹 {np.median(dt_g):.4e} | 生成/真实 = {ratio:.3f}")
        if ratio < 2.0:
            print("    → ❌ H_θ 在真实轨迹上并不守恒：该 critic 测的是训练不足，"
                  "其失败**不能**归因于 HNN 方法")
        else:
            print("    → ✅ H_θ 在真实轨迹上守恒，判别力不足属方法本身的问题")


def main():
    t0 = time.time()
    diagnose("figure-eight 三体（N=3, 2D）",
             lambda r: system_fig8(r, jitter=0.02), np.ones(3), seed=0)
    diagnose("Kepler 二体（N=2, 2D）",
             lambda r: system_kepler(r), np.ones(2), seed=1)
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
