"""实验 K：用**哈密顿方程残差**重做 HNN 对照（实验 J 定位的原因）

为什么换打分方式
----------------
实验 H / J 都用「H_θ 沿轨迹是否恒定」给候选打分，结果 HNN 三次都不成立。
`diagnose_hnn_standard.py` 定位到原因：

    HNN 的**训练目标**是哈密顿方程残差
        r = ‖∂H/∂p − dq/dt‖² + ‖∂H/∂q + dp/dt‖²
    而「H_θ 近似恒定」是比它**更弱的判据** —— 一条不满足运动方程的轨迹
    仍然可以有近似恒定的 H_θ。

因此本实验直接用**方程残差**打分，这也是与 HNN 训练目标一致的用法。

同时报告守恒 critic 与两种 HNN 打分的公平性诊断（真实 vs 生成轨迹的分离度）。
"""
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

from lumyn.experiments.experiment_j_standard_benchmarks import (
    system_fig8, system_kepler, rollout, rollout_nn, train_force,
    train_hnn_local, hnn_energy, cons_channels, spearman, K, SIGMAS, DT,
)

STEPS = 300
N_TR, N_TE = 100, 50
HNN_EPOCHS = 2000


def hnn_eq_residual_series(net, pos_t, vel_t, mass, dt=DT):
    """沿轨迹逐时刻的哈密顿方程残差（相对量）。返回标量：均值。"""
    T, Nn, _ = pos_t.shape
    q = pos_t.reshape(T, Nn * 3)
    p = (vel_t * np.asarray(mass)[None, :, None]).reshape(T, Nn * 3)
    dq = (q[1:] - q[:-1]) / dt
    dp = (p[1:] - p[:-1]) / dt
    qt = torch.tensor(q[:-1], dtype=torch.float32)
    pt = torch.tensor(p[:-1], dtype=torch.float32)
    with torch.enable_grad():
        pq, pp = net.time_deriv(qt, pt)
    r1 = ((pq - torch.tensor(dq, dtype=torch.float32)) ** 2).sum(1)
    r2 = ((pp - torch.tensor(dp, dtype=torch.float32)) ** 2).sum(1)
    # 按各自尺度归一，两个方程量纲不同
    s1 = float((torch.tensor(dq, dtype=torch.float32) ** 2).sum(1).mean()) + 1e-30
    s2 = float((torch.tensor(dp, dtype=torch.float32) ** 2).sum(1).mean()) + 1e-30
    return float((r1.mean() / s1 + r2.mean() / s2) / 2.0)


def build_system(name, ic_fn, mass, seed, steps=STEPS):
    rng = np.random.default_rng(seed)
    PT, VT = [], []
    for _ in range(N_TR + N_TE):
        p, v, m = ic_fn(rng)
        pt, vt = rollout(p, v, m, steps)
        PT.append(pt); VT.append(vt)
    return np.stack(PT), np.stack(VT)


def evaluate(name, PT, VT, mass, steps, seed):
    TRp, TRv = PT[:N_TR], VT[:N_TR]
    TEp, TEv = PT[N_TR:], VT[N_TR:]

    gen = train_force(TRp, TRv, mass, epochs=80, h=64, seed=seed)
    hnet = train_hnn_local(TRp, TRv, mass, mass.size, epochs=HNN_EPOCHS, seed=seed)

    # ---------------- 候选 ----------------
    R = np.random.default_rng(seed + 7)
    P, V, E = [], [], []
    for i in range(N_TE):
        for s in SIGMAS:
            cp, cv = rollout_nn(TEp[i][0], TEv[i][0], mass, gen, s, R, steps)
            P.append(cp); V.append(cv)
            E.append(float(np.sqrt(((cp - TEp[i]) ** 2).mean())))
    P, V = np.stack(P), np.stack(V)
    E = np.array(E).reshape(N_TE, K)

    # ---------------- critics ----------------
    def cons(pt, vt):
        ch = cons_channels(pt, vt, mass)
        return -(ch["angular"] + ch["momentum"])

    def hnn_drift(pt, vt):
        h = hnn_energy(hnet, pt, vt, mass, mass.size)
        return -float(np.abs(h - h[0]).max())

    def hnn_eq(pt, vt):
        return -hnn_eq_residual_series(hnet, pt, vt, mass)

    crits = [("守恒 (zero-shot)", cons),
             ("HNN 能量漂移(旧)", hnn_drift),
             ("HNN 方程残差(新)", hnn_eq)]

    # ---------------- 公平性诊断 ----------------
    print()
    print("=" * 80)
    print(f"[{name}]")
    print("=" * 80)
    print("  公平性诊断（真实轨迹 vs 生成轨迹的分离度，越大越有判别力）：")
    n = min(N_TE, 30)
    print(f"    {'critic':<20}{'真实(中位)':>16}{'生成(中位)':>16}{'分离度':>14}")
    for cn, fn in crits:
        a = np.median([fn(TEp[i], TEv[i]) for i in range(n)])
        b = np.median([fn(P[i * K], V[i * K]) for i in range(n)])   # 取 sigma=0 的候选
        sep = (a - b) / (abs(a) + 1e-300)
        print(f"    {cn:<20}{a:>16.5e}{b:>16.5e}{sep:>14.3f}")
    print("    （分离度 = (真实 − 生成)/|真实|；>0 表示真实轨迹得分更高）")

    # ---------------- 选择指标 ----------------
    S = {}
    for cn, fn in crits:
        S[cn] = np.array([[fn(P[i * K + j], V[i * K + j]) for j in range(K)]
                          for i in range(N_TE)])
    rand = float(np.median(E.mean(1))); orac = float(np.median(E.min(1)))
    print()
    print(f"  {'critic':<20}{'within-IC ρ':>14}{'IC内更优率':>12}"
          f"{'选中/随机':>11}{'选中/oracle':>12}")
    print("  " + "-" * 70)
    res = {}
    for cn, _ in crits:
        per = [spearman(S[cn][i], E[i]) for i in range(N_TE)]
        per = [x for x in per if not np.isnan(x)]
        wr = float(np.mean([E[i][int(np.argmax(S[cn][i]))] < np.median(E[i])
                            for i in range(N_TE)]))
        pk = float(np.median([E[i][int(np.argmax(S[cn][i]))] for i in range(N_TE)]))
        res[cn] = (float(np.mean(per)), wr, pk / rand, pk / orac)
        print(f"  {cn:<20}{res[cn][0]:>+14.4f}{wr:>11.1%}{pk/rand:>10.3f}x{pk/orac:>11.2f}x")
    print(f"  {'[参照] 随机':<20}{'—':>14}{'50.0%':>12}{1.0:>10.3f}x{rand/orac:>11.2f}x")
    print(f"  {'[参照] oracle':<20}{'—':>14}{'100%':>12}{orac/rand:>10.3f}x{1.0:>11.2f}x")
    return res


def main():
    t0 = time.time()
    print("=" * 80)
    print("实验 K：HNN 方程残差 critic vs 守恒 critic（标准基准）")
    print("=" * 80)

    PT, VT = build_system("fig8", lambda r: system_fig8(r, jitter=0.02), np.ones(3), 0)
    evaluate("figure-eight 三体（N=3, 2D）", PT, VT, np.ones(3), STEPS, seed=0)

    PT, VT = build_system("kepler", lambda r: system_kepler(r), np.ones(2), 1)
    evaluate("Kepler 二体（N=2, 2D）", PT, VT, np.ones(2), STEPS, seed=1)

    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
