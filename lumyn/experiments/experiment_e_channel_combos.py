"""实验 E：确定守恒 critic 的**最优通道组合**（只做打分，不训练，因此很快）。

背景（实验 D 的结果）
---------------------
现行 `cons_drift` 是 energy/angular/momentum/com 四通道**等权相加**。实测：
  - `angular` 单通道（更优率 88.7%）**优于**四通道等权（84.0%）
  - `com` 单通道更优率仅 25.3%，**比随机还差** —— 它在主动拖后腿
  - 在强生成器上，`angular` 单通道恰好达到 oracle（0.057×），
    而四通道等权是 0.070×，**等权相加使性能退化 23%**

因此必须重新选择组合，而不是沿用任意加权。
"""
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np
from lumyn.experiments.experiment_a1_critic_quality import (
    prior_ic, rollout_verlet,
)
from lumyn.experiments.experiment_a2_neural_generator import train_force_model
from lumyn.experiments.experiment_d_channels_and_generators import (
    channels, CH, K, SIGMAS, N, MASS, DT, F_TRAIN, F_TEST, spearman,
    rollout_net, train_meanfield,
)

N_IC = 120


def build(n_ic, fam, net, seed):
    rng = np.random.default_rng(seed)
    P, V, E = [], [], []
    for _ in range(n_ic):
        p, v = prior_ic(rng.uniform(*fam), rng)
        tt, _ = rollout_verlet(p, v, np.ones(N) * MASS)
        for s in SIGMAS:
            cp, cv = rollout_net(p, v, net, s, rng)
            P.append(cp); V.append(cv); E.append(float(np.sqrt(((cp - tt) ** 2).mean())))
    return np.stack(P), np.stack(V), np.array(E).reshape(n_ic, K)


def evaluate(combos, P, V, E):
    n = len(E)
    rand = float(np.median(E.mean(1))); orac = float(np.median(E.min(1)))
    rows = []
    for label, chs in combos:
        S = np.zeros((n, K))
        for i in range(n):
            for j in range(K):
                ch = channels(P[i * K + j], V[i * K + j])
                S[i, j] = -sum(ch[c] for c in chs)
        per = [spearman(S[i], E[i]) for i in range(n)]
        per = [r for r in per if not np.isnan(r)]
        wr = float(np.mean([E[i][int(np.argmax(S[i]))] < np.median(E[i]) for i in range(n)]))
        pk = float(np.median([E[i][int(np.argmax(S[i]))] for i in range(n)]))
        rows.append((label, float(np.mean(per)), wr, pk, pk / rand, pk / orac))
    return rows, rand, orac


def main():
    t0 = time.time()
    rng = np.random.default_rng(1)
    Ptr, Vtr = [], []
    for _ in range(320):
        p, v = prior_ic(rng.uniform(*F_TRAIN), rng)
        pt, vt = rollout_verlet(p, v, np.ones(N) * MASS)
        Ptr.append(pt); Vtr.append(vt)
    Ptr, Vtr = np.stack(Ptr), np.stack(Vtr)
    M = np.ones(N) * MASS
    gA = train_force_model(Ptr[:120], Vtr[:120], M, epochs=12, h=32, lr=3e-3)
    gB = train_force_model(Ptr, Vtr, M, epochs=120, h=64, lr=2e-3)
    print(f"生成器就绪（耗时 {time.time()-t0:.0f}s）")

    combos = [
        ("energy", ("energy",)),
        ("angular", ("angular",)),
        ("momentum", ("momentum",)),
        ("com", ("com",)),
        ("E+A", ("energy", "angular")),
        ("E+M", ("energy", "momentum")),
        ("A+M", ("angular", "momentum")),
        ("A+M+E", ("angular", "momentum", "energy")),
        ("E+A+M+C（现行）", CH),
        ("A+M+C", ("angular", "momentum", "com")),
    ]

    for gname, gnet in (("G_A 成对GNN(中预算)", gA), ("G_B 成对GNN(强预算)", gB)):
        print()
        print("=" * 78)
        print(f"[{gname}] 通道组合排名")
        print("=" * 78)
        for split, fam, seed in (("ID", F_TRAIN, 202), ("OOD", F_TEST, 303)):
            P, V, E = build(N_IC, fam, gnet, seed)
            rows, rand, orac = evaluate(combos, P, V, E)
            rows.sort(key=lambda r: r[3])          # 按选中误差升序
            print(f"\n   [{split}] 候选 RMSE 中位={np.median(E):.4f}  "
                  f"IC内 max/min 中位={np.median(E.max(1)/(E.min(1)+1e-12)):.2f}x  "
                  f"（oracle/随机 = {orac/rand:.3f}x）")
            print(f"   {'组合':<18}{'within-IC ρ':>13}{'更优率':>9}"
                  f"{'选中误差':>12}{'/随机':>9}{'/oracle':>10}")
            print("   " + "-" * 72)
            for label, rho, wr, pk, rr, ro in rows:
                mark = "  <- 最优" if pk == rows[0][3] else ""
                print(f"   {label:<18}{rho:>+13.4f}{wr:>8.1%}{pk:>12.5f}"
                      f"{rr:>8.3f}x{ro:>9.2f}x{mark}")
        print(f"\n   （耗时累计 {time.time()-t0:.0f}s）")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
