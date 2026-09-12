"""实验 F：用**修正后的守恒 critic** 重跑主对照。

动机
----
实验 D 得出「在强生成器 G_B 上，守恒 critic 略逊于学习式 baseline 约 10%」。
但 D 用的守恒 critic 是**四通道等权相加**（energy+angular+momentum+com）。
实验 E 的通道消融显示该定义**次优**：
  - `com` 单通道更优率仅 16.7–24.2%，远低于随机（在挑最差候选）
  - `energy` 单通道仅 55.8–60.0%
  - 而 `angular`/`momentum` 单通道达 88–100%
  - A+M（angular+momentum）在 4 组设置中 3 组优于现行定义，G_B 上 0.056× vs 0.067×

因此「守恒略逊」这一结论可能是 critic 定义不当造成的假象。本实验用修正后的
critic（A+M）重跑，判定该结论是否成立。
"""
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

from lumyn.experiments.experiment_a1_critic_quality import (   # noqa: E402
    prior_ic, rollout_verlet, raw_tensor, score_net,
)
from lumyn.experiments.experiment_a2_neural_generator import train_force_model  # noqa: E402
from lumyn.experiments.experiment_d_channels_and_generators import (   # noqa: E402
    channels, CH, K, SIGMAS, N, MASS, DT, F_TRAIN, F_TEST, spearman,
    rollout_net, train_abs_binary, train_pairwise, train_rel_reg,
)

N_TR, N_TE = 200, 120
COMBOS = {
    "守恒 A+M（修正后）": ("angular", "momentum"),
    "守恒 angular 单通道": ("angular",),
    "守恒 四通道等权（旧）": CH,
}


def build(n_ic, fam, net, seed):
    rng = np.random.default_rng(seed)
    P, V, E, TP, TV = [], [], [], [], []
    for _ in range(n_ic):
        p, v = prior_ic(rng.uniform(*fam), rng)
        tt, tv = rollout_verlet(p, v, np.ones(N) * MASS)
        TP.append(tt); TV.append(tv)
        for s in SIGMAS:
            cp, cv = rollout_net(p, v, net, s, rng)
            P.append(cp); V.append(cv); E.append(float(np.sqrt(((cp - tt) ** 2).mean())))
    return (np.stack(P), np.stack(V), np.array(E).reshape(n_ic, K),
            np.stack(TP), np.stack(TV))


def main():
    t0 = time.time()
    print("=" * 80)
    print("准备强生成器 G_B 与数据")
    print("=" * 80)
    rng = np.random.default_rng(1)
    Ptr = []; Vtr = []
    for _ in range(320):
        p, v = prior_ic(rng.uniform(*F_TRAIN), rng)
        pt, vt = rollout_verlet(p, v, np.ones(N) * MASS)
        Ptr.append(pt); Vtr.append(vt)
    Ptr, Vtr = np.stack(Ptr), np.stack(Vtr)
    gB = train_force_model(Ptr, Vtr, np.ones(N) * MASS, epochs=120, h=64, lr=2e-3)
    TR = build(N_TR, F_TRAIN, gB, 101)
    ID = build(N_TE, F_TRAIN, gB, 202)
    OD = build(N_TE, F_TEST, gB, 303)
    print(f"   完成（{time.time()-t0:.0f}s）  ID 候选 RMSE 中位={np.median(ID[2]):.5f}  "
          f"IC内 max/min 中位={np.median(ID[2].max(1)/(ID[2].min(1)+1e-12)):.1f}x")

    print()
    print("训练学习式 baseline（全部只用 train 切分）")
    n1, mu1, sd1 = train_abs_binary(TR[3], TR[4], TR[0], TR[1])
    n2, mu2, sd2 = train_pairwise(TR[0], TR[1], TR[2])
    n4, mu4, sd4 = train_rel_reg(TR[0], TR[1], TR[2])
    print(f"   完成（{time.time()-t0:.0f}s）")

    def mk(net_, mu_, sd_, sign):
        def f(pt, vt):
            return sign * float(score_net(net_, mu_, sd_,
                                          np.stack([raw_tensor(pt, vt)]))[0])
        return f

    for split_name, TE in (("ID", ID), ("OOD-族", OD)):
        P, V, E = TE[0], TE[1], TE[2]
        n = len(E)
        crits = []
        for label, chs in COMBOS.items():
            def mc(chs=chs):
                def f(pt, vt):
                    ch = channels(pt, vt)
                    return -sum(ch[c] for c in chs)
                return f
            crits.append((label, mc()))
        crits += [("L1 绝对二分类", mk(n1, mu1, sd1, +1.0)),
                  ("L2 成对排序", mk(n2, mu2, sd2, +1.0)),
                  ("L4 IC内相对回归", mk(n4, mu4, sd4, -1.0))]

        # 融合：用**修正后**的 A+M 守恒 + L2 + L4
        by = dict(crits)
        ADOPTED = "守恒 A+M（修正后）"
        def feat(pt, vt):
            return [by[ADOPTED](pt, vt), by["L2 成对排序"](pt, vt),
                    by["L4 IC内相对回归"](pt, vt)]
        F, y = [], []
        for i in range(len(TR[2])):
            med = np.median(TR[2][i])
            for j in range(K):
                F.append(feat(TR[0][i * K + j], TR[1][i * K + j]))
                y.append(1.0 if TR[2][i][j] <= med else 0.0)
        Ft = torch.tensor(np.array(F), dtype=torch.float32)
        yt = torch.tensor(np.array(y), dtype=torch.float32)
        wb = torch.zeros(4, dtype=torch.float32, requires_grad=True)
        of = torch.optim.Adam([wb], lr=0.05)
        for _ in range(600):
            of.zero_grad()
            nn.functional.binary_cross_entropy_with_logits(Ft @ wb[:3] + wb[3], yt).backward()
            of.step()
        W = wb.detach().numpy()

        def fus(pt, vt):
            return float(np.dot(W[:3], feat(pt, vt)) + W[3])
        crits.append(("L5 最优融合(A+M)", fus))

        print()
        print("=" * 80)
        print(f"[{split_name}] 修正后的守恒 critic vs 学习式 baseline")
        print("=" * 80)
        S = {}
        for cn, fn in crits:
            S[cn] = np.array([[fn(P[i * K + j], V[i * K + j]) for j in range(K)]
                              for i in range(n)])
        rand = float(np.median(E.mean(1))); orac = float(np.median(E.min(1)))
        rows = []
        for cn, _ in crits:
            per = [spearman(S[cn][i], E[i]) for i in range(n)]
            per = [r for r in per if not np.isnan(r)]
            wr = float(np.mean([E[i][int(np.argmax(S[cn][i]))] < np.median(E[i])
                                for i in range(n)]))
            pk = float(np.median([E[i][int(np.argmax(S[cn][i]))] for i in range(n)]))
            rows.append((cn, float(np.mean(per)), wr, pk))
        rows.sort(key=lambda r: r[3])
        print(f"   融合权重: A+M守恒 {W[0]:+.3f} | L2 {W[1]:+.3f} | L4 {W[2]:+.3f}")
        print(f"   {'critic':<24}{'within-IC ρ':>13}{'更优率':>9}"
              f"{'选中误差':>12}{'/随机':>9}{'/oracle':>10}")
        print("   " + "-" * 76)
        for cn, rho, wr, pk in rows:
            mark = "  <- 最优" if pk == rows[0][3] else ""
            print(f"   {cn:<24}{rho:>+13.4f}{wr:>8.1%}{pk:>12.5f}"
                  f"{pk/rand:>8.3f}x{pk/orac:>9.2f}x{mark}")
        print(f"   {'[参照] 随机':<24}{'—':>13}{'50.0%':>9}{rand:>12.5f}"
              f"{1.0:>8.3f}x{rand/orac:>9.2f}x")
        print(f"   {'[参照] oracle':<24}{'—':>13}{'100%':>9}{orac:>12.5f}"
              f"{orac/rand:>8.3f}x{1.0:>9.2f}x")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
