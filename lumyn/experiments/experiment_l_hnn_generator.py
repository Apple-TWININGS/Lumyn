"""实验 L：换一个**架构根本不同**的生成器（HNN 作为生成器），结论是否仍然成立？

动机
----
审稿人必问：全部结论是否只对「成对力模型 + 有限训练预算」这一个生成器成立？
此前用平均场 MLP 试过，但它质量过差（单步加速度相对误差 0.81、IC 内候选差异仅
1.16×、oracle/随机 = 1.09×），**几乎没有选择空间，不提供任何信息**。

本实验改用 **HNN 作为生成器**（Greydanus et al. 2019）。它与成对力模型有
**根本不同的归纳偏置**：

  · 成对力模型：预测逐对力，逐点破坏/满足牛顿第三定律；
  · HNN 生成器：学一个黑箱标量 H(q,p)，由哈密顿方程导出向量场。
    它对坐标**不具平移/旋转不变性**（不是成对的），因此失效模式不同 ——
    这是对论文「噪声地板」论证能否外推的直接检验。

同时报告生成轨迹的守恒指纹，与成对力模型对比。
"""
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

from lumyn.experiments.experiment_j_standard_benchmarks import (
    system_fig8, system_kepler, rollout, cons_channels, spearman, K, DT,
    train_hnn_local, hnn_eq_residual,
)
from lumyn.experiments.experiment_a1_critic_quality import raw_tensor, score_net, DeepSetsCritic

STEPS = 300
N_TR, N_TE = 100, 50
HNN_EPOCHS = 2000
SIGMAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.6, 1.0)


def rollout_hnn(net, pos, vel, mass, steps, sigma, rng, dt=DT):
    """用 HNN 学到的向量场推进。sigma>0 时在向量场上注入过程噪声。"""
    Nn = pos.shape[0]
    q = torch.tensor(pos.reshape(1, -1), dtype=torch.float32)
    p = torch.tensor((vel * np.asarray(mass)[None, :, None]).reshape(1, -1),
                     dtype=torch.float32)
    ps, vs = [pos.copy()], [vel.copy()]
    for _ in range(steps):
        with torch.enable_grad():
            dq, dp = net.time_deriv(q, p)
        dq = dq.detach(); dp = dp.detach()
        if sigma > 0:
            sc = float(dq.abs().mean() + 1e-8)
            dq = dq + torch.tensor(rng.normal(0, sigma, dq.shape) * sc,
                                   dtype=torch.float32)
            dp = dp + torch.tensor(rng.normal(0, sigma, dp.shape) * sc,
                                   dtype=torch.float32)
        q = q + dt * dq
        p = p + dt * dp
        ps.append(q.numpy().reshape(Nn, 3).astype(np.float64))
        vs.append((p.numpy().reshape(Nn, 3) / np.asarray(mass)[:, None])
                  .astype(np.float64))
    return np.stack(ps), np.stack(vs)


# ---------------------------------------------------------------- 学习式 baseline
def _prep(P, V, mu=None, sd=None):
    X = np.stack([raw_tensor(P[i], V[i]) for i in range(len(P))])
    if mu is None:
        mu = X.mean(axis=(0, 1, 2), keepdims=True)
        sd = X.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    return torch.tensor((X - mu) / sd, dtype=torch.float32), mu, sd


def _net(seed=0):
    torch.manual_seed(seed)
    return DeepSetsCritic(h=128).to(torch.float32)


def train_pairwise(P, V, E, epochs=100, seed=0, batch=32):
    X, mu, sd = _prep(P, V)
    n = len(E)
    X = X.reshape(n, K, *X.shape[1:])
    Ee = torch.tensor(E, dtype=torch.float32)
    net = _net(seed)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    for _ in range(epochs):
        idx = torch.randperm(n)[:batch]
        s = net(X[idx].reshape(-1, *X.shape[2:])).reshape(len(idx), K)
        eb = Ee[idx]
        loss = 0.0
        for a, b in pairs:
            loss = loss + nn.functional.binary_cross_entropy_with_logits(
                s[:, a] - s[:, b], (eb[:, a] < eb[:, b]).float())
        (loss / len(pairs)).backward()
        opt.step(); opt.zero_grad()
    net.eval()
    return net, mu, sd


def main():
    t0 = time.time()
    print("=" * 84)
    print("实验 L：HNN 作为**生成器**（架构不同于成对力模型）")
    print("=" * 84)

    for name, ic_fn, mass, seed in (
            ("figure-eight 三体（N=3, 2D）", lambda r: system_fig8(r, jitter=0.02),
             np.ones(3), 0),
            ("Kepler 二体（N=2, 2D）", lambda r: system_kepler(r), np.ones(2), 1)):

        rng = np.random.default_rng(seed)
        PT, VT = [], []
        for _ in range(N_TR + N_TE):
            p, v, m = ic_fn(rng)
            pt, vt = rollout(p, v, m, STEPS)
            PT.append(pt); VT.append(vt)
        PT, VT = np.stack(PT), np.stack(VT)
        TRp, TRv = PT[:N_TR], VT[:N_TR]
        TEp, TEv = PT[N_TR:], VT[N_TR:]

        print()
        print("-" * 84)
        print(f"[{name}]")
        print("-" * 84)
        hnet = train_hnn_local(TRp, TRv, mass, mass.size, epochs=HNN_EPOCHS, seed=seed)
        r1, r2 = hnn_eq_residual(hnet, TEp, TEv, mass)
        print(f"  HNN 生成器：哈密顿方程残差（真实轨迹上）dq {r1:.4f} | dp {r2:.4f}")

        # ---------------- 候选 ----------------
        R = np.random.default_rng(seed + 21)
        P, V, E = [], [], []
        for i in range(N_TE):
            for s in SIGMAS:
                cp, cv = rollout_hnn(hnet, TEp[i][0], TEv[i][0], mass, STEPS, s, R)
                P.append(cp); V.append(cv)
                E.append(float(np.sqrt(((cp - TEp[i]) ** 2).mean())))
        P, V = np.stack(P), np.stack(V)
        E = np.array(E).reshape(N_TE, K)
        print(f"  候选 RMSE 中位 = {np.median(E):.5f}  "
              f"IC 内 max/min 中位 = {np.median(E.max(1)/(E.min(1)+1e-12)):.2f}x  "
              f"oracle/随机 = {np.median(E.min(1))/np.median(E.mean(1)):.3f}x")

        # ---------------- 指纹对比 ----------------
        ch_gen = np.median([cons_channels(P[i * K], V[i * K], mass)["angular"]
                            for i in range(N_TE)])
        ch_true = np.median([cons_channels(TEp[i], TEv[i], mass)["angular"]
                             for i in range(N_TE)])
        print(f"  守恒指纹：角动量分离度 = {ch_gen/(ch_true+1e-300):.3e}"
              f"（真实 {ch_true:.3e} → 生成 {ch_gen:.3e}）")

        # ---------------- critics ----------------
        # train_pairwise 内部按 (n_ic*K, ...) 取特征后自行 reshape 成 (n_ic, K, ...)，
        # 因此这里必须传**扁平**的 P / V（长度 N_TE*K），不能再预先 reshape。
        n2, mu2, sd2 = train_pairwise(P, V, E, seed=seed)

        def cons(pt, vt):
            ch = cons_channels(pt, vt, mass)
            return -(ch["angular"] + ch["momentum"])

        def pw(pt, vt):
            return float(score_net(n2, mu2, sd2,
                                   np.stack([raw_tensor(pt, vt)]))[0])

        crits = [("守恒 (zero-shot)", cons), ("L2 成对排序(监督)", pw)]
        S = {cn: np.zeros((N_TE, K)) for cn, _ in crits}
        for i in range(N_TE):
            for j in range(K):
                pt, vt = P[i * K + j], V[i * K + j]
                for cn, fn in crits:
                    S[cn][i, j] = fn(pt, vt)

        rand = float(np.median(E.mean(1))); orac = float(np.median(E.min(1)))
        print()
        print(f"  {'critic':<22}{'within-IC ρ':>14}{'IC内更优率':>12}"
              f"{'选中/随机':>11}{'选中/oracle':>12}")
        print("  " + "-" * 72)
        for cn, _ in crits:
            per = [spearman(S[cn][i], E[i]) for i in range(N_TE)]
            per = [x for x in per if not np.isnan(x)]
            wr = float(np.mean([E[i][int(np.argmax(S[cn][i]))] < np.median(E[i])
                                for i in range(N_TE)]))
            pk = float(np.median([E[i][int(np.argmax(S[cn][i]))] for i in range(N_TE)]))
            print(f"  {cn:<22}{np.mean(per):>+14.4f}{wr:>11.1%}"
                  f"{pk/rand:>10.3f}x{pk/orac:>11.2f}x")
        print(f"  {'[参照] 随机':<22}{'—':>14}{'50.0%':>12}{1.0:>10.3f}x{rand/orac:>11.2f}x")
        print(f"  {'[参照] oracle':<22}{'—':>14}{'100%':>12}{orac/rand:>10.3f}x{1.0:>11.2f}x")

    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
