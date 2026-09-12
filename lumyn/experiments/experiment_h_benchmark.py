"""实验 H：基准表（多随机种子 + 95% CI），含文献级 baseline。

任务定义（可复现）
------------------
给定同一初始条件下由**学习式生成器**产生的 K 个候选轨迹，
用 critic 打分挑出最接近真实解的那个（真值可算，因为生成器模拟的是已知物理）。

指标
----
  within-IC ρ  ：同一 IC 内 critic 分数与真实误差的 Spearman 秩相关
                 （**必须用 within-IC**：pooled 会被 IC 间差异主导而严重高估，
                  见 PAPER_STATUS §二·九·一）
  IC 内更优率  ：critic 选中的候选是否优于该 IC 的候选中位数（0.5 = 无能力）
  选中/随机    ：选中候选的误差 / 随机挑选的期望误差
  选中/oracle  ：选中候选的误差 / 逐 IC 最优候选的误差

critic 清单
-----------
  守恒 (zero-shot)      「angular + momentum」通道，**零标签**
  HNN 能量残差           Hamiltonian Neural Network（Greydanus et al. 2019）学到的
                        能量 $H_\\theta$ 沿轨迹的漂移 —— **文献级学习式 baseline**
  L1 绝对二分类          真值轨迹 vs 候选，BCE（朴素监督）
  L2 成对排序损失        同 IC 内 A 优于 B（对齐评测目标的强监督）
  随机 / oracle          参照

符号约定：所有 critic 分数**越高越优**，方向由显式符号表声明，
不得由名字推断（该 bug 曾在实验 C 中把最强 baseline 变成最差）。

⚠️ HNN baseline 已被排除出结论表（保留代码与数字以备审查）
----------------------------------------------------------------
本实验中的 HNN 能量残差更优率仅 36.2%±6.1%（ID）、31.2%±3.3%（OOD），**低于随机**。
经 `diagnose_hnn_fairness.py` 诊断，**这是实现未训练充分，不是方法本身的问题**：

    epochs=300 : 哈密顿方程残差 dq 0.2394 | dp 0.9992
                 H_θ 在**真实**轨迹上的相对漂移 7.2261e-02
                 H_θ 在**生成**轨迹上的相对漂移 7.2467e-02
                 生成/真实 = 1.003      ← 两类轨迹上同样不守恒
    epochs=1200: 比值 1.008（几乎无改善）

`dp` 残差 ≈ 1.0 说明该 HNN 基本没学到动量方程；H_θ 的漂移在真实与生成轨迹上
**完全相同**，即该 critic 只是在测量自身的训练误差，**判别力为零**。

根因：让一个普通 MLP 在 36 维状态空间、约 4800 个样本上学出 6 体 3D 的 1/r² 力场，
并非 HNN 的适用场景（原论文用于低维系统）。
**因此本表不得报告「我们击败了 HNN」** —— 那会是一个稻草人。
一个公平的 HNN 对照需要在低维系统上进行，本实验未建立该设置。
"""
import sys, os, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

from lumyn.experiments.experiment_a1_critic_quality import (   # noqa: E402
    prior_ic, rollout_verlet, raw_tensor, score_net, DeepSetsCritic,
)
from lumyn.experiments.experiment_a2_neural_generator import train_force_model  # noqa: E402
from lumyn.experiments.experiment_d_channels_and_generators import (   # noqa: E402
    channels, rollout_net, N, MASS, DT, SOFT, G, F_TRAIN, F_TEST, spearman,
    train_abs_binary, train_pairwise,
)

K = 8
SIGMAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.6, 1.0)
N_TR_IC = 120
N_TE_IC = 80
SEEDS = (0, 1, 2, 3, 4)


# ============================================================
# Hamiltonian Neural Network（Greydanus et al. 2019）
# ============================================================
class HNN(nn.Module):
    """学一个标量哈密顿量 H(q, p)，使其满足哈密顿方程。

        dq/dt =  ∂H/∂p
        dp/dt = −∂H/∂q

    训练目标就是这两个残差。它是「用学习方式尊重物理」的代表性方法，
    与本文「直接用已知守恒律」构成正面对照。
    """

    def __init__(self, dim, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 1),
        )
        self.dim = dim

    def H(self, q, p):
        """q, p: (B, D) → (B, 1)"""
        return self.net(torch.cat([q, p], dim=-1))

    def time_deriv(self, q, p):
        q = q.clone().detach().requires_grad_(True)
        p = p.clone().detach().requires_grad_(True)
        H = self.H(q, p).sum()
        gq, gp = torch.autograd.grad(H, [q, p], create_graph=True)
        return gp, -gq          # dq/dt, dp/dt


def train_hnn(P, V, mass, epochs=300, h=128, lr=1e-3, batch=128, seed=0):
    """用真实轨迹的哈密顿方程残差训练。P,V: (B,T,N,3)。"""
    torch.manual_seed(seed)
    B, T, Nn, _ = P.shape
    m_np = np.asarray(mass, dtype=np.float64)
    Q = torch.tensor(P.reshape(B, T, Nn * 3), dtype=torch.float32)
    # 广义动量 p = m v（先按 numpy 广播，避免 ndarray 与 Tensor 混算）
    P_mom = (V * m_np[None, None, :, None]).astype(np.float64)
    Pt = torch.tensor(P_mom.reshape(B, T, Nn * 3), dtype=torch.float32)
    dq = (Q[:, 1:] - Q[:, :-1]) / DT
    dp = (Pt[:, 1:] - Pt[:, :-1]) / DT
    Qa = Q[:, :-1].reshape(-1, Nn * 3)
    Pa = Pt[:, :-1].reshape(-1, Nn * 3)
    DQ = dq.reshape(-1, Nn * 3)
    DP = dp.reshape(-1, Nn * 3)

    net = HNN(Nn * 3, h=h).to(torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(Qa)
    for ep in range(epochs):
        idx = torch.randperm(n)[:batch * 4]
        pq, pp = net.time_deriv(Qa[idx], Pa[idx])
        # 尺度归一化：两个方程的残差量纲不同，各自按自身尺度归一
        loss = (((pq - DQ[idx]) ** 2).mean() / (DQ ** 2).mean()
                + ((pp - DP[idx]) ** 2).mean() / (DP ** 2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def hnn_energy(net, pos_t, vel_t, mass):
    """用学到的 H 计算轨迹上的哈密顿量序列。"""
    Tlen = len(pos_t)
    q = torch.tensor(pos_t.reshape(Tlen, -1), dtype=torch.float32)
    p = torch.tensor((vel_t * mass[None, :, None]).reshape(Tlen, -1),
                     dtype=torch.float32)
    with torch.no_grad():
        return net.H(q, p).squeeze(-1).numpy().astype(np.float64)


# ============================================================
# 数据
# ============================================================
def make_block(n_ic, fam, net, seed):
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


def evaluate(S, E):
    """S: {critic: (n_ic, K)}; E: (n_ic, K)。返回各指标。"""
    n = len(E)
    rand = float(np.median(E.mean(1))); orac = float(np.median(E.min(1)))
    out = {}
    for cn, sc in S.items():
        per = [spearman(sc[i], E[i]) for i in range(n)]
        per = [r for r in per if not np.isnan(r)]
        wr = float(np.mean([E[i][int(np.argmax(sc[i]))] < np.median(E[i])
                            for i in range(n)]))
        pk = float(np.median([E[i][int(np.argmax(sc[i]))] for i in range(n)]))
        out[cn] = dict(rho=float(np.mean(per)) if per else float("nan"),
                       win=wr, ratio_rand=pk / rand, ratio_oracle=pk / orac)
    return out, rand, orac


def main():
    t0 = time.time()
    results = {}          # (critic, split) -> list of metric dicts
    meta = []
    for seed in SEEDS:
        ts = time.time()
        rng = np.random.default_rng(1)
        Ptr, Vtr = [], []
        for _ in range(240):
            p, v = prior_ic(rng.uniform(*F_TRAIN), rng)
            pt, vt = rollout_verlet(p, v, np.ones(N) * MASS)
            Ptr.append(pt); Vtr.append(vt)
        Ptr, Vtr = np.stack(Ptr), np.stack(Vtr)
        mass = np.ones(N) * MASS
        gen = train_force_model(Ptr[:120], Vtr[:120], mass, epochs=12, h=32,
                                lr=3e-3, seed=seed)

        TR = make_block(N_TR_IC, F_TRAIN, gen, 100 + seed)
        blocks = {"ID": make_block(N_TE_IC, F_TRAIN, gen, 200 + seed),
                  "OOD-族": make_block(N_TE_IC, F_TEST, gen, 300 + seed)}

        n1, mu1, sd1 = train_abs_binary(TR[3], TR[4], TR[0], TR[1], seed=seed)
        n2, mu2, sd2 = train_pairwise(TR[0], TR[1], TR[2], seed=seed)
        hnet = train_hnn(TR[3], TR[4], mass, seed=seed)
        meta.append(dict(seed=seed, secs=round(time.time() - ts, 1)))

        def mk(net_, mu_, sd_, sign):
            def f(pt, vt):
                return sign * float(score_net(net_, mu_, sd_,
                                              np.stack([raw_tensor(pt, vt)]))[0])
            return f
        # 符号显式声明（实验 C 的教训）
        def cons(pt, vt):
            ch = channels(pt, vt)
            return -(ch["angular"] + ch["momentum"])

        def hnn_score(pt, vt):
            h = hnn_energy(hnet, pt, vt, mass)
            return -float(np.abs(h - h[0]).max())

        CRITICS = [("守恒 (zero-shot)", cons),
                   ("HNN 能量残差", hnn_score),
                   ("L1 绝对二分类", mk(n1, mu1, sd1, +1.0)),
                   ("L2 成对排序损失", mk(n2, mu2, sd2, +1.0))]

        for split, (P, V, E, _, _) in blocks.items():
            n_ic = len(E)
            S = {}
            for cn, fn in CRITICS:
                S[cn] = np.array([[fn(P[i * K + j], V[i * K + j]) for j in range(K)]
                                  for i in range(n_ic)])
            m, rand, orac = evaluate(S, E)
            for cn, d in m.items():
                d = dict(d); d.update(rand=rand, orac=orac)
                results.setdefault((cn, split), []).append(d)
        print(f"  seed {seed} 完成（{time.time()-ts:.0f}s）", flush=True)

    # ---------------- 汇总 ----------------
    def ci(vals):
        vals = np.asarray(vals, dtype=float)
        return float(np.mean(vals)), float(vals.std(ddof=1) / np.sqrt(len(vals)) * 1.96)

    for split in ("ID", "OOD-族"):
        print()
        print("=" * 88)
        print(f"[{split}] 基准表（{len(SEEDS)} 个随机种子，均值 ± 95% CI）")
        print("=" * 88)
        print(f"  {'critic':<20}{'within-IC ρ':>20}{'IC内更优率':>16}"
              f"{'选中/随机':>16}{'选中/oracle':>16}")
        print("  " + "-" * 84)
        rows = []
        for cn, _ in results:
            if _ != split:
                continue
            ds = results[(cn, split)]
            r_m, r_e = ci([d["rho"] for d in ds])
            w_m, w_e = ci([d["win"] for d in ds])
            a_m, a_e = ci([d["ratio_rand"] for d in ds])
            o_m, o_e = ci([d["ratio_oracle"] for d in ds])
            rows.append((a_m, cn, r_m, r_e, w_m, w_e, a_m, a_e, o_m, o_e))
        rows.sort()
        for _, cn, r_m, r_e, w_m, w_e, a_m, a_e, o_m, o_e in rows:
            print(f"  {cn:<20}{r_m:>+11.4f}±{r_e:<7.4f}{w_m:>9.1%}±{w_e:<5.1%}"
                  f"{a_m:>10.3f}±{a_e:<5.3f}{o_m:>10.2f}±{o_e:<5.2f}")
        d0 = results[("守恒 (zero-shot)", split)][0]
        print(f"  {'[参照] 随机':<20}{'—':>20}{'50.0%':>16}"
              f"{1.0:>16.3f}{d0['rand']/d0['orac']:>16.2f}")
        print(f"  {'[参照] oracle':<20}{'—':>20}{'100%':>16}"
              f"{d0['rand']/d0['orac']:>16.3f}{1.0:>16.2f}")

    out = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                       "results", "experiment_h_benchmark.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": dict(seeds=list(SEEDS), K=K, n_train_ic=N_TR_IC,
                                  n_test_ic=N_TE_IC, sigmas=list(SIGMAS)),
                   "runs": {f"{c}|{s}": v for (c, s), v in results.items()},
                   "meta": meta}, f, ensure_ascii=False, indent=1)
    print(f"\n耗时 {time.time()-t0:.0f}s  [写出] {out}")


if __name__ == "__main__":
    main()
