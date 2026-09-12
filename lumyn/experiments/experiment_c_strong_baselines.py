"""实验 C：把学习式 baseline 做到**尽可能强**，再检验零样本守恒 critic 是否仍不劣。

为什么必须做这个
----------------
实验 B 的结论是「守恒 critic 不劣于训练过的 critic」。但那个「训练过的 critic」是用
**绝对标签**（真值轨迹 vs 候选轨迹，二分类）训练的，而评测任务是**相对选择**
（同一个 IC 的 8 个候选里挑最好的）。训练目标与评测任务不匹配 ——
这是一个**低估 baseline** 的稻草人风险。本实验把它补掉。

这里加入四种「称职实践者会用的」强 baseline：

  L1  二分类（绝对）        —— 实验 B 的基线，作为参照
  L2  **成对排序损失**      —— 直接优化「同 IC 内 A 优于 B」，与选择任务同构
  L3  绝对误差回归          —— 回归 log(RMSE)
  L4  **IC 内相对回归**     —— 回归 log(RMSE / 该 IC 的候选中位误差)，去掉 IC 间差异
  L5  **最优融合**          —— 各 critic 分数经逻辑回归学习权重
  L6  列表损失（ListNet 简化）—— 对同一 IC 的 8 个候选做 softmax 分布的交叉熵

若守恒 critic（零样本、无标签）在这批强 baseline 面前仍不劣，结论才站得住。
若某个 baseline 明显更优 → 如实报告「守恒只是廉价代理，劣于监督排序模型」。

评测指标
--------
- **within-IC 秩相关**（实验 B 已证明 pooled 指标会严重高估；见 §二·九）
- **IC 内更优率**：critic 选中的候选是否优于该 IC 的候选中位数（0.5 = 无能力）
- **选择误差**：critic 选中候选的误差，相对随机 / oracle
"""
import sys, os, json, time, itertools
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

N, STEPS, DT, SOFT, G = 6, 40, 0.01, 0.05, 1.0
MASS = 0.3

F_TRAIN = (0.0, 0.3)
F_TEST = (0.5, 0.8)

from lumyn.experiments.experiment_a1_critic_quality import (   # noqa: E402
    prior_ic, rollout_verlet, cons_drift, raw_tensor,
    train_critic, score_net, auroc, boot_ci, DeepSetsCritic,
)
from lumyn.experiments.experiment_a2_neural_generator import (   # noqa: E402
    NeuralForce, train_force_model,
)

N_TRAIN_IC = 600
N_ID_IC = 200
N_OOD_IC = 200
K = 8
SIGMAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.6, 1.0)


# ============================================================
# 工具
# ============================================================
def spearman(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(1, len(a) + 1)
        return r
    rx, ry = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
    return float((rx * ry).sum() / (np.sqrt((rx ** 2).sum() * (ry ** 2).sum()) + 1e-30))


def rms(traj, ref):
    return float(np.sqrt(((traj - ref) ** 2).mean()))


def rollout_noisy(pos, vel, net, sigma, rng):
    p = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)
    v = torch.tensor(vel, dtype=torch.float32).unsqueeze(0)
    m = torch.tensor(np.ones(N) * MASS, dtype=torch.float32)
    ps, vs = [p[0].numpy().astype(np.float64)], [v[0].numpy().astype(np.float64)]
    with torch.no_grad():
        for _ in range(STEPS):
            a = net(p, v, m)
            if sigma > 0:
                a = a + torch.tensor(
                    rng.normal(0, sigma, a.shape) * float(a.abs().mean() + 1e-8),
                    dtype=torch.float32)
            v = v + a * DT
            p = p + v * DT
            ps.append(p[0].numpy().astype(np.float64))
            vs.append(v[0].numpy().astype(np.float64))
    return np.stack(ps), np.stack(vs)


def make_block(n_ic, fam, net, seed):
    """返回 (cand_pos, cand_vel, errors, true_pos, true_vel)。

    cand_*: (n_ic*K, T, N, 3)；errors: (n_ic, K)；
    true_*: (n_ic, T, N, 3) —— 供训练绝对二分类 baseline 用。
    """
    rng = np.random.default_rng(seed)
    P, V, E, TP, TV = [], [], [], [], []
    for _ in range(n_ic):
        p, v = prior_ic(rng.uniform(*fam), rng)
        true_traj, true_vel = rollout_verlet(p, v, np.ones(N) * MASS)
        TP.append(true_traj); TV.append(true_vel)
        for s in SIGMAS:
            cp, cv = rollout_noisy(p, v, net, s, rng)
            P.append(cp); V.append(cv); E.append(rms(cp, true_traj))
    return np.stack(P), np.stack(V), np.array(E).reshape(n_ic, K), \
        np.stack(TP), np.stack(TV)


def train_L1_binary(TP, TV, Ptr, Vtr, h=128, lr=1e-3, epochs=120, seed=0, batch=128):
    """L1：绝对二分类（正=真实轨迹，负=候选轨迹）。实验 B 用的就是这一设定。"""
    Xp, mu, sd = _prep(TP, TV)
    Xn, _, _ = _prep(Ptr, Vtr, mu, sd)
    X = torch.cat([Xp, Xn], 0)
    Y = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))]).float()
    net = _new_net(h, seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(X)
    for ep in range(epochs):
        idx = torch.randperm(n)[:batch * 4]
        loss = nn.functional.binary_cross_entropy_with_logits(net(X[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, mu, sd


# ============================================================
# 强 baseline 训练器
# ============================================================
def _prep(P, V, mu=None, sd=None):
    X = np.stack([raw_tensor(P[i], V[i]) for i in range(len(P))])
    if mu is None:
        mu = X.mean(axis=(0, 1, 2), keepdims=True)
        sd = X.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    return torch.tensor((X - mu) / sd, dtype=torch.float32), mu, sd


def _new_net(h, seed):
    torch.manual_seed(seed)
    return DeepSetsCritic(h=h).to(torch.float32)


def train_L2_ranking(Ptr, Vtr, Etr, h=128, lr=1e-3, epochs=120, seed=0, batch=32):
    """成对排序损失：同一 IC 内，误差更小的候选应得更高分。"""
    X, mu, sd = _prep(Ptr, Vtr)
    n_ic = len(Etr)
    X = X.reshape(n_ic, K, *X.shape[1:])
    E = torch.tensor(Etr, dtype=torch.float32)
    # 预算：每个 IC 内所有 (i,j) 且 err_i < err_j 的对
    pairs = []
    for a in range(K):
        for b in range(K):
            if a < b:
                pairs.append((a, b))
    net = _new_net(h, seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        idx = torch.randperm(n_ic)[:batch]
        xb = X[idx].reshape(-1, *X.shape[2:])
        sb = net(xb).reshape(len(idx), K)
        eb = E[idx]
        loss = 0.0
        for a, b in pairs:
            better = (eb[:, a] < eb[:, b]).float()
            d = sb[:, a] - sb[:, b]
            # better=1 → 希望 d>0；用 BCE-with-logits 形式的对称损失
            loss = loss + nn.functional.binary_cross_entropy_with_logits(
                d, better, reduction="mean")
        loss = loss / len(pairs)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, mu, sd


def train_L4_relative_reg(Ptr, Vtr, Etr, h=128, lr=1e-3, epochs=120, seed=0, batch=32):
    """IC 内相对误差回归：目标 = log(RMSE / 该 IC 的候选中位 RMSE)。"""
    X, mu, sd = _prep(Ptr, Vtr)
    n_ic = len(Etr)
    X = X.reshape(n_ic, K, *X.shape[1:])
    med = Etr.mean(axis=1, keepdims=True)          # 用均值作基线更稳
    Yt = torch.tensor(np.log(Etr / med), dtype=torch.float32)
    net = _new_net(h, seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        idx = torch.randperm(n_ic)[:batch]
        xb = X[idx].reshape(-1, *X.shape[2:])
        s = net(xb).reshape(len(idx), K)
        loss = ((s - Yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, mu, sd


def train_L6_listnet(Ptr, Vtr, Etr, h=128, lr=1e-3, epochs=120, seed=0, batch=32):
    """列表损失：同一 IC 的 8 个候选中，分数经 softmax 应匹配「越小越好的」目标分布。"""
    X, mu, sd = _prep(Ptr, Vtr)
    n_ic = len(Etr)
    X = X.reshape(n_ic, K, *X.shape[1:])
    # 目标分布：按 -log(err) 做 softmax，误差越小权重越大
    logit_t = -np.log(Etr + 1e-12)
    logit_t = logit_t - logit_t.max(axis=1, keepdims=True)
    Tt = torch.tensor(np.exp(logit_t) / np.exp(logit_t).sum(axis=1, keepdims=True),
                      dtype=torch.float32)
    net = _new_net(h, seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        idx = torch.randperm(n_ic)[:batch]
        xb = X[idx].reshape(-1, *X.shape[2:])
        s = net(xb).reshape(len(idx), K)
        logp = torch.log_softmax(s, dim=1)
        loss = -(Tt[idx] * logp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, mu, sd


def train_L3_absolute_reg(Ptr, Vtr, Etr, h=128, lr=1e-3, epochs=120, seed=0, batch=32):
    X, mu, sd = _prep(Ptr, Vtr)
    Yt = torch.tensor(np.log(Etr.reshape(-1) + 1e-12), dtype=torch.float32)
    net = _new_net(h, seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(X)
    for ep in range(epochs):
        idx = torch.randperm(n)[:batch * K]
        loss = ((net(X[idx]) - Yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, mu, sd


def main():
    t0 = time.time()
    print("=" * 84)
    print("1. 生成器与候选数据")
    print("=" * 84)
    rng = np.random.default_rng(1)
    Ptr0, Vtr0 = [], []
    for _ in range(120):
        p, v = prior_ic(rng.uniform(*F_TRAIN), rng)
        pt, vt = rollout_verlet(p, v, np.ones(N) * MASS)
        Ptr0.append(pt); Vtr0.append(vt)
    net = train_force_model(np.stack(Ptr0), np.stack(Vtr0),
                            np.ones(N) * MASS, epochs=12, h=32, lr=3e-3)
    print(f"   生成器：120 条轨迹 / 12 轮 / h=32（受限预算，刻意留出质量差异）")

    TR_P, TR_V, TR_E, TR_TP, TR_TV = make_block(N_TRAIN_IC, F_TRAIN, net, 101)
    ID_P, ID_V, ID_E, _, _ = make_block(N_ID_IC, F_TRAIN, net, 202)
    OD_P, OD_V, OD_E, _, _ = make_block(N_OOD_IC, F_TEST, net, 303)
    print(f"   训练 {N_TRAIN_IC} IC | ID {N_ID_IC} IC | OOD {N_OOD_IC} IC，每 IC {K} 候选")
    for nm, E in (("train", TR_E), ("ID", ID_E), ("OOD", OD_E)):
        print(f"   [{nm}] 候选 RMSE 中位={np.median(E):.4f}  "
              f"IC 内 max/min 中位={np.median(E.max(1)/(E.min(1)+1e-12)):.2f}x")

    print()
    print("=" * 84)
    print("2. 训练强 baseline（全部只用 train 切分）")
    print("=" * 84)
    models = {}
    t1 = time.time()
    m1, mu1, sd1 = train_L1_binary(TR_TP, TR_TV, TR_P, TR_V, h=128, lr=1e-3, epochs=120)
    models["L1 二分类(绝对)"] = (m1, mu1, sd1)
    print(f"   {'L1 二分类(绝对)':<20} 训练完成 ({time.time()-t1:.0f}s)", flush=True)

    for name, fn in (("L2 成对排序损失", train_L2_ranking),
                     ("L3 绝对误差回归", train_L3_absolute_reg),
                     ("L4 IC内相对回归", train_L4_relative_reg),
                     ("L6 列表损失 ListNet", train_L6_listnet)):
        t1 = time.time()
        m, mu, sd = fn(TR_P, TR_V, TR_E, h=128, lr=1e-3, epochs=120, seed=0, batch=32)
        models[name] = (m, mu, sd)
        print(f"   {name:<20} 训练完成 ({time.time()-t1:.0f}s)", flush=True)

    # ---------------- critic 集合 ----------------
    # 符号必须**显式**声明，不能用名字子串推断：
    # 早先用 `-1.0 if "误差回归" in name else 1.0`，而 "L4 IC内相对回归" 不含
    # 「误差回归」四个连续字，被误判为 +1 —— 于是把一个「输出 log 误差、越高越差」
    # 的模型当成「越高越好」用，结果 within-IC ρ 变成 +0.70、更优率 0.5%（比随机还差），
    # 并且污染了融合权重（L4 被赋 −4.622 去补偿）。这类 bug 会直接翻转结论。
    SIGN = {
        "L1 二分类(绝对)": +1.0,     # 输出 logit，越高=越像真实
        "L2 成对排序损失": +1.0,     # 输出分数，越高=越好
        "L3 绝对误差回归": -1.0,     # 输出 log(误差)，越高=越差
        "L4 IC内相对回归": -1.0,     # 输出 log(误差/IC均值)，越高=越差
        "L6 列表损失 ListNet": +1.0,  # 输出分数，越高=越好
    }

    def mk(net_, mu_, sd_, sign):
        def f(pos_t, vel_t):
            return sign * float(score_net(net_, mu_, sd_,
                                          np.stack([raw_tensor(pos_t, vel_t)]))[0])
        return f

    def sc_cons(pos_t, vel_t):
        return -cons_drift(pos_t, vel_t)

    CRITICS = [("守恒 (zero-shot)", sc_cons)]
    for name, (m, mu, sd) in models.items():
        assert name in SIGN, f"未声明符号的 baseline：{name}"
        CRITICS.append((name, mk(m, mu, sd, SIGN[name])))

    # 最优融合：守恒 + 两个最强的学习式（成对排序、IC 内相对回归）做逻辑回归。
    # 用**按名查找**而不是下标，避免列表顺序变动后静默取错 critic。
    by_name = dict(CRITICS)
    F_NAMES = ("L2 成对排序损失", "L4 IC内相对回归")
    for nm in F_NAMES:
        assert nm in by_name, f"融合所需的 critic 不存在：{nm}"

    def feat(pos_t, vel_t):
        return [sc_cons(pos_t, vel_t)] + [by_name[nm](pos_t, vel_t) for nm in F_NAMES]

    Ftr, ytr = [], []
    for i in range(len(TR_E)):
        med = np.median(TR_E[i])
        for j in range(K):
            Ftr.append(feat(TR_P[i * K + j], TR_V[i * K + j]))
            ytr.append(1.0 if TR_E[i][j] <= med else 0.0)
    Ftr_t = torch.tensor(np.array(Ftr), dtype=torch.float32)
    ytr_t = torch.tensor(np.array(ytr), dtype=torch.float32)
    wb = torch.zeros(4, dtype=torch.float32, requires_grad=True)
    of = torch.optim.Adam([wb], lr=0.05)
    for _ in range(800):
        of.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(
            Ftr_t @ wb[:3] + wb[3], ytr_t).backward()
        of.step()
    W = wb.detach().numpy()
    print()
    print(f"   最优融合权重：守恒 {W[0]:+.3f} | {F_NAMES[0]} {W[1]:+.3f} | "
          f"{F_NAMES[1]} {W[2]:+.3f} | b {W[3]:+.3f}")

    def sc_fusion(pos_t, vel_t):
        return float(np.dot(W[:3], feat(pos_t, vel_t)) + W[3])
    CRITICS.append(("L5 最优融合", sc_fusion))

    # ---------------- 评估 ----------------
    for fam_name, P, V, E in (("ID", ID_P, ID_V, ID_E),
                              ("OOD-族", OD_P, OD_V, OD_E)):
        n_ic = len(E)
        S = {cn: np.zeros((n_ic, K)) for cn, _ in CRITICS}
        for i in range(n_ic):
            for j in range(K):
                pt, vt = P[i * K + j], V[i * K + j]
                for cn, fn in CRITICS:
                    S[cn][i, j] = fn(pt, vt)

        print()
        print("=" * 84)
        print(f"[{fam_name}] 强 baseline 对照")
        print("=" * 84)
        rand_err = float(np.median(E.mean(1)))
        orac_err = float(np.median(E.min(1)))
        print(f"   {'critic':<22}{'within-IC ρ':>13}{'更优率':>10}"
              f"{'选中误差':>12}{'/随机':>9}{'/oracle':>9}")
        print("   " + "-" * 75)
        print(f"   {'随机挑选':<22}{'—':>13}{'50.0%':>10}{rand_err:>12.5f}"
              f"{1.0:>8.3f}x{rand_err/orac_err:>8.2f}x")
        print(f"   {'oracle(上界)':<22}{'—':>13}{'100%':>10}{orac_err:>12.5f}"
              f"{orac_err/rand_err:>8.3f}x{1.0:>8.2f}x")
        rows = {}
        for cn, _ in CRITICS:
            per = [spearman(S[cn][i], E[i]) for i in range(n_ic)]
            per = [r for r in per if not np.isnan(r)]
            within = float(np.mean(per)) if per else float("nan")
            better = float(np.mean([E[i][int(np.argmax(S[cn][i]))] < np.median(E[i])
                                    for i in range(n_ic)]))
            pick = float(np.median([E[i][int(np.argmax(S[cn][i]))] for i in range(n_ic)]))
            rows[cn] = dict(within=within, better=better, pick=pick)
            print(f"   {cn:<22}{within:>+13.4f}{better:>9.1%}{pick:>12.5f}"
                  f"{pick/rand_err:>8.3f}x{pick/orac_err:>8.2f}x")
        print()
        best = min((v["pick"], k) for k, v in rows.items())
        print(f"   -> 选择误差最低：{best[1]} = {best[0]:.5f}")

        # ---------------- 配对检验：守恒 vs 各 baseline ----------------
        # 注意：中位差在多数 IC 上恰好为 0 —— 因为强 critic 们在多数 IC 上
        # **挑的是同一个候选**。中位数会掩盖这个结构，因此这里报告三件事：
        #   1) 分歧比例（挑的是不同候选）
        #   2) 分歧时守恒赢/输的次数（二项检验的直观形式）
        #   3) 均值差及其 bootstrap CI
        sel = {cn: np.array([E[i][int(np.argmax(S[cn][i]))] for i in range(n_ic)])
               for cn, _ in CRITICS}
        cons = sel["守恒 (zero-shot)"]
        print()
        print(f"   配对比较（{n_ic} 个 IC）：守恒 vs 各 baseline")
        print(f"   {'对比':<24}{'分歧率':>9}{'守恒赢/输':>13}"
              f"{'均值差':>12}{'95% CI':>22}{'结论':>10}")
        print("   " + "-" * 92)
        rg = np.random.default_rng(0)
        for cn, _ in CRITICS:
            if cn == "守恒 (zero-shot)":
                continue
            d = cons - sel[cn]
            diff = np.abs(d) > 1e-12
            nb, nw = int((d[diff] < 0).sum()), int((d[diff] > 0).sum())
            bs = np.array([d[rg.integers(0, n_ic, n_ic)].mean() for _ in range(2000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            verdict = "守恒更差*" if lo > 0 else ("守恒更好*" if hi < 0 else "无差异")
            print(f"   {'守恒 vs ' + cn:<24}{diff.mean():>8.1%}{f'{nb}/{nw}':>13}"
                  f"{d.mean():>+12.7f}{f'[{lo:+.7f}, {hi:+.7f}]':>22}{verdict:>10}")
        print("   分歧率=两者挑到不同候选的 IC 占比；* = 均值差的 95% CI 不跨 0")

    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
