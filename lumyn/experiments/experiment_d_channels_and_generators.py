"""实验 D：(A) 守恒通道消融  (B) 跨生成器架构/质量验证

两个问题
--------
A. **守恒 critic 到底是什么？**
   当前 `cons_drift` 是四个通道的**等权相加**：energy / angular / momentum / com。
   这个权重是任意选的，而各通道的量纲与量级完全不同。
   且此前实测：物理轨迹上 `angular` 与 `momentum` 的漂移中位是 **1e-16（机器精度）** ——
   它们是浮点噪声，不携带信息。
   若如此，则「守恒 critic」实际上只是 energy + com 通道，消融应当证实这一点；
   若某单通道或某组合更强，则应据实修改 critic 定义，而不是保留任意加权。

B. **结论是否只是某一个生成器的产物？**
   三种生成器，覆盖「架构不同」与「质量不同」两个轴：
     G_A  成对 GNN，中等预算   —— 前面实验用的那个
     G_B  成对 GNN，强预算     —— 质量好得多，候选间差异更小（更难的选择任务）
     G_C  平均场 MLP           —— 无成对结构，归纳偏置完全不同
   对每个生成器分别重跑主对照，看守恒 critic 的位次是否稳定。

符号约定（吸取实验 C 的教训）
----------------------------
所有 critic 的分数方向**显式声明**，不得由名字推断。
凡期望「越大越好」的 critic 直接使用；输出 log(误差) 的一律取负。
"""
import sys, os, json, time
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
    prior_ic, rollout_verlet, raw_tensor, score_net, DeepSetsCritic,
)
from lumyn.experiments.experiment_a2_neural_generator import (   # noqa: E402
    NeuralForce, train_force_model,
)

K = 8
SIGMAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.6, 1.0)
N_TR_IC, N_TE_IC = 300, 150


# ============================================================
# 工具
# ============================================================
def spearman(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    def rank(a):
        o = np.argsort(a, kind="mergesort")
        r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
        return r
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / (np.sqrt((rx ** 2).sum() * (ry ** 2).sum()) + 1e-30))


def rms(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


# ---------------- 四个守恒通道 ----------------
def channels(pos_t, vel_t):
    m = np.ones(N) * MASS
    ke = 0.5 * (m * (vel_t * vel_t).sum(-1)).sum(-1)
    d = pos_t[:, None, :, :] - pos_t[:, :, None, :]
    r = np.sqrt((d * d).sum(-1) + SOFT ** 2)
    eye = 1.0 - np.eye(N)
    mm = m[:, None] * m[None, :]
    pe = -(mm[None] / r * eye[None]).sum((1, 2)) / 2.0
    E = ke + pe
    L = (m[None, :, None] * np.cross(pos_t, vel_t)).sum(1)
    P = (m[None, :, None] * vel_t).sum(1)
    com = (m[None, :, None] * pos_t).sum(1) / m.sum()

    E_scale = np.abs(E).mean() + 1e-12
    L_scale = np.linalg.norm(L, axis=-1).mean() + 1e-12
    P_scale = (m * np.linalg.norm(vel_t, axis=-1)).sum(-1).mean() + 1e-12
    c_scale = np.linalg.norm(pos_t, axis=-1).mean() + 1e-12
    return {
        "energy":   float((E.max() - E.min()) / E_scale),
        "angular":  float(np.linalg.norm(L.max(0) - L.min(0)) / L_scale),
        "momentum": float(np.linalg.norm(P.max(0) - P.min(0)) / P_scale),
        "com":      float(np.linalg.norm(com.max(0) - com.min(0)) / c_scale),
    }


CH = ("energy", "angular", "momentum", "com")


# ============================================================
# 第二个生成器：平均场 MLP（无成对结构）
# ============================================================
class MeanFieldForce(nn.Module):
    """逐粒子 MLP，只用「自身状态 + 全局均值上下文」。

    **完全没有成对交互**：不显式建模 r_ij，因此归纳偏置与成对 GNN 完全不同。
    它仍能从数据里学到「越靠近质心加速度越大」这类平均场规律，但学不到
    具体某两个粒子之间的近距离交会。
    """

    def __init__(self, h=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(16, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 3),
        )

    def forward(self, pos, vel, mass):
        B, Nn, _ = pos.shape
        mp = pos.mean(1, keepdim=True).expand(-1, Nn, -1)
        mv = vel.mean(1, keepdim=True).expand(-1, Nn, -1)
        rel = pos - mp
        feat = torch.cat([pos, vel, mp, mv, rel, rel.norm(dim=-1, keepdim=True)], -1)
        return self.mlp(feat)                     # (B,N,3)


def train_meanfield(P, V, M, epochs=60, h=128, lr=2e-3, batch=64, seed=0):
    torch.manual_seed(seed)
    net = MeanFieldForce(h=h).to(torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Pt = torch.tensor(P, dtype=torch.float32); Vt = torch.tensor(V, dtype=torch.float32)
    A = ((Vt[:, 1:] - Vt[:, :-1]) / DT).reshape(-1, N, 3)
    Xp = Pt[:, :-1].reshape(-1, N, 3); Xv = Vt[:, :-1].reshape(-1, N, 3)
    Mt = torch.tensor(M, dtype=torch.float32)
    n = len(Xp)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            ((net(Xp[idx], Xv[idx], Mt) - A[idx]) ** 2).mean().backward()
            opt.step()
    net.eval()
    return net


def one_step_err(net, P, V):
    with torch.no_grad():
        Pt = torch.tensor(P[:32], dtype=torch.float32)
        Vt = torch.tensor(V[:32], dtype=torch.float32)
        At = ((Vt[:, 1:] - Vt[:, :-1]) / DT).reshape(-1, N, 3)
        Ap = net(Pt[:, :-1].reshape(-1, N, 3), Vt[:, :-1].reshape(-1, N, 3),
                 torch.tensor(np.ones(N) * MASS, dtype=torch.float32))
        return float(((Ap - At) ** 2).mean().sqrt() / (At ** 2).mean().sqrt())


def rollout_net(pos, vel, net, sigma, rng):
    p = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)
    v = torch.tensor(vel, dtype=torch.float32).unsqueeze(0)
    m = torch.tensor(np.ones(N) * MASS, dtype=torch.float32)
    ps = [p[0].numpy().astype(np.float64)]; vs = [v[0].numpy().astype(np.float64)]
    with torch.no_grad():
        for _ in range(STEPS):
            a = net(p, v, m)
            if sigma > 0:
                a = a + torch.tensor(rng.normal(0, sigma, a.shape)
                                     * float(a.abs().mean() + 1e-8), dtype=torch.float32)
            v = v + a * DT
            p = p + v * DT
            ps.append(p[0].numpy().astype(np.float64))
            vs.append(v[0].numpy().astype(np.float64))
    return np.stack(ps), np.stack(vs)


def make_block(n_ic, fam, net, seed):
    rng = np.random.default_rng(seed)
    P, V, E, TP, TV = [], [], [], [], []
    for _ in range(n_ic):
        p, v = prior_ic(rng.uniform(*fam), rng)
        tt, tv = rollout_verlet(p, v, np.ones(N) * MASS)
        TP.append(tt); TV.append(tv)
        for s in SIGMAS:
            cp, cv = rollout_net(p, v, net, s, rng)
            P.append(cp); V.append(cv); E.append(rms(cp, tt))
    return (np.stack(P), np.stack(V), np.array(E).reshape(n_ic, K),
            np.stack(TP), np.stack(TV))


# ============================================================
# 学习式 baseline（沿用实验 C 的两个最强 + 一个朴素）
# ============================================================
def _prep(P, V, mu=None, sd=None):
    X = np.stack([raw_tensor(P[i], V[i]) for i in range(len(P))])
    if mu is None:
        mu = X.mean(axis=(0, 1, 2), keepdims=True)
        sd = X.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    return torch.tensor((X - mu) / sd, dtype=torch.float32), mu, sd


def _net(seed=0):
    torch.manual_seed(seed)
    return DeepSetsCritic(h=128).to(torch.float32)


def train_abs_binary(TP, TV, P, V, epochs=100, seed=0):
    Xp, mu, sd = _prep(TP, TV)
    Xn, _, _ = _prep(P, V, mu, sd)
    X = torch.cat([Xp, Xn], 0)
    Y = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))]).float()
    net = _net(seed); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        idx = torch.randperm(len(X))[:512]
        nn.functional.binary_cross_entropy_with_logits(net(X[idx]), Y[idx]).backward()
        opt.step(); opt.zero_grad()
    net.eval(); return net, mu, sd


def train_pairwise(P, V, E, epochs=100, seed=0, batch=32):
    X, mu, sd = _prep(P, V)
    n = len(E); X = X.reshape(n, K, *X.shape[1:])
    Ee = torch.tensor(E, dtype=torch.float32)
    net = _net(seed); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    for _ in range(epochs):
        idx = torch.randperm(n)[:batch]
        s = net(X[idx].reshape(-1, *X.shape[2:])).reshape(len(idx), K)
        eb = Ee[idx]
        loss = 0.0
        for a, b in pairs:
            loss = loss + nn.functional.binary_cross_entropy_with_logits(
                s[:, a] - s[:, b], (eb[:, a] < eb[:, b]).float())
        (loss / len(pairs)).backward(); opt.step(); opt.zero_grad()
    net.eval(); return net, mu, sd


def train_rel_reg(P, V, E, epochs=100, seed=0, batch=32):
    X, mu, sd = _prep(P, V)
    n = len(E); X = X.reshape(n, K, *X.shape[1:])
    Y = torch.tensor(np.log(E / E.mean(axis=1, keepdims=True)), dtype=torch.float32)
    net = _net(seed); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        idx = torch.randperm(n)[:batch]
        s = net(X[idx].reshape(-1, *X.shape[2:])).reshape(len(idx), K)
        ((s - Y[idx]) ** 2).mean().backward(); opt.step(); opt.zero_grad()
    net.eval(); return net, mu, sd


# ============================================================
def eval_critics(crits, TR, TE, tag):
    def score_all(P, V, E):
        n = len(E)
        S = {cn: np.zeros((n, K)) for cn, _ in crits}
        for i in range(n):
            for j in range(K):
                pt, vt = P[i * K + j], V[i * K + j]
                for cn, fn in crits:
                    S[cn][i, j] = fn(pt, vt)
        return S

    # TE 是 (P, V, E, TP, TV) 五元组，score_all 只要前三项
    S = score_all(TE[0], TE[1], TE[2])
    E = TE[2]
    n = len(E)
    rand = float(np.median(E.mean(1))); orac = float(np.median(E.min(1)))
    out = {}
    print(f"   {'critic':<26}{'within-IC ρ':>13}{'更优率':>9}"
          f"{'选中/随机':>11}{'选中/oracle':>13}")
    print("   " + "-" * 72)
    for cn, _ in crits:
        per = [spearman(S[cn][i], E[i]) for i in range(n)]
        per = [r for r in per if not np.isnan(r)]
        wr = float(np.mean([E[i][int(np.argmax(S[cn][i]))] < np.median(E[i])
                            for i in range(n)]))
        pk = float(np.median([E[i][int(np.argmax(S[cn][i]))] for i in range(n)]))
        out[cn] = dict(rho=float(np.mean(per)) if per else float("nan"),
                       win=wr, pick=pk)
        print(f"   {cn:<26}{out[cn]['rho']:>+13.4f}{wr:>8.1%}"
              f"{pk/rand:>10.3f}x{pk/orac:>12.2f}x")
    print(f"   {'[参照] 随机':<26}{'—':>13}{'50.0%':>9}{1.0:>10.3f}x{rand/orac:>12.2f}x")
    print(f"   {'[参照] oracle':<26}{'—':>13}{'100%':>9}{orac/rand:>10.3f}x{1.0:>12.2f}x")
    return out


def main():
    t0 = time.time()
    print("=" * 84)
    print("1. 三个生成器：成对 GNN（中/强预算）与平均场 MLP")
    print("=" * 84)
    rng = np.random.default_rng(1)
    Ptr, Vtr = [], []
    for _ in range(320):
        p, v = prior_ic(rng.uniform(*F_TRAIN), rng)
        pt, vt = rollout_verlet(p, v, np.ones(N) * MASS)
        Ptr.append(pt); Vtr.append(vt)
    Ptr, Vtr = np.stack(Ptr), np.stack(Vtr)
    M = np.ones(N) * MASS

    gens = {}
    gA = train_force_model(Ptr[:120], Vtr[:120], M, epochs=12, h=32, lr=3e-3)
    gens["G_A 成对GNN(中预算)"] = gA
    print(f"   G_A 单步加速度相对误差 = {one_step_err(gA, Ptr, Vtr):.4f}")
    gB = train_force_model(Ptr, Vtr, M, epochs=120, h=64, lr=2e-3)
    gens["G_B 成对GNN(强预算)"] = gB
    print(f"   G_B 单步加速度相对误差 = {one_step_err(gB, Ptr, Vtr):.4f}")
    gC = train_meanfield(Ptr, Vtr, M, epochs=60, h=128, lr=2e-3)
    gens["G_C 平均场MLP"] = gC
    print(f"   G_C 单步加速度相对误差 = {one_step_err(gC, Ptr, Vtr):.4f}")

    # ---------------- A. 通道消融 ----------------
    print()
    print("=" * 84)
    print("2. 通道消融：守恒 critic 到底是什么？")
    print("=" * 84)
    for gname, gnet in gens.items():
        TR = make_block(N_TR_IC, F_TRAIN, gnet, 101)
        ID = make_block(N_TE_IC, F_TRAIN, gnet, 202)
        OD = make_block(N_TE_IC, F_TEST, gnet, 303)

        print(f"\n   ---- {gname} ----")
        # 通道基准：物理轨迹 vs 生成轨迹
        print(f"   [通道基准] {'通道':<10}{'物理(中位)':>15}{'生成(中位)':>15}{'比值':>10}")
        base = {}
        for c in CH:
            a = np.median([channels(TR[3][i], TR[4][i])[c] for i in range(len(TR[3]))])
            b = np.median([channels(TR[0][i], TR[1][i])[c] for i in range(len(TR[0]))])
            base[c] = (a, b)
            print(f"              {c:<10}{a:>15.3e}{b:>15.3e}{b/(a+1e-30):>10.3f}")

        combos = [("energy 单通道", ("energy",)),
                  ("angular 单通道", ("angular",)),
                  ("momentum 单通道", ("momentum",)),
                  ("com 单通道", ("com",)),
                  ("energy+com", ("energy", "com")),
                  ("四通道等权（现行）", CH)]
        combos_sorted = sorted(combos, key=lambda x: (x[1] != CH,))

        crits = []
        for label, chs in combos:
            def mk(chs=chs):
                def f(pt, vt):
                    ch = channels(pt, vt)
                    return -sum(ch[c] for c in chs)
                return f
            crits.append((label, mk()))
        print(f"\n   [ID] 通道组合的判别力")
        res_id = eval_critics(crits, TR, ID, "ID")
        print(f"\n   [OOD] 通道组合的判别力")
        res_od = eval_critics(crits, TR, OD, "OOD")

    # ---------------- B. 跨生成器主对照 ----------------
    print()
    print("=" * 84)
    print("3. 跨生成器主对照（守恒 vs 学习式强 baseline）")
    print("=" * 84)
    summary = {}
    for gname, gnet in gens.items():
        TR = make_block(N_TR_IC, F_TRAIN, gnet, 101)
        ID = make_block(N_TE_IC, F_TRAIN, gnet, 202)
        print(f"\n   ==== {gname} ====")
        print(f"   候选 RMSE 中位: train={np.median(TR[2]):.5f} "
              f"ID={np.median(ID[2]):.5f} | "
              f"IC 内 max/min 中位={np.median(ID[2].max(1)/(ID[2].min(1)+1e-12)):.2f}x")

        n1, mu1, sd1 = train_abs_binary(TR[3], TR[4], TR[0], TR[1])
        n2, mu2, sd2 = train_pairwise(TR[0], TR[1], TR[2])
        n4, mu4, sd4 = train_rel_reg(TR[0], TR[1], TR[2])

        def cons(pt, vt):
            ch = channels(pt, vt)
            return -sum(ch[c] for c in CH)

        def mk(net_, mu_, sd_, sign):
            def f(pt, vt):
                return sign * float(score_net(net_, mu_, sd_,
                                              np.stack([raw_tensor(pt, vt)]))[0])
            return f
        # 符号显式声明（吸取实验 C 的教训）
        crits = [("守恒 (zero-shot)", cons),
                 ("L1 绝对二分类", mk(n1, mu1, sd1, +1.0)),
                 ("L2 成对排序", mk(n2, mu2, sd2, +1.0)),
                 ("L4 IC内相对回归", mk(n4, mu4, sd4, -1.0))]

        # 最优融合（守恒 + L2 + L4）
        by = dict(crits)
        def feat(pt, vt):
            return [cons(pt, vt), by["L2 成对排序"](pt, vt), by["L4 IC内相对回归"](pt, vt)]
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
        print(f"   融合权重: 守恒 {W[0]:+.3f} | L2 {W[1]:+.3f} | L4 {W[2]:+.3f}")

        def fus(pt, vt):
            return float(np.dot(W[:3], feat(pt, vt)) + W[3])
        crits.append(("L5 最优融合", fus))

        r_id = eval_critics(crits, TR, ID, "ID")
        summary[gname] = r_id

    print()
    print("=" * 84)
    print("4. 汇总：守恒在各生成器上的位次（ID）")
    print("=" * 84)
    print(f"   {'生成器':<24}{'守恒 选中/随机':>16}{'最好 critic':>24}{'其选中/随机':>14}")
    print("   " + "-" * 80)
    for gname, r in summary.items():
        best = min((v["pick"], k) for k, v in r.items())
        cons_pick = r["守恒 (zero-shot)"]["pick"]
        # 相对随机需要基准，这里用守恒 pick / best pick 作粗略位次指标
        print(f"   {gname:<24}{r['守恒 (zero-shot)']['win']:>15.1%}"
              f"{best[1]:>24}{best[0]:>14.5f}")
        rel = cons_pick / best[0]
        print(f"   {'':<24}守恒选中误差={cons_pick:.5f}  最优={best[0]:.5f}  "
              f"比值={rel:.3f}x  {'守恒最优' if rel <= 1.001 else '守恒略逊'}")

    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
