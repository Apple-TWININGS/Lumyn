"""实验 A1：critic 的判别质量（zero-shot 守恒 vs 学习式），ID / OOD。

范式主张（修订版，互补型）
--------------------------
守恒律 critic 与学习式 critic 的失效模式**互补**：
  - 守恒 critic：精确掌握守恒律 → 对结构性物理违规近完美、零训练成本；
    但对**未违反守恒的**微妙动力学误差盲。
  - 学习式 critic：能捕捉动力学细节，但对结构性不一致盲。
  - 二者结合应在所有分布切分上一致优于任一单项。

可证伪判据
----------
1. 若两者盲区**不**互补（同一类违规上一起失效）→ 互补型叙事不成立。
2. 若混合 critic 不能在多数切分上优于单项 → 结合无收益。
3. 若混合的收益小于「多跑一个 critic」的代价 → 工程上不值得。

状态口径（v2 修正）
-------------------
v1 用 `np.gradient` 从位置反推速度来算守恒量。该估计的 O(dt²) 误差
（dt=0.01 时约 1e-2 量级）会**污染守恒残差本身**：物理轨迹的"残差"
中位 0.31 主要来自这个伪影，而非真实漂移。低严重度下伪影盖过真信号，
可能正是守恒 critic 掉到随机以下（AUROC 0.39–0.46）的原因。

v2 改为 critic 接收**完整状态 (pos, vel)**——这与真实仿真器的输出一致
（仿真器本来就同时产出位置和速度），损坏操作同时作用于两者，
因此不再需要有限差分。这是消除该混淆项的正确做法。

学习式 baseline 的公平性（必须守住）
-----------------------------------
- 学习式 critic 输入仅原始状态（位置/速度/有限差分加速度），**不含任何守恒量**
- 置换不变 DeepSets（拼接式 MLP 是稻草人：ID AUROC 仅 0.54~0.57）
- 超参在 train 分布上选择，不接触 OOD

评分约定：AUROC 1.0 = 完美，0.5 = 无判别力。
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
T = STEPS + 1
SNAP = np.array([0, 8, 16, 24, 32, 40])

F_TRAIN = (0.0, 0.3)
F_TEST = (0.5, 0.8)

N_TRAIN = 2000
N_ID = 500
N_OOD = 500
SEV_SUBTLE = 0.25


# ============================================================
# 物理：状态 = (pos, vel)，两者都返回
# ============================================================
def prior_ic(e, rng):
    ang = rng.uniform(0, 2 * np.pi, N)
    r = (1 - e ** 2) / (1 + e * np.cos(ang))
    pos = np.stack([r * np.cos(ang), r * np.sin(ang), rng.normal(0, 0.05, N)], axis=1)
    v = np.sqrt(1.0 / np.maximum(r, 1e-6))
    vel = np.stack([-v * np.sin(ang), v * np.cos(ang), rng.normal(0, 0.02, N)], axis=1)
    return pos, vel


def accel(pos, mass):
    d = pos[None, :, :] - pos[:, None, :]
    r2 = (d * d).sum(-1) + SOFT ** 2
    inv_r3 = r2 ** -1.5
    np.fill_diagonal(inv_r3, 0.0)
    return G * (mass[None, :, None] * d * inv_r3[..., None]).sum(1)


def rollout_verlet(pos, vel, mass, steps=STEPS, dt=DT):
    """辛欧拉（kick-drift）。返回 (pos_traj, vel_traj)，均为 (T,N,3)。"""
    p, v = pos.copy(), vel.copy()
    ps, vs = [p.copy()], [v.copy()]
    for _ in range(steps):
        v = v + accel(p, mass) * dt
        p = p + v * dt
        ps.append(p.copy()); vs.append(v.copy())
    return np.stack(ps), np.stack(vs)


def rollout_euler(pos, vel, mass, steps=STEPS, dt=DT):
    """显式欧拉（非辛）。返回 (pos_traj, vel_traj)。"""
    p, v = pos.copy(), vel.copy()
    ps, vs = [p.copy()], [v.copy()]
    for _ in range(steps):
        a = accel(p, mass)
        p = p + v * dt
        v = v + a * dt
        ps.append(p.copy()); vs.append(v.copy())
    return np.stack(ps), np.stack(vs)


# ============================================================
# 物理违规：同时作用于位置与速度
# ============================================================
CORRUPTIONS = ("noise", "euler", "freeze", "splice", "warp")
BASE_SEV = {"noise": 0.02, "euler": 1.0, "freeze": 1.0, "splice": 1.0, "warp": 1.35}


def corrupt(pos_t, vel_t, kind, rng, sev=1.0):
    p, v = pos_t.copy(), vel_t.copy()
    if kind == "noise":
        s = BASE_SEV["noise"] * sev
        return p + rng.normal(0, s, p.shape), v + rng.normal(0, s / DT, v.shape)
    if kind == "euler":
        return rollout_euler(p[0], v[0], np.ones(N) * MASS, steps=STEPS, dt=DT * sev)
    if kind == "freeze":
        k = max(1, T - int((T // 2) * sev))
        p[k:] = p[k]; v[k:] = v[k]
        return p, v
    if kind == "splice":
        k = T // 2
        return (np.concatenate([p[:k], np.roll(p[k:], 1, axis=1)], axis=0),
                np.concatenate([v[:k], np.roll(v[k:], 1, axis=1)], axis=0))
    if kind == "warp":
        idx = np.clip((np.arange(T) * (1.0 + (BASE_SEV["warp"] - 1.0) * sev)).astype(int),
                      0, T - 1)
        return p[idx], v[idx]
    raise ValueError(kind)


# ============================================================
# 守恒残差（zero-shot critic）—— 使用**精确速度**
# ============================================================
def cons_drift(pos_t, vel_t):
    """四个守恒量在轨迹上的相对漂移之和。越小越物理。

    与 v1 的区别：速度直接来自状态，不再用 np.gradient 反推，
    因此残差里不含有限差分伪影。
    """
    m = np.ones(N) * MASS
    ke = 0.5 * (m * (vel_t * vel_t).sum(-1)).sum(-1)                 # (T,)
    d = pos_t[:, None, :, :] - pos_t[:, :, None, :]                  # (T,N,N,3)
    r = np.sqrt((d * d).sum(-1) + SOFT ** 2)
    eye = 1.0 - np.eye(N)
    mm = m[:, None] * m[None, :]
    pe = -(mm[None] / r * eye[None]).sum((1, 2)) / 2.0
    E = ke + pe

    L = (m[None, :, None] * np.cross(pos_t, vel_t)).sum(1)           # (T,3)
    P = (m[None, :, None] * vel_t).sum(1)                            # (T,3)
    com = (m[None, :, None] * pos_t).sum(1) / m.sum()

    dE = (E.max() - E.min()) / (abs(E.mean()) + 1e-8)
    dL = np.linalg.norm(L.max(0) - L.min(0))
    # 动量：除以典型动量尺度 |m*v| 之和，避免净动量接近 0 时除以噪声
    p_scale = (m * np.linalg.norm(vel_t, axis=-1)).mean() + 1e-12
    dP = np.linalg.norm(P.max(0) - P.min(0)) / p_scale
    dC = np.linalg.norm(com.max(0) - com.min(0))
    return dE + dL + dP + dC


# ============================================================
# 学习式 critic：置换不变 + 时间感知（DeepSets），只看原始状态
# ============================================================
def raw_tensor(pos_t, vel_t):
    """(S, N, 9) = 位置3 + 速度3 + 有限差分加速度3。**不含守恒量。**"""
    a = np.gradient(vel_t, DT, axis=0)
    return np.concatenate([pos_t[SNAP], vel_t[SNAP], a[SNAP]], axis=-1).astype(np.float64)


class DeepSetsCritic(nn.Module):
    def __init__(self, feat=9, h=64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(feat, h), nn.Tanh(),
                                 nn.Linear(h, h), nn.Tanh(), nn.Linear(h, h))
        self.rho = nn.Sequential(nn.Linear(4 * h, h), nn.Tanh(),
                                 nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, x):
        B, S, Nn, _ = x.shape
        e = self.phi(x.reshape(B * S * Nn, -1)).reshape(B, S, Nn, -1)
        pooled = torch.cat([e.mean(2), e.amax(2)], dim=-1)
        g = torch.cat([pooled.mean(1), pooled.amax(1)], dim=-1)
        return self.rho(g).squeeze(-1)


def train_critic(X, Y, Xval, Yval, h, lr, epochs, wd, seed=0, batch=256):
    torch.manual_seed(seed)
    mu = X.mean(axis=(0, 1, 2), keepdims=True)
    sd = X.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor(Y.astype(np.float32))
    Xv = torch.tensor((Xval - mu) / sd, dtype=torch.float32)
    net = DeepSetsCritic(h=h).to(torch.float32)   # 顶部 set_default_dtype(float64) 的坑
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    n = len(Xn)
    best, best_state = -1.0, None
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lossf(net(Xn[idx]), Yt[idx]).backward()
            opt.step()
        if (ep + 1) % 5 == 0:
            net.eval()
            with torch.no_grad():
                a = auroc(net(Xv).numpy(), Yval)
            net.train()
            if a > best:
                best, best_state = a, {k: v.clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, mu, sd, best


def score_net(net, mu, sd, X):
    with torch.no_grad():
        return net(torch.tensor((X - mu) / sd, dtype=torch.float32)).numpy()


# ============================================================
# AUROC
# ============================================================
def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_pos = int((labels == 1).sum()); n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def boot_ci(scores, labels, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    k = len(scores)
    out = [auroc(np.asarray(scores)[i], np.asarray(labels)[i])
           for i in (rng.integers(0, k, k) for _ in range(n))]
    out = np.array([x for x in out if not np.isnan(x)])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ============================================================
# 数据
# ============================================================
def build(fam, n_per_class, rng, sev=1.0):
    """返回 (pos, vel, labels, kinds)，pos/vel 形状 (B,T,N,3)。"""
    P, V, labels, kinds = [], [], [], []
    for _ in range(n_per_class):
        p, v = prior_ic(rng.uniform(*fam), rng)
        pt, vt = rollout_verlet(p, v, np.ones(N) * MASS)
        P.append(pt); V.append(vt); labels.append(1); kinds.append("physical")
    for _ in range(n_per_class):
        p, v = prior_ic(rng.uniform(*fam), rng)
        pt, vt = rollout_verlet(p, v, np.ones(N) * MASS)
        k = CORRUPTIONS[rng.integers(0, len(CORRUPTIONS))]
        cp, cv = corrupt(pt, vt, k, rng, sev)
        P.append(cp); V.append(cv); labels.append(0); kinds.append(k)
    return np.stack(P), np.stack(V), np.array(labels), np.array(kinds)


def main():
    t0 = time.time()
    print("=" * 78); print("构造数据（critic 接收完整状态 pos+vel）"); print("=" * 78)
    ptr, vtr, Ytr, Ktr = build(F_TRAIN, N_TRAIN, np.random.default_rng(1))
    pva, vva, Yva, Kva = build(F_TRAIN, 400, np.random.default_rng(2))
    pid, vid, Yid, Kid = build(F_TRAIN, N_ID, np.random.default_rng(11))
    poe, voe, Yoe, Koe = build(F_TEST, N_OOD, np.random.default_rng(22))
    pow_, vow, Yow, Kow = build(F_TRAIN, N_OOD, np.random.default_rng(33), sev=SEV_SUBTLE)
    pob, vob, Yob, Kob = build(F_TEST, N_OOD, np.random.default_rng(44), sev=SEV_SUBTLE)
    print(f"  train  {len(Ytr):>5}  e∈{F_TRAIN} sev=1.0")
    print(f"  ID     {len(Yid):>5}  e∈{F_TRAIN} sev=1.0")
    print(f"  OOD-族 {len(Yoe):>5}  e∈{F_TEST}  sev=1.0")
    print(f"  OOD-弱 {len(Yow):>5}  e∈{F_TRAIN} sev={SEV_SUBTLE}")
    print(f"  OOD-双 {len(Yob):>5}  e∈{F_TEST}  sev={SEV_SUBTLE}")

    print()
    print("=" * 78); print("守恒残差是否还有伪影？（v1 vs v2 口径对比）"); print("=" * 78)
    phys = np.array([cons_drift(ptr[i], vtr[i]) for i in range(len(Ytr)) if Ytr[i] == 1])
    print(f"  [v2 精确速度] 物理轨迹残差: 中位={np.median(phys):.3e}  "
          f"90分位={np.percentile(phys,90):.3e}")
    print(f"  （v1 用 np.gradient 时为 中位 3.137e-01 —— 该数值几乎全是差分伪影）")

    print()
    print("=" * 78); print("学习式 critic：超参选择（仅用 train 分布）"); print("=" * 78)
    Ttr = np.stack([raw_tensor(ptr[i], vtr[i]) for i in range(len(ptr))])
    Tva = np.stack([raw_tensor(pva[i], vva[i]) for i in range(len(pva))])
    best_cfg, best_score, best_pack = None, -1, None
    for h, lr in itertools.product((64, 128), (3e-3, 1e-3)):
        t_cfg = time.time()
        net, mu, sd, v = train_critic(Ttr, Ytr, Tva, Yva, h, lr, 40, 1e-5)
        print(f"  h={h:<4} lr={lr:<7} val_AUROC={v:.4f}  ({time.time()-t_cfg:.0f}s)", flush=True)
        if v > best_score:
            best_score, best_cfg, best_pack = v, (h, lr, 1e-5), (net, mu, sd)
    net, mu, sd = best_pack
    print(f"  -> 选定 {best_cfg}, val AUROC = {best_score:.4f}")

    def ts(P, V):
        return np.stack([raw_tensor(P[i], V[i]) for i in range(len(P))])

    def sc_learned(P, V):
        return score_net(net, mu, sd, ts(P, V))

    def sc_cons(P, V):
        return -np.array([cons_drift(P[i], V[i]) for i in range(len(P))])

    _hl = (sc_learned(ptr, vtr).mean(), sc_learned(ptr, vtr).std() + 1e-12)
    _hc = (sc_cons(ptr, vtr).mean(), sc_cons(ptr, vtr).std() + 1e-12)

    def sc_hybrid(P, V):
        return (sc_learned(P, V) - _hl[0]) / _hl[1] + (sc_cons(P, V) - _hc[0]) / _hc[1]

    CRITICS = (("守恒 (zero-shot)", sc_cons),
               ("学习式 (DeepSets)", sc_learned),
               ("混合", sc_hybrid))
    sets = [("ID", pid, vid, Yid, Kid),
            ("OOD-族", poe, voe, Yoe, Koe),
            ("OOD-弱", pow_, vow, Yow, Kow),
            ("OOD-双", pob, vob, Yob, Kob)]

    print()
    print("=" * 78); print("判别力 AUROC"); print("=" * 78)
    print(f"  {'critic':<20}" + "".join(f"{nm:>23}" for nm, *_ in sets))
    print("  " + "-" * (20 + 23 * len(sets)))
    results, scores = {}, {}
    for cname, fn in CRITICS:
        cells, row = [], {}
        for nm, P, V, Y, K in sets:
            s = fn(P, V)
            scores[(cname, nm)] = s
            a = auroc(s, Y); lo, hi = boot_ci(s, Y)
            row[nm] = dict(auc=a, ci=[lo, hi])
            cells.append(f"{a:.4f} [{lo:.3f},{hi:.3f}]")
        results[cname] = row
        print(f"  {cname:<20}" + "".join(f"{c:>23}" for c in cells))

    print()
    print("  OOD 相对 ID 的变化：")
    for cname, _ in CRITICS:
        base = results[cname]["ID"]["auc"]
        print(f"    {cname:<20}" + "".join(
            f"{nm}:{results[cname][nm]['auc']-base:+.4f}   " for nm, *_ in sets if nm != "ID"))

    print()
    print("=" * 78); print("分类别 AUROC —— 盲区是否互补？"); print("=" * 78)
    table = {}
    for nm, P, V, Y, K in sets:
        print(f"  [{nm}]")
        print(f"    {'违规':<9}" + "".join(f"{cn:>19}" for cn, _ in CRITICS))
        for k in CORRUPTIONS:
            m = (K == k) | (Y == 1)
            vals = [auroc(scores[(cn, nm)][m], Y[m]) for cn, _ in CRITICS]
            table.setdefault(k, {})[nm] = vals
            print(f"    {k:<9}" + "".join(f"{v:>19.4f}" for v in vals))
        print()

    print("  " + "-" * 74)
    print("  互补性汇总（混合是否在每个切分都不低于单项？）")
    wins = 0
    for nm, *_ in sets:
        hyb = results["混合"][nm]["auc"]
        singles = max(results["守恒 (zero-shot)"][nm]["auc"],
                      results["学习式 (DeepSets)"][nm]["auc"])
        ok = hyb >= singles
        wins += int(ok)
        print(f"    {nm:<8} 混合={hyb:.4f}  最佳单项={singles:.4f}  "
              f"{'混合更优/持平' if ok else '单项更优'}  ({hyb-singles:+.4f})")
    print(f"  -> {wins}/{len(sets)} 个切分上混合不低于最佳单项")

    # 严重度扫描
    print()
    print("=" * 78); print("严重度扫描（违规越隐蔽，谁先失效）"); print("=" * 78)
    sevs = [0.1, 0.15, 0.25, 0.4, 0.6, 1.0, 2.0]
    print(f"  {'sev':<8}" + "".join(f"{cn:>20}" for cn, _ in CRITICS))
    sweep = {cn: [] for cn, _ in CRITICS}
    for sv in sevs:
        P, V, Y, K = build(F_TRAIN, 400, np.random.default_rng(int(sv * 1000) + 7), sev=sv)
        cells = []
        for cn, fn in CRITICS:
            a = auroc(fn(P, V), Y)
            sweep[cn].append(a)
            cells.append(f"{a:>20.4f}")
        print(f"  {sv:<8}" + "".join(cells), flush=True)

    out = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                       "results", "experiment_a1_critic_quality.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": dict(N=N, steps=STEPS, dt=DT, f_train=list(F_TRAIN),
                                  f_test=list(F_TEST), n_train=N_TRAIN,
                                  sev_subtle=SEV_SUBTLE, state="pos+vel (v2)",
                                  learned_cfg=dict(h=best_cfg[0], lr=best_cfg[1],
                                                   wd=best_cfg[2]),
                                  learned_val_auroc=best_score,
                                  cons_drift_median_physical=float(np.median(phys))),
                   "auroc": results,
                   "per_corruption": {k: v for k, v in table.items()},
                   "severity_sweep": dict(sevs=sevs, **sweep)},
                  f, ensure_ascii=False, indent=1)
    print(f"\n耗时 {time.time()-t0:.0f}s  [写出] {out}")


if __name__ == "__main__":
    main()
