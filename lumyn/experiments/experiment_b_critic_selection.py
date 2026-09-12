"""实验 B：critic 作为**监督信号**是否真的有用？（选择实验）

为什么换掉 AUROC
----------------
A2 用「区分类别」的 AUROC 作指标，需要预先定义「什么算无效」。实测：
神经生成器单步加速度相对误差仅 0.0032，生成轨迹相对真实解的 RMSE 中位 0.0002，
按阈值 0.02 判定，**0.00% 的生成轨迹是无效的** —— 负样本集为空，AUROC 必然 ≈0.5。
那是我自己的方法失败，不是发现。

本实验改问一个不需要阈值、且直接对应论文主张的问题：

    「给定同一个初始条件下的一批候选生成轨迹，
      用 critic 打分挑出的那个，是否真的更接近真实解？」

这就是 critic 作为**监督信号**的定义。若用 critic 选出的候选与随机挑选无异，
则 critic 无监督价值，无论它的 AUROC 多高。

指标
----
1. **Spearman 相关**：critic 分数 vs 候选的真实轨迹误差（越高越好，理想 ≈ +1）
2. **选择收益**：critic 选出的候选误差，相对随机挑选的降幅
3. **互补性**：混合 critic 的选择收益是否高于任一单项
4. **oracle 上界**：若完美挑选能达到的误差（衡量还有多少空间）

生成器质量谱
------------
A2 的生成器太准了。本实验刻意用一个**受限训练预算**的生成器（数据少、轮次少），
并在自回归推进时注入过程噪声 σ 形成候选分布 —— 这正对应真实生成模型
「多次采样得到一批质量不一的样本」的情形。
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

F_TRAIN = (0.0, 0.3)
F_TEST = (0.5, 0.8)

from lumyn.experiments.experiment_a1_critic_quality import (   # noqa: E402
    prior_ic, rollout_verlet, cons_drift, raw_tensor,
    train_critic, score_net, auroc, boot_ci, DeepSetsCritic,
)
from lumyn.experiments.experiment_a2_neural_generator import (  # noqa: E402
    NeuralForce, train_force_model,
)

N_IC = 200            # 测试用初始条件数
K = 8                 # 每个初始条件生成多少候选
SIGMAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.6, 1.0)   # 过程噪声谱


# ============================================================
# 带过程噪声的生成
# ============================================================
def rollout_noisy(pos, vel, net, sigma, rng):
    """用学到的力场推进，每步注入 sigma 比例的过程噪声。"""
    p = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)
    v = torch.tensor(vel, dtype=torch.float32).unsqueeze(0)
    m = torch.tensor(np.ones(N) * MASS, dtype=torch.float32)
    ps, vs = [p[0].numpy().astype(np.float64)], [v[0].numpy().astype(np.float64)]
    with torch.no_grad():
        for _ in range(STEPS):
            a = net(p, v, m)
            if sigma > 0:
                a_scale = a.abs().mean() + 1e-8
                a = a + torch.tensor(
                    rng.normal(0, sigma, a.shape) * float(a_scale),
                    dtype=torch.float32)
            v = v + a * DT
            p = p + v * DT
            ps.append(p[0].numpy().astype(np.float64))
            vs.append(v[0].numpy().astype(np.float64))
    return np.stack(ps), np.stack(vs)


def rms(traj, ref):
    return float(np.sqrt(((traj - ref) ** 2).mean()))


# ============================================================
# 最强监督基线：直接回归真实误差
# ============================================================
def train_error_predictor(X, Yerr, Xval, Yerrval, h=128, lr=1e-3, epochs=80,
                          seed=0, batch=128):
    """同样的 DeepSets 架构，但目标是**回归真实轨迹误差的对数**。

    这是本题最强的监督基线：它直接用了「真实误差」这个标签。
    若守恒 critic（零样本、无需任何标签）能与之接近，则是强结论；
    若明显不如，则如实报告「守恒只是廉价的代理，劣于监督回归」。
    """
    torch.manual_seed(seed)
    mu = X.mean(axis=(0, 1, 2), keepdims=True)
    sd = X.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    Xn = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor(np.log(Yerr + 1e-12), dtype=torch.float32)
    Xv = torch.tensor((Xval - mu) / sd, dtype=torch.float32)
    Yv = np.log(np.asarray(Yerrval) + 1e-12)
    net = DeepSetsCritic(h=h).to(torch.float32)
    # 输出层改成回归头（去掉原来的 1 维分类头即可复用同一架构）
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(Xn)
    best, best_state = -1.0, None
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            ((net(Xn[idx]) - Yt[idx]) ** 2).mean().backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                rho = spearman(net(Xv).numpy(), Yv)
            net.train()
            if rho > best:
                best, best_state = rho, {k: v.clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, mu, sd, best


# ============================================================
# Spearman 秩相关
# ============================================================
def spearman(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(1, len(a) + 1)
        return r
    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / (den + 1e-30))


def main():
    t0 = time.time()
    print("=" * 82)
    print("1. 受限预算的生成器（刻意不做准，以产生有质量差异的候选）")
    print("=" * 82)
    rng = np.random.default_rng(1)
    Ptr, Vtr = [], []
    for _ in range(120):                       # 刻意少的数据
        p, v = prior_ic(rng.uniform(*F_TRAIN), rng)
        pt, vt = rollout_verlet(p, v, np.ones(N) * MASS)
        Ptr.append(pt); Vtr.append(vt)
    Ptr, Vtr = np.stack(Ptr), np.stack(Vtr)
    net = train_force_model(Ptr, Vtr, np.ones(N) * MASS, epochs=12, h=32, lr=3e-3)
    with torch.no_grad():
        Pt = torch.tensor(Ptr[:32], dtype=torch.float32)
        Vt = torch.tensor(Vtr[:32], dtype=torch.float32)
        At = ((Vt[:, 1:] - Vt[:, :-1]) / DT).reshape(-1, N, 3)
        Ap = net(Pt[:, :-1].reshape(-1, N, 3), Vt[:, :-1].reshape(-1, N, 3),
                 torch.tensor(np.ones(N) * MASS, dtype=torch.float32))
        rel = float(((Ap - At) ** 2).mean().sqrt() / (At ** 2).mean().sqrt())
    print(f"   训练轨迹 {len(Ptr)} 条, 12 轮, h=32")
    print(f"   单步加速度相对误差 = {rel:.4f}  (A2 的完整预算为 0.0032)")

    # ---------------- 生成候选 ----------------
    print()
    print("=" * 82)
    print(f"2. 生成候选：{N_IC} 个初始条件 × {K} 个候选（过程噪声谱 {SIGMAS}）")
    print("=" * 82)
    datasets = {}
    for fam_name, fam, seed in (("ID", F_TRAIN, 11), ("OOD-族", F_TEST, 22)):
        rgen = np.random.default_rng(seed)
        ICs, TRUES, CANDS, ERRORS = [], [], [], []
        for _ in range(N_IC):
            p, v = prior_ic(rgen.uniform(*fam), rgen)
            true_traj, _ = rollout_verlet(p, v, np.ones(N) * MASS)
            cands = []
            for s in SIGMAS:
                cp, cv = rollout_noisy(p, v, net, s, rgen)
                cands.append((cp, cv))
            ICs.append((p, v)); TRUES.append(true_traj); CANDS.append(cands)
            ERRORS.append([rms(cp, true_traj) for cp, _ in cands])
        ERRORS = np.array(ERRORS)
        datasets[fam_name] = (ICs, TRUES, CANDS, ERRORS)
        print(f"   [{fam_name}] 候选误差 RMSE: 中位={np.median(ERRORS):.4f} "
              f"min={ERRORS.min():.5f} max={ERRORS.max():.4f}")
        print(f"        最佳候选(逐 IC 取 min)中位 = {np.median(ERRORS.min(1)):.4f}  "
              f"随机候选(逐 IC 取均值)中位 = {np.median(ERRORS.mean(1)):.4f}")
        print(f"        候选间质量差异（逐 IC 的 max/min 比）中位 = "
              f"{np.median(ERRORS.max(1) / (ERRORS.min(1) + 1e-12)):.2f}x")

    # ---------------- 学习式 critic ----------------
    print()
    print("=" * 82)
    print("3. 训练学习式 critic（正=真实轨迹，负=候选轨迹）")
    print("=" * 82)
    ICs, TRUES, CANDS, ERRORS = datasets["ID"]
    n_ic = len(ICs)
    n_tr = n_ic // 2

    def stack(P, V):
        return np.stack([raw_tensor(P[i], V[i]) for i in range(len(P))])

    def build_split(lo, hi):
        """按**初始条件**切分，保证真实轨迹与它的候选落在同一侧。

        早先按样本数对半切会把某个 IC 的真实轨迹与候选分开，导致
        验证集里正负样本分布错位、误差标签对不上。
        """
        Pp, Vv, Yy, Ee = [], [], [], []
        for i in range(lo, hi):
            p, v = ICs[i]
            tp, tv = rollout_verlet(p, v, np.ones(N) * MASS)
            Pp.append(tp); Vv.append(tv); Yy.append(1.0); Ee.append(0.0)
            for j, (cp, cv) in enumerate(CANDS[i]):
                Pp.append(cp); Vv.append(cv); Yy.append(0.0); Ee.append(ERRORS[i][j])
        return Pp, Vv, np.array(Yy), np.array(Ee)

    Xtr_p, Xtr_v, Ytr, Etr = build_split(0, n_tr)
    Xva_p, Xva_v, Yva, Eva = build_split(n_tr, n_ic)
    Xtr, Xva = stack(Xtr_p, Xtr_v), stack(Xva_p, Xva_v)
    print(f"   训练样本 {len(Ytr)}（{n_tr} 个 IC） | 验证样本 {len(Yva)}"
          f"（{n_ic - n_tr} 个 IC）")

    best = (-1, None)
    for h, lr in itertools.product((64, 128), (3e-3, 1e-3)):
        netc, mu, sd, v_auc = train_critic(Xtr, Ytr, Xva, Yva, h, lr, 60, 1e-5)
        print(f"   [二分类] h={h:<4} lr={lr:<7} val_AUROC={v_auc:.4f}", flush=True)
        if v_auc > best[0]:
            best = (v_auc, (netc, mu, sd, h, lr))
    val_auc, (critic, mu, sd, bh, blr) = best
    print(f"   -> 二分类 critic 选定 h={bh} lr={blr}, val AUROC={val_auc:.4f}")

    def sc_learned(pos_t, vel_t):
        return float(score_net(critic, mu, sd,
                               np.stack([raw_tensor(pos_t, vel_t)]))[0])

    # 最强监督基线：直接回归真实误差
    ep_net, ep_mu, ep_sd, ep_rho = train_error_predictor(
        Xtr, Etr, Xva, Eva, h=128, lr=1e-3, epochs=80)
    print(f"   [误差回归] h=128 lr=1e-3  val Spearman(log误差)={ep_rho:+.4f}")

    def sc_errpred(pos_t, vel_t):
        # 网络输出 log(误差)，取负使其与其它 critic 方向一致：分数越高越好
        return -float(score_net(ep_net, ep_mu, ep_sd,
                                np.stack([raw_tensor(pos_t, vel_t)]))[0])

    def sc_cons(pos_t, vel_t):
        return -cons_drift(pos_t, vel_t)

    # 混合的归一化常数：只在 ID 训练切分上定
    l_tr = np.array([sc_learned(Xtr_p[i], Xtr_v[i]) for i in range(len(Xtr_p))])
    c_tr = np.array([sc_cons(Xtr_p[i], Xtr_v[i]) for i in range(len(Xtr_p))])
    e_tr = np.array([sc_errpred(Xtr_p[i], Xtr_v[i]) for i in range(len(Xtr_p))])
    l_mu, l_sd = l_tr.mean(), l_tr.std() + 1e-12
    c_mu, c_sd = c_tr.mean(), c_tr.std() + 1e-12
    e_mu, e_sd = e_tr.mean(), e_tr.std() + 1e-12

    def sc_hybrid(pos_t, vel_t):
        return (sc_learned(pos_t, vel_t) - l_mu) / l_sd + \
               (sc_cons(pos_t, vel_t) - c_mu) / c_sd

    def sc_hybrid3(pos_t, vel_t):
        return ((sc_learned(pos_t, vel_t) - l_mu) / l_sd
                + (sc_cons(pos_t, vel_t) - c_mu) / c_sd
                + (sc_errpred(pos_t, vel_t) - e_mu) / e_sd)

    # ---------------- 学习式融合：互补型主张的决定性检验 ----------------
    # 前两个「混合」只是把标准化后的分数等权相加，权重是任意的。
    # 这里改为**在训练集上学出最优线性组合**（逻辑回归，标签=该候选误差是否
    # 低于同一 IC 的候选误差中位数）。
    # 若连这个最优组合都打不过守恒单项，则三个 critic 携带的信号是**冗余**的，
    # 而非互补的 —— 互补型叙事不成立。
    def feat(pos_t, vel_t):
        return [(sc_learned(pos_t, vel_t) - l_mu) / l_sd,
                (sc_cons(pos_t, vel_t) - c_mu) / c_sd,
                (sc_errpred(pos_t, vel_t) - e_mu) / e_sd]

    Ftr, ytr = [], []
    for i in range(n_tr):
        errs = ERRORS[i]
        med = np.median(errs)
        for j, (cp, cv) in enumerate(CANDS[i]):
            Ftr.append(feat(cp, cv))
            ytr.append(1.0 if errs[j] <= med else 0.0)
    Ftr_t = torch.tensor(np.array(Ftr), dtype=torch.float32)
    ytr_t = torch.tensor(np.array(ytr), dtype=torch.float32)
    # 必须显式 float32：本模块顶部 set_default_dtype(float64) 会让 torch.zeros 也是
    # float64，与 float32 特征相加时报 addmv dtype 不一致。
    wb = torch.zeros(4, dtype=torch.float32, requires_grad=True)
    optf = torch.optim.Adam([wb], lr=0.05)
    for _ in range(800):
        optf.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(
            Ftr_t @ wb[:3] + wb[3], ytr_t).backward()
        optf.step()
    W = wb.detach().numpy()
    print(f"   [融合] 训练集上学到的权重："
          f"二分类 {W[0]:+.3f} | 守恒 {W[1]:+.3f} | 误差回归 {W[2]:+.3f} | b {W[3]:+.3f}")
    print(f"   [融合] 训练集样本 {len(ytr)}（{n_tr} 个 IC × {K} 候选）")

    def sc_fusion(pos_t, vel_t):
        return float(np.dot(W[:3], feat(pos_t, vel_t)) + W[3])

    CRITICS = (("守恒 (zero-shot)", sc_cons),
               ("学习式 (二分类)", sc_learned),
               ("监督回归 (最强基线)", sc_errpred),
               ("混合 守恒+学习式", sc_hybrid),
               ("混合 三者", sc_hybrid3),
               ("融合 (逻辑回归-最优组合)", sc_fusion))

    # ---------------- 评估 ----------------
    for fam_name in ("ID", "OOD-族"):
        ICs, TRUES, CANDS, ERRORS = datasets[fam_name]
        print()
        print("=" * 82)
        print(f"[{fam_name}] critic 能否预测真实误差 / 能否挑出更好的候选")
        print("=" * 82)

        # 逐候选收集分数，**保留 (IC, 候选) 的二维结构**。
        #
        # pooled 与 within-IC 是两个不同的量，不能用错：
        #   pooled   : 把全部 IC 的全部候选混在一起排序，主要反映「IC 之间的质量差异」
        #              （σ 大的候选整体就更差），因此即使 critic 完全不会在 IC 内排序，
        #              pooled 相关也可能很高。
        #   within-IC: 只在同一 IC 的 K 个候选内排序 —— 这才对应选择任务。
        # 早先版本只报告 pooled，会高估 critic 的实际选择能力。
        S = {cn: np.zeros((len(ICs), K)) for cn, _ in CRITICS}
        for i in range(len(ICs)):
            for j, (cp, cv) in enumerate(CANDS[i]):
                for cn, fn in CRITICS:
                    S[cn][i, j] = fn(cp, cv)
        flat_err = ERRORS.reshape(-1)

        print(f"   {'critic':<22}{'pooled ρ':>11}{'within-IC ρ':>14}{'IC内更优率':>12}")
        print("   " + "-" * 60)
        for cn, _ in CRITICS:
            pooled = spearman(S[cn].reshape(-1), flat_err)
            per_ic = [spearman(S[cn][i], ERRORS[i]) for i in range(len(ICs))]
            per_ic = [r for r in per_ic if not np.isnan(r)]
            within = float(np.mean(per_ic)) if per_ic else float("nan")
            better = np.mean([ERRORS[i][int(np.argmax(S[cn][i]))] < np.median(ERRORS[i])
                              for i in range(len(ICs))])
            print(f"   {cn:<22}{pooled:>+11.4f}{within:>+14.4f}{better:>11.1%}")
        print("   分数越高=越物理/越准，故 ρ 应为负；IC内更优率 0.5 = 无选择能力")

        # 选择实验
        print()
        print(f"   {'critic':<22}{'选中误差(中位)':>16}{'相对随机':>11}{'相对oracle':>13}")
        print("   " + "-" * 62)
        rand_err = np.median(ERRORS.mean(1))
        orac_err = np.median(ERRORS.min(1))
        print(f"   {'随机挑选':<22}{rand_err:>16.5f}{1.0:>10.3f}x{rand_err/orac_err:>12.2f}x")
        print(f"   {'oracle(上界)':<22}{orac_err:>16.5f}{orac_err/rand_err:>10.3f}x{1.0:>12.2f}x")
        for cn, fn in CRITICS:
            picked = []
            for i in range(len(ICs)):
                ss = [fn(cp, cv) for cp, cv in CANDS[i]]
                picked.append(ERRORS[i][int(np.argmax(ss))])
            med = float(np.median(picked))
            print(f"   {cn:<22}{med:>16.5f}{med/rand_err:>10.3f}x{med/orac_err:>12.2f}x")

    # ---------------- 检出：候选误差 > 中位数 ----------------
    print()
    print("=" * 82)
    print("4. 相对标签下的检出能力（误差 > 该 IC 的候选误差中位数 → 视为劣质）")
    print("=" * 82)
    for fam_name in ("ID", "OOD-族"):
        ICs, TRUES, CANDS, ERRORS = datasets[fam_name]
        lab, sc = {cn: [] for cn, _ in CRITICS}, {cn: [] for cn, _ in CRITICS}
        for i in range(len(ICs)):
            thr = np.median(ERRORS[i])
            for j, (cp, cv) in enumerate(CANDS[i]):
                for cn, fn in CRITICS:
                    sc[cn].append(fn(cp, cv))
                    lab[cn].append(1.0 if ERRORS[i][j] <= thr else 0.0)
        print(f"   [{fam_name}]")
        for cn, _ in CRITICS:
            a = auroc(np.array(sc[cn]), np.array(lab[cn]))
            lo, hi = boot_ci(np.array(sc[cn]), np.array(lab[cn]))
            print(f"     {cn:<22} AUROC = {a:.4f} [{lo:.3f},{hi:.3f}]")

    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
