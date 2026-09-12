"""实验 A2：用**真实生成模型**的输出检验互补型主张。

为什么必须换掉 A1 的损坏集
--------------------------
A1 用人工损坏（noise/euler/freeze/splice/warp）制造负样本，被证伪两次：
  1. 守恒通道里 angular/momentum 的物理基线是 1e-16（机器精度），其「漂移」
     是浮点噪声，AUROC 无意义。
  2. freeze 反而使测得的残差**变小**，所有通道 AUROC 低于随机 —— 它不是
     有效的物理违规。
  3. 一个**不含任何物理知识**的通用平滑度统计量 (jerk) 在 splice 上得 0.95，
     高于任何守恒通道 —— 说明那些损坏可被通用启发式识破。

本实验改为：**训练一个神经动力学模型（成对 GNN），用它自回归生成的轨迹做负样本。**

为什么这更正确
--------------
- 生成轨迹是**平滑可信**的：它由同一个积分器推进，只是力模型是学出来的，
  通用平滑度启发式难以识破。
- 失效模式**自然涌现**：不是人工设计的 splice/freeze，而是学到的力场在
  分布外、近距离交会、误差累积处的真实偏差。
- 标签**明确定义**：每条生成轨迹都有对应的真实解（同一初始条件），
  可直接算轨迹误差。
- 有**天然的严重度轴**：控制生成器的训练量 → 生成质量连续变化。

判据
----
  互补型主张成立需同时满足：
  1. 存在生成轨迹被守恒 critic 抓住、而学习式 critic 漏掉（反之亦然）
  2. 二者结合（混合）优于任一单项
  3. 上述成立在**分布外**也成立
  若两个 critic 的失效实例几乎重合（无互补），主张不成立。
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

from lumyn.experiments.experiment_a1_critic_quality import (   # noqa: E402
    prior_ic, accel, rollout_verlet, cons_drift, raw_tensor,
    DeepSetsCritic, train_critic, score_net, auroc, boot_ci,
)

# 判「物理合法」的阈值：生成轨迹相对真实解的末端误差超过该值即视为无效
INVALID_RMSE = 0.02


# ============================================================
# 神经动力学模型：成对 GNN，置换不变
# ============================================================
class NeuralForce(nn.Module):
    """逐对力模型：f_ij = MLP([r_ij, |r_ij|, v_ij, m_i, m_j])，沿 r̂_ij 方向。

    置换不变（对粒子求和），且按对计算，因此规模可泛化。
    只预测**力**，不预测位置增量的黑箱 —— 这样其误差集中在物理量上。
    """

    def __init__(self, h=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, 1),
        )

    def forward(self, pos, vel, mass):
        """pos, vel: (B,N,3); mass: (N,) 或 (B,N)。返回 (B,N,3) 加速度。"""
        B, Nn, _ = pos.shape
        if mass.dim() == 1:
            mass = mass.unsqueeze(0).expand(B, -1)
        d = pos[:, None, :, :] - pos[:, :, None, :]              # (B,N,N,3) r_ij
        dv = vel[:, None, :, :] - vel[:, :, None, :]             # (B,N,N,3)
        r = d.norm(dim=-1, keepdim=True)                         # (B,N,N,1)
        mi = mass[:, :, None, None].expand(B, Nn, Nn, 1)
        mj = mass[:, None, :, None].expand(B, Nn, Nn, 1)
        feat = torch.cat([d, r, dv, mi, mj], dim=-1)             # (B,N,N,9)
        f = self.mlp(feat).squeeze(-1)                           # (B,N,N) 力的大小
        eye = torch.eye(Nn, dtype=torch.bool)
        f = f.masked_fill(eye, 0.0)
        r_hat = d / (r + 1e-6)
        # 除以粒子 i 自己的质量：(B,N,1)；此前误用 mi.squeeze(-1) 得 (B,N,N)，维度不匹配
        a = (f.unsqueeze(-1) * r_hat).sum(dim=2) / mass.unsqueeze(-1)
        return a


def train_force_model(P, V, M, epochs=60, h=64, lr=2e-3, batch=64, seed=0):
    """用真实轨迹做单步监督训练。P,V: (B,T,N,3)。"""
    torch.manual_seed(seed)
    net = NeuralForce(h=h).to(torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Pt = torch.tensor(P, dtype=torch.float32)
    Vt = torch.tensor(V, dtype=torch.float32)
    B, Tn, Nn, _ = Pt.shape
    Mt = torch.tensor(M, dtype=torch.float32)
    # 监督信号：真实加速度 a = (v_{t+1} - v_t)/dt
    A = (Vt[:, 1:] - Vt[:, :-1]) / DT                            # (B,T-1,N,3)
    Xp, Xv, Ya = Pt[:, :-1].reshape(-1, Nn, 3), Vt[:, :-1].reshape(-1, Nn, 3), \
        A.reshape(-1, Nn, 3)
    n = len(Xp)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            a_pred = net(Xp[idx], Xv[idx], Mt)
            loss = ((a_pred - Ya[idx]) ** 2).mean() * 1e4
            loss.backward()
            opt.step()
            tot += float(loss)
    net.eval()
    return net


def rollout_neural(pos, vel, mass, net, steps=STEPS, dt=DT):
    """用学到的力场做辛欧拉推进，返回 (pos_traj, vel_traj)。"""
    p = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)
    v = torch.tensor(vel, dtype=torch.float32).unsqueeze(0)
    m = torch.tensor(mass, dtype=torch.float32)
    ps, vs = [p[0].numpy().astype(np.float64)], [v[0].numpy().astype(np.float64)]
    with torch.no_grad():
        for _ in range(steps):
            a = net(p, v, m)
            v = v + a * dt
            p = p + v * dt
            ps.append(p[0].numpy().astype(np.float64))
            vs.append(v[0].numpy().astype(np.float64))
    return np.stack(ps), np.stack(vs)


# ============================================================
# 数据
# ============================================================
def make_traj(n, fam, rng):
    P, V = [], []
    for _ in range(n):
        p, v = prior_ic(rng.uniform(*fam), rng)
        pt, vt = rollout_verlet(p, v, np.ones(N) * MASS)
        P.append(pt); V.append(vt)
    return np.stack(P), np.stack(V)


def gen_pairs(P, V, net):
    """对每条真实轨迹，用神经生成器从同一初始条件生成一条轨迹。"""
    GP, GV, ERR = [], [], []
    for i in range(len(P)):
        gp, gv = rollout_neural(P[i][0], V[i][0], np.ones(N) * MASS, net)
        GP.append(gp); GV.append(gv)
        ERR.append(float(np.sqrt(((gp - P[i]) ** 2).mean())))
    return np.stack(GP), np.stack(GV), np.array(ERR)


def main():
    t0 = time.time()
    print("=" * 80)
    print("1. 训练神经动力学生成器（仅用 train 族 e∈[0,0.3]）")
    print("=" * 80)
    Ptr, Vtr = make_traj(500, F_TRAIN, np.random.default_rng(1))
    print(f"   训练轨迹 {len(Ptr)} 条 × {T} 步")
    net = train_force_model(Ptr, Vtr, np.ones(N) * MASS, epochs=60)
    # 训练集上的单步精度
    # net 期望 (B, N, 3)，而 Pt[:, :-1] 是 (B, T-1, N, 3)：必须先把时间维摊平。
    with torch.no_grad():
        B0, Nn = Ptr.shape[0], N
        Pt = torch.tensor(Ptr[:64], dtype=torch.float32)
        Vt = torch.tensor(Vtr[:64], dtype=torch.float32)
        Atrue = ((Vt[:, 1:] - Vt[:, :-1]) / DT).reshape(-1, Nn, 3)
        Apred = net(Pt[:, :-1].reshape(-1, Nn, 3), Vt[:, :-1].reshape(-1, Nn, 3),
                    torch.tensor(np.ones(N) * MASS, dtype=torch.float32))
        rel = float(((Apred - Atrue) ** 2).mean().sqrt()
                    / (Atrue ** 2).mean().sqrt())
    print(f"   单步加速度相对误差 = {rel:.4f}")

    print()
    print("=" * 80)
    print("2. 生成轨迹的质量（相对真实解的 RMSE）")
    print("=" * 80)
    sets = {}
    for nm, fam, seed in (("ID", F_TRAIN, 11), ("OOD-族", F_TEST, 22)):
        P, V = make_traj(400, fam, np.random.default_rng(seed))
        GP, GV, ERR = gen_pairs(P, V, net)
        sets[nm] = (P, V, GP, GV, ERR)
        print(f"   [{nm}] 生成轨迹 RMSE: 中位={np.median(ERR):.4f} "
              f"90分位={np.percentile(ERR,90):.4f} "
              f"max={ERR.max():.4f}  |  超阈值({INVALID_RMSE})比例="
              f"{(ERR > INVALID_RMSE).mean():.2%}")

    print()
    print("=" * 80)
    print("3. 训练学习式 critic：正=真实轨迹，负=神经生成轨迹")
    print("=" * 80)
    # 用 ID 分布的另一批数据训练（生成器的训练集之外）
    P0, V0 = make_traj(600, F_TRAIN, np.random.default_rng(2))
    G0P, G0V, _ = gen_pairs(P0, V0, net)
    Pv, Vv = make_traj(150, F_TRAIN, np.random.default_rng(3))
    GvP, GvV, _ = gen_pairs(Pv, Vv, net)

    def stack(P, V):
        return np.stack([raw_tensor(P[i], V[i]) for i in range(len(P))])

    Xtr = np.concatenate([stack(P0, V0), stack(G0P, G0V)])
    Ytr = np.concatenate([np.ones(len(P0)), np.zeros(len(G0P))])
    Xva = np.concatenate([stack(Pv, Vv), stack(GvP, GvV)])
    Yva = np.concatenate([np.ones(len(Pv)), np.zeros(len(GvP))])

    best = (-1, None)
    for h, lr in itertools.product((64, 128), (3e-3, 1e-3)):
        netc, mu, sd, v = train_critic(Xtr, Ytr, Xva, Yva, h, lr, 40, 1e-5)
        print(f"   h={h:<4} lr={lr:<7} val_AUROC={v:.4f}", flush=True)
        if v > best[0]:
            best = (v, (netc, mu, sd, h, lr))
    val_auc, (critic, mu, sd, bh, blr) = best
    print(f"   -> 选定 h={bh} lr={blr}, val AUROC={val_auc:.4f}")

    def sc_learned(P, V):
        return score_net(critic, mu, sd, stack(P, V))

    def sc_cons(P, V):
        return -np.array([cons_drift(P[i], V[i]) for i in range(len(P))])

    _l = (sc_learned(P0, V0).mean(), sc_learned(P0, V0).std() + 1e-12)
    _c = (sc_cons(P0, V0).mean(), sc_cons(P0, V0).std() + 1e-12)

    def sc_hybrid(P, V):
        return (sc_learned(P, V) - _l[0]) / _l[1] + (sc_cons(P, V) - _c[0]) / _c[1]

    print()
    print("=" * 80)
    print("4. 判别力：区分「真实轨迹」与「神经生成轨迹」")
    print("=" * 80)
    print(f"   {'critic':<20}{'ID':>20}{'OOD-族':>20}")
    print("   " + "-" * 58)
    res = {}
    for cname, fn in (("守恒 (zero-shot)", sc_cons),
                      ("学习式 (DeepSets)", sc_learned),
                      ("混合", sc_hybrid)):
        row = []
        for nm in ("ID", "OOD-族"):
            P, V, GP, GV, ERR = sets[nm]
            PX = np.concatenate([P, GP]); VX = np.concatenate([V, GV])
            YX = np.concatenate([np.ones(len(P)), np.zeros(len(GP))])
            s = fn(PX, VX)
            a = auroc(s, YX); lo, hi = boot_ci(s, YX)
            res[(cname, nm)] = (a, s, YX)
            row.append(f"{a:.4f} [{lo:.3f},{hi:.3f}]")
        print(f"   {cname:<20}" + "".join(f"{c:>20}" for c in row))

    print()
    print("=" * 80)
    print("5. 互补性：两个 critic 抓住的是不是**不同的**实例？")
    print("=" * 80)
    print("   把「生成轨迹」按两个 critic 是否判为无效分类（阈值取各自中位数）。")
    for nm in ("ID", "OOD-族"):
        P, V, GP, GV, ERR = sets[nm]
        PX = np.concatenate([P, GP]); VX = np.concatenate([V, GV])
        nreal = len(P)
        sl = sc_learned(PX, VX)[nreal:]
        sc = sc_cons(PX, VX)[nreal:]
        tl, tc = np.median(sl), np.median(sc)
        flag_l = sl < tl          # 学习式判为「低分」→ 认为是生成物
        flag_c = sc < tc
        both = int((flag_l & flag_c).sum()); onlyl = int((flag_l & ~flag_c).sum())
        onlyc = int((~flag_l & flag_c).sum()); neither = int((~flag_l & ~flag_c).sum())
        n = len(sl)
        print(f"   [{nm}] n={n}")
        print(f"      两个都抓到      : {both:>4} ({both/n:.1%})")
        print(f"      仅学习式抓到    : {onlyl:>4} ({onlyl/n:.1%})")
        print(f"      仅守恒抓到      : {onlyc:>4} ({onlyc/n:.1%})")
        print(f"      两个都漏掉      : {neither:>4} ({neither/n:.1%})")
        agree = (flag_l == flag_c).mean()
        print(f"      判定一致率      : {agree:.1%}  "
              f"（若接近 100%，说明两者**冗余**而非互补）")

    print()
    print("=" * 80)
    print("6. 逐通道守恒残差：生成的轨迹真的违反守恒吗？")
    print("=" * 80)
    from lumyn.experiments.audit_cons_channels import channels, CH
    for nm in ("ID", "OOD-族"):
        P, V, GP, GV, ERR = sets[nm]
        print(f"   [{nm}]")
        print(f"      {'通道':<10}{'真实(中位)':>16}{'生成(中位)':>16}{'比值':>10}")
        for c in CH:
            a = np.median([channels(P[i], V[i])[c] for i in range(len(P))])
            b = np.median([channels(GP[i], GV[i])[c] for i in range(len(GP))])
            print(f"      {c:<10}{a:>16.4e}{b:>16.4e}{b/(a+1e-30):>10.3f}")

    out = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                       "results", "experiment_a2_neural_generator.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": dict(N=N, steps=STEPS, dt=DT, one_step_rel_err=rel,
                                  invalid_rmse=INVALID_RMSE, critic_cfg=[bh, blr],
                                  critic_val_auroc=val_auc),
                   "auroc": {f"{c}|{n}": res[(c, n)][0] for c, n in res}},
                  f, ensure_ascii=False, indent=1)
    print(f"\n耗时 {time.time()-t0:.0f}s  [写出] {out}")


if __name__ == "__main__":
    main()
