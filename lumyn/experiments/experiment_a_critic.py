"""实验 A：守恒律 critic vs 学习式 critic（主指标为跨分布泛化）。

问题设定（反问题）
------------------
给定 N 体演化的**目标末态形态**，反演**初始条件**。
这是 Lumyn 的"梯度引导生成"所解的问题。

三种 critic
-----------
  A1 learned    ：小型 MLP，在"真实物理轨迹 vs 扰动轨迹"上训练二分类
  A2 conservation：守恒律残差（能量/动量/角动量/质心），无需任何训练数据
  A3 hybrid     ：两者相加

跨分布协议（关键）
-----------------
  critic 只在 **train 族**（椭圆离心率 e ∈ [0, 0.3]）上训练；
  在 **test 族**（e ∈ [0.5, 0.8]）上评估 —— 后者在 critic 训练时从未出现。
  守恒律 critic 因为不需要训练数据，理论上应在此占优。

可证伪的预测
-----------
  A2 在同分布上可能与 A1 相当，但在跨分布上应显著更优。
  若 A2 在跨分布上也不占优 → 范式不成立。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from lumyn.physics.differentiable import DifferentiableNBody
from lumyn.physics.losses import (
    PhysicsLosses, angular_momentum, center_of_mass, total_energy, total_momentum,
)

torch.set_default_dtype(torch.float64)

# ---------------- 配置 ----------------
N = 6
STEPS = 40          # 可微前向步数
DT = 0.01
SOFT = 0.05
OPT_STEPS = 50      # 反演优化步数（瓶颈：384 次反演 × 每步一次 40 步可微 N 体）
LR = 0.05
SEEDS = (0, 1, 2)
N_TRAIN_T = 5       # train 族目标数
N_TEST_T = 5        # test 族目标数

F_TRAIN = (0.0, 0.3)    # 训练分布
F_TEST = (0.5, 0.8)     # 测试分布（critic 未见）
N_EVAL = 3              # 每个目标重复评估次数


# ---------------- 目标族 ----------------
def make_gt_ic(e: float, rng) -> tuple:
    """按离心率 e 生成一组"真值初始条件"。"""
    ang = rng.uniform(0, 2 * np.pi, N)
    a = 1.0
    r = a * (1 - e ** 2) / (1 + e * np.cos(ang))
    pos = np.stack([r * np.cos(ang), r * np.sin(ang),
                    rng.normal(0, 0.05, N)], axis=1)
    v = np.sqrt(1.0 / np.maximum(r, 1e-6))
    vel = np.stack([-v * np.sin(ang), v * np.cos(ang),
                    rng.normal(0, 0.02, N)], axis=1)
    mass = np.ones(N) * 0.3
    return pos, vel, mass


def target_final_state(e: float, rng) -> torch.Tensor:
    """目标 = 真值初始条件演化后的末态位置。"""
    pos, vel, mass = make_gt_ic(e, rng)
    sysm = DifferentiableNBody(pos, vel, mass, G=1.0, softening=SOFT,
                               dt=DT, steps=STEPS)
    with torch.no_grad():
        out = sysm.forward()
    return out["final_pos"].detach()


# ---------------- 学习式 critic ----------------
class LearnedCritic(nn.Module):
    """输入轨迹，输出"物理可信度"分数。在 train 族上训练。"""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def featurize(self, traj: torch.Tensor) -> torch.Tensor:
        """把轨迹压成 6 维特征（守恒量漂移 + 统计量）。

        批量向量化：接受 (B, T, N, 3) 或单条 (T, N, 3)。
        此前逐时刻 + 逐样本做 Python 循环，240 样本 × 300 epoch 要跑几十分钟。
        """
        single = traj.dim() == 3
        if single:
            traj = traj.unsqueeze(0)
        B, T, N, _ = traj.shape
        mass = torch.ones(N) * 0.3

        v = (traj[:, 1:] - traj[:, :-1]) / DT                  # (B,T-1,N,3)
        v = torch.cat([v, v[:, -1:]], dim=1)                   # (B,T,N,3)

        ke = 0.5 * (mass * (v * v).sum(-1)).sum(-1)            # (B,T)

        d = traj.unsqueeze(2) - traj.unsqueeze(3)              # (B,T,N,N,3)
        r = ((d * d).sum(-1) + SOFT ** 2).sqrt()               # (B,T,N,N)
        eye = 1.0 - torch.eye(N, dtype=r.dtype)
        mm = mass[:, None] * mass[None, :]                     # (N,N)
        pe = -(mm / r * eye).sum(dim=(2, 3)) / 2.0             # (B,T)
        E = ke + pe

        cross = torch.cross(traj, v, dim=-1)                   # (B,T,N,3)
        # mass 是 (N,)，必须显式插轴到 (1,1,N,1)——否则广播会去对齐最后一维
        m4 = mass[None, None, :, None]
        Ln = (m4 * cross).sum(dim=2).norm(dim=-1)              # (B,T)
        Pn = (m4 * v).sum(dim=2).norm(dim=-1)                  # (B,T)

        def rel(x):
            return (x - x[:, :1]).abs().mean(dim=1) / (x[:, :1].abs().squeeze(1) + 1e-8)

        feats = torch.stack([
            rel(E), rel(Ln), rel(Pn),
            E.std(dim=1) / (E.abs().mean(dim=1) + 1e-8),
            traj.std(dim=(1, 2, 3)), traj.abs().mean(dim=(1, 2, 3)),
        ], dim=-1)                                             # (B,6)
        return feats[0] if single else feats

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        return self.net(self.featurize(traj))


def train_learned_critic(f_train, rng, epochs=300):
    """在 train 族上训练：正样本 = 真值 IC 的轨迹；负样本 = 扰动 IC 的轨迹。"""
    critic = LearnedCritic()
    opt = torch.optim.Adam(critic.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()
    X, Y = [], []
    for _ in range(60):
        e = rng.uniform(*f_train)
        pos, vel, mass = make_gt_ic(e, rng)
        base = DifferentiableNBody(pos, vel, mass, G=1.0, softening=SOFT,
                                   dt=DT, steps=STEPS)
        with torch.no_grad():
            X.append(base.forward()["trajectory"]); Y.append(1.0)
            for scale in (0.15, 0.3, 0.6):
                pert = DifferentiableNBody(
                    pos + rng.normal(0, scale, pos.shape),
                    vel + rng.normal(0, scale, vel.shape), mass,
                    G=1.0, softening=SOFT, dt=DT, steps=STEPS)
                X.append(pert.forward()["trajectory"]); Y.append(0.0)
    X = torch.stack(X); Y = torch.tensor(Y)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(critic(X).squeeze(-1), Y)
        loss.backward()
        opt.step()
    critic.eval()
    return critic, float(loss)


# ---------------- 反演 ----------------
def invert(target, critic, mode, rng, opt_steps=OPT_STEPS):
    """从随机 IC 出发，反演使末态接近 target 的初始位置。

    mode: 'none' | 'learned' | 'conservation' | 'hybrid'
    """
    pos0 = torch.tensor(rng.normal(0, 0.3, (N, 3)))
    vel0 = torch.tensor(rng.normal(0, 0.05, (N, 3)))
    mass = torch.ones(N) * 0.3
    pos0 = pos0.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([pos0], lr=LR)
    for _ in range(opt_steps):
        opt.zero_grad()
        sysm = DifferentiableNBody(pos0, vel0, mass, G=1.0, softening=SOFT,
                                   dt=DT, steps=STEPS)
        out = sysm.forward()
        shape = (out["final_pos"] - target).pow(2).mean()
        loss = shape
        if mode in ("conservation", "hybrid"):
            pl = PhysicsLosses(sysm, G=1.0, softening=SOFT)
            loss = loss + 1.0 * pl(out)
        if mode in ("learned", "hybrid"):
            loss = loss + 1.0 * torch.relu(1.0 - critic(out["trajectory"])[0])
        loss.backward()
        opt.step()

    # 评估：末态形状误差 + 守恒残差
    with torch.no_grad():
        sysm = DifferentiableNBody(pos0.detach(), vel0, mass, G=1.0,
                                   softening=SOFT, dt=DT, steps=STEPS)
        out = sysm.forward()
        shape_err = float((out["final_pos"] - target).pow(2).mean())
        pl = PhysicsLosses(sysm, G=1.0, softening=SOFT)
        cons = float(pl(out))
    return shape_err, cons


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)

    print("=== 训练学习式 critic（仅在 train 族 e∈[0, 0.3]）===")
    critic, closs = train_learned_critic(F_TRAIN, rng)
    print(f"  critic 训练 loss = {closs:.4f}")

    # 生成两族目标
    def gen_targets(fam, n):
        r = np.random.default_rng(1234)
        return [target_final_state(r.uniform(*fam), r) for _ in range(n)]

    tgt_train = gen_targets(F_TRAIN, N_TRAIN_T)
    tgt_test = gen_targets(F_TEST, N_TEST_T)

    MODES = ("none", "learned", "conservation", "hybrid")
    rows = []
    rng_eval = np.random.default_rng(7)

    for fam_name, tgts in (("同分布(train族)", tgt_train),
                           ("跨分布(test族)", tgt_test)):
        print(f"\n=== {fam_name} ===")
        print(f"{'critic':<14}{'形状误差↓':>14}{'守恒残差↓':>14}")
        print("-" * 44)
        for mode in MODES:
            se, cs = [], []
            for ti, t in enumerate(tgts):
                for _ in range(N_EVAL):
                    a, b = invert(t, critic, mode, rng_eval)
                    se.append(a); cs.append(b)
                print(f"    [{mode}] target {ti+1}/{len(tgts)}  "
                      f"shape={np.mean(se):.5f}", flush=True)
            rows.append(dict(family=fam_name, mode=mode,
                             shape_err=float(np.mean(se)),
                             shape_err_sem=float(np.std(se) / np.sqrt(len(se))),
                             cons=float(np.mean(cs)),
                             cons_sem=float(np.std(cs) / np.sqrt(len(cs)))))
            print(f"{mode:<14}{np.mean(se):>14.5f}{np.mean(cs):>14.5f}")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "experiment_a.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"config": dict(N=N, STEPS=STEPS, DT=DT, OPT_STEPS=OPT_STEPS,
                                  f_train=list(F_TRAIN), f_test=list(F_TEST),
                                  n_eval=N_EVAL),
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"\n耗时 {time.time()-t0:.0f}s  [写出] {out}")


if __name__ == "__main__":
    main()
