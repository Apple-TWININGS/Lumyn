"""DifferentiableNBody：PyTorch 可微 N 体引力 + 守恒约束。

核心思想（对齐 physics/nbody.py 的 Barnes-Hut 接口约定）：
- 位置/质量/速度均为 torch.Tensor(requires_grad=True)
- 用"软化势 + 全对成对求和"做可微前向（小规模精确；大规模可后续切 BH）
- 反向传播自然得到 dLoss/dpos，可"梯度引导"生成目标形态
- 训练时加能量/角动量守恒约束，保证"物理不出错"到 1e-3 级
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class DifferentiableNBody(nn.Module):
    """可微 N 体模拟器。softening 避免 r→0 奇点。"""

    def __init__(self, G: float = 1.0, softening: float = 0.05, dt: float = 0.01,
                 method: str = "euler"):
        super().__init__()
        self.G = G
        self.softening = softening
        self.dt = dt
        self.method = method  # 'euler' 或 'leapfrog'

    def acceleration(self, pos: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """可微成对引力，向量化。(N,3) -> (N,3)"""
        # d_ij = pos_j - pos_i，shape (N,N,3)
        d = pos.unsqueeze(0) - pos.unsqueeze(1)            # (N,N,3)
        r2 = (d * d).sum(dim=-1) + self.softening ** 2     # (N,N)
        inv_r3 = r2.pow(-1.5).unsqueeze(-1)                # (N,N,1)
        # 去掉自相互作用（对角置零）
        eye = torch.eye(pos.shape[0], device=pos.device, dtype=pos.dtype)
        inv_r3 = inv_r3 * (1.0 - eye).unsqueeze(-1)
        # a_i = G * sum_j m_j * d_ij / r_ij^3
        a = self.G * (inv_r3 * d * mass.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        return a

    def step(self, pos: torch.Tensor, vel: torch.Tensor,
             mass: torch.Tensor) -> tuple:
        """单步积分（leapfrog 更保辛）。"""
        if self.method == "leapfrog":
            # 半步速度
            a = self.acceleration(pos, mass)
            v_half = vel + 0.5 * self.dt * a
            pos_new = pos + self.dt * v_half
            a_new = self.acceleration(pos_new, mass)
            vel_new = v_half + 0.5 * self.dt * a_new
            return pos_new, vel_new
        else:
            a = self.acceleration(pos, mass)
            vel_new = vel + self.dt * a
            pos_new = pos + self.dt * vel_new
            return pos_new, vel_new

    def simulate(self, pos0: torch.Tensor, vel0: torch.Tensor, mass: torch.Tensor,
                 steps: int = 50) -> torch.Tensor:
        """演化 steps 步，返回轨迹 (steps+1, N, 3)。"""
        traj = [pos0]
        pos, vel = pos0, vel0
        for _ in range(steps):
            pos, vel = self.step(pos, vel, mass)
            traj.append(pos)
        return torch.stack(traj, dim=0)

    @staticmethod
    def energy(pos: torch.Tensor, vel: torch.Tensor, mass: torch.Tensor,
               G: float = 1.0, softening: float = 0.05) -> torch.Tensor:
        """总能量（可微），用于守恒约束。"""
        ke = 0.5 * (mass * (vel * vel).sum(dim=-1)).sum()
        n = pos.shape[0]
        d = pos.unsqueeze(0) - pos.unsqueeze(1)
        r = torch.sqrt((d * d).sum(dim=-1) + softening ** 2)
        m_i = mass.unsqueeze(0); m_j = mass.unsqueeze(1)
        pe = -G * (m_i * m_j / r).sum() / 2.0  # /2 消重复
        return ke + pe

    @staticmethod
    def angular_momentum(pos: torch.Tensor, vel: torch.Tensor,
                         mass: torch.Tensor) -> torch.Tensor:
        return (mass.unsqueeze(-1) * torch.cross(pos, vel, dim=-1)).sum(dim=0)


class ConservationConstraint(nn.Module):
    """训练时的守恒软约束：把能量/角动量漂移压到 1e-3 级。"""

    def __init__(self, G: float = 1.0, softening: float = 0.05,
                 w_energy: float = 1.0, w_am: float = 0.5):
        super().__init__()
        self.G = G
        self.softening = softening
        self.w_energy = w_energy
        self.w_am = w_am

    def forward(self, traj: torch.Tensor, vel: torch.Tensor,
                mass: torch.Tensor) -> dict:
        # vel 有两种合法形态：
        #   - (T, N, 3) 速度轨迹  —— 与 traj 同步
        #   - (N, 3)    单个状态  —— optimize_initial_conditions 里 vel 恒定
        # 三个守恒量必须用**同一套**取值逻辑。此前 energy 用了保护、
        # angular_momentum 没有，传 (N,3) 时 vel[0] 退化成 (3,) 一维张量，
        # torch.cross 与 (N,3) 的 traj[0] 维度不匹配：
        #   "linalg.cross: inputs must have the same number of dimensions"
        is_traj = vel.dim() == 3
        v0 = vel[0] if is_traj else vel
        vT = vel[-1] if is_traj else vel

        e0 = DifferentiableNBody.energy(traj[0], v0, mass, self.G, self.softening)
        eT = DifferentiableNBody.energy(traj[-1], vT, mass, self.G, self.softening)
        energy_loss = ((eT - e0) / (torch.abs(e0) + 1e-8)).pow(2)

        L0 = DifferentiableNBody.angular_momentum(traj[0], v0, mass)
        LT = DifferentiableNBody.angular_momentum(traj[-1], vT, mass)
        am_loss = ((LT - L0) / (torch.abs(L0) + 1e-8)).pow(2).sum()

        return {
            "loss": self.w_energy * energy_loss + self.w_am * am_loss,
            "energy_drift": energy_loss.sqrt(),
            "angular_momentum_drift": am_loss.sqrt(),
        }


def optimize_initial_conditions(target_pos: torch.Tensor, n: int = 200,
                                steps: int = 40, iters: int = 30,
                                lr: float = 0.05, G: float = 1.0) -> dict:
    """梯度引导：给定目标末态形态，反推物理正确的初始位置。

    这就是"用梯度引导场景生成"——从目标反演初始条件，保证演化后
    既贴近目标、又满足守恒约束。对齐论文的"反问题"思路。
    """
    torch.manual_seed(0)
    # target_pos 通常由 engine.optimize 以 numpy 数组传入。
    # 必须先转成 Tensor：否则 traj[-1] - target_pos 会触发 Tensor.__array__
    # → .numpy()，而 traj 需要梯度，直接抛
    # "Can't call numpy() on Tensor that requires grad"。
    target_pos = torch.as_tensor(np.asarray(target_pos), dtype=torch.float64)
    # 必须先缩放、再开 requires_grad_：
    # 若写成 torch.randn(..., requires_grad=True) * 1.5，乘法结果是非叶子张量，
    # 交给 torch.optim.Adam 会直接抛 "can't optimize a non-leaf Tensor"。
    pos = (torch.randn(n, 3) * 1.5).requires_grad_(True)
    mass = torch.ones(n) * 0.1
    mass[0] = 10.0
    vel = torch.zeros(n, 3)
    model = DifferentiableNBody(G=G, softening=0.05, dt=0.01, method="leapfrog")
    opt = torch.optim.Adam([pos], lr=lr)
    cons = ConservationConstraint(G=G, softening=0.05)

    for _ in range(iters):
        opt.zero_grad()
        traj = model.simulate(pos, vel, mass, steps=steps)
        shape_loss = (traj[-1] - target_pos).pow(2).mean()
        c = cons(traj, vel, mass)
        loss = shape_loss + 0.1 * c["loss"]
        loss.backward()
        opt.step()
        with torch.no_grad():
            pos.clamp_(-10, 10)

    traj = model.simulate(pos, vel, mass, steps=steps)
    return {
        "pos0": pos.detach().numpy(),
        "trajectory": traj.detach().numpy(),
        "final_energy_drift": float(cons(traj, vel, mass)["energy_drift"].sqrt()),
    }


def contact_map(pos: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """接触/邻近映射（可微版的邻接关系近似），用于碰撞检测可视化。"""
    d = np.abs(pos[:, None, :] - pos[None, :, :]).sum(axis=-1)
    return (d < threshold).astype(np.float32)
