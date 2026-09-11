"""DifferentiableNBody (NumPy 参考实现) —— torch 不可用时的数值梯度兜底。

接口与 differentiable.py 完全一致：energy / angular_momentum / simulate / optimize_initial_conditions。
梯度用中心差分实现，保证"反问题 / 梯度引导"逻辑可验证；精度低于 torch.autograd，
但无需任何额外依赖。科研级精度请安装 torch 后使用 differentiable.py。
"""
from __future__ import annotations
import numpy as np
from typing import Dict


class DifferentiableNBodyNumPy:
    """NumPy 版可微 N 体：forward 用 leapfrog，gradient 用中心差分。"""

    def __init__(self, G: float = 1.0, softening: float = 0.05, dt: float = 0.01,
                 method: str = "leapfrog", eps: float = 1e-3):
        self.G = G
        self.softening = softening
        self.dt = dt
        self.method = method
        self.eps = eps

    def acceleration(self, pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
        d = pos[None, :, :] - pos[:, None, :]            # (N,N,3)
        r2 = (d * d).sum(axis=-1) + self.softening ** 2   # (N,N)
        inv_r3 = r2 ** (-1.5)
        np.fill_diagonal(inv_r3, 0.0)
        return self.G * (inv_r3[:, :, None] * d * mass[None, :, None]).sum(axis=1)

    def step(self, pos, vel, mass):
        if self.method == "leapfrog":
            a = self.acceleration(pos, mass)
            v_half = vel + 0.5 * self.dt * a
            pos_new = pos + self.dt * v_half
            a_new = self.acceleration(pos_new, mass)
            vel_new = v_half + 0.5 * self.dt * a_new
            return pos_new, vel_new
        else:
            a = self.acceleration(pos, mass)
            vel_new = vel + self.dt * a
            return pos + self.dt * vel_new, vel_new

    def simulate(self, pos0, vel0, mass, steps: int = 40):
        traj = [pos0.copy()]
        pos, vel = pos0.copy(), vel0.copy()
        for _ in range(steps):
            pos, vel = self.step(pos, vel, mass)
            traj.append(pos.copy())
        return np.stack(traj, axis=0)

    def energy(self, pos, vel, mass):
        ke = 0.5 * (mass * (vel * vel).sum(axis=-1)).sum()
        n = len(pos)
        d = pos[None, :, :] - pos[:, None, :]
        r = np.sqrt((d * d).sum(axis=-1) + self.softening ** 2)
        m_i = mass[None, :]; m_j = mass[:, None]
        pe = -self.G * (m_i * m_j / r).sum() / 2.0
        return ke + pe

    def angular_momentum(self, pos, vel, mass):
        return (mass[:, None] * np.cross(pos, vel)).sum(axis=0)

    def gradient(self, pos, vel, mass, target, steps: int = 40):
        """对 pos (初始位置) 求 d|traj[-1]-target|²/dpos 的数值梯度。"""
        eps = self.eps
        n, d = pos.shape
        grad = np.zeros_like(pos)
        base = self.simulate(pos, vel, mass, steps=steps)
        base_loss = ((base[-1] - target) ** 2).sum()
        for i in range(n):
            for k in range(d):
                pos_p = pos.copy(); pos_p[i, k] += eps
                pos_m = pos.copy(); pos_m[i, k] -= eps
                lp = ((self.simulate(pos_p, vel, mass, steps=steps)[-1] - target) ** 2).sum()
                lm = ((self.simulate(pos_m, vel, mass, steps=steps)[-1] - target) ** 2).sum()
                grad[i, k] = (lp - lm) / (2 * eps)
        return grad, base_loss


def optimize_initial_conditions(target: np.ndarray, n: int = 40, steps: int = 10,
                                iters: int = 8, lr: float = 0.05,
                                G: float = 1.0) -> Dict:
    """NumPy 梯度下降反演初始条件（接口对齐 torch 版）。"""
    rng = np.random.default_rng(0)
    pos = rng.normal(0, 1, (n, 3)) * 1.0
    mass = np.ones(n) * 0.1; mass[0] = 5.0
    vel = np.zeros((n, 3))
    model = DifferentiableNBodyNumPy(G=G, softening=0.05, dt=0.01, method="leapfrog")

    for _ in range(iters):
        grad, loss = model.gradient(pos, vel, mass, np.asarray(target, dtype=np.float32), steps=steps)
        pos = np.clip(pos - lr * grad, -10, 10)

    traj = model.simulate(pos, vel, mass, steps=steps)
    e0 = model.energy(traj[0], vel, mass)
    eT = model.energy(traj[-1], vel, mass)
    return {
        "pos0": pos,
        "trajectory": traj,
        "final_energy_drift": float(np.abs(eT - e0) / (np.abs(e0) + 1e-8)),
    }


class ConservationConstraintNumPy:
    """守恒约束（NumPy 版，用于无 torch 环境的验证）。"""

    def __call__(self, traj: np.ndarray, vel: np.ndarray, mass: np.ndarray,
                 G: float = 1.0, softening: float = 0.05) -> Dict:
        model = DifferentiableNBodyNumPy(G=G, softening=softening)
        e0 = model.energy(traj[0], vel, mass)
        eT = model.energy(traj[-1], vel, mass)
        L0 = model.angular_momentum(traj[0], vel, mass)
        LT = model.angular_momentum(traj[-1], vel, mass)
        e_drift = float(np.abs(eT - e0) / (np.abs(e0) + 1e-8))
        am_drift = float(np.linalg.norm(LT - L0) / (np.linalg.norm(L0) + 1e-8))
        return {"loss": e_drift ** 2 + 0.5 * am_drift ** 2,
                "energy_drift": e_drift, "angular_momentum_drift": am_drift}
