"""N 体引力模拟器：用 Barnes-Hut 做省算力演化。

每步（Velocity-Verlet，比朴素 Euler 守恒好得多）：
  acc = tree.compute_acceleration(theta)          # O(N log N)
  vel += 0.5 * G * acc * dt
  pos += vel * dt
  acc' = tree.compute_acceleration(theta)
  vel += 0.5 * G * acc' * dt
可选 repulsion（短程斥力）防止粒子堆叠，向量化实现。
"""
from __future__ import annotations

import numpy as np
from .barnes_hut import BarnesHutTree


class NBodySimulator:
    """省算力的 N 体引力模拟器（NumPy 版）。

    Parameters
    ----------
    G : float
        引力常数（模拟单位下可调）。
    theta : float
        Barnes-Hut 打开判据，越小越精确、越慢。
    softening : float
        软化长度，避免 r→0 时力发散。
    """

    def __init__(self, G: float = 1.0, theta: float = 0.5, softening: float = 1e-4):
        self.G = G
        self.theta = theta
        self.softening = softening
        self.tree = BarnesHutTree(max_leaf=16, max_depth=24)

    def _acceleration(self, pos, mass):
        """计算当前位置的加速度 (N,3)。含可选短程斥力（向量化）。"""
        center = pos.mean(axis=0)
        size = float(np.abs(pos - center).max()) * 2 + 1e-6
        self.tree.build(pos, mass, center, size)
        acc = self.tree.compute_acceleration(theta=self.theta)

        # 斥力：仅在极近距离起作用（ soften 已软化），此处叠加一个短程排斥，向量化
        # 用 pairwise 会被 O(N^2) 抵消省算力优势，故仅在粒子数较小时启用
        return acc

    def step(self, pos, mass, vel, dt=0.01, repulsion=0.0):
        """单步 Velocity-Verlet。返回 (new_pos, new_vel)。"""
        acc = self._acceleration(pos, mass)

        if repulsion > 0.0 and len(pos) <= 512:
            # 短程斥力只在小规模场景启用（避免 O(N^2) 拖慢）
            d = pos[:, None, :] - pos[None, :, :]          # (N,N,3)
            dist = np.sqrt((d * d).sum(axis=2)) + self.softening
            inv_r3 = dist ** (-3)
            np.fill_diagonal(inv_r3, 0.0)
            rep = repulsion * (mass[None, :] * inv_r3)[:, :, None] * d
            acc = acc + rep.sum(axis=1)

        # Velocity-Verlet
        vel_half = vel + 0.5 * self.G * acc * dt
        new_pos = pos + vel_half * dt
        acc_new = self._acceleration(new_pos, mass)
        new_vel = vel_half + 0.5 * self.G * acc_new * dt
        return new_pos, new_vel

    def simulate(self, pos, mass, vel, n_steps=100, dt=0.01, repulsion=0.0):
        """模拟 n_steps 步，返回轨迹 (n_steps+1, N, 3)。"""
        traj = [pos.copy()]
        for _ in range(n_steps):
            pos, vel = self.step(pos, mass, vel, dt, repulsion)
            traj.append(pos.copy())
        return np.stack(traj, axis=0)

    def compute(self, pos, mass, dt=0.01):
        """一步演化（兼容 differentiable 接口）：返回 new_pos, new_vel。

        供 scientific/nbody.py 的 NBodySim 统一调用，也供可微模块做单步前向。
        """
        return self.step(pos, mass, np.zeros_like(pos), dt=dt)


class NBodySimulatorConfig:
    """默认配置（供 guided_generation / 示例脚本统一读取）。"""
    DEFAULT_G = 1.0
    DEFAULT_DT = 0.01
    DEFAULT_THETA = 0.5
    DEFAULT_STEPS = 100
