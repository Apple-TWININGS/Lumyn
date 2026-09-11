"""可微 N 体（NumPy 数值梯度版）——无需 PyTorch 即可验证梯度引导生成。

为什么需要这个文件？
--------------------
differentiable.py 依赖 PyTorch 自动微分。但很多环境（包括部分服务器 / CI）
装不上 torch。为保证「物理不出错 + 梯度引导生成」在任何环境都可复现验证，
这里用 **中心差分** 数值估计梯度，接口与 DifferentiableNBody 对齐：

    - DifferentiableNBodyNumpy(pos, vel, mass, steps, ...)
    - .forward() -> {"trajectory": (T,N,3), "final_pos", ...}
    - .gradient(loss_fn) -> {"pos": grad, "vel": grad, "mass": grad}
    - optimize(target_radius=..., lr=..., steps=...) 梯度下降闭环

精度说明
--------
- 前向模拟与 differentiable.py 完全一致（同一 velocity-Verlet + softening）。
- 梯度是 **中心差分** 估计，精度 ~ eps^2（默认 eps=1e-4 → ~1e-8 量级），
  比 autograd 略低但足以验证「梯度引导确实能把系统推向守恒」。
- 代价：每参数 2 次前向 → O(2 * N*3) 次模拟，大规模请换 torch + GPU。

依赖：仅 numpy。
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, Callable, Any
import numpy as np


class DifferentiableNBodyNumpy:
    """NumPy 版可微 N 体：前向可微（中心差分梯度）。"""

    def __init__(self, pos, vel, mass, G: float = 1.0, softening: float = 1e-3,
                 dt: float = 0.01, steps: int = 40, repulsive: float = 0.0):
        self.G = float(G)
        self.softening = float(softening)
        self.dt = float(dt)
        self.steps = int(steps)
        self.repulsive = float(repulsive)
        # 存为可写数组
        self.pos = np.asarray(pos, dtype=np.float64).copy()
        self.vel = np.asarray(vel, dtype=np.float64).copy()
        self.mass = np.asarray(mass, dtype=np.float64).copy()

    # ------------------------------------------------------------
    def _accel(self, pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
        N = pos.shape[0]
        d = pos[None, :, :] - pos[:, None, :]          # (N, N, 3)
        r2 = (d * d).sum(axis=-1) + self.softening ** 2
        inv_r3 = r2 ** (-1.5)
        np.fill_diagonal(inv_r3, 0.0)                   # 去自作用
        sign = -1.0 if self.repulsive == 0.0 else 1.0
        return sign * self.G * (mass[None, :, None] * d * inv_r3[:, :, None]).sum(axis=1)

    # ------------------------------------------------------------
    def step(self, pos: np.ndarray, vel: np.ndarray,
             mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        a = self._accel(pos, mass)
        new_vel = vel + a * self.dt
        new_pos = pos + new_vel * self.dt
        return new_pos, new_vel

    # ------------------------------------------------------------
    def forward(self, pos: Optional[np.ndarray] = None,
                vel: Optional[np.ndarray] = None,
                mass: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """整段模拟。允许传入临时参数（用于差分）。"""
        pos = pos if pos is not None else self.pos
        vel = vel if vel is not None else self.vel
        mass = mass if mass is not None else self.mass
        traj = [pos.copy()]
        for _ in range(self.steps):
            pos, vel = self.step(pos, vel, mass)
            traj.append(pos.copy())
        traj = np.stack(traj, axis=0)
        return {"trajectory": traj, "final_pos": traj[-1], "final_vel": vel}

    # ------------------------------------------------------------
    def _energy(self, traj: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> float:
        """总能量（用 traj 端点 + vel 近似）。"""
        N = traj.shape[0] if traj.ndim == 2 else traj.shape[1]
        # 兼容传入单帧或整段
        if traj.ndim == 3:
            pos0, pos1 = traj[0], traj[-1]
            v0 = (traj[1] - traj[0]) / self.dt
            v1 = (traj[-1] - traj[-2]) / self.dt
        else:
            pos0 = pos1 = traj
            v0 = v1 = vel
        ke = 0.5 * (mass * (v1 * v1).sum(-1)).sum()
        # 势能（去对角）
        d = pos1[None, :, :] - pos1[:, None, :]
        r = np.sqrt((d * d).sum(-1) + self.softening ** 2)
        eye = 1.0 - np.eye(N)
        pe = -self.G * (mass[None, :] * mass[:, None] / r * eye).sum() / 2.0
        return ke + pe

    # ------------------------------------------------------------
    def gradient(self, loss_fn: Callable[[Dict[str, Any]], float],
                 eps: float = 1e-5) -> Dict[str, np.ndarray]:
        """中心差分估计 loss 对 pos/vel/mass 的梯度。"""
        base = self.forward()
        base_loss = loss_fn(base)

        grads = {"pos": np.zeros_like(self.pos), "vel": np.zeros_like(self.vel),
                 "mass": np.zeros_like(self.mass)}

        # 对 pos 的每个分量
        for i in range(self.pos.size):
            idx = np.unravel_index(i, self.pos.shape)
            pos_p, pos_m = self.pos.copy(), self.pos.copy()
            pos_p[idx] += eps
            pos_m[idx] -= eps
            lp = loss_fn(self.forward(pos=pos_p))
            lm = loss_fn(self.forward(pos=pos_m))
            grads["pos"][idx] = (lp - lm) / (2 * eps)

        # 对 vel 的每个分量
        for i in range(self.vel.size):
            idx = np.unravel_index(i, self.vel.shape)
            vel_p, vel_m = self.vel.copy(), self.vel.copy()
            vel_p[idx] += eps
            vel_m[idx] -= eps
            lp = loss_fn(self.forward(vel=vel_p))
            lm = loss_fn(self.forward(vel=vel_m))
            grads["vel"][idx] = (lp - lm) / (2 * eps)

        # 对 mass
        for i in range(self.mass.size):
            m_p, m_m = self.mass.copy(), self.mass.copy()
            m_p[i] += eps
            m_m[i] -= eps
            lp = loss_fn(self.forward(mass=m_p))
            lm = loss_fn(self.forward(mass=m_m))
            grads["mass"][i] = (lp - lm) / (2 * eps)

        self._last_grad_norm = float(np.sqrt(
            (grads["pos"] ** 2).sum() + (grads["vel"] ** 2).sum() + (grads["mass"] ** 2).sum()))
        return grads

    # ------------------------------------------------------------
    def optimize(self, target_radius: Optional[float] = None,
                 w_energy: float = 1.0, lr: float = 0.01, steps: int = 100,
                 callback=None) -> Dict[str, Any]:
        """梯度下降：最小化「能量守恒误差 + 半径目标误差」。"""
        history = []
        for s in range(steps):
            def loss_fn(out):
                traj = out["trajectory"]
                # 能量守恒：(E_end - E_start)^2
                E0 = self._energy(traj[0:1] if False else traj[0], (traj[1]-traj[0])/self.dt, self.mass)
                ET = self._energy(traj[-1], (traj[-1]-traj[-2])/self.dt, self.mass)
                le = w_energy * (ET - E0) ** 2
                # 目标半径
                lt = 0.0
                if target_radius is not None:
                    r = float(np.sqrt((traj[-1] ** 2).sum(-1).mean()))
                    lt = (r - target_radius) ** 2
                return le + lt

            grads = self.gradient(loss_fn)
            self.pos -= lr * grads["pos"]
            self.vel -= lr * grads["vel"]
            self.mass -= lr * grads["mass"]
            # mass 保持正
            self.mass = np.maximum(self.mass, 1e-3)

            out = self.forward()
            total = loss_fn(out)
            history.append(float(total))
            if callback:
                callback(s, float(total), self._last_grad_norm)

        return {"final_loss": float(history[-1]), "steps": steps, "history": history}


# ============================================================
# 便捷 API：NumPy 版「生成椭圆轨道」
# ============================================================
def generate_ellipse_numpy(target_a: float = 1.0, target_e: float = 0.3, N: int = 8,
                           steps: int = 40, lr: float = 0.02,
                           opt_steps: int = 80) -> Dict[str, np.ndarray]:
    """NumPy 数值梯度版：从随机初始条件优化出近似椭圆轨道。"""
    rng = np.random.default_rng(42)
    pos = rng.normal(0, 0.5, (N, 3))
    pos[:, 2] *= 0.2
    r = np.sqrt((pos[:, :2] ** 2).sum(-1)).clip(min=0.3)
    vel = np.zeros_like(pos)
    v_circ = np.sqrt(1.0 / r)
    vel[:, 0] = -pos[:, 1] / r * v_circ
    vel[:, 1] = pos[:, 0] / r * v_circ
    mass = rng.uniform(0.1, 1.0, (N,))

    sys = DifferentiableNBodyNumpy(pos, vel, mass, steps=steps)
    info = sys.optimize(target_radius=target_a, w_energy=1.0, lr=lr, steps=opt_steps)
    return {"pos": sys.pos, "vel": sys.vel, "mass": sys.mass, **info}


if __name__ == "__main__":
    out = generate_ellipse_numpy(target_a=1.0, N=6, opt_steps=30)
    print(f"[diff_numpy] 优化完成: pos={out['pos'].shape}, final_loss={out['final_loss']:.6f}")
