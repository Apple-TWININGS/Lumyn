"""物理量诊断：能量、角动量、质心漂移、半质量半径。"""
from __future__ import annotations
import numpy as np
from typing import Dict


class PhysicsMetrics:
    """对 N 体轨迹做守恒量诊断，输出科研可复现的数值。"""

    def __init__(self, G: float = 1.0):
        self.G = G

    def kinetic(self, vel: np.ndarray, mass: np.ndarray) -> float:
        return 0.5 * float((mass * (vel ** 2).sum(axis=1)).sum())

    def potential(self, pos: np.ndarray, mass: np.ndarray) -> float:
        n = len(pos)
        pe = 0.0
        for i in range(n):
            d = pos - pos[i]
            r = np.sqrt((d * d).sum(axis=1)) + 1e-8
            pe -= self.G * mass[i] * (mass / r).sum()
        return 0.5 * pe  # 消除重复计数

    def total_energy(self, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> float:
        return self.kinetic(vel, mass) + self.potential(pos, mass)

    def angular_momentum(self, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> np.ndarray:
        return ((mass[:, None] * np.cross(pos, vel)).sum(axis=0))

    def center_of_mass(self, pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
        return (mass[:, None] * pos).sum(axis=0) / (mass.sum() + 1e-12)

    def half_mass_radius(self, pos: np.ndarray, mass: np.ndarray) -> float:
        com = self.center_of_mass(pos, mass)
        r = np.sqrt(((pos - com) ** 2).sum(axis=1))
        order = np.argsort(r)
        cum = np.cumsum(mass[order])
        idx = np.searchsorted(cum, 0.5 * mass.sum())
        return float(r[order[min(idx, len(r) - 1)]])

    def diagnose(self, traj: np.ndarray, mass: np.ndarray,
                 dt: float = 0.01) -> Dict:
        """对整条轨迹 (T,N,3) 诊断守恒漂移。"""
        T = len(traj)
        energies, amags, com_drift, hmr = [], [], [], []
        # 说明：此处原有一行 `fixed_mask = getattr(mass, "fixed_mask", None)`，
        # 但 mass 是 ndarray，该属性永远取不到（恒为 None），其后的分支体又乘以 0，
        # 是彻底的死代码。已删除。若要支持「排除固定粒子」，应由调用方显式传入掩码。

        m = mass.values if hasattr(mass, "values") else np.asarray(mass)

        for t in range(T):
            pos = traj[t]
            # 速度由位置差商得到。此前写成
            #     vel = (traj[min(t + 1, T - 1)] - pos) / dt
            # 在最后一帧 t = T-1 时 min(T, T-1) = T-1，于是 vel = 0 ——
            # **最后一帧所有粒子的速度被置零**，后果是：
            #   · amags[-1] = 0 ⇒ angular_momentum_drift 恒等于 1.0（实测 0.9999999829）
            #   · energy_final 的动能被清零
            #   · energy_drift 只因该类场景 |E| 被势能项支配而侥幸未暴露
            # 现在末帧沿用倒数第二帧的速度。
            if t + 1 < T:
                vel = (traj[t + 1] - pos) / dt
            else:
                vel = (traj[T - 1] - traj[T - 2]) / dt if T > 1 else np.zeros_like(pos)
            energies.append(self.total_energy(pos, vel, m))
            L = self.angular_momentum(pos, vel, m)
            amags.append(float(np.linalg.norm(L)))
            com_drift.append(self.center_of_mass(pos, m))
            hmr.append(self.half_mass_radius(pos, m))

        energies = np.array(energies)
        com_drift = np.array(com_drift)
        e0 = energies[0] if len(energies) else 1.0
        com0 = com_drift[0] if len(com_drift) else np.zeros(3)

        return {
            "energy_drift": float(np.abs(energies[-1] - energies[0]) / (np.abs(e0) + 1e-8)),
            "energy_final": float(energies[-1]),
            "angular_momentum_drift": float(np.abs(amags[-1] - amags[0]) / (np.abs(amags[0]) + 1e-8)),
            "com_drift": float(np.linalg.norm(com_drift[-1] - com0)),
            "half_mass_radius_final": float(hmr[-1]),
            "half_mass_radius_shrink": float(hmr[0] - hmr[-1]),
            "energies": energies,
            "angular_momenta": np.array(amags),
        }

    # ---- 兼容别名（供 test_scientific & end_to_end 使用） ----
    def energy_drift(self, traj: np.ndarray, mass: np.ndarray, dt: float = 0.01) -> float:
        """总能量相对漂移。"""
        return self.diagnose(traj, mass, dt=dt)["energy_drift"]

    def momentum_conserved(self, traj: np.ndarray, mass: np.ndarray,
                           dt: float = 0.01, tol: float = 0.5) -> bool:
        """**线动量** P = Σ m v 是否守恒。

        此前这个方法名实不符：它比较的是**质心**位置（Σ m r / Σ m）的首末之差，
        而不是动量。二者的关系是 ``d(com)/dt = P / Σ m``，因此「质心漂移小」
        与「动量守恒」并不等价 —— 一个匀速平移的系统质心一直在动，但动量严格守恒。

        现改为真正的动量判据：用末帧与首帧的**平均动量**之差，
        并以 ``Σ m‖v‖``（恒为正的固有尺度）归一，避免净动量接近 0 时除以噪声。
        `center_of_mass_conserved()` 保留了原来的行为。
        """
        m = np.asarray(mass, dtype=np.float64)
        traj = np.asarray(traj, dtype=np.float64)
        T = traj.shape[0]
        if T < 2:
            return True
        v0 = (traj[1] - traj[0]) / dt
        vT = (traj[T - 1] - traj[T - 2]) / dt
        P0 = (m[:, None] * v0).sum(axis=0)
        PT = (m[:, None] * vT).sum(axis=0)
        scale = float((m * np.linalg.norm(vT, axis=-1)).sum()) + 1e-12
        return bool(np.linalg.norm(PT - P0) / scale < tol + 1e-8)

    # ---- 保留原语义，改名为它实际在做的事 ----
    def center_of_mass_conserved(self, traj: np.ndarray, mass: np.ndarray,
                                 tol: float = 0.5) -> bool:
        """质心位置是否长期稳定（**不是**动量守恒；见 `momentum_conserved`）。

        这正是旧版 `momentum_conserved` 的实际行为，保留以免破坏既有调用方。
        """
        m = np.asarray(mass)
        com = (m[:, None] * traj[0]).sum(axis=0) / (m.sum() + 1e-12)
        comT = (m[:, None] * traj[-1]).sum(axis=0) / (m.sum() + 1e-12)
        return bool(np.linalg.norm(comT - com) < tol + 1e-8)


def energy_drift(traj: np.ndarray, mass: np.ndarray, G: float = 1.0) -> float:
    return PhysicsMetrics(G=G).energy_drift(traj, mass)


def momentum_conserved(traj: np.ndarray, mass: np.ndarray, G: float = 1.0,
                       tol: float = 0.5, dt: float = 0.01) -> bool:
    """线动量是否守恒（真正的动量判据，见 `PhysicsMetrics.momentum_conserved`）。"""
    return PhysicsMetrics(G=G).momentum_conserved(traj, mass, dt=dt, tol=tol)


def center_of_mass_conserved(traj: np.ndarray, mass: np.ndarray,
                             G: float = 1.0, tol: float = 0.5) -> bool:
    """质心位置是否长期稳定（**不是**动量守恒）。"""
    return PhysicsMetrics(G=G).center_of_mass_conserved(traj, mass, tol=tol)
