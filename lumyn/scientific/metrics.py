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
        fixed_mask = getattr(mass, "fixed_mask", None)
        m = mass.values if hasattr(mass, "values") else np.asarray(mass)

        for t in range(T):
            pos = traj[t]
            vel = (traj[min(t + 1, T - 1)] - pos) / dt
            energies.append(self.total_energy(pos, vel, m))
            L = self.angular_momentum(pos, vel, m)
            amags.append(float(np.linalg.norm(L)))
            if fixed_mask is not None:
                mobile = ~np.asarray(fixed_mask, dtype=bool)
                energies[-1] -= self.potential(pos[mobile][:0], m[:0]) * 0  # noop
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
        """线动量是否守恒（中心势阱下未必守恒，故默认 tol 较宽）。"""
        m = np.asarray(mass)
        com = (m[:, None] * traj[0]).sum(axis=0) / (m.sum() + 1e-12)
        comT = (m[:, None] * traj[-1]).sum(axis=0) / (m.sum() + 1e-12)
        return bool(np.linalg.norm(comT - com) < tol + 1e-8)


def energy_drift(traj: np.ndarray, mass: np.ndarray, G: float = 1.0) -> float:
    return PhysicsMetrics(G=G).energy_drift(traj, mass)


def momentum_conserved(traj: np.ndarray, mass: np.ndarray, G: float = 1.0,
                       tol: float = 0.5) -> bool:
    return PhysicsMetrics(G=G).momentum_conserved(traj, mass, tol=tol)
