"""N 体模拟包装：把 physics.NBodySimulator 封装为带「场景名称」的可调用。

供 visualization / metrics / export 统一使用：它们只关心
`sim.run()` 返回的轨迹 (T, N, 3) 与质量 (N,)。
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from lumyn.physics.nbody import NBodySimulator, NBodySimulatorConfig


class NBodySim:
    """带场景语义的 N 体模拟。

    Parameters
    ----------
    scene_name : str
        预设场景名（binary_star / spiral_galaxy / globular_cluster / galaxy_collision）
        或 "custom"。决定初始条件的选取。
    n_steps : int
        演化步数。
    dt : float
        时间步长。
    G, theta, softening : float
        转发给底层 `NBodySimulator`。
    """

    def __init__(self, scene_name: str = "custom", n_steps: int = 100,
                 dt: float = NBodySimulatorConfig.DEFAULT_DT, G: float = NBodySimulatorConfig.DEFAULT_G,
                 theta: float = NBodySimulatorConfig.DEFAULT_THETA, softening: float = 1e-4,
                 mass: Optional[np.ndarray] = None):
        self.scene_name = scene_name
        self.n_steps = n_steps
        self.dt = dt
        self.mass = mass
        self.sim = NBodySimulator(G=G, theta=theta, softening=softening)

    def _default_init(self, n: int = 32):
        """无预设时的兜底初始条件：小随机扰动 + 中心大质量。"""
        rng = np.random.default_rng(0)
        pos = rng.normal(0, 1.0, (n, 3)).astype(np.float64)
        vel = rng.normal(0, 0.2, (n, 3)).astype(np.float64)
        mass = np.ones(n, dtype=np.float64)
        mass[0] = 10.0
        pos[0] = 0.0
        vel[0] = 0.0
        return pos, vel, mass

    def run(self, pos: Optional[np.ndarray] = None, vel: Optional[np.ndarray] = None,
            mass: Optional[np.ndarray] = None, seed: Optional[int] = None) -> dict:
        """执行演化。

        Returns
        -------
        dict
            {
              "trajectory": (T, N, 3) ndarray,
              "mass": (N,) ndarray,
              "pos": (N,3), "vel": (N,3),
              "scene": scene_name,
            }
        """
        if seed is not None:
            np.random.seed(seed)

        if mass is None:
            mass = self.mass
        if pos is None or vel is None:
            n = len(mass) if mass is not None else 32
            pos, vel, mass = self._default_init(n)
        if mass is None:
            mass = np.ones(len(pos), dtype=np.float64)

        traj = self.sim.simulate(pos.astype(np.float64), mass.astype(np.float64),
                                 vel.astype(np.float64), n_steps=self.n_steps, dt=self.dt)
        return {
            "trajectory": traj,
            "mass": mass,
            "pos": traj[-1],
            "vel": (traj[-1] - traj[-2]) / self.dt if len(traj) > 1 else vel,
            "scene": self.scene_name,
        }