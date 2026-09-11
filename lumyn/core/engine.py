"""Lumyn 主引擎：编排 scientific / physics / video / eval 各层。"""
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np

from ..physics.nbody import NBodySimulator
from ..scientific import (
    ScientificVisualizer, ScientificExporter, PhysicsMetrics, presets,
)


def _optimize(target, n, steps, iters, G, lr):
    """延迟导入可微模块：优先 torch，缺失时走 NumPy 数值梯度兜底。"""
    from ..scientific import HAS_TORCH
    if HAS_TORCH:
        from ..scientific.differentiable import optimize_initial_conditions
    else:
        from ..scientific.differentiable_numpy import optimize_initial_conditions as optimize_initial_conditions_numpy
    fn = optimize_initial_conditions if HAS_TORCH else optimize_initial_conditions_numpy
    return fn(target, n=n, steps=steps, iters=iters, G=G, lr=lr)


class LumynEngine:
    """省算力 + 物理可信的天体/场景生成引擎。"""

    def __init__(self, G: float = 1.0, theta: float = 0.5):
        self.sim = NBodySimulator(G=G, theta=theta)
        self.validator = PhysicsMetrics(G=G)
        self._G = G

    def generate(self, scene_type: str = "spiral_galaxy", steps: int = 40, **kwargs) -> Dict[str, Any]:
        scene = presets.get(scene_type, **kwargs)
        traj, mass = scene.run(steps=steps)
        validation = self.validator.diagnose(traj, mass, dt=scene.dt)
        return {
            "trajectory": traj,
            "mass": mass,
            "scene": scene,
            "validation": validation,
            "meta": {"scene_type": scene_type, "G": self._G},
        }

    def render(self, result: Dict, output_path: str = "out.mp4") -> str:
        traj = result["trajectory"]
        mass = result["mass"]
        scene = result["scene"]
        vis = ScientificVisualizer(
            {"pos": traj, "vel": _vel(traj, scene.dt), "mass": mass},
            metadata={"dt": scene.dt},
        )
        return vis.render_3d(output_path=output_path, fps=15)

    def export(self, result: Dict, output_path: str = "out.csv") -> str:
        traj = result["trajectory"]
        return ScientificExporter.to_csv(
            {"pos": traj, "vel": _vel(traj, result["scene"].dt), "mass": result["mass"]},
            output_path, metadata={"G": self._G},
        )

    def optimize(self, target: Any, n: int = 40, steps: int = 10, iters: int = 8,
                  lr: float = 0.05) -> Dict:
        """可微反问题：从目标末态反推初始条件（torch 或 NumPy 兜底）。"""
        return _optimize(np.asarray(target, dtype=np.float32), n, steps, iters, self._G, lr)


def _vel(traj: np.ndarray, dt: float) -> np.ndarray:
    T = traj.shape[0]
    vel = np.zeros_like(traj)
    for t in range(T - 1):
        vel[t] = (traj[t + 1] - traj[t]) / dt
    vel[-1] = vel[-2]
    return vel
