"""TeachingMode：同一场景不同参数并排对比（质量 / 角速度）。"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..physics.nbody import NBodySimulator


class TeachingMode:
    """物理教学演示：参数滑块式对比，导出并排 PNG。"""

    @staticmethod
    def compare_mass(scene_factory, base_params: dict, mass_factors=(0.5, 1.0, 2.0),
                     output_path: str = "mass_comparison.png", steps: int = 40) -> str:
        """质量翻倍 → 轨道收缩的可视化对比。"""
        fig, axes = plt.subplots(1, len(mass_factors), figsize=(6 * len(mass_factors), 6))
        if len(mass_factors) == 1:
            axes = [axes]

        for i, factor in enumerate(mass_factors):
            params = dict(base_params)
            params["masses"] = np.asarray(base_params["masses"], dtype=np.float64) * factor
            traj, mass = scene_factory(params).run(steps)
            ax = axes[i]
            ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=8, alpha=0.7)
            ax.set_title(f"Mass × {factor}")
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
            ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(output_path, dpi=110)
        plt.close(fig)
        return output_path

    @staticmethod
    def compare_angular_velocity(scene_factory, base_params: dict,
                                  omegas=(0.5, 1.0, 1.5),
                                  output_path: str = "omega_comparison.png",
                                  steps: int = 40) -> str:
        """角速度不同 → 旋臂松紧对比。"""
        fig, axes = plt.subplots(1, len(omegas), figsize=(6 * len(omegas), 6))
        if len(omegas) == 1:
            axes = [axes]

        for i, w in enumerate(omegas):
            params = dict(base_params)
            params["omega"] = w
            traj, mass = scene_factory(params).run(steps)
            ax = axes[i]
            ax.scatter(traj[-1, :, 0], traj[-1, :, 1], s=8, alpha=0.7, c=range(len(mass)), cmap="plasma")
            ax.set_title(f"Ω = {w}")
            ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
            ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(output_path, dpi=110)
        plt.close(fig)
        return output_path
