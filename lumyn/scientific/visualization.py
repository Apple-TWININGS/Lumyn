"""ScientificVisualizer：科研级可视化（粒子大小=质量，颜色=速度）。"""
from __future__ import annotations
import os
import numpy as np


class ScientificVisualizer:
    """渲染 3D 粒子动画 / 2D 俯视 + 守恒量曲线叠加。"""

    def __init__(self, trajectories: dict, metadata: dict = None):
        self.pos = np.asarray(trajectories["pos"], dtype=np.float64)
        self.vel = np.asarray(trajectories.get("vel", np.zeros_like(self.pos)), dtype=np.float64)
        self.mass = np.asarray(trajectories["mass"], dtype=np.float64)
        self.metadata = metadata or {}

    def _writer(self, output_path: str, fps: int):
        """优先 ffmpeg，缺失则降级到 GIF/图片。"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
        except ImportError:
            raise ImportError("需要 matplotlib")

        if output_path.lower().endswith(".gif"):
            return PillowWriter(fps=fps), "gif"
        try:
            return FFMpegWriter(fps=fps, codec="libx264"), "mp4"
        except Exception:
            return PillowWriter(fps=fps), "gif"

    def render_3d(self, output_path: str = "galaxy.mp4", fps: int = 20,
                  every: int = 1, point_scale: float = 40.0) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        T, N, _ = self.pos.shape
        speeds = np.linalg.norm(self.vel, axis=-1)  # (T, N)
        vmax = max(float(speeds.max()), 1e-8)
        sizes = point_scale * (0.2 + 0.8 * self.mass / (float(self.mass.max()) + 1e-8))
        writer, ext = self._writer(output_path, fps)
        if ext == "gif":
            output_path = output_path.rsplit(".", 1)[0] + ".gif"

        fig = plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111, projection="3d")
        frames = range(0, T, every)

        def update(frame):
            ax.clear()
            color = (speeds[frame] / vmax).ravel()  # (N,) 逐粒子速度色
            ax.scatter(self.pos[frame, :, 0], self.pos[frame, :, 1], self.pos[frame, :, 2],
                       s=sizes, c=color, cmap="viridis", alpha=0.75, depthshade=False)
            lim = float(np.abs(self.pos).max()) * 1.05
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.set_title(f"t = {frame * self.metadata.get('dt', 0.01):.3f}")

        anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
        anim.save(output_path, writer=writer)
        plt.close(fig)
        return output_path

    def render_2d_with_physics(self, output_path: str, fps: int = 20,
                               diagnostics: dict = None) -> str:
        """2D 俯视图 + 能量/角动量曲线（需先由 PhysicsMetrics.diagnose 得到）。"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        T, N, _ = self.pos.shape
        diagnostics = diagnostics or {}
        energies = diagnostics.get("energies")
        amags = diagnostics.get("angular_momenta")
        writer, ext = self._writer(output_path, fps)
        if ext == "gif":
            output_path = output_path.rsplit(".", 1)[0] + ".gif"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10),
                                       gridspec_kw={"height_ratios": [3, 1]})
        sizes = 10 + 40 * (self.mass / (float(self.mass.max()) + 1e-8))

        def update(frame):
            ax1.clear()
            ax1.scatter(self.pos[frame, :, 0], self.pos[frame, :, 1], s=sizes, alpha=0.7)
            ax1.set_title(f"t = {frame * self.metadata.get('dt', 0.01):.3f}")
            ax1.set_aspect("equal")
            ax2.clear()
            if energies is not None:
                ax2.plot(energies[:frame + 1], "b-", label="Energy")
            if amags is not None:
                ax2.plot(amags[:frame + 1], "r-", label="|L|")
            ax2.legend(loc="upper right", fontsize=8)

        anim = FuncAnimation(fig, update, frames=range(0, T, max(1, T // 60)), interval=1000 / fps)
        anim.save(output_path, writer=writer)
        plt.close(fig)
        return output_path
