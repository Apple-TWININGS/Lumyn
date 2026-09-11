"""生成「梯度引导生成」论文 Figure：优化过程中损失与能量漂移的收敛曲线。

输出：lumyn_output/gradient_guided_optimization.png

用 NumPy 数值梯度版（无需 torch），真实跑出数据再绘图。
两张子图：
  (a) 总损失 vs 优化步数（应单调下降）
  (b) 能量相对漂移 vs 优化步数（应改善 → 守恒）
"""
from __future__ import annotations
import os
import sys
# 确保脚本可直接 `python lumyn/physics/make_figure.py` 运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lumyn.physics.differentiable_numpy import DifferentiableNBodyNumpy

OUT = os.path.join(os.path.dirname(__file__), "..", "lumyn_output")
os.makedirs(OUT, exist_ok=True)


def energy_drift(pos_traj, dt, mass, G=1.0, softening=1e-3):
    """计算轨迹每帧相对初始帧的能量漂移。"""
    N = pos_traj.shape[1]
    eye = 1.0 - np.eye(N)

    def energy(pos, vel):
        ke = 0.5 * (mass * (vel * vel).sum(-1)).sum()
        d = pos[None, :, :] - pos[:, None, :]
        r = np.sqrt((d * d).sum(-1) + softening ** 2)
        pe = -G * (mass[None, :] * mass[:, None] / r * eye).sum() / 2.0
        return ke + pe

    drifts = []
    for t in range(1, len(pos_traj)):
        vel_t = (pos_traj[t] - pos_traj[t - 1]) / dt
        vel_0 = (pos_traj[1] - pos_traj[0]) / dt
        E0 = energy(pos_traj[0], vel_0)
        Et = energy(pos_traj[t], vel_t)
        drifts.append(abs(Et - E0) / (abs(E0) + 1e-12))
    return np.array(drifts)


def main():
    rng = np.random.default_rng(7)
    N = 8
    pos = rng.normal(0, 0.5, (N, 3)).astype(np.float64)
    pos[:, 2] *= 0.2
    r = np.sqrt((pos[:, :2] ** 2).sum(-1)).clip(min=0.3)
    vel = np.zeros_like(pos)
    v_circ = np.sqrt(1.0 / r)
    vel[:, 0] = -pos[:, 1] / r * v_circ
    vel[:, 1] = pos[:, 0] / r * v_circ
    mass = np.abs(rng.normal(0.5, 0.3, (N,))) + 0.1

    target_radius = 1.0
    sys = DifferentiableNBodyNumpy(pos, vel, mass, steps=12)

    losses, drifts = [], []

    def step_callback(s, loss, grad_norm):
        losses.append(loss)
        # 记录当前轨迹的能量漂移（采样）
        out = sys.forward()
        d = energy_drift(out["trajectory"], sys.dt, sys.mass)
        drifts.append(float(d.max()) if len(d) else 0.0)

    info = sys.optimize(target_radius=target_radius, w_energy=1.0, lr=0.03, steps=120,
                        callback=step_callback)
    steps = np.arange(len(losses))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.semilogy(steps, losses, color="#c0392b", lw=1.8)
    ax1.set_xlabel("Optimization step")
    ax1.set_ylabel("Total loss (log)")
    ax1.set_title("(a) Gradient-guided loss converges")
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(steps, drifts, color="#2980b9", lw=1.8)
    ax2.set_xlabel("Optimization step")
    ax2.set_ylabel("Max energy drift (log)")
    ax2.set_title("(b) Conservation improves")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Gradient-guided generation (NumPy central-difference gradients)\n"
        f"N={N} particles, target radius={target_radius}, final loss={info['final_loss']:.2e}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(OUT, "gradient_guided_optimization.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[figure] saved: {path}")
    print(f"[figure] 初始损失={losses[0]:.4f}, 最终={losses[-1]:.2e}")
    print(f"[figure] 初始漂移={drifts[0]:.2e}, 最终={drifts[-1]:.2e}")


if __name__ == "__main__":
    main()
