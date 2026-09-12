"""端到端演示：四场景 → MP4 → 抽帧预览图 → 守恒诊断 → 可微优化。

运行：python -m lumyn.examples.scientific_demo
产物输出到 ../../lumyn_output/（与上一版 galaxy_collision.mp4 同目录）
"""
from __future__ import annotations
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from lumyn.scientific import (  # noqa: E402
    ScientificVisualizer, ScientificExporter, FrameExtractor,
    TeachingMode, PhysicsMetrics, presets,
)
# NumPy 兜底实现：**始终导入**，并显式别名。
# 此前该脚本在 torch 可用时 `DifferentiableNBody` 绑定到 torch 版
# （scientific/differentiable.py），却把 numpy 数组喂进去，于是报
# `'numpy.ndarray' object has no attribute 'unsqueeze'` —— 注释写着「统一用 NumPy 版」，
# 但代码并没有这么做。现在一律用 NumPy 版，行为与注释一致。
from lumyn.scientific.differentiable_numpy import (  # noqa: E402
    DifferentiableNBodyNumPy,
    ConservationConstraintNumPy,
    optimize_initial_conditions as optimize_initial_conditions_numpy,
)

try:
    import torch as _torch  # noqa: E402
    _HAS_TORCH = True
except ImportError:
    _torch = None  # type: ignore
    _HAS_TORCH = False

import numpy as _np  # noqa: E402（脚本内部沿用 _np 别名）

# 产物目录。此前写成 os.path.join(ROOT, "..", "lumyn_output")，会把文件写到
# **仓库之外**（ROOT 已经是仓库根，再加 ".." 就跑出去了）。
OUT = os.path.abspath(os.path.join(ROOT, "lumyn_output"))
os.makedirs(OUT, exist_ok=True)


def run_scene(name: str, steps: int = 40):
    print(f"\n[Scene] {name}")
    scene = presets.get(name)
    traj, mass = scene.run(steps=steps)
    scene_dt = scene.dt

    trajectories = {"pos": traj, "vel": _estimate_vel(traj, scene_dt), "mass": mass}
    meta = {"name": name, "G": scene.G, "dt": scene_dt, "steps": steps,
            "n_particles": scene.n_particles}

    # MP4（降低采样与粒子数以适配无 GPU 环境）
    vis = ScientificVisualizer(trajectories, metadata={"dt": scene_dt})
    mp4 = vis.render_3d(output_path=os.path.join(OUT, f"{name}.mp4"),
                         fps=8, every=max(2, steps // 8), point_scale=20.0)
    print(f"  → {mp4}")

    # 抽帧预览（先抽后拼）
    try:
        from PIL import Image
        fe = FrameExtractor(mp4)
        frames = fe.extract(n=9, strategy="uniform")
        labels = [f"t={i * steps // max(len(frames), 1)}" for i in range(len(frames))]
        grid = FrameExtractor.make_grid(frames, cols=3,
                                        output_path=os.path.join(OUT, f"{name}_frames.png"),
                                        labels=labels)
        print(f"  → 抽帧预览: {grid}")
    except Exception as e:
        print(f"  ⚠ 抽帧跳过: {e}")

    # 守恒诊断
    d = PhysicsMetrics(G=scene.G).diagnose(traj, mass, dt=scene_dt)
    print(f"  energy_drift={d['energy_drift']:.4f}  am_drift={d['angular_momentum_drift']:.4f}  "
          f"com_drift={d['com_drift']:.4f}")

    # CSV 导出
    csv_path = os.path.join(OUT, f"{name}.csv")
    ScientificExporter.to_csv(trajectories, csv_path, metadata=meta, time_step=scene_dt)
    print(f"  → {csv_path}")

    return d


def _plot_diagnostics(d: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    energies = d.get("energies")
    if energies is None or len(energies) < 2:
        return
    amags = d.get("angular_momenta")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(energies, "b-"); ax1.set_ylabel("Total Energy")
    ax1.set_title(f"E drift = {d['energy_drift']:.4f}")
    if amags is not None and len(amags) > 0:
        ax2.plot(amags, "r-"); ax2.set_ylabel("|Angular Momentum|")
        ax2.set_title(f"AM drift = {d['angular_momentum_drift']:.4f}")
    ax2.set_xlabel("Frame")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  → {path}")


def _estimate_vel(traj: np.ndarray, dt: float) -> np.ndarray:
    """逐帧速度 (T, N, 3)：第 t 帧 = (pos[t+1] - pos[t]) / dt，末帧用前一帧值填充。"""
    T = traj.shape[0]
    vel = np.zeros_like(traj)
    for t in range(T - 1):
        vel[t] = (traj[t + 1] - traj[t]) / dt
    vel[-1] = vel[-2]
    return vel


def run_differentiable():
    print("\n[Differentiable] 梯度引导：从目标形态反推初始条件")
    print(f"  backend: numpy（中心差分数值梯度）"
          f"{'；torch 已安装但本演示统一走 NumPy 版以保证接口一致' if _HAS_TORCH else ''}")

    target = _np.random.default_rng(0).normal(0, 0.5, (40, 3)).astype(_np.float32)
    result = optimize_initial_conditions_numpy(target, n=40, steps=10, iters=6,
                                               lr=0.05, G=1.0)
    print(f"  final_energy_drift = {result['final_energy_drift']:.4f}")
    print(f"  trajectory shape   = {result['trajectory'].shape}")

    model = DifferentiableNBodyNumPy(G=1.0, softening=0.05, dt=0.01,
                                     method="leapfrog")
    rng = _np.random.default_rng(1)
    pos = rng.normal(0, 1, (30, 3)).astype(_np.float64)
    mass = _np.ones(30) * 0.1
    mass[0] = 5.0
    vel = _np.zeros((30, 3))
    traj = model.simulate(pos, vel, mass, steps=6)
    print(f"  trajectory shape   = {traj.shape}")

    # 守恒诊断（用与 PhysicsMetrics 相同的口径，避免依赖可选类）
    d = PhysicsMetrics(G=1.0).diagnose(traj, mass, dt=0.01)
    print(f"  E_drift={d['energy_drift']:.6f}  "
          f"AM_drift={d['angular_momentum_drift']:.6f}")
    try:
        c = ConservationConstraintNumPy()
        out = c(traj, vel, mass, G=1.0, softening=0.05)
        print(f"  constraint loss = {out['loss']:.6f}")
    except Exception as e:      # pragma: no cover - 可选组件
        print(f"  （ConservationConstraintNumPy 未采用：{type(e).__name__}）")


def run_teaching():
    print("\n[Teaching] 参数对比：质量 × {0.5, 1.0, 2.0}")
    base = {"n_particles": 150, "masses": np.ones(150) * 0.1, "G": 1.0, "dt": 0.02, "steps": 30}
    base["masses"][0] = 5.0
    # 构造初始圆盘（ TeachingMode 需要 scene_factory(pos,vel) 接口）
    rng = np.random.default_rng(1)
    r = rng.uniform(0.3, 3.0, 150); th = rng.uniform(0, 2 * np.pi, 150)
    pos = np.stack([r * np.cos(th), r * np.sin(th), np.zeros(150)], axis=1)
    vc = np.sqrt(5.0 / np.maximum(r, 0.2))
    vel = np.stack([-vc * np.sin(th), vc * np.cos(th), np.zeros(150)], axis=1)
    base["positions"] = pos; base["velocities"] = vel

    from dataclasses import dataclass

    @dataclass
    class Wrapper:
        params: dict
        def run(self, steps):
            from lumyn.physics.nbody import NBodySimulator
            p = self.params
            sim = NBodySimulator(G=p["G"], theta=0.5)
            pp, vv = p["positions"].copy(), p["velocities"].copy()
            m = p["masses"].copy()
            traj = [pp.copy()]
            for _ in range(steps):
                pp, vv = sim.step(pp, m, vv, dt=p["dt"])
                traj.append(pp.copy())
            return np.stack(traj, axis=0), m

    def factory(params):
        return Wrapper(params)

    path = TeachingMode.compare_mass(factory, base, mass_factors=(0.5, 1.0, 2.0),
                                     output_path=os.path.join(OUT, "mass_comparison.png"), steps=30)
    print(f"  → {path}")

    # 守恒诊断图（能量/角动量曲线）
    scene = presets.spiral_galaxy(n_particles=200)
    traj, mass = scene.run(steps=60)
    d = PhysicsMetrics(G=scene.G).diagnose(traj, mass, dt=scene.dt)
    _plot_diagnostics(d, os.path.join(OUT, "galaxy_diagnostics.png"))


def main():
    for name in ["binary_star", "spiral_galaxy", "globular_cluster", "galaxy_collision"]:
        run_scene(name, steps=24)
    run_teaching()
    try:
        run_differentiable()
    except ImportError as e:
        print(f"\n[Differentiable] 跳过: {e}")

    print(f"\n{'=' * 60}")
    print(f"产物目录: {OUT}")
    for f in sorted(os.listdir(OUT)):
        print(f"  {f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
