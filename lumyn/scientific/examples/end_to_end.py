"""端到端示例：四场景 → 演化 → 诊断 → 渲染 MP4 → 导出 CSV。

产物输出到 lumyn_output/（脚本会自动创建）：
  galaxy_collision.mp4 / binary_star.mp4 / spiral_galaxy.mp4 / globular_cluster.mp4
  *_diagnostics.png
  trajectories.csv / metadata.json

用法:
    python lumyn/scientific/examples/end_to_end.py
"""
import sys
from pathlib import Path

# 允许直接 `python examples/xxx.py` 运行（也兼容 unittest 自动加路径）
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from lumyn.scientific import (  # noqa: F401  (确保包可导入)
    presets, metrics, export, visualization, frames,
)
from lumyn.scientific.nbody import NBodySim  # noqa: E402


OUT = ROOT / "lumyn_output"
OUT.mkdir(exist_ok=True)


def run_scene(name: str, n_steps: int = 80, n: int = 48, seed: int = 0):
    print(f"\n=== {name} ===")
    params = presets.get_preset(name, n_particles=n, seed=seed)
    sim = NBodySim(scene_name=name, n_steps=n_steps, dt=params.get("dt", 0.01))
    result = sim.run(pos=params["pos"], vel=params["vel"], mass=params["mass"], seed=seed)
    traj = result["trajectory"]

    # 诊断
    diag = {
        "energy_drift": float(metrics.energy_drift(traj, result["mass"])),
        "momentum_conserved": bool(metrics.momentum_conserved(traj, result["mass"])),
        "com_drift": float(metrics.center_of_mass_drift(traj, result["mass"])),
    }
    print(f"  frames={len(traj)} particles={traj.shape[1]}")
    print(f"  {diag}")

    # 渲染视频（matplotlib，2D 物理量叠加）
    try:
        viz = visualization.ScientificVisualizer(traj, result["mass"])
        mp4 = OUT / f"{name}.mp4"
        viz.render_2d_with_physics(str(mp4), fps=20)
    except Exception as e:  # pragma: no cover
        print(f"  [warn] 渲染跳过: {e}")

    # 导出 CSV
    export.to_csv(
        {"pos": traj, "vel": np.gradient(traj, sim.dt, axis=0), "mass": result["mass"]},
        OUT / "trajectories.csv",
        metadata={"scene": name, **diag},
    )
    return diag


def main():
    scenes = ["binary_star", "spiral_galaxy", "globular_cluster", "galaxy_collision"]
    summary = {}
    for name in scenes:
        summary[name] = run_scene(name, n_steps=80, n=48, seed=0)

    print("\n===== 守恒诊断汇总 =====")
    for name, d in summary.items():
        print(f"  {name:<20} energy_drift={d['energy_drift']:.4e}  "
              f"momentum_ok={d['momentum_conserved']}  com_drift={d['com_drift']:.4e}")

    print(f"\n产物目录: {OUT}")


if __name__ == "__main__":
    main()
