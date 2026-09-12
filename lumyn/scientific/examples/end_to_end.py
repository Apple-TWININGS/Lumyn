"""端到端示例：四场景 → 演化 → 诊断 → 渲染 MP4 → 导出 CSV。

用法:
    PYTHONPATH=. python lumyn/scientific/examples/end_to_end.py

产物输出到 <repo>/lumyn_output/。

本脚本此前**完全无法运行**，原因有六处：

1. `_ROOT = _HERE.parents[2]` 指向 `lumyn/` 而非仓库根，`import lumyn` 必然失败；
2. `OUT = ROOT / ...` 中的 `ROOT` 从未定义（实际变量名是 `_ROOT`）；
3. `presets.get_preset()` 不存在（真实接口是 `presets.get()`），且它返回
   `SceneParams` 数据类而非 dict，因此 `params["pos"]` 等索引也错；
4. `metrics.center_of_mass_drift()` 不是模块级函数（是 `PhysicsMetrics.center_of_mass`）；
5. `ScientificVisualizer(traj, mass)` 需要的是**轨迹字典**，不是裸数组；
6. `export.to_csv(traj_dict, ...)` 同样需要字典。

现全部按真实接口重写。
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
# 包根是仓库根：end_to_end.py → examples → scientific → lumyn → <repo>
_ROOT = _HERE.parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lumyn.scientific import (  # noqa: E402
    PhysicsMetrics, ScientificExporter, ScientificVisualizer, presets,
)

SCENES = ("binary_star", "spiral_galaxy", "globular_cluster", "galaxy_collision")
OUT = _ROOT / "lumyn_output"
OUT.mkdir(exist_ok=True)


def _vel_from_pos(traj: np.ndarray, dt: float) -> np.ndarray:
    """由位置差分估计速度；末帧沿用倒数第二帧（避免零速度伪影）。"""
    vel = np.zeros_like(traj)
    vel[:-1] = (traj[1:] - traj[:-1]) / dt
    vel[-1] = vel[-2] if len(traj) > 1 else 0.0
    return vel


def run_scene(name: str, n_steps: int = 80, n: int = 48) -> dict:
    print(f"\n=== {name} ===")
    scene = presets.get(name, n_particles=n)
    traj, mass = scene.run(steps=n_steps)
    dt = scene.dt

    # 诊断（全部通过 PhysicsMetrics，其 diagnose() 一次返回所有量）
    mt = PhysicsMetrics(G=scene.G)
    d = mt.diagnose(traj, mass, dt=dt)
    diag = {
        "energy_drift": float(d["energy_drift"]),
        "angular_momentum_drift": float(d["angular_momentum_drift"]),
        "com_drift": float(d["com_drift"]),
        "momentum_conserved": bool(mt.momentum_conserved(traj, mass, dt=dt)),
    }
    print(f"  frames={traj.shape[0]} particles={traj.shape[1]} dt={dt}")
    print(f"  energy_drift={diag['energy_drift']:.4e}  "
          f"am_drift={diag['angular_momentum_drift']:.4e}  "
          f"com_drift={diag['com_drift']:.4e}")

    trajectories = {"pos": traj, "vel": _vel_from_pos(traj, dt), "mass": mass}

    # 渲染（ScientificVisualizer 需要轨迹字典）
    try:
        viz = ScientificVisualizer(trajectories, metadata={"dt": dt, "scene": name})
        mp4 = viz.render_2d_with_physics(str(OUT / f"{name}.mp4"), fps=20,
                                         diagnostics=d)
        print(f"  → {mp4}")
    except Exception as e:      # pragma: no cover - 渲染可选
        print(f"  [warn] 渲染跳过: {type(e).__name__}: {e}")

    # 导出 CSV（同样需要轨迹字典）
    csv_path = OUT / f"{name}.csv"
    ScientificExporter.to_csv(trajectories, str(csv_path),
                              metadata={"scene": name, **diag}, time_step=dt)
    print(f"  → {csv_path}")
    return diag


def main():
    summary = {}
    for name in SCENES:
        summary[name] = run_scene(name, n_steps=80, n=48)

    print("\n===== 守恒诊断汇总 =====")
    print(f"  {'scene':<20}{'energy_drift':>16}{'am_drift':>16}"
          f"{'com_drift':>16}{'momentum_ok':>14}")
    for name, d in summary.items():
        print(f"  {name:<20}{d['energy_drift']:>16.4e}"
              f"{d['angular_momentum_drift']:>16.4e}"
              f"{d['com_drift']:>16.4e}{str(d['momentum_conserved']):>14}")

    print(f"\n产物目录: {OUT}")


if __name__ == "__main__":
    main()
