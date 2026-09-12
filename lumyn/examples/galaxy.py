"""单场景快速示例：生成一个物理星系并打印守恒诊断。

运行：
    PYTHONPATH=. python lumyn/examples/galaxy.py

本脚本此前调用的是一个**不存在的 API**：`LumynEngine(img_size=...)`、
把中文句子当作 `scene_type` 传入、读取 `result["validation"]["conservation"]`
与 `result["video"]`，并写入硬编码的 Linux 绝对路径 `/data/workspace/...`。
现按 `LumynEngine` 的真实接口重写。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# 包根是 lumyn/ 的**父目录**（仓库根），不是本文件所在目录
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from lumyn import LumynEngine  # noqa: E402

#: `presets.get()` 接受的场景名。注意不存在 "galaxy" 这个键。
VALID_SCENES = ("binary_star", "spiral_galaxy", "globular_cluster", "galaxy_collision")


def main(scene_type: str = "spiral_galaxy", n_particles: int = 200,
         steps: int = 40) -> None:
    if scene_type not in VALID_SCENES:
        raise SystemExit(f"未知场景 {scene_type!r}；可用：{', '.join(VALID_SCENES)}")

    engine = LumynEngine(G=1.0, theta=0.5)
    result = engine.generate(scene_type, n_particles=n_particles, steps=steps)

    traj = result["trajectory"]
    v = result["validation"]

    print(f"场景:     {result['meta']['scene_type']}")
    print(f"轨迹:     {traj.shape}  (步数+1, 粒子数, 3)")
    print(f"质量:     {result['mass'].shape}")
    print()
    print("守恒诊断（各量均为相对漂移，越小越物理）：")
    for key in ("energy_drift", "momentum_drift", "angular_momentum_drift",
                "com_drift"):
        if key in v:
            print(f"  {key:<24} {v[key]:.6e}")
    print()
    print(f"质心漂移:      {v.get('com_drift', float('nan')):.6e}")
    print(f"半质量半径:    {v.get('half_mass_radius_final', float('nan')):.4f}")
    print()
    print("注：`validation` 是 PhysicsMetrics.diagnose() 的输出，**没有 `verdict` 键**；")
    print("    判定请用 eval.ConservationCritic 或 eval.ConservationChecker。")


if __name__ == "__main__":
    main()
