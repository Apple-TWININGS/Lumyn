"""探测：哪个守恒通道适合做「积分是否充分解析」的收敛检查？

背景
----
angrmor/momentum 通道对任何像样的积分器都在**机器精度**附近（实测物理轨迹
基线 4e-16），因此它们测的是**浮点累积**而非离散误差 —— 细化 dt 反而让残差变大。
若验证门要用守恒量判断积分质量，必须选一个随 dt 单调收敛的通道。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
from lumyn.scientific import presets
from lumyn.eval.conservation_critic import conservation_channels

KS = (1, 2, 4, 8, 16)
SCENES = ("binary_star", "spiral_galaxy", "globular_cluster", "galaxy_collision")


def residual(name, n, steps, k, ch):
    sc = presets.get(name, n_particles=n)
    sc.dt = sc.dt / k
    traj, mass = sc.run(steps=steps * k)
    vel = np.zeros_like(traj)
    vel[:-1] = (traj[1:] - traj[:-1]) / sc.dt
    vel[-1] = vel[-2]
    return conservation_channels(traj, vel, mass)[ch]


def main():
    for ch in ("energy", "angular", "momentum"):
        print("=" * 84)
        print(f"通道 {ch}")
        print("=" * 84)
        head = f"{'scene':<18}" + "".join(f"{'k=' + str(k):>15}" for k in KS)
        print(head + f"{'单调收敛?':>12}")
        print("-" * 84)
        for name in SCENES:
            vals = [residual(name, 80, 20, k, ch) for k in KS]
            mono = all(vals[i + 1] <= vals[i] * 1.001 for i in range(len(vals) - 1))
            print(f"{name:<18}" + "".join(f"{v:>15.4e}" for v in vals)
                  + f"{('是' if mono else '否'):>12}")
        print()


if __name__ == "__main__":
    main()
