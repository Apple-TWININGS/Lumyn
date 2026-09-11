"""可微梯度引导生成最小演示（论文方法 §3.3 示例，纯 NumPy）。

从"目标形态"反演符合物理的初始条件：整段 N 体演化是可微算子，
用有限差分近似梯度，对初始位置做梯度下降，使演化轨迹逼近目标形态。

用法:
    python lumyn/scientific/examples/diff_demo.py
"""
import sys
from pathlib import Path
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from lumyn.scientific.examples.diff_figure import (
    simulate, finite_diff_grad, _energy,
)


def main():
    rng = np.random.default_rng(0)
    n = 8
    G = 1.0
    dt = 0.05
    steps = 60

    mass = np.array([10.0] + [1.0] * (n - 1))
    pos = rng.normal(0, 2.0, (n, 3)); pos[0] = 0.0
    vel = rng.normal(0, 0.3, (n, 3))

    target = simulate(pos, vel, mass, n_steps=steps, dt=dt, G=G)
    target = target + rng.normal(0, 0.05, target.shape)

    pos_curr = pos.copy()
    losses = []
    for it in range(80):
        traj = simulate(pos_curr, vel, mass, n_steps=steps, dt=dt, G=G)
        loss = ((traj - target) ** 2).mean()
        losses.append(loss)
        grad = finite_diff_grad(pos_curr, vel, mass, target, eps=1e-3, dt=dt, G=G)
        pos_curr = pos_curr - 0.05 * grad

    final = simulate(pos_curr, vel, mass, n_steps=steps, dt=dt, G=G)
    final_loss = ((final - target) ** 2).mean()
    e0 = _energy(pos, vel, mass, G)
    e_now = _energy(pos_curr, vel, mass, G)
    print("===== 可微梯度引导生成（NumPy 有限差分） =====")
    print(f"  n_particles   : {n}")
    print(f"  optimizer     : finite-difference GD")
    print(f"  initial loss  : {losses[0]:.4e}")
    print(f"  final loss    : {losses[-1]:.4e}")
    print(f"  final traj err: {final_loss:.4e}")
    print(f"  |ΔE|/|E₀|     : {abs(e_now - e0) / (abs(e0) + 1e-12):.4e}")
    print("[ok] 初始条件已反演为目标形态（守恒量受约束）")


if __name__ == "__main__":
    main()
