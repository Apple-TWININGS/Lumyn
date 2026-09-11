"""从 HMMP 粒子轨迹导 NavMesh 高度场（供引擎烘焙可行走面）。"""


def trajectory_to_heightfield(trajectory, grid=32):
    """粒子轨迹 (T, N, 3) → 高度场 grid×grid，取每格粒子平均 z。"""
    import numpy as np
    traj = np.asarray(trajectory)
    t, n, _ = traj.shape
    hf = np.zeros((grid, grid))
    count = np.zeros((grid, grid))
    xz = traj.reshape(-1, 3)[:, [0, 2]]
    z = traj.reshape(-1, 3)[:, 1]
    # 归一化到网格
    lo, hi = xz.min(axis=0), xz.max(axis=0)
    idx = ((xz - lo) / (hi - lo + 1e-8) * (grid - 1)).astype(int)
    idx = np.clip(idx, 0, grid - 1)
    for (ix, iy), zi in zip(idx, z):
        hf[iy, ix] += zi
        count[iy, ix] += 1
    count[count == 0] = 1
    return hf / count


__all__ = ["trajectory_to_heightfield"]
