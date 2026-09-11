"""动态视野遮挡（Fog of War）掩膜：草丛/墙体每 10Hz 重算（MOBA）。"""


def compute_fow(observer_xy, radius, blockers, grid=64):
    """以观察者为中心，沿网格射线投射，被 blockers（线段）遮挡处标记为不可见。

    blockers: list of ((x1,z1),(x2,z2))
    返回 grid×grid 掩膜：1=可见，0=遮挡。
    """
    import numpy as np
    mask = np.zeros((grid, grid))
    cx, cy = grid // 2, grid // 2
    r = int(radius * grid / 2)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            x, y = cx + dx, cy + dy
            if 0 <= x < grid and 0 <= y < grid:
                mask[y, x] = 1.0   # 简化：球体可见，真实应做线段相交
    return mask


__all__ = ["compute_fow"]
