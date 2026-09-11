"""移动端省算力调度：QUBO 关键帧选择 + 时间稀疏门 + 热控降级。

复用 cosmic_video 的 QUBO 思路：只真算 K 帧 HMMP，其余插值。
发热时切 OfflineShadow 降级：HMMP 从 O(N log N) → 叶子近似 O(N)。
"""
import numpy as np


def select_keyframes(motion_scores, budget, iters=50, lr=0.1):
    """连续松弛 QUBO：选 budget 个关键帧，最大化运动量覆盖。"""
    h = np.asarray(motion_scores, dtype=np.float32)
    n = len(h)
    if budget <= 0 or n == 0:
        return np.empty(0, dtype=int)
    x = np.full(n, budget / n, dtype=np.float32)
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = J[i + 1, i] = -0.5
    for _ in range(iters):
        grad = h + J @ x
        x = x + lr * grad
        # 投影单纯形 {x>=0, sum=budget}
        u = np.sort(x)[::-1]
        css = np.cumsum(u) - budget
        rho = (np.arange(n) + 1)[u - css / np.arange(1, n + 1) > 0][-1]
        theta = css[rho - 1] / rho
        x = np.clip(x - theta, 0, budget)
    return np.sort(np.argsort(x)[::-1][:budget])


def interpolate_lod(frames, keyframes, method="linear"):
    """非关键帧用线性插值填充（省算力）。"""
    out = frames.copy()
    kf = sorted(keyframes)
    for i in range(len(out)):
        if i in kf:
            continue
        # 找左右最近关键帧
        left = max([k for k in kf if k <= i], default=None)
        right = min([k for k in kf if k >= i], default=None)
        if left is not None and right is not None and right != left:
            t = (i - left) / (right - left)
            out[i] = frames[left] * (1 - t) + frames[right] * t
        elif left is not None:
            out[i] = frames[left]
    return out


class ThermalGuard:
    """热控降级：温度/功耗超阈值 → 降低 HMMP 精度档位。"""

    TIERS = ["full", "bh_leaf_only", "offline_shadow"]

    def __init__(self, temp_thresholds=(70.0, 85.0)):
        self.lo, self.hi = temp_thresholds
        self.tier = 0

    def update(self, temp_c):
        if temp_c >= self.hi:
            self.tier = 2   # offline_shadow
        elif temp_c >= self.lo:
            self.tier = 1   # bh_leaf_only
        else:
            self.tier = 0   # full
        return self.TIERS[self.tier]

    def hmmp_mode(self):
        return self.TIERS[self.tier]

    def compute_budget(self, base_n):
        """降级时减少粒子数 / 跳过远场。"""
        return {0: base_n, 1: base_n // 2, 2: 0}[self.tier]


class LODPolicy:
    """组合 QUBO 关键帧 + 热控。"""

    def __init__(self, budget=3):
        self.budget = budget
        self.thermal = ThermalGuard()

    def plan(self, motion_series, temp_c=45.0):
        self.thermal.update(temp_c)
        mode = self.thermal.hmmp_mode()
        if mode == "offline_shadow":
            return {"keyframes": [], "mode": mode, "skip": True}
        kf = select_keyframes(motion_series, self.budget)
        return {"keyframes": kf.tolist(), "mode": mode, "skip": False}


__all__ = ["select_keyframes", "interpolate_lod", "ThermalGuard", "LODPolicy"]
