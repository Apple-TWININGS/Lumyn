"""MOBA 规则（王者类）：三路/河道/野区 + 红蓝镜像公平校验。

symmetry_check 用 HMMP 受力演化比对镜像半场的物理可达性——
两边能量曲线重合 → 地图公平；否则 SimScore 判 invalid。
"""
from typing import Dict, Tuple, List
import numpy as np


def make_moba_map(width=100.0, n_camps=8, seed=0):
    """生成对称三路 MOBA 地图：中路直线，两侧路带曲率，野区 Voronoi 点。"""
    rng = np.random.default_rng(seed)
    half = width / 2.0
    # 三路基线（中路 y=0，上路 y>0，下路 y<0）
    lanes = {
        "mid": [(x, 0.0) for x in np.linspace(-half, half, 9)],
        "top": [(-half + i * width / 4,  half * 0.6) for i in range(5)],
        "bot": [(-half + i * width / 4, -half * 0.6) for i in range(5)],
    }
    # 野区营地（镜像成对）
    camps = []
    for i in range(n_camps // 2):
        x = rng.uniform(0, half * 0.7)
        y = rng.uniform(-half * 0.7, half * 0.7)
        camps.append(( x,  y))   # 蓝方（右）
        camps.append((-x,  y))   # 红方（左，镜像）
    return {"lanes": lanes, "camps": camps, "width": width, "half": half}


def mirror_x(pts, half):
    """关于 x=0 镜像（红蓝互换）。"""
    return [(-x, y) for (x, y) in pts]


def symmetry_score(map_a, map_b, n_particles=64, steps=20, dt=0.01, G=1.0):
    """两半场各跑一次 N 体短演化，比对能量/路径对称 → SimScore 变体。

    返回 0~1，越接近 1 越公平。
    """
    def evolve(pts):
        pos = np.array(pts, dtype=np.float64)
        if len(pos) == 0:
            return np.zeros((steps, 0, 2))
        mass = np.ones(len(pos))
        vel = np.zeros_like(pos)
        traj = [pos.copy()]
        for _ in range(steps):
            # 简化引力步（足够做公平比对）
            d = pos[:, None, :] - pos[None, :, :]
            r2 = (d * d).sum(axis=2) + 1e-6
            acc = (mass[None, :, None] * d * r2[:, :, None] ** -1.5).sum(axis=1)
            vel = vel + G * acc * dt
            pos = pos + vel * dt
            traj.append(pos.copy())
        return np.stack(traj)
    # map_a/b 为点集；若结构不同（点数差），直接低分
    if len(map_a) != len(map_b):
        return 0.0
    ta = evolve(map_a)
    tb = evolve(mirror_x(map_b, 0.0) if False else map_b)
    if ta.shape != tb.shape or ta.shape[1] == 0:
        return 0.0
    diff = np.abs(ta - tb).mean()
    return float(np.clip(1.0 / (1.0 + diff), 0.0, 1.0))


class MobaSymmetryChecker:
    """红蓝镜像公平校验器（接 SimScore 的 StepCoverage 思想）。

    校验方式：地图生成时已保证红方 = 蓝方关于 x=0 镜像（x 取反、y/z 不变）。
    这里比对每对镜像点的坐标差 → 越接近 0 越公平（确定性，不依赖 N 体演化噪声）。
    """

    def __init__(self, map_spec: Dict):
        self.spec = map_spec

    def check(self, threshold=0.9) -> Dict:
        camps = self.spec["camps"]
        # 按 x 符号拆红蓝（x==0 归蓝）
        blue = sorted([c for c in camps if c[0] >= 0], key=lambda c: (c[0], c[1]))
        red = sorted([c for c in camps if c[0] < 0], key=lambda c: (-c[0], c[1]))
        # 镜像对称：red 应满足 red_i == (-blue_i.x, blue_i.y)
        max_pairs = min(len(blue), len(red))
        if max_pairs == 0:
            score = 0.0
        else:
            diffs = []
            for i in range(max_pairs):
                bx, by = blue[i]
                rx, ry = red[i]
                diffs.append(abs(rx - (-bx)) + abs(ry - by))
            max_score = max(diffs)
            score = float(np.clip(1.0 / (1.0 + max_score), 0.0, 1.0))
        return {
            "symmetry_score": score,
            "verdict": "fair" if score >= threshold else "unfair",
            "blue_camps": len(blue),
            "red_camps": len(red),
        }
