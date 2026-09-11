"""场景预设：把「物理参数 + 初始条件」封装成可复用的生成器。

每个函数返回 (pos, mass, vel)，可直接喂给 NBodySimulator.simulate。
"""
from __future__ import annotations

import numpy as np


def galaxy(n_particles=500, seed=42, G=1.0):
    """螺旋星系：中心大质量 + 盘面开普勒轨道 + 螺旋臂扰动。"""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.1, 1.0, n_particles)
    theta = rng.uniform(0, 2 * np.pi, n_particles)
    theta = theta + 2.0 * r                       # 螺旋臂
    pos = np.stack([
        r * np.cos(theta),
        r * np.sin(theta),
        rng.normal(0, 0.02, n_particles),
    ], axis=1)

    mass = 10.0 / (1.0 + r ** 2)                  # 中心大、边缘小

    v_circ = np.sqrt(G * 10.0 / np.maximum(r, 0.1))
    vel = np.stack([
        -v_circ * np.sin(theta),
        v_circ * np.cos(theta),
        np.zeros(n_particles),
    ], axis=1)

    # 中心黑洞作为固定势阱（不参与动力学），速度=0。
    # fixed_mask 存为模块属性，供守恒检查排除黑洞（避免伪动量漂移）。
    pos = np.concatenate([np.array([[0.0, 0.0, 0.0]]), pos], axis=0)
    mass = np.concatenate([np.array([1000.0]), mass], axis=0)
    vel = np.concatenate([np.zeros((1, 3)), vel], axis=0)
    fixed_mask = np.zeros(len(mass), dtype=bool)
    fixed_mask[0] = True
    galaxy.fixed_mask = fixed_mask  # 也存为函数属性，便于外部查询
    return pos, mass, vel, fixed_mask


__all__ = ["galaxy", "explosion", "collision", "SCENES"]


def explosion(n_particles=200, seed=7):
    """爆炸：所有粒子从原点以随机速度飞散，加斥力防堆叠。"""
    rng = np.random.default_rng(seed)
    pos = np.zeros((n_particles, 3))
    vel = rng.normal(0, 1.0, (n_particles, 3))
    mass = rng.uniform(0.1, 1.0, n_particles)
    return pos, mass, vel


def collision(n_particles=300, seed=99):
    """两团粒子对撞：用于流体/碰撞特效。"""
    rng = np.random.default_rng(seed)
    n1 = n_particles // 2
    n2 = n_particles - n1

    pos1 = rng.normal([-2, 0, 0], 0.3, (n1, 3))
    pos2 = rng.normal([2, 0, 0], 0.3, (n2, 3))
    pos = np.concatenate([pos1, pos2], axis=0)

    vel1 = rng.normal([1, 0, 0], 0.1, (n1, 3))
    vel2 = rng.normal([-1, 0, 0], 0.1, (n2, 3))
    vel = np.concatenate([vel1, vel2], axis=0)

    mass = rng.uniform(0.5, 1.5, n_particles)
    return pos, mass, vel


SCENES = {
    "galaxy": galaxy,
    "explosion": explosion,
    "collision": collision,
}


class HMMPSceneGenerator:
    """场景生成器类封装（兼容 physics/__init__.py 的历史导入路径）。

    把 ``galaxy / explosion / collision`` 三个工厂函数包装成统一的
    ``generate(scene_type, **kwargs)`` 接口，返回 dict：

        {"pos": (N,3), "mass": (N,), "vel": (N,3),
         "fixed_mask": (N,)bool, "type": str}
    """

    def __init__(self, G: float = 1.0):
        self.G = float(G)

    def generate(self, scene_type: str = "galaxy", **kwargs):
        if scene_type not in SCENES:
            raise ValueError(
                f"Unknown scene_type={scene_type!r}. Choose from {list(SCENES)}")
        factory = SCENES[scene_type]
        result = factory(G=self.G, **kwargs)
        # galaxy 返回 4 元组 (含 fixed_mask)，其余返回 3 元组
        if scene_type == "galaxy":
            pos, mass, vel, fixed_mask = result
        else:
            pos, mass, vel = result
            fixed_mask = np.zeros(len(mass), dtype=bool)
        return {
            "pos": pos, "mass": mass, "vel": vel,
            "fixed_mask": fixed_mask, "type": scene_type,
        }

    def __call__(self, scene_type: str = "galaxy", **kwargs):
        return self.generate(scene_type, **kwargs)
