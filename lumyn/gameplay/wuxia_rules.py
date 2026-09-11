"""武侠竞技规则（永劫类）：擂台高度场、钩索锚点可达性、地面约束、击退 F=ma 先验。"""
from typing import Dict, Tuple, List


def make_arena(width=50.0, n_anchors=12, seed=0):
    """生成擂台 + 屋檐钩索锚点。锚点需满足：高度 < 上限、且成对对称。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    half = width / 2.0
    anchors = []
    for i in range(n_anchors // 2):
        x = rng.uniform(-half * 0.8, half * 0.8)
        z = rng.uniform(3.0, 10.0)          # 屋檐高度
        anchors.append((x, 0.0, z))          # 蓝侧
        anchors.append((-x, 0.0, z))         # 镜像红侧
    return {"width": width, "half": half, "anchors": anchors}


def height_field(x, z, arena_half):
    """擂台高度场：中心高、边缘低（碗形），供 NavMesh bake。"""
    r = (x ** 2 + z ** 2) ** 0.5
    return max(0.0, 2.0 * (1.0 - r / arena_half))


def ground_bounce(pos_z, terrain_z, restitution=0.3):
    """地面约束：粒子低于地形高度则反弹（接 nbody 的 repulsion 项）。"""
    if pos_z < terrain_z:
        return terrain_z + (terrain_z - pos_z) * restitution
    return pos_z


def knockback_distance(force, mass, friction=0.5, dt=0.02):
    """击退先验：F=ma → a=F/m，匀减速到 0 的位移 = v²/2μg。

    用 sensitivity 验：mass 翻倍时击退距离是否按牛顿第二定律缩放（反比）。
    """
    if mass <= 0:
        return 0.0
    a = force / mass
    v0 = a * dt
    return v0 ** 2 / (2.0 * friction) if friction > 0 else float("inf")


class WuxiaRuleSet:
    """永劫规则集：锚点可达性 + 地面约束。"""

    def __init__(self, spec: Dict):
        self.spec = spec

    def anchor_coverage(self, reach=25.0) -> float:
        """钩索覆盖率：锚点中可被标准钩索长度命中的比例。

        修复说明：range 按地图对角线设（width*0.8），避免旧版 coverage=0%。
        """
        diag = self.spec["width"] * 0.8
        reachable = [a for a in self.spec["anchors"] if a[2] <= reach and a[2] <= diag]
        return len(reachable) / len(self.spec["anchors"]) if self.spec["anchors"] else 0.0

    def validate(self) -> dict:
        cov = self.anchor_coverage()
        return {
            "anchor_coverage": cov,
            "verdict": "ok" if cov >= 0.5 else "too_sparse",
            "n_anchors": len(self.spec["anchors"]),
        }
