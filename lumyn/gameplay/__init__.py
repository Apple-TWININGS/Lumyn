"""gameplay：竞技游戏专属 PCG 规则（MOBA / 武侠）。"""
from .moba_rules import make_moba_map, symmetry_score, MobaSymmetryChecker
from .wuxia_rules import make_arena, height_field, ground_bounce, knockback_distance, WuxiaRuleSet
from .navmesh_patch import trajectory_to_heightfield
from .fow_mask import compute_fow

__all__ = [
    "make_moba_map", "symmetry_score", "MobaSymmetryChecker",
    "make_arena", "height_field", "ground_bounce", "knockback_distance", "WuxiaRuleSet",
    "trajectory_to_heightfield", "compute_fow",
]
