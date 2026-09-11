"""mobile：移动端省算力调度（LOD / 热控 / NPU 委托）。"""
from .lod_policy import select_keyframes, interpolate_lod, ThermalGuard, LODPolicy

__all__ = ["select_keyframes", "interpolate_lod", "ThermalGuard", "LODPolicy"]
