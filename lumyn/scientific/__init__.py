"""科研/教学专用模块：导出、渲染、抽帧、教学对比、诊断、可微 N 体。"""
from .export import ScientificExporter
from .visualization import ScientificVisualizer
from .frames import FrameExtractor
from .teaching import TeachingMode
from .metrics import PhysicsMetrics
from . import presets

# 可微版：优先 PyTorch（differentiable.py），不可用时降级到 NumPy 数值梯度兜底
try:
    from .differentiable import DifferentiableNBody, ConservationConstraint
    HAS_TORCH = True
except Exception:
    from .differentiable_numpy import (
        DifferentiableNBodyNumPy as DifferentiableNBody,
        ConservationConstraintNumPy as ConservationConstraint,
    )
    HAS_TORCH = False

from .differentiable_numpy import (
    DifferentiableNBodyNumPy,
    ConservationConstraintNumPy,
    optimize_initial_conditions as optimize_initial_conditions_numpy,
)

__all__ = [
    "ScientificExporter",
    "ScientificVisualizer",
    "FrameExtractor",
    "TeachingMode",
    "PhysicsMetrics",
    "presets",
    "DifferentiableNBody",
    "ConservationConstraint",
    "DifferentiableNBodyNumPy",
    "ConservationConstraintNumPy",
    "optimize_initial_conditions_numpy",
    "HAS_TORCH",
]
