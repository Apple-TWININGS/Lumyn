"""physics 子包：可微 + 数值 N 体物理引擎。"""
from .barnes_hut import BarnesHutTree, Cell
from .nbody import NBodySimulator
from .scenes import HMMPSceneGenerator, SCENES, galaxy, explosion, collision

# 可微模块依赖 PyTorch：仅在 torch 可用时导出，否则保持 _DIFFERENTIABLE_AVAILABLE=False
# 这样上层（tests / scientific）可据此 skip，避免运行时才报错。
try:
    import torch  # noqa: F401
    _TORCH_OK = True
except ImportError:  # pragma: no cover
    _TORCH_OK = False

_DIFFERENTIABLE_AVAILABLE = False

if _TORCH_OK:
    try:
        from .differentiable import (
            DifferentiableNBody, EnergyConservingIntegrator, make_ellipse,
        )
        from .losses import (
            PhysicsLosses, ConservationBounds,
            total_energy, total_momentum, angular_momentum, center_of_mass,
            trajectory_loss, symmetry_loss, smoothness_loss, target_loss,
        )
        from .guided_generation import (
            ShapeGuidedGenerator, PhysicsGuidedSampler, generate_ellipse,
        )
        _DIFFERENTIABLE_AVAILABLE = True
    except Exception as e:  # pragma: no cover
        import warnings
        warnings.warn(f"可微模块导入失败（将降级为 NumPy 数值版）: {e}")


# ----------------------------------------------------------------------
# torch 不可用时：提供延迟报错占位，保持 API 表面一致。
# 任何调用都会得到明确的 "pip install torch" 提示，而非 NameError。
# ----------------------------------------------------------------------
if not _DIFFERENTIABLE_AVAILABLE:
    import types

    class _MissingTorch(AttributeError):
        def __init__(self, name):
            super().__init__(
                f"{name} 需要 PyTorch：pip install torch。"
                f"或使用 NumPy 数值版 physics.differentiable_numpy"
            )

    class _RequiresTorch:
        def __init__(self, name):
            self._name = name

        def __call__(self, *a, **kw):
            raise _MissingTorch(self._name)

        def __getattr__(self, item):
            raise _MissingTorch(self._name)

    for _name in ("DifferentiableNBody", "EnergyConservingIntegrator", "make_ellipse",
                  "PhysicsLosses", "ConservationBounds",
                  "ShapeGuidedGenerator", "PhysicsGuidedSampler", "generate_ellipse",
                  "total_energy", "total_momentum", "angular_momentum", "center_of_mass",
                  "trajectory_loss", "symmetry_loss", "smoothness_loss", "target_loss"):
        if _name not in globals():
            globals()[_name] = _RequiresTorch(_name)

# NumPy 数值梯度版（无需 torch，始终可用）——用于环境不可装 PyTorch 时验证梯度引导
from .differentiable_numpy import DifferentiableNBodyNumpy, generate_ellipse_numpy

__all__ = [
    "BarnesHutTree", "Cell", "NBodySimulator",
    "HMMPSceneGenerator", "SCENES", "galaxy", "explosion", "collision",
    "DifferentiableNBody", "EnergyConservingIntegrator", "make_ellipse",
    "PhysicsLosses", "ConservationBounds",
    "total_energy", "total_momentum", "angular_momentum", "center_of_mass",
    "trajectory_loss", "symmetry_loss", "smoothness_loss", "target_loss",
    "ShapeGuidedGenerator", "PhysicsGuidedSampler", "generate_ellipse",
    "DifferentiableNBodyNumpy", "generate_ellipse_numpy",
    "_DIFFERENTIABLE_AVAILABLE",
]

__all__ = [
    "BarnesHutTree", "Cell", "NBodySimulator",
    "HMMPSceneGenerator", "SCENES", "galaxy", "explosion", "collision",
    "DifferentiableNBody", "EnergyConservingIntegrator", "make_ellipse",
    "PhysicsLosses", "ConservationBounds",
    "total_energy", "total_momentum", "angular_momentum", "center_of_mass",
    "trajectory_loss", "symmetry_loss", "smoothness_loss", "target_loss",
    "ShapeGuidedGenerator", "PhysicsGuidedSampler", "generate_ellipse",
    "_DIFFERENTIABLE_AVAILABLE",
]
