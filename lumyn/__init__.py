"""Lumyn —— 省算力 · 物理可信的 AI 场景 / 视频生成引擎。

对外只暴露一个主入口 `LumynEngine`；其余符号（`NBodySimulator`、`SimScore`、
`BarnesHutTree` 等）按需从各自子模块延迟加载，**不在包级别重复导出**。

这样做的原因：`physics/` 与 `scientific/` 下存在同名或近似的双胞胎实现，
包级"花哨地全量导出"会让 `from lumyn import NBodySimulator` 拿到哪一个变得
依赖导入顺序，用户导入时容易踩坑。

推荐用法：

    from lumyn import LumynEngine
    engine = LumynEngine()
    result = engine.generate("galaxy", n_particles=500)

    # 需要更细的符号时，从子模块显式取，语义清晰：
    from lumyn.physics import NBodySimulator, BarnesHutTree
    from lumyn.eval import SimScore, ErrorClassifier

历史说明：本文件此前误存为一段中文待办笔记（非代码），
导致 `import lumyn` 抛 SyntaxError、整个测试套件无法收集。
"""
from __future__ import annotations

from ._version import __version__
from .core.engine import LumynEngine

# 兼容别名：README 的 API 示例使用 `HMMPGameEngine`，
# 而 `explain/causal_tracing.py` 内部也这样别名。
# 保留它以免破坏既有文档与调用方。
HMMPGameEngine = LumynEngine

# ---------------------------------------------------------------------------
# 延迟加载（PEP 562）：子模块符号按需导入，避免包级命名冲突与循环导入。
# ---------------------------------------------------------------------------
_LAZY = {
    # physics
    "NBodySimulator": ("lumyn.physics", "NBodySimulator"),
    "BarnesHutTree": ("lumyn.physics", "BarnesHutTree"),
    "HMMPSceneGenerator": ("lumyn.physics", "HMMPSceneGenerator"),
    "SCENES": ("lumyn.physics", "SCENES"),
    # eval
    "SimScore": ("lumyn.eval", "SimScore"),
    "ConservationChecker": ("lumyn.eval", "ConservationChecker"),
    "ErrorClassifier": ("lumyn.eval", "ErrorClassifier"),
    "evaluate_scene": ("lumyn.eval", "evaluate_scene"),
    # explain
    "CausalTracer": ("lumyn.explain", "CausalTracer"),
}

__all__ = ["LumynEngine", "HMMPGameEngine", "__version__", *sorted(_LAZY)]


def __getattr__(name: str):
    """按需从子模块解析延迟符号。"""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(
            f"module 'lumyn' has no attribute {name!r}。"
            f"可用的延迟符号：{sorted(_LAZY)}。"
            f"其他符号请从子模块显式导入，例如 `from lumyn.physics import ...`。"
        )
    import importlib

    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value  # 缓存，后续访问不再走 __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
