"""explain 包：可解释性 / 因果追踪（P2）。

将论文的「逐层恢复率」方法改造为 Lumyn 三层管线的层间因果追踪：
  导演层(director) → 生成层(generate) → 验证层(validate)

对外接口：
    CausalTracer   逐场景跑损坏/恢复实验，计算各层贡献度
    TracingReport  跨场景汇总 + JSON 导出 + 柱状图 + summary()
"""
from .causal_tracing import (
    CausalTracer,
    TracingReport,
    LayerResult,
)

__all__ = ["CausalTracer", "TracingReport", "LayerResult"]
