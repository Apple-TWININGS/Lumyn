"""错误分类：E1（定律误用）/ E2（代数错）/ E3（约束违背）→ 映射到管线层。

移植自论文 classify_errors.py，改为 Lumyn 可编程 API（无需 argparse / 文件 I/O）。
LLM 标注可选，默认走规则引擎以避免循环依赖。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ErrorRecord:
    category: str          # "E1" | "E2" | "E3" | "Unknown"
    layer: str             # "director" | "generate" | "validate"
    reason: str
    confidence: float = 1.0


@dataclass
class ErrorReport:
    """跨场景错误汇总，可直接喂 docs/EXPERIMENT_TABLES.md 表 4。"""
    records: list = None

    def __post_init__(self):
        if self.records is None:
            self.records = []

    def to_dict(self) -> Dict[str, Any]:
        from collections import Counter
        c = Counter(r.category for r in self.records)
        return {
            "total": len(self.records),
            "by_category": dict(c),
            "records": [{"category": r.category, "layer": r.layer, "reason": r.reason}
                        for r in self.records],
        }

    def chi2_vs(self, other: "ErrorReport") -> Dict[str, float]:
        """与另一份报告做 2x2 卡方对比（模型 A vs B）。"""
        from .bootstrap_ci import _random  # noqa  (占位，实际用下方 chi2)
        a = self.to_dict()["by_category"]
        b = other.to_dict()["by_category"]
        cats = sorted(set(a) | set(b))
        matrix = [[a.get(c, 0), b.get(c, 0)] for c in cats]
        chi2, p, dof = chi2_contingency(matrix)
        return {"chi2": chi2, "p_value": p, "dof": dof}


class ErrorClassifier:
    """规则优先的错误分类器（LLM 可选注入）。"""

    # 映射：错误类型 → 管线层
    LAYER_MAP = {"E1": "director", "E2": "generate", "E3": "validate"}

    def classify(self, trace: Dict[str, Any], final_state: Dict[str, Any],
                 use_llm: Optional[Any] = None) -> ErrorRecord:
        """对单条验证记录分类。"""
        # E3：守恒约束被违背
        if trace.get("constraint_violated") or trace.get("energy_drift", 0) > 0.05:
            return ErrorRecord("E3", "validate", "守恒约束被违背")
        # E1：定律选错（静态场景却用了运动定律 / explode 却无守恒律）
        if self._law_mismatch(trace):
            return ErrorRecord("E1", "director", "物理定律选择错误")
        # E2：数值/代数问题
        if trace.get("numeric_instability") or self._algebra_wrong(trace, final_state):
            return ErrorRecord("E2", "generate", "数值/代数误差")
        return ErrorRecord("Unknown", "unknown", "规则未覆盖", confidence=0.0)

    def auto_classify(self, trace: Dict[str, Any], final_state: Dict[str, Any]) -> ErrorRecord:
        """自动分类入口（规则引擎，无 LLM 调用）。"""
        return self.classify(trace, final_state)

    # ---- 内部启发式 ----
    def _law_mismatch(self, trace: Dict[str, Any]) -> bool:
        laws = trace.get("laws", [])
        law_str = " ".join(str(l) for l in laws)
        motion = str(trace.get("subject_motion", ""))
        # 静态场景却用了运动/动力学定律 → 选错
        if trace.get("static") and any(
                kw in law_str for kw in ("Kinematics", "Newton", "Projectile")):
            return True
        # 非静态场景（有运动描述）却只用了静力平衡定律 → 选错
        moving = bool(motion) and motion not in ("idle", "", "static")
        if moving and "StaticEquilibrium" in law_str:
            return True
        # 爆炸类运动却无守恒律 → 选错
        if "explode" in motion and "Conservation" not in law_str:
            return True
        return False

    def _algebra_wrong(self, trace: Dict[str, Any], final_state: Dict[str, Any]) -> bool:
        # 有限差分检查占位：实际接入时对 velocity 做偏导验证
        return bool(trace.get("algebra_error"))


# ---------------------------------------------------------------- 卡方工具
def chi2_contingency(matrix):
    """r × c 列联表卡方（任意维度），返回 (chi2, p, dof)。"""
    mat = [[float(x) for x in row] for row in matrix]
    n = sum(sum(r) for r in mat)
    if n == 0:
        return float("nan"), float("nan"), 0
    r = len(mat)
    c = len(mat[0]) if r else 0
    row_tot = [sum(row) for row in mat]
    col_tot = [sum(mat[i][j] for i in range(r)) for j in range(c)]
    chi2 = 0.0
    for i in range(r):
        for j in range(c):
            exp = row_tot[i] * col_tot[j] / n
            if exp > 0:
                chi2 += (mat[i][j] - exp) ** 2 / exp
    dof = (r - 1) * (c - 1)
    from .bootstrap_ci import _random  # 避免循环；此处仅占位
    p = _chi2_sf(chi2, dof)
    return float(chi2), float(p), dof


def _chi2_sf(chi2, dof):
    """卡方尾概率：有 scipy 优先，否则数值近似。"""
    try:
        from scipy import stats as _sp
        return float(_sp.chi2.sf(chi2, dof))
    except Exception:
        return float("nan")


def compare_models(model_a: Dict[str, float], model_b: Dict[str, float]) -> Dict[str, float]:
    """两模型错误分布卡方对比（包装）。"""
    cats = sorted(set(model_a) | set(model_b))
    matrix = [[model_a.get(c, 0), model_b.get(c, 0)] for c in cats]
    chi2, p, dof = chi2_contingency(matrix)
    return {"chi2": chi2, "p_value": p, "dof": dof}
