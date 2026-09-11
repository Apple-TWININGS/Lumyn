"""答案判定：数值相对误差 / MCQ 字母 / 科学计数法 / 分数解析。

移植自未发表论文代码 grading.py，适配 Lumyn 包结构（相对导入 + 不依赖 src.*）。
"""
from __future__ import annotations
import re
from typing import Tuple, Optional


_UNIT = r"[A-Za-zμ°%]*"
_EXP = r"(?:\s*\^\s*[-+]?\d+)?"
_PAT = re.compile(
    r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"\s*" + _UNIT + r"(?:/[A-Za-zμ°%]+)?" + _EXP
)


def extract_number(text: str) -> Optional[float]:
    """从文本中抽取最后一个数值（支持 10^3、m/s^2、分数）。"""
    if text is None:
        return None
    # 分数
    m = re.search(r"(-?\d+)\s*/\s*(\d+)", str(text))
    if m:
        return float(m.group(1)) / float(m.group(2))
    # 纯科学计数：如 "10^3" → 1000（先处理，避免被带单位分支吞掉指数）
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*\^\s*(-?\d+)", str(text))
    if m:
        try:
            return float(m.group(1)) ** float(m.group(2))
        except ValueError:
            return None
    # 科学计数 / 带单位（如 "3.0 m/s^2"）
    m = _PAT.search(str(text))
    if m:
        s = m.group(0).strip()
        # 去掉单位/指数部分，但保留数字本身（含小数与负号）
        s = re.sub(r"[A-Za-zμ°%].*", "", s).rstrip().rstrip("^")
        try:
            return float(s)
        except ValueError:
            return None
    # 纯数字
    m = re.search(r"-?\d+\.\d+|-?\d+", str(text))
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def judge(pred: str, ref: str, tol: float = 0.05,
          abs_tol: float = 0.01) -> Tuple[bool, Optional[float]]:
    """判定 pred 是否匹配 ref。

    返回 (correct, pred_val)。规则：
      - 均为 MCQ 单字母 → 严格相等；
      - 否则按数值相对误差（近零值用 abs_tol）。
    """
    pred_v = extract_number(pred)
    ref_v = extract_number(ref)

    # MCQ：两边都像单字母选项
    if _is_choice(pred) and _is_choice(ref):
        return (pred.strip().upper()[0] == ref.strip().upper()[0]), pred_v

    if ref_v is None or pred_v is None:
        return False, pred_v

    if abs(ref_v) < abs_tol:
        correct = abs(pred_v - ref_v) < abs_tol
    else:
        correct = abs(pred_v - ref_v) / abs(ref_v) <= tol
    return bool(correct), pred_v


def _is_choice(s: str) -> bool:
    return bool(re.match(r"^\s*[A-Da-d]\s*[.)]?\s*$", str(s)))


def self_test() -> bool:
    cases = [
        ("答案是 3.14", "3.14", True),
        ("= 1/2", "0.5", True),
        ("速度 10 m/s^2", "10", True),
        ("结果 100", "106", False),   # 6% > 5%
        ("A", "A", True),
        ("B", "A", False),
    ]
    for pred, ref, expected in cases:
        ok, _ = judge(pred, ref)
        if ok != expected:
            return False
    return True
