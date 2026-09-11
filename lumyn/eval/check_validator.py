"""验证层自检：null 率统计 + 坏 ref 检测 + 样例人工核对提示。

移植自论文 check_grading.py，改为直接消费 validation detail 列表（不依赖文件 I/O）。
"""
from __future__ import annotations
from typing import List, Dict, Any


def check_validation(detail: List[Dict[str, Any]]) -> Dict[str, Any]:
    """扫描验证明细，返回质量报告。

    detail 每项: {id, pred_raw, ref, pred_val, ...}
    """
    if not detail:
        return {"n": 0, "null_count": 0, "null_rate": 0.0, "bad_refs": []}

    n = len(detail)
    null_count = sum(1 for d in detail if d.get("pred_val") is None)
    bad_refs = [d["id"] for d in detail if d.get("pred_val") is None
                and not _looks_like_unit_error(d.get("ref", ""))]

    return {
        "n": n,
        "null_count": null_count,
        "null_rate": null_count / n,
        "bad_refs": bad_refs,
        "warning": null_count / n > 0.1,  # >10% null 告警
    }


def _looks_like_unit_error(ref: str) -> bool:
    """ref 本身是单位/符号（如 "m/s^2"）时，pred 为 null 不算坏 ref。"""
    import re
    return bool(re.match(r"^[\s\^A-Za-zμ°%/*]+$", str(ref)))


def sample_for_manual_check(detail: List[Dict[str, Any]], k: int = 5,
                            seed: int = 42) -> List[Dict[str, Any]]:
    """抽取 k 条供人工核对（固定 seed 可复现）。"""
    rng = __import__("random").Random(seed)
    idx = list(range(len(detail)))
    rng.shuffle(idx)
    return [detail[i] for i in idx[:k]]
