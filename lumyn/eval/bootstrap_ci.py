"""bootstrap 95% 置信区间（纯 Python，跨机器可复现，seed=42）。

移植自论文代码 stats.py，去掉对 src.* 的绝对导入。
不依赖 numpy：统一用 random.Random + 顺序统计量，任何环境结果一致。
"""
from __future__ import annotations
import random as _random

__all__ = ["bootstrap_ci", "summarize_accuracy"]


def _sample(arr, rng: _random.Random, n: int):
    return [arr[rng.randrange(n)] for _ in range(n)]


def _mean(arr):
    return (sum(arr) / len(arr)) if arr else float("nan")


def bootstrap_ci(scores, n_iter: int = 10000, ci: float = 0.95,
                  seed: int = 42, statistic: str = "mean"):
    """一维样本 bootstrap CI。返回 (lo, hi, mean)。"""
    arr = list(scores)
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = _random.Random(seed)
    fn = _mean if statistic == "mean" else _median
    boot = sorted(fn(_sample(arr, rng, n)) for _ in range(n_iter))
    lo_idx = int((1 - ci) / 2 * n_iter)
    hi_idx = int((1 + ci) / 2 * n_iter) - 1
    lo_idx = max(0, min(lo_idx, n_iter - 1))
    hi_idx = max(lo_idx, min(hi_idx, n_iter - 1))
    return float(boot[lo_idx]), float(boot[hi_idx]), float(fn(arr))


def _median(arr):
    if not arr:
        return float("nan")
    s = sorted(arr)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def summarize_accuracy(correct_flags, n_iter: int = 10000, seed: int = 42):
    """0/1 数组 → '0.695 [0.66, 0.73]' + 原始 CI。"""
    flags = list(correct_flags)
    if not flags:
        return "N/A", (float("nan"), float("nan"), float("nan"))
    lo, hi, mean = bootstrap_ci(flags, n_iter=n_iter, seed=seed)
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]", (lo, hi, mean)
