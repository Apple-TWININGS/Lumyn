"""敏感性分析（PhysBench Method C2）：防"背答案"。

改变 prompt/初始条件中的物理量（如质量、初速度），观察生成轨迹的最终状态
是否按物理规律变化。若输出几乎不变 → 模型在 memorization，返回低分。
"""
from __future__ import annotations

import numpy as np


def sensitivity_test(scene_fn, variable="mass", values=None, n_trials=4) -> dict:
    """对一个场景生成器做敏感性扫描。

    scene_fn(mass, ...) -> traj，我们改变 mass 看最终速度/位置如何变化。

    Returns
    -------
    {"std": float, "verdict": "physics_dependent" | "likely_memorization"}
    """
    if values is None:
        values = [1.0, 2.0, 5.0, 10.0]

    outputs = []
    for v in values:
        traj = scene_fn(**{variable: v})
        outputs.append(_extract_final(traj))

    outputs = np.array(outputs, dtype=np.float64)
    std = float(np.std(outputs))
    return {
        "std": std,
        "verdict": "physics_dependent" if std > 1e-6 else "likely_memorization",
    }


def _extract_final(traj) -> float:
    """从轨迹里取一个标量最终状态（位置范数代理）。"""
    return float(np.linalg.norm(traj[-1].mean(axis=0)))
