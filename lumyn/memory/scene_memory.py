"""场景记忆（陈深 MemoryStore 思想）：保持同一世界的物理规则一致。

- 每个场景注册一组物理规则 (G, theta, 守恒约束)
- 生成新场景前先查记忆：同一 world_id 复用规则，避免"厨房里牛顿定律突然变了"
"""
from __future__ import annotations

from typing import Dict, Optional


class SceneMemory:
    def __init__(self, decay_days=30):
        self._store: Dict[str, dict] = {}
        self.decay_days = decay_days

    def register(self, world_id: str, laws: list, constraints: list, params: dict) -> None:
        self._store[world_id] = {
            "laws": laws,
            "constraints": constraints,
            "params": params,
            "last_seen": 0,  # 轮次计数，超 decay_days 自动降权
        }

    def get(self, world_id: str) -> Optional[dict]:
        entry = self._store.get(world_id)
        if entry is None:
            return None
        # 时间衰减：超过 decay_days 降权（此处仅标记，供上层决策）
        if entry["last_seen"] > self.decay_days:
            entry["decayed"] = True
        return entry

    def touch(self, world_id: str) -> None:
        if world_id in self._store:
            self._store[world_id]["last_seen"] = 0

    def step(self) -> None:
        """每轮调用：老化所有条目。"""
        for entry in self._store.values():
            entry["last_seen"] = entry.get("last_seen", 0) + 1

    def validate_consistency(self, world_id: str, new_constraints: list) -> bool:
        """检查新场景的约束是否与已注册规则一致。"""
        entry = self.get(world_id)
        if entry is None:
            return True
        return set(new_constraints).issubset(set(entry["constraints"]))
