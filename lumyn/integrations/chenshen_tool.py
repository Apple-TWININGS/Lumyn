"""陈深集成：把 LumynEngine 注册为一个 Tool。

用法：
    from chenshen_tool import make_lumyn_tool
    tool = make_lumyn_tool()
    result = tool.run(prompt="一个星系围绕黑洞旋转", world_id="game_alpha")
"""
from __future__ import annotations

from typing import Optional

try:
    from ..core.engine import LumynEngine
except ImportError:
    LumynEngine = None  # 允许独立运行


def make_lumyn_tool(api_key=None, **engine_kwargs):
    """工厂：返回一个符合陈深 ToolRegistry 接口的 Tool。"""
    if LumynEngine is None:
        raise ImportError("Lumyn 未安装，无法创建 Tool")

    engine = LumynEngine(api_key=api_key, **engine_kwargs)

    class LumynTool:
        name = "lumyn_physics_scene"
        description = "生成物理正确的游戏场景/视频（省算力，Barnes-Hut + 守恒律验证）"

        def run(self, prompt: str, world_id: str = "default", **kwargs) -> dict:
            return engine.generate(prompt, world_id=world_id, **kwargs)

        @property
        def engine(self) -> LumynEngine:
            return engine

    return LumynTool()


# 兼容直接调用
def generate_scene(prompt: str, world_id: str = "default", **kwargs) -> dict:
    tool = make_lumyn_tool()
    return tool.run(prompt, world_id=world_id, **kwargs)
