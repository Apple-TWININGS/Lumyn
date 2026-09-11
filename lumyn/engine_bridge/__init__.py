"""engine_bridge：对接游戏引擎（UE5/Unity）+ 帧同步协议。"""
from .protocol.deterministic import Fixed16, FixedVec3, sorted_interaction_list, make_frame_packet

__all__ = ["Fixed16", "FixedVec3", "sorted_interaction_list", "make_frame_packet"]
