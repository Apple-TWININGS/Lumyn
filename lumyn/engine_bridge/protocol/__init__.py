"""protocol：帧同步协议与序列化。"""
from .deterministic import Fixed16, FixedVec3, sorted_interaction_list, make_frame_packet

__all__ = ["Fixed16", "FixedVec3", "sorted_interaction_list", "make_frame_packet"]
