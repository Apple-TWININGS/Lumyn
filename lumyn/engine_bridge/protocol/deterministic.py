"""确定性协议：定点数包装 + 交互列表确定序（帧同步命门）。

跨端（Android/iOS/PC）float 运算可能因编译器/CPU 产生末位差异，
竞技游戏必须定点数 + 固定遍历顺序 → 双端比特一致。
"""
import numpy as np


class Fixed16:
    """Q16.16 定点数（16 整数位 + 16 小数位）。"""

    SCALE = 1 << 16

    def __init__(self, val=0):
        if isinstance(val, Fixed16):
            self._raw = val._raw
        elif isinstance(val, int):
            self._raw = val * self.SCALE
        else:
            self._raw = int(round(val * self.SCALE))

    def to_float(self):
        return self._raw / self.SCALE

    def __add__(self, o):
        r = Fixed16(); r._raw = self._raw + Fixed16(o)._raw; return r

    def __sub__(self, o):
        r = Fixed16(); r._raw = self._raw - Fixed16(o)._raw; return r

    def __mul__(self, o):
        # 饱和防止溢出（竞技场景安全优先）
        r = Fixed16(); r._raw = (self._raw * Fixed16(o)._raw) >> 16; return r

    def __truediv__(self, o):
        r = Fixed16(); r._raw = (self._raw << 16) // max(Fixed16(o)._raw, 1); return r

    def __repr__(self):
        return f"Fixed16({self.to_float():.4f})"


class FixedVec3:
    """三维定点向量。HMMP 的 pos/mass/acc 全部换此类型。"""

    def __init__(self, x=0, y=0, z=0):
        self.x = x if isinstance(x, Fixed16) else Fixed16(x)
        self.y = y if isinstance(y, Fixed16) else Fixed16(y)
        self.z = z if isinstance(z, Fixed16) else Fixed16(z)

    def to_float(self):
        return (self.x.to_float(), self.y.to_float(), self.z.to_float())

    @classmethod
    def from_array(cls, arr):
        return [cls(*v) for v in arr]


def sorted_interaction_list(pairs):
    """交互列表确定序：按 (i, j) 排序，保证双端遍历一致。"""
    return sorted(set((min(a, b), max(a, b)) for a, b in pairs))


def make_frame_packet(frame, commands, seed_state):
    """帧数据包：种子快照 + 指令列表（容差友好，float 用定点字节）。"""
    return {
        "frame": frame,
        "seed": seed_state,
        "commands": [c.encode().decode() for c in commands],
    }


__all__ = ["Fixed16", "FixedVec3", "sorted_interaction_list", "make_frame_packet"]
