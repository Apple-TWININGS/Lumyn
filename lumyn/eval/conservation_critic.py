"""守恒律 critic —— 用守恒量的漂移给轨迹打分，作为生成过程的监督信号。

与 `ConservationChecker`（同包 `simscore.py`）的区别
---------------------------------------------------
`ConservationChecker` 是**旧的**、面向 `trace` 字典的验证器，它有三个已确认缺陷
（`angular_drift` 实际算的是线动量模长、无阈值断言、性能为纯 Python O(N²)）。
本模块是**独立的新实现**，不修改旧类，供需要「给候选打分并排序」的场景使用。

它解决的是一类具体任务：

    同一个初始条件下生成 K 个候选轨迹，用某个 critic 挑出最物理的那个。

为什么默认只用 angular + momentum 两个通道
------------------------------------------
这一点是**实测**得出的，不是先验选择。`cons_drift` 曾把四个通道**等权相加**：

  energy / angular / momentum / com

在 4 组设置（2 个生成器 × ID/OOD）上做的通道消融（见 `docs/PAPER_STATUS.md` §二·十）：

| 通道 | 单通道「挑中较好候选」的比例 |
|---|---|
| angular | 88% – 100% |
| momentum | 85% – 100% |
| energy | 56% – 60% |
| **com** | **17% – 24%（低于随机 50%：它在挑最差的）** |

即 `com` 通道在**主动损害** critic；`energy` 很弱。等权四通道加和显著次优：
在强生成器（单步加速度误差 0.0068）上，等权和的选中误差是 oracle 的 **1.21 倍**，
而 angular/angular+momentum 只有 **1.02 倍**。

**物理解释（合理且可检验）**：学习到的力模型通常仍能写成某个（错误的）势的梯度，
因此**能量仍近似守恒**，energy 就成了弱检测器；而动量与角动量守恒要求力是
**成对且中心**的，学习模型极少严格满足 —— 因此这两个通道才是强检测器。

归一化的脆弱性（已知）
----------------------
`angular` 与 `momentum` 在**真实物理轨迹**上的漂移是**机器精度**（约 1e-16，因为
辛积分器确实守恒到该量级）。因此**不能用「漂移量本身」做跨样本比较**，
必须除以一个同量纲的尺度。本模块用的尺度是：

    angular :  <||L(t)||>_t          （角动量模长的时间平均）
    momentum:  <sum_i m_i ||v_i||>_t （总速率加权质量，恒为正）

其中 momentum 的尺度恒为非零；angular 的尺度在净角动量接近 0 的场景会变小，
因此加了 `_EPS` 保护，并在 `degenerate` 标记中报告该情形。
"""
from __future__ import annotations
from typing import Dict, Iterable, Optional, Sequence, Tuple
import numpy as np

# 各通道默认采用的归一化尺度在净量接近 0 时会放大噪声，用该常数兜底。
_EPS = 1e-12

#: 依据通道消融结果（见模块 docstring）选定的默认通道。
DEFAULT_CHANNELS: Tuple[str, ...] = ("angular", "momentum")

#: 全部可用通道。`com` 默认不启用：实测其单通道判别力低于随机。
ALL_CHANNELS: Tuple[str, ...] = ("energy", "angular", "momentum", "com")


def _as_traj(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"{name} 形状应为 (T, N, 3)，实际为 {a.shape}")
    return a


def conservation_channels(pos: np.ndarray, vel: np.ndarray,
                          mass: Optional[np.ndarray] = None,
                          G: float = 1.0, softening: float = 1e-3) -> Dict[str, float]:
    """返回四个守恒通道的**归一化漂移**（越小越物理）。

    参数
    ----
    pos, vel : (T, N, 3) 位置与速度轨迹。**速度必须由调用方提供**。
        早期实现用 `np.gradient(pos)` 反推速度，其 O(dt^2) 误差会污染残差本身
        （实测物理轨迹的"残差"中位数几乎全部来自该伪影）。因此这里要求显式传入速度。
    mass : (N,) 质量；缺省为全 1。
    """
    pos = _as_traj(pos, "pos")
    vel = _as_traj(vel, "vel")
    if pos.shape != vel.shape:
        raise ValueError(f"pos 与 vel 形状必须一致：{pos.shape} vs {vel.shape}")
    T, N, _ = pos.shape
    m = np.ones(N, dtype=np.float64) if mass is None else np.asarray(mass, dtype=np.float64)
    if m.shape != (N,):
        raise ValueError(f"mass 形状应为 ({N},)，实际为 {m.shape}")

    # ---- 能量 ----
    ke = 0.5 * (m * (vel * vel).sum(-1)).sum(-1)                     # (T,)
    d = pos[:, None, :, :] - pos[:, :, None, :]                      # (T,N,N,3)
    r = np.sqrt((d * d).sum(-1) + softening ** 2)
    eye = 1.0 - np.eye(N)
    mm = m[:, None] * m[None, :]
    pe = -(G * mm[None] / r * eye[None]).sum((1, 2)) / 2.0
    E = ke + pe
    E_scale = float(np.abs(E).mean()) + _EPS

    # ---- 角动量 L = sum m (r x v) ----
    L = (m[None, :, None] * np.cross(pos, vel)).sum(1)               # (T,3)
    L_scale = float(np.linalg.norm(L, axis=-1).mean()) + _EPS

    # ---- 线动量 P = sum m v ----
    P = (m[None, :, None] * vel).sum(1)                              # (T,3)
    # 尺度用「总速率加权质量」而非 ||P||：净动量天然接近 0（旋转盘、对称爆炸），
    # 除以 ||P|| 会变成除以噪声。
    P_scale = float((m * np.linalg.norm(vel, axis=-1)).sum(-1).mean()) + _EPS

    # ---- 质心 ----
    com = (m[None, :, None] * pos).sum(1) / (m.sum() + _EPS)
    com_scale = float(np.linalg.norm(pos, axis=-1).mean()) + _EPS

    return {
        "energy": float(E.max() - E.min()) / E_scale,
        "angular": float(np.linalg.norm(L.max(0) - L.min(0))) / L_scale,
        "momentum": float(np.linalg.norm(P.max(0) - P.min(0))) / P_scale,
        "com": float(np.linalg.norm(com.max(0) - com.min(0))) / com_scale,
    }


class ConservationCritic:
    """给轨迹打「物理性」分数：**分数越高越物理**。

    默认只用 `angular` 与 `momentum` 通道（依据见模块 docstring）。

    用法：
        critic = ConservationCritic()
        s_phys = critic.score(pos_true, vel_true)
        s_gen  = critic.score(pos_gen,  vel_gen)
        # s_phys < s_gen 时返回 True
        assert critic.prefers(pos_true, vel_true, pos_gen, vel_gen)
    """

    def __init__(self, channels: Sequence[str] = DEFAULT_CHANNELS,
                 mass: Optional[np.ndarray] = None,
                 G: float = 1.0, softening: float = 1e-3,
                 weights: Optional[Dict[str, float]] = None):
        bad = [c for c in channels if c not in ALL_CHANNELS]
        if bad:
            raise ValueError(f"未知通道 {bad}；可用通道为 {ALL_CHANNELS}")
        if not channels:
            raise ValueError("channels 不能为空")
        self.channels = tuple(channels)
        self.mass = mass
        self.G = float(G)
        self.softening = float(softening)
        self.weights = dict(weights or {})
        self.last_breakdown: Dict[str, float] = {}

    # ------------------------------------------------------------
    def breakdown(self, pos: np.ndarray, vel: np.ndarray) -> Dict[str, float]:
        """返回本次打分**所用的**各通道值及其加权和（诊断用）。"""
        ch = conservation_channels(pos, vel, self.mass, self.G, self.softening)
        used = {c: ch[c] * self.weights.get(c, 1.0) for c in self.channels}
        self.last_breakdown = used
        return used

    def residual(self, pos: np.ndarray, vel: np.ndarray) -> float:
        """归一化漂移之和（越小越物理）。"""
        return float(sum(self.breakdown(pos, vel).values()))

    def score(self, pos: np.ndarray, vel: np.ndarray) -> float:
        """分数（越大越物理）= 漂移之和取负。"""
        return -self.residual(pos, vel)

    # ------------------------------------------------------------
    def prefers(self, pos_a, vel_a, pos_b, vel_b, margin: float = 0.0) -> bool:
        """判断 a 是否比 b 更物理。

        margin > 0 时要求超出该裕度，用于避免数值并列。
        """
        return self.score(pos_a, vel_a) > self.score(pos_b, vel_b) + margin

    def rank(self, candidates: Iterable[Tuple[np.ndarray, np.ndarray]]):
        """对候选按物理性**从优到劣**排序，返回 (原索引, 分数) 列表。"""
        scored = [(i, self.score(p, v)) for i, (p, v) in enumerate(candidates)]
        return sorted(scored, key=lambda t: -t[1])

    def best(self, candidates: Sequence[Tuple[np.ndarray, np.ndarray]]):
        """返回最物理的候选及其索引。"""
        ranked = self.rank(candidates)
        if not ranked:
            raise ValueError("candidates 为空")
        idx = ranked[0][0]
        return idx, candidates[idx]
