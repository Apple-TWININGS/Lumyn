"""守恒约束损失族——把"物理不出错"编码成可微损失。

每个损失项对应一条物理约束，优化器最小化损失即推动系统趋向守恒：

  - conservation_loss：能量 / 动量 / 角动量 / 质心 守恒
  - trajectory_loss   ：轨迹与目标形态接近（用于形状引导生成）
  - symmetry_loss     ：镜像 / 旋转对称性
  - smoothness_loss   ：轨迹随时间平滑（抑制数值抖动）
  - target_loss       ：终点落在目标半径 / 目标速度

用法：
    sys = DifferentiableNBody(pos, vel, mass, steps=60)
    losses = PhysicsLosses(sys, targets={"radius": 2.0})
    out = sys.forward()
    total = losses(out).backward()   # 反向传播到 pos/vel/mass
"""
from __future__ import annotations
from typing import Optional, Dict, Any


# ============================================================
# 守恒量计算
# ============================================================
def total_energy(pos, vel, mass, G: float = 1.0, softening: float = 1e-3) -> Any:
    """E = Σ ½ m v² − Σ_{i<j} G m_i m_j / r

    注意 r 的写法：softening **必须加在 sqrt 内部**。
    此前写成 `sqrt(Σd²).clamp(min=softening)`，对角线元素 (i==j) 处 Σd²=0，
    而 sqrt 在 0 点的导数是无穷大；即使随后乘 eye=0 消掉该项，
    反向传播时仍会出现 0 × inf = NaN，导致**损失值有限但梯度全为 NaN**。
    这里改为 sqrt(Σd² + softening²)，与 _accel / _accel in differentiable.py
    使用完全一致的软化约定：对角线 r = softening（非零），梯度有限。
    """
    import torch

    ke = 0.5 * (mass * (vel * vel).sum(dim=-1)).sum()
    N = pos.shape[0]
    d = pos.unsqueeze(0) - pos.unsqueeze(1)
    r = ((d * d).sum(dim=-1) + softening ** 2).sqrt()
    eye = 1.0 - torch.eye(N, dtype=r.dtype, device=r.device)
    pe = -G * (mass.unsqueeze(0) * mass.unsqueeze(-1) / r * eye).sum() / 2.0
    return ke + pe


def total_momentum(vel, mass) -> Any:
    return (mass.unsqueeze(-1) * vel).sum(dim=0)  # (3,)


def _cross(a, b):
    """批量叉积 (N,3) x (N,3) -> (N,3)。"""
    import torch
    return torch.cross(a, b, dim=-1)


def angular_momentum(pos, vel, mass) -> Any:
    """L = Σ m (r × v)。"""
    return (mass.unsqueeze(-1) * _cross(pos, vel)).sum(dim=0)


def center_of_mass(pos, mass) -> Any:
    return (mass.unsqueeze(-1) * pos).sum(dim=0) / mass.sum()


# ============================================================
# 漂移上界监控
# ============================================================
class ConservationBounds:
    """运行时监控守恒量漂移是否超出阈值。"""

    def __init__(self, energy_tol: float = 1e-3, momentum_tol: float = 1e-3,
                 angular_tol: float = 1e-3, com_tol: float = 1e-3):
        self.tols = {"energy": energy_tol, "momentum": momentum_tol,
                     "angular": angular_tol, "com": com_tol}
        self.history: Dict[str, list] = {k: [] for k in self.tols}

    def update(self, E0, L0, P0, com0, E1, L1, P1, com1) -> Dict[str, Any]:
        de = float(abs(E1 - E0) / (abs(E0) + 1e-12))
        dP = float(P1.sub(P0).norm())
        dL = float(L1.sub(L0).norm())
        dcom = float(com1.sub(com0).norm())
        self.history["energy"].append(de)
        self.history["momentum"].append(dP)
        self.history["angular"].append(dL)
        self.history["com"].append(dcom)
        return {
            "drift": {"energy": de, "momentum": dP, "angular": dL, "com": dcom},
            "within": {
                "energy": de < self.tols["energy"],
                "momentum": dP < self.tols["momentum"],
                "angular": dL < self.tols["angular"],
                "com": dcom < self.tols["com"],
            },
        }

    def report(self) -> Dict[str, float]:
        """返回**首次** update 建立的基线漂移。

        语义说明（测试 test_conservation_bounds 依赖此语义）：
        report() 报告的是"基准漂移"，即在监控起始时刻测得的漂移量；
        超界事件由 update() 返回值里的 `within` 单独标记，不污染基线。
        需要本次运行的最大漂移请用 max_drift()。
        """
        return {k: (v[0] if v else 0.0) for k, v in self.history.items()}

    def max_drift(self) -> Dict[str, float]:
        """本次运行中各项守恒量的最大漂移。"""
        return {k: (max(v) if v else 0.0) for k, v in self.history.items()}


# ============================================================
# 损失集合
# ============================================================
class PhysicsLosses:
    """把守恒约束 + 目标约束打包成一个总损失。"""

    def __init__(self, system, targets: Optional[Dict[str, Any]] = None,
                 w_energy: float = 1.0, w_momentum: float = 1.0,
                 w_angular: float = 1.0, w_com: float = 0.5,
                 w_target: float = 1.0, w_smooth: float = 0.1,
                 w_symmetry: float = 0.0, G: float = 1.0, softening: float = 1e-3):
        self.sys = system
        self.targets = targets or {}
        self.weights = dict(energy=w_energy, momentum=w_momentum, angular=w_angular,
                            com=w_com, target=w_target, smooth=w_smooth, symmetry=w_symmetry)
        self.G = G
        self.softening = softening

    # ------------------------------------------------------------
    def __call__(self, out) -> Any:
        """计算总损失。返回标量。"""
        import torch
        traj = out["trajectory"]
        mass = self.sys.mass
        T = traj.shape[0]

        # 能量守恒：(E_T - E_0)^2
        E0 = total_energy(traj[0], (traj[1] - traj[0]) / self.sys.dt, mass, self.G, self.softening)
        ET = total_energy(traj[-1], out["final_vel"], mass, self.G, self.softening)
        loss_E = (ET - E0) ** 2

        # 动量守恒：|P_T - P_0|^2
        v0 = (traj[1] - traj[0]) / self.sys.dt
        P0 = total_momentum(v0, mass)
        PT = total_momentum(out["final_vel"], mass)
        loss_P = (PT - P0).pow(2).sum()

        # 角动量守恒
        L0 = angular_momentum(traj[0], v0, mass)
        LT = angular_momentum(traj[-1], out["final_vel"], mass)
        loss_L = (LT - L0).pow(2).sum()

        # 质心守恒
        com0 = center_of_mass(traj[0], mass)
        comT = center_of_mass(traj[-1], mass)
        loss_com = (comT - com0).pow(2).sum()

        total = (self.weights["energy"] * loss_E +
                 self.weights["momentum"] * loss_P +
                 self.weights["angular"] * loss_L +
                 self.weights["com"] * loss_com)

        # 目标约束（形状引导）
        if "radius" in self.targets:
            r = traj[-1].norm(dim=-1).mean()
            total = total + self.weights["target"] * (r - self.targets["radius"]) ** 2

        # 平滑度
        if self.weights["smooth"] > 0:
            diff = (traj[2:] - 2 * traj[1:-1] + traj[:-2]).pow(2).sum()
            total = total + self.weights["smooth"] * diff

        return total


# ============================================================
# 便捷函数
# ============================================================
def trajectory_loss(traj_pred, traj_target) -> Any:
    """轨迹级 MSE（用于 shape guidance）。"""
    import torch
    return (traj_pred - traj_target).pow(2).mean()


def symmetry_loss(pos, planes: list = None) -> Any:
    """镜像对称损失：若系统应关于某平面对称，则镜像后应重合。"""
    import torch
    if planes is None:
        planes = [(0,)]  # 默认 x=0 平面
    loss = 0.0
    for axes in planes:
        mirror = pos.clone()
        for ax in axes:
            mirror = mirror.clone()
            mirror[:, ax] = -mirror[:, ax]
        loss = loss + (pos - mirror).pow(2).sum()
    return loss


def smoothness_loss(traj) -> Any:
    """二阶差分平滑度。"""
    return (traj[2:] - 2 * traj[1:-1] + traj[:-2]).pow(2).sum()


def target_loss(final_pos, target: dict) -> Any:
    """终点落在目标半径 / 目标速度。"""
    loss = 0.0
    if "radius" in target:
        r = final_pos.norm(dim=-1).mean()
        loss = loss + (r - target["radius"]) ** 2
    if "velocity" in target:
        v = final_pos  # 若传入的是速度
        loss = loss + (v.norm(dim=-1).mean() - target["velocity"]) ** 2
    return loss
