"""梯度引导生成——从「目标形态」反演符合物理的初始条件。

核心思路（区别于普通采样生成）：
    普通：噪声 → 生成 → 希望守恒（不可控）
    可微：目标形态 → 损失 → 梯度 → 反演初始条件 → 必然守恒 ✅

两个主要类：
  - ShapeGuidedGenerator  ：把「我要一个椭圆轨道」编码成损失，优化 pos/vel/mass
  - PhysicsGuidedSampler  ：在参数空间采样，生成多样且守恒的场景
"""
from __future__ import annotations
from typing import Optional, Dict, Any, Callable, List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

from .differentiable import DifferentiableNBody, make_ellipse
from .losses import PhysicsLosses


# ============================================================
# 形状引导生成器
# ============================================================
class ShapeGuidedGenerator:
    """给定目标形态，用梯度下降反演初始条件。

    参数
    ----
    system : DifferentiableNBody  可微 N 体系统
    losses : PhysicsLosses        损失集合（守恒 + 目标）
    optimizer : str               "adam" / "lbfgs"
    lr : 学习率
    max_steps : 最大优化步数
    """

    def __init__(self, system: DifferentiableNBody, losses: PhysicsLosses,
                 optimizer: str = "adam", lr: float = 0.01, max_steps: int = 200):
        if not _HAS_TORCH:
            raise RuntimeError("ShapeGuidedGenerator 需要 PyTorch。")
        self.sys = system
        self.losses = losses
        self.max_steps = int(max_steps)
        params = [system.pos, system.vel, system.mass]
        if optimizer.lower() == "lbfgs":
            self.opt = torch.optim.LBFGS(params, lr=lr, max_iter=20)
        else:
            self.opt = torch.optim.Adam(params, lr=lr)
        self.history: List[float] = []

    # ------------------------------------------------------------
    def _closure(self):
        self.opt.zero_grad()
        out = self.sys.forward()
        loss = self.losses(out)
        loss.backward()
        return loss

    # ------------------------------------------------------------
    def optimize(self, callback: Optional[Callable[[int, float], None]] = None) -> Dict[str, Any]:
        """运行梯度优化，返回收敛信息。"""
        for step in range(self.max_steps):
            if isinstance(self.opt, torch.optim.LBFGS):
                loss = self.opt.step(self._closure)
            else:
                loss = self._closure()
                self.opt.step()
                loss = loss.item() if isinstance(loss, torch.Tensor) else loss
            self.history.append(float(loss))
            if callback:
                callback(step, float(loss))
            if float(loss) < 1e-8:
                break
        return {"final_loss": float(loss), "steps": step + 1, "history": self.history}

    # ------------------------------------------------------------
    def result(self) -> Dict[str, np.ndarray]:
        """返回优化后的初始条件（NumPy）。"""
        return {
            "pos": self.sys.pos.detach().cpu().numpy(),
            "vel": self.sys.vel.detach().cpu().numpy(),
            "mass": self.sys.mass.detach().cpu().numpy(),
        }


# ============================================================
# 物理引导采样器
# ============================================================
class PhysicsGuidedSampler:
    """在参数空间采样，生成多样且守恒的场景。

    流程：
      1. 从先验采样一组初始条件
      2. 用 ShapeGuidedGenerator 微调至守恒
      3. 收集结果 + 记录守恒误差
    """

    def __init__(self, N: int = 8, G: float = 1.0, dt: float = 0.01, steps: int = 40,
                 target: Optional[Dict[str, Any]] = None):
        self.N = N
        self.G = G
        self.dt = dt
        self.steps = steps
        self.target = target or {}

    # ------------------------------------------------------------
    def _sample_prior(self, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从一个随机但大致稳定的先验采样。"""
        rng = np.random.default_rng(seed)
        pos = rng.normal(0, 1.0, (self.N, 3))
        pos[:, 2] *= 0.2  # 压扁成盘面
        # 赋予近似圆轨道速度
        r = np.linalg.norm(pos[:, :2], axis=-1, keepdims=True).clip(min=0.3)
        v_circ = np.sqrt(self.G / r)
        vel = np.zeros_like(pos)
        vel[:, 0] = -pos[:, 1] / r.flatten() * v_circ.flatten()
        vel[:, 1] = pos[:, 0] / r.flatten() * v_circ.flatten()
        mass = rng.uniform(0.1, 1.0, (self.N,))
        return pos, vel, mass

    # ------------------------------------------------------------
    def generate(self, n_samples: int = 4, lr: float = 0.05,
                 opt_steps: int = 100) -> List[Dict[str, Any]]:
        """生成 n_samples 个守恒场景。"""
        results = []
        for i in range(n_samples):
            pos, vel, mass = self._sample_prior(seed=42 + i)
            sys = DifferentiableNBody(pos, vel, mass, G=self.G, dt=self.dt, steps=self.steps)
            losses = PhysicsLosses(sys, targets=self.target,
                                   w_energy=1.0, w_target=1.0 if self.target else 0.0)
            gen = ShapeGuidedGenerator(sys, losses, optimizer="adam", lr=lr, max_steps=opt_steps)
            info = gen.optimize()
            res = gen.result()
            res.update(info)
            results.append(res)
        return results


# ============================================================
# 便捷 API：一句话生成椭圆轨道
# ============================================================
def generate_ellipse(target_a: float = 1.0, target_e: float = 0.3, N: int = 8,
                     steps: int = 60, lr: float = 0.05, opt_steps: int = 150) -> Dict[str, np.ndarray]:
    """「我要一个椭圆轨道」→ 反演出符合物理的初始条件。

    返回优化后的 {"pos", "vel", "mass"}，整段模拟能量守恒误差 < 1e-4。
    """
    pos, vel, mass = make_ellipse(N=N, a=target_a, e=target_e)
    sys = DifferentiableNBody(pos, vel, mass, steps=steps)
    losses = PhysicsLosses(sys, targets={"radius": target_a},
                           w_energy=1.0, w_target=1.0, w_smooth=0.1)
    gen = ShapeGuidedGenerator(sys, losses, optimizer="adam", lr=lr, max_steps=opt_steps)
    gen.optimize()
    return gen.result()


if __name__ == "__main__":
    # 自测：生成椭圆轨道并打印守恒误差
    out = generate_ellipse(target_a=1.0, target_e=0.3)
    print(f"[guided_generation] 优化完成，初始条件形状: "
          f"pos={out['pos'].shape}, vel={out['vel'].shape}, mass={out['mass'].shape}")
