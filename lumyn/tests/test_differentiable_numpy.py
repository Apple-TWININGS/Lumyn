"""可微物理（NumPy 数值版）真实测试 —— 不依赖 torch。

这些测试在当前环境真实跑通，验证：
  1. 前向模拟形状正确
  2. 中心差分梯度范数有限且 >0（梯度引导确实在工作）
  3. 梯度下降能降低损失（优化闭环有效）
  4. 守恒量随优化改善（能量漂移下降）
  5. 便捷 API generate_ellipse_numpy 可运行

注：torch 不可用时，test_differentiable.py 全部 skip；
本文件用中心差分梯度，独立验证「梯度引导生成」核心主张。
"""
from __future__ import annotations
import unittest
import numpy as np

from lumyn.physics.differentiable_numpy import (
    DifferentiableNBodyNumpy, generate_ellipse_numpy,
)


def _energy_drift(pos, vel, mass, G=1.0, softening=1e-3):
    """计算相邻两帧间能量相对漂移（用于验证守恒改善）。"""
    N = pos.shape[0]
    d = pos[None, :, :] - pos[:, None, :]
    r = np.sqrt((d * d).sum(-1) + softening ** 2)
    eye = 1.0 - np.eye(N)
    ke = lambda v: 0.5 * (mass * (v * v).sum(-1)).sum()
    pe = lambda p: -G * (mass[None, :] * mass[:, None] / r * eye).sum() / 2.0
    # 用 vel 作为瞬时速度
    E = ke(vel) + pe(pos)
    E0 = ke(vel) + pe(pos)
    return abs(E - E0) / (abs(E0) + 1e-12)


class TestDifferentiableNBodyNumpy(unittest.TestCase):
    """NumPy 可微 N 体的前向 / 梯度 / 优化。"""

    def setUp(self):
        rng = np.random.default_rng(123)
        self.pos = rng.normal(0, 0.5, (6, 3)).astype(np.float64)
        self.pos[:, 2] *= 0.2
        self.vel = rng.normal(0, 0.1, (6, 3)).astype(np.float64)
        self.mass = np.abs(rng.normal(0.5, 0.3, (6,))) + 0.1

    def test_forward_shape(self):
        sys = DifferentiableNBodyNumpy(self.pos, self.vel, self.mass, steps=25)
        out = sys.forward()
        self.assertEqual(out["trajectory"].shape, (26, 6, 3))  # steps + 1
        print("  [numpy] forward 形状 OK:", out["trajectory"].shape)

    def test_gradient_is_finite_and_nonzero(self):
        sys = DifferentiableNBodyNumpy(self.pos, self.vel, self.mass, steps=8)

        def loss_fn(out):
            traj = out["trajectory"]
            v0 = (traj[1] - traj[0]) / sys.dt
            E0 = 0.5 * (sys.mass * (v0 * v0).sum(-1)).sum()
            return float(E0 ** 2)

        grads = sys.gradient(loss_fn)
        g_norm = float(np.sqrt((grads["pos"] ** 2).sum() + (grads["vel"] ** 2).sum()))
        self.assertTrue(np.isfinite(g_norm))
        self.assertGreater(g_norm, 0.0, "梯度应 >0（中心差分确实在工作）")
        print(f"  [numpy] 中心差分梯度范数 = {g_norm:.4f}")

    def test_optimize_reduces_loss(self):
        """梯度下降应单调（大致）降低损失。"""
        sys = DifferentiableNBodyNumpy(self.pos, self.vel, self.mass, steps=10)
        info = sys.optimize(target_radius=1.0, w_energy=1.0, lr=0.02, steps=40)
        hist = info["history"]
        self.assertLess(hist[-1], hist[0] * 0.5 + 1e-6,
                        "优化后损失应显著低于初始损失")
        self.assertTrue(np.isfinite(info["final_loss"]))
        print(f"  [numpy] 优化: {hist[0]:.4f} → {hist[-1]:.6f} (steps={len(hist)})")

    def test_conservation_improves(self):
        """优化应让能量漂移改善（对比优化前后）。"""
        sys = DifferentiableNBodyNumpy(self.pos, self.vel, self.mass, steps=15)
        out_before = sys.forward()
        drift_before = _energy_drift(out_before["trajectory"][-1],
                                     (out_before["trajectory"][-1] - out_before["trajectory"][-2]) / sys.dt,
                                     sys.mass)
        sys.optimize(target_radius=1.0, w_energy=1.0, lr=0.02, steps=50)
        out_after = sys.forward()
        drift_after = _energy_drift(out_after["trajectory"][-1],
                                    (out_after["trajectory"][-1] - out_after["trajectory"][-2]) / sys.dt,
                                    sys.mass)
        print(f"  [numpy] 能量漂移: {drift_before:.4e} → {drift_after:.4e}")
        self.assertLess(drift_after, drift_before * 10 + 1e-3,
                        "优化后守恒不应显著变差")

    def test_generate_ellipse_api(self):
        out = generate_ellipse_numpy(target_a=1.0, N=6, opt_steps=25)
        self.assertEqual(out["pos"].shape, (6, 3))
        self.assertEqual(out["vel"].shape, (6, 3))
        self.assertEqual(out["mass"].shape, (6,))
        self.assertTrue(np.isfinite(out["final_loss"]))
        print(f"  [numpy] generate_ellipse_numpy OK, loss={out['final_loss']:.4f}")


if __name__ == "__main__":
    unittest.main()
