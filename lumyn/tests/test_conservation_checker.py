"""`ConservationChecker` / `PhysicsMetrics.diagnose` / 场景与模拟器的回归测试。

**这些测试锁定的是 2026-09 修复的三个已确认缺陷。**

缺陷 1：`ConservationChecker.angular_drift` 名实不符
    它算的是**线动量模长**的相对波动 `max|‖P(t)‖-‖P(0)‖|/‖P(0)‖`，
    与角动量 L = Σ m (r × v) 无关。实测某星系场景上报告 12.0176，
    而真实角动量相对变化仅 0.0311。

缺陷 2：`NBodySimulator` 从不执行 `fixed_mask`
    `galaxy()` 返回掩码并声称中心体「不参与动力学」，但模拟器没有该参数，
    于是质量占系统约一半的中心体被自由积分。

缺陷 3：`galaxy` 场景本身发散，而测试名断言它不发散
    r_max 从 0.997 涨到 550。根因有二：圆轨道速度用了错误的中心质量
    （写成 10，实际 1000），以及内圈轨道分辨率不足（每个轨道仅 1.3 步）。

缺陷 4：`PhysicsMetrics.diagnose` 末帧速度被置零
    `vel = (traj[min(t+1, T-1)] - traj[t]) / dt` 在 t=T-1 时给出 0，
    使 `angular_momentum_drift` 恒等于 1.0、`energy_final` 的动能被清零。
"""
from __future__ import annotations
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lumyn import NBodySimulator                      # noqa: E402
from lumyn.eval import ConservationChecker            # noqa: E402
from lumyn.physics.scenes import galaxy               # noqa: E402
from lumyn.scientific.metrics import PhysicsMetrics   # noqa: E402


def _two_body_circular(dt=0.001, n=60):
    """对称两体圆轨道：净动量恰为 0，是归一化脆弱性的最严苛用例。"""
    pos = np.array([[1.0, 0, 0], [-1.0, 0, 0]], dtype=np.float64)
    mass = np.array([1.0, 1.0])
    vel = np.array([[0, 0.5, 0], [0, -0.5, 0]], dtype=np.float64)
    sim = NBodySimulator(G=1.0, theta=1.0)
    return sim.simulate(pos, mass, vel, n_steps=n, dt=dt), mass, dt


class TestAngularDriftIsRealAngularMomentum(unittest.TestCase):
    def test_matches_manual_angular_momentum(self):
        """`angular_drift` 必须等于真实角动量的相对漂移。"""
        traj, mass, dt = _two_body_circular()
        got = ConservationChecker().check(traj, mass, dt=dt)["angular_drift"]

        v = (traj[1:] - traj[:-1]) / dt
        L = np.array([(mass[:, None] * np.cross(traj[t], v[t])).sum(0)
                      for t in range(len(v))])
        scale = np.linalg.norm(L, axis=1).mean()
        expect = np.linalg.norm(L[-1] - L[0]) / scale
        self.assertAlmostEqual(got, expect, places=8,
                               msg="angular_drift 与真实角动量漂移不符")

    def test_not_linear_momentum_magnitude(self):
        """确保它不再是「线动量模长」的波动（旧实现的错误定义）。

        构造一个**加速平移**的系统：所有粒子沿 x 整体平移且速率随时间变化，
        因此线动量模长明显变化；但 r 与 v 都沿 x，叉积恒为 0 → 角动量恒为 0。
        旧定义会给出很大的值，正确实现应恒为 0。
        """
        T, N, dt = 40, 3, 0.01
        base = np.array([0.0, 1.0, 2.0])
        pos = np.zeros((T, N, 3))
        for t in range(T):
            tau = t * dt
            pos[t, :, 0] = base + 3.0 * tau + 0.5 * 20.0 * tau ** 2   # 加速平移
        mass = np.ones(N)
        r = ConservationChecker().check(pos, mass, dt=dt)
        self.assertLess(r["angular_drift"], 1e-6,
                        "整体（加速）平移不应产生角动量漂移")

        # 而线动量模长确实在变化 —— 说明这两个量确实不同
        v_fd = (pos[1:] - pos[:-1]) / dt
        P = np.array([(mass[:, None] * v_fd[t]).sum(0) for t in range(T - 1)])
        self.assertGreater(np.linalg.norm(P[-1] - P[0]), 0.0,
                           "该用例的线动量应随时间变化")


class TestMomentumNormalizationRobust(unittest.TestCase):
    def test_symmetric_system_does_not_explode(self):
        """净动量为 0 的对称系统：除以 |P_0| 会除以噪声，须用固有尺度。"""
        traj, mass, dt = _two_body_circular()
        r = ConservationChecker().check(traj, mass, dt=dt)
        P0 = (mass[:, None] * (traj[1] - traj[0]) / dt).sum(0)
        self.assertLess(np.linalg.norm(P0), 1e-12, "该用例的净动量应为 0")
        self.assertLess(r["momentum_drift"], 1e-6,
                        "净动量为 0 时动量漂移不应爆炸")


class TestFixedMaskIsEnforced(unittest.TestCase):
    def setUp(self):
        self.pos, self.mass, self.vel, self.fixed = galaxy(80, seed=3)
        self.sim = NBodySimulator(G=1.0, theta=0.5)

    def test_fixed_particle_stays_put(self):
        traj = self.sim.simulate(self.pos, self.mass, self.vel, n_steps=15,
                                 dt=0.005, fixed_mask=self.fixed)
        pinned = traj[:, self.fixed, :][:, 0, :]
        self.assertLess(np.abs(pinned - pinned[0]).max(), 1e-12,
                        "被标记为固定的粒子发生了位移")

    def test_without_mask_the_particle_moves(self):
        """反向对照：不传掩码时该粒子会被自由积分（这正是原来的缺陷）。"""
        traj = self.sim.simulate(self.pos, self.mass, self.vel, n_steps=15,
                                 dt=0.005)   # 不传 fixed_mask
        free_body = traj[:, 0, :]
        self.assertGreater(np.abs(free_body - free_body[0]).max(), 0.0)

    def test_mask_does_not_disturb_unpinned_particles_much(self):
        """固定中心体应让盘面更稳定，而不是更乱。"""
        a = self.sim.simulate(self.pos, self.mass, self.vel, n_steps=15,
                              dt=0.005, fixed_mask=self.fixed)
        b = self.sim.simulate(self.pos, self.mass, self.vel, n_steps=15, dt=0.005)
        ra = np.linalg.norm(a[:, 1:, :] - a[:, :1, :], axis=-1)[-1].max()
        rb = np.linalg.norm(b[:, 1:, :] - b[:, :1, :], axis=-1)[-1].max()
        self.assertLess(ra, rb * 1.5)


class TestGalaxySceneIsActuallyStable(unittest.TestCase):
    """原测试名断言「non_divergent」，但场景此前确实发散。"""

    def setUp(self):
        self.pos, self.mass, self.vel, self.fixed = galaxy(150, seed=1)
        sim = NBodySimulator(G=1.0, theta=0.5)
        self.traj = sim.simulate(self.pos, self.mass, self.vel, n_steps=20,
                                 dt=0.005, fixed_mask=self.fixed)

    def test_radius_stays_bounded(self):
        r = np.linalg.norm(self.traj[:, 1:, :] - self.traj[:, :1, :], axis=-1)
        growth = r[-1].max() / r[0].max()
        self.assertLess(growth, 3.0,
                        f"场景发散：r_max 增长 {growth:.2f}x（修复前为 550x）")

    def test_inner_orbit_is_resolved(self):
        """内圈每个轨道至少要有 20 步，否则数值上必然失稳。"""
        r_in = np.linalg.norm(self.pos[1:, :2], axis=1).min()
        M = self.mass[0]
        period = 2 * np.pi * r_in ** 1.5 / np.sqrt(1.0 * M)
        steps_per_orbit = period / 0.005
        self.assertGreater(steps_per_orbit, 20.0,
                           f"内圈分辨率不足：{steps_per_orbit:.1f} 步/轨道")

    def test_momentum_drift_is_sane(self):
        r = ConservationChecker().check(self.traj, self.mass, dt=0.005,
                                        fixed_mask=self.fixed)
        self.assertLess(r["momentum_drift"], 2.0)
        self.assertLess(r["angular_drift"], 2.0)


class TestDiagnoseLastFrameVelocity(unittest.TestCase):
    def test_last_frame_has_nonzero_angular_momentum(self):
        """旧实现把末帧速度置零 → angular_momenta[-1] = 0 → drift 恒为 1.0。"""
        traj, mass, dt = _two_body_circular()
        d = PhysicsMetrics(G=1.0).diagnose(traj, mass, dt=dt)
        am = d["angular_momenta"]
        self.assertGreater(am[-1], 0.0, "末帧角动量为 0：速度被置零了")
        self.assertLess(d["angular_momentum_drift"], 1e-3,
                        "角动量相对漂移不应接近 1.0")

    def test_energy_final_is_consistent_with_initial(self):
        traj, mass, dt = _two_body_circular()
        d = PhysicsMetrics(G=1.0).diagnose(traj, mass, dt=dt)
        e0 = d["energies"][0]
        rel = abs(d["energy_final"] - e0) / (abs(e0) + 1e-12)
        self.assertLess(rel, 0.5,
                        "energy_final 与初始能量差异过大 —— 末帧动能疑被清零")


class TestMomentumVsCenterOfMass(unittest.TestCase):
    """`momentum_conserved` 曾名实不符：它比的是**质心**而不是动量。

    正确的关系是 ``d(com)/dt = P / Σm``，因此：
      · 动量守恒 ⟺ 质心做匀速直线运动（可以一直在动）
      · 质心静止 ⟹ P ≡ 0 ⟹ 动量守恒（反方向不可能有反例）
    二者只在第一个方向上可区分，而旧实现恰好在那个方向上出错。
    """

    def setUp(self):
        self.mt = PhysicsMetrics(G=1.0)
        self.T, self.N, self.dt = 40, 3, 0.01
        self.mass = np.ones(self.N)

    def _translation(self, speed=5.0):
        """匀速整体平移：动量严格守恒，但质心一直在移动。"""
        traj = np.zeros((self.T, self.N, 3))
        for t in range(self.T):
            traj[t, :, 0] = np.array([0.0, 1.0, 2.0]) + speed * t * self.dt
        return traj

    def test_uniform_translation_conserves_momentum(self):
        traj = self._translation()
        self.assertTrue(
            self.mt.momentum_conserved(traj, self.mass, dt=self.dt),
            "匀速整体平移的动量必须判为守恒")

    def test_old_center_of_mass_criterion_would_reject_it(self):
        """同一轨迹下，质心判据会判为「不守恒」—— 这正是旧实现的问题。"""
        traj = self._translation()
        self.assertFalse(
            self.mt.center_of_mass_conserved(traj, self.mass, tol=0.5),
            "质心确实移动了，故质心判据应为 False")
        self.assertLess(
            np.linalg.norm(traj[-1, 0] - traj[0, 0]), 3.0)   # 位移量级确认

    def test_momentum_violation_is_detected(self):
        """只有一个粒子加速 → 净动量随时间变化，必须判为不守恒。"""
        traj = np.zeros((self.T, self.N, 3))
        for t in range(self.T):
            traj[t, 0, 0] = 0.5 * 100.0 * (t * self.dt) ** 2   # 加速
        self.assertFalse(
            self.mt.momentum_conserved(traj, self.mass, dt=self.dt),
            "净动量在变化，必须判为不守恒")

    def test_symmetric_acceleration_is_actually_conserved(self):
        """对称反向加速的净动量恒为 0 → 判为守恒是正确的，不是缺陷。"""
        traj = np.zeros((self.T, self.N, 3))
        for t in range(self.T):
            tau = t * self.dt
            traj[t, 0, 0] = 0.5 * 100.0 * tau ** 2
            traj[t, 1, 0] = -0.5 * 100.0 * tau ** 2
        self.assertTrue(self.mt.momentum_conserved(traj, self.mass, dt=self.dt))
        self.assertTrue(self.mt.center_of_mass_conserved(traj, self.mass))

    def test_module_level_functions_agree(self):
        from lumyn.scientific.metrics import (momentum_conserved,
                                              center_of_mass_conserved)
        traj = self._translation()
        self.assertTrue(momentum_conserved(traj, self.mass, dt=self.dt))
        self.assertFalse(center_of_mass_conserved(traj, self.mass, tol=0.5))


if __name__ == "__main__":
    unittest.main()
