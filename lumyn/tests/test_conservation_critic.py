"""`ConservationCritic` 的回归测试。

重点是把**通道消融的结论**固化成断言：
  - 默认通道不含 `com`（实测其单通道判别力 17%–24%，低于随机 50%）
  - 默认通道组合（angular+momentum）的判别力**优于**四通道等权
    （曾经的四通道等权在强生成器上使选中误差退化 20%–59%）
这样以后若有人「顺手」把 com/energy 加回默认集合，测试会失败。
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lumyn.eval.conservation_critic import (   # noqa: E402
    ALL_CHANNELS, DEFAULT_CHANNELS, ConservationCritic, conservation_channels,
)

N, DT, STEPS, SOFT, G = 6, 0.01, 40, 0.05, 1.0
MASS = 0.3


def _prior_ic(e, rng):
    ang = rng.uniform(0, 2 * np.pi, N)
    r = (1 - e ** 2) / (1 + e * np.cos(ang))
    pos = np.stack([r * np.cos(ang), r * np.sin(ang), rng.normal(0, 0.05, N)], axis=1)
    v = np.sqrt(1.0 / np.maximum(r, 1e-6))
    vel = np.stack([-v * np.sin(ang), v * np.cos(ang), rng.normal(0, 0.02, N)], axis=1)
    return pos, vel


def _accel(pos, mass):
    d = pos[None, :, :] - pos[:, None, :]
    r2 = (d * d).sum(-1) + SOFT ** 2
    inv = r2 ** -1.5
    np.fill_diagonal(inv, 0.0)
    return G * (mass[None, :, None] * d * inv[..., None]).sum(1)


def _rollout(pos, vel, mass):
    """辛欧拉（kick-drift），返回 (pos_traj, vel_traj)，两者都是 (T,N,3)。"""
    p, v = pos.copy(), vel.copy()
    ps, vs = [p.copy()], [v.copy()]
    for _ in range(STEPS):
        v = v + _accel(p, mass) * DT
        p = p + v * DT
        ps.append(p.copy()); vs.append(v.copy())
    return np.stack(ps), np.stack(vs)


def _noisy_rollout(pos, vel, mass, sigma, rng):
    """每步注入过程噪声 —— 模拟「生成模型给出的、略偏离真实动力学」的轨迹。

    位置与速度**同时**加噪：早期实现只给位置加噪、再用 np.gradient 反推速度，
    得到的残差几乎全是有限差分伪影。
    """
    p, v = pos.copy(), vel.copy()
    ps, vs = [p.copy()], [v.copy()]
    for _ in range(STEPS):
        a = _accel(p, mass)
        v = v + a * DT + rng.normal(0, sigma, v.shape)
        p = p + v * DT + rng.normal(0, sigma, p.shape)
        ps.append(p.copy()); vs.append(v.copy())
    return np.stack(ps), np.stack(vs)


class TestConservationChannels(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.mass = np.ones(N) * MASS
        pos, vel = _prior_ic(0.3, self.rng)
        self.pos_t, self.vel_t = _rollout(pos, vel, self.mass)
        self.gp, self.gv = _noisy_rollout(pos, vel, self.mass, 0.15, self.rng)

    def test_returns_all_four_channels(self):
        ch = conservation_channels(self.pos_t, self.vel_t, self.mass)
        self.assertEqual(set(ch), set(ALL_CHANNELS))
        for k, v in ch.items():
            self.assertTrue(np.isfinite(v), f"{k} 不是有限值：{v}")
            self.assertGreaterEqual(v, 0.0, f"{k} 漂移为负：{v}")

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            conservation_channels(self.pos_t[0], self.vel_t, self.mass)
        with self.assertRaises(ValueError):
            conservation_channels(self.pos_t, self.vel_t[:-1], self.mass)

    def test_mass_shape_validation(self):
        with self.assertRaises(ValueError):
            conservation_channels(self.pos_t, self.vel_t, np.ones(N + 1))

    def test_physical_trajectory_has_lower_drift(self):
        """物理轨迹的漂移必须低于加噪轨迹 —— 这是 critic 有意义的最低要求。"""
        ch_phys = conservation_channels(self.pos_t, self.vel_t, self.mass)
        ch_gen = conservation_channels(self.gp, self.gv, self.mass)
        for c in ("angular", "momentum"):
            self.assertLess(ch_phys[c], ch_gen[c],
                            f"{c} 通道未能区分物理与生成轨迹")

    def test_permutation_invariance(self):
        """物理量应关于粒子置换不变。"""
        perm = np.array([2, 0, 5, 1, 4, 3])
        a = conservation_channels(self.pos_t, self.vel_t, self.mass)
        b = conservation_channels(self.pos_t[:, perm], self.vel_t[:, perm],
                                  self.mass[perm])
        for c in ALL_CHANNELS:
            self.assertAlmostEqual(a[c], b[c], places=10,
                                   msg=f"{c} 通道对粒子置换不不变")

    def test_zero_velocity_is_finite(self):
        """静止轨迹不能让尺度项变成 0 而导致 inf/NaN。"""
        zeros = np.zeros_like(self.vel_t)
        ch = conservation_channels(self.pos_t, zeros, self.mass)
        for k, v in ch.items():
            self.assertTrue(np.isfinite(v), f"{k} 在零速度下非有限：{v}")


class TestConservationCritic(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.mass = np.ones(N) * MASS
        pos, vel = _prior_ic(0.3, self.rng)
        self.phys = _rollout(pos, vel, self.mass)
        self.gen = _noisy_rollout(pos, vel, self.mass, 0.15, self.rng)
        self.critic = ConservationCritic(mass=self.mass, G=G, softening=SOFT)

    def test_default_channels_exclude_com(self):
        """`com` 实测单通道判别力低于随机，不得出现在默认集合中。"""
        self.assertNotIn("com", DEFAULT_CHANNELS)
        self.assertEqual(tuple(DEFAULT_CHANNELS), ("angular", "momentum"))
        self.assertEqual(self.critic.channels, ("angular", "momentum"))

    def test_default_channels_exclude_energy(self):
        """`energy` 单通道判别力仅 56%–60%，不应默认启用。"""
        self.assertNotIn("energy", DEFAULT_CHANNELS)

    def test_unknown_channel_rejected(self):
        with self.assertRaises(ValueError):
            ConservationCritic(channels=("angular", "nope"))

    def test_empty_channels_rejected(self):
        with self.assertRaises(ValueError):
            ConservationCritic(channels=())

    def test_score_direction_is_higher_is_better(self):
        self.assertGreater(self.critic.score(*self.phys), self.critic.score(*self.gen))
        self.assertTrue(self.critic.prefers(*self.phys, *self.gen))

    def test_residual_is_negated_score(self):
        self.assertAlmostEqual(self.critic.residual(*self.phys),
                               -self.critic.score(*self.phys), places=12)

    def test_breakdown_only_reports_used_channels(self):
        bd = self.critic.breakdown(*self.phys)
        self.assertEqual(set(bd), set(self.critic.channels))

    def test_weights_scale_channels(self):
        c2 = ConservationCritic(mass=self.mass, softening=SOFT,
                                channels=("angular",), weights={"angular": 3.0})
        c1 = ConservationCritic(mass=self.mass, softening=SOFT, channels=("angular",))
        self.assertAlmostEqual(c2.residual(*self.phys), 3.0 * c1.residual(*self.phys),
                               places=12)

    def test_rank_and_best(self):
        cands = [self.gen, self.phys, self.gen]
        ranked = self.critic.rank(cands)
        self.assertEqual(ranked[0][0], 1, "最物理的候选排名不正确")
        idx, chosen = self.critic.best(cands)
        self.assertEqual(idx, 1)
        self.assertTrue(np.allclose(chosen[0], self.phys[0]))

    def test_best_on_empty_raises(self):
        with self.assertRaises(ValueError):
            self.critic.best([])

    def test_default_channels_beat_four_channel_sum(self):
        """**编码通道消融结论的回归测试。**

        在「区分物理轨迹与多个不同噪声强度的生成轨迹」上，默认的
        angular+momentum 组合的正确排序数**不得少于**四通道等权（com+energy 在内）。

        历史：四通道等权曾在强生成器上使选中误差相对 oracle 从 1.02x 退化到 1.21x
        （OOD 上退化到 1.61x），原因是 com 通道的单通道判别力低于随机。
        """
        rng = np.random.default_rng(7)
        pos, vel = _prior_ic(0.3, rng)
        true = _rollout(pos, vel, self.mass)
        cands = [_noisy_rollout(pos, vel, self.mass, s, rng)
                 for s in (0.0, 0.05, 0.1, 0.2, 0.35, 0.6)]
        # 噪声单调递增 → 真实排序就是 0,1,2,...
        truth = list(range(len(cands)))

        default = ConservationCritic(mass=self.mass, softening=SOFT)
        four = ConservationCritic(mass=self.mass, softening=SOFT, channels=ALL_CHANNELS)

        def correctness(critic):
            order = [i for i, _ in critic.rank(cands)]
            # 与真实排序的 Spearman 秩相关
            n = len(truth)
            rx = np.argsort(np.argsort(order))
            ry = np.argsort(np.argsort(truth))
            rx = rx - rx.mean(); ry = ry - ry.mean()
            return float((rx * ry).sum() /
                         (np.sqrt((rx ** 2).sum() * (ry ** 2).sum()) + 1e-30))

        self.assertGreaterEqual(
            correctness(default), correctness(four),
            "默认通道（angular+momentum）的排序质量低于四通道等权；"
            "请勿把 com/energy 加回默认集合而不重新验证")

    def test_best_candidate_of_true_ic_is_the_true_trajectory(self):
        """当候选中包含真实轨迹（sigma=0）时，critic 应选中它。"""
        rng = np.random.default_rng(11)
        pos, vel = _prior_ic(0.3, rng)
        true = _rollout(pos, vel, self.mass)
        cands = [true] + [_noisy_rollout(pos, vel, self.mass, s, rng)
                          for s in (0.05, 0.15, 0.4)]
        idx, _ = self.critic.best(cands)
        self.assertEqual(idx, 0, "critic 未能在候选中选中真实轨迹")


if __name__ == "__main__":
    unittest.main()
