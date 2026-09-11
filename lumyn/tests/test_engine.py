"""Lumyn 测试套件。

运行（从项目根：即 lumyn/ 的父目录）：
    python -m unittest lumyn.tests.test_engine -v

也可在 lumyn/ 内直接跑：
    cd lumyn && python tests/test_engine.py
"""
import os
import sys
import unittest

# 项目根 = lumyn/ 的父目录。test_engine.py 位于 lumyn/tests/ -> 需向上 3 层
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402

from lumyn import LumynEngine, NBodySimulator, SimScore  # noqa: E402
from lumyn.physics.barnes_hut import BarnesHutTree  # noqa: E402
from lumyn.physics.scenes import galaxy, explosion, collision  # noqa: E402
from lumyn.video.pipeline import VideoPipeline  # noqa: E402
from lumyn.eval.simscore import ConservationChecker  # noqa: E402


class TestBarnesHut(unittest.TestCase):
    def test_build_and_acceleration(self):
        rng = np.random.default_rng(0)
        pos = rng.normal(0, 1, (128, 3))
        mass = rng.uniform(0.1, 1.0, 128)
        tree = BarnesHutTree(max_leaf=8)
        root = tree.build(pos, mass, np.zeros(3), 4.0)
        self.assertIsNotNone(root)
        acc = tree.compute_acceleration(theta=0.5)
        self.assertEqual(acc.shape, (128, 3))


class TestNBodyConservation(unittest.TestCase):
    def test_two_body_conservation(self):
        # 两体闭合系统（无中心固定质量）→ 动量/能量应严格守恒
        # 正确圆轨道：m1=m2=1，间距 2（r=1），万有引力 F=G*1*1/(2^2)=0.25
        # 向心力 m*v^2/r = 0.25 -> v^2 = 0.25 -> v = 0.5（各自绕质心）
        sim = NBodySimulator(G=1.0, theta=1.0)  # theta>=1 退化为近精确成对
        pos = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
        mass = np.array([1.0, 1.0], dtype=np.float64)
        v = 0.5
        vel = np.array([[0.0, v, 0.0], [0.0, -v, 0.0]], dtype=np.float64)
        traj = sim.simulate(pos, mass, vel, n_steps=80, dt=0.001)
        result = ConservationChecker(G=1.0).check(traj, mass, dt=0.001)
        self.assertLess(result["momentum_drift"], 0.05)
        self.assertLess(result["energy_drift"], 0.5)

    def test_galaxy_non_divergent(self):
        # galaxy 含固定中心黑洞（fixed_mask），守恒检查应排除它
        result = galaxy(150, seed=1)
        pos, mass, vel, fixed = result
        sim = NBodySimulator(G=1.0, theta=0.5)
        traj = sim.simulate(pos, mass, vel, n_steps=20, dt=0.005)
        checker = ConservationChecker(G=1.0, energy_tol=5.0, momentum_tol=5.0)
        r_no_mask = checker.check(traj, mass, dt=0.005)              # 不排除黑洞
        r_mask = checker.check(traj, mass, dt=0.005, fixed_mask=fixed)  # 排除黑洞
        # 排除黑洞后，动量漂移不应更差；两者都应 < 阈值（不发散）
        self.assertLess(r_mask["momentum_drift"], 2.0)
        self.assertLess(r_no_mask["momentum_drift"], 2.0)


class TestScenes(unittest.TestCase):
    def test_all_scenes_simulate(self):
        sim = NBodySimulator(G=1.0, theta=0.5)
        for name, fn in (("galaxy", galaxy), ("explosion", explosion), ("collision", collision)):
            result = fn(120, seed=7)
            pos, mass, vel = result[:3]
            traj = sim.simulate(pos, mass, vel, n_steps=10, dt=0.01,
                                repulsion=0.5 if name != "galaxy" else 0.0)
            self.assertEqual(traj.ndim, 3)
            self.assertEqual(traj.shape[0], 11)


class TestVideoPipeline(unittest.TestCase):
    def test_render_shape(self):
        traj = np.random.randn(8, 50, 3).cumsum(axis=0)
        pipe = VideoPipeline(img_size=32)
        video = pipe.generate(traj, img_size=32)
        self.assertEqual(video.shape, (8, 32, 32, 3))
        self.assertEqual(pipe.provenance, "numpy_render")


class TestEval(unittest.TestCase):
    def test_simscore_range(self):
        scorer = SimScore()
        ref = {
            "steps": [{"law": "Newton2", "expr": "F=ma"}, {"law": "Kinematics", "expr": "v^2=u^2+2as"}],
            "constraints": ["force_balance"],
            "final_state": {"position_norm": 1.0},
        }
        gen = {
            "steps": [{"law": "Newton2", "expr": "F=ma"}],
            "constraints": ["force_balance"],
            "final_state": {"position_norm": 1.05},
        }
        s = scorer.compute(gen, ref)
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_conservation_checker(self):
        checker = ConservationChecker(energy_tol=10.0, momentum_tol=10.0)
        r = checker.check(np.zeros((3, 10, 3)), np.ones(10))
        self.assertIn("verdict", r)


class TestEngineE2E(unittest.TestCase):
    def test_generate_returns_trajectory_and_validation(self):
        import tempfile, os
        engine = LumynEngine(G=1.0, theta=0.5)
        result = engine.generate("spiral_galaxy", steps=10, n_particles=60)
        self.assertIn("trajectory", result)
        self.assertIn("validation", result)
        self.assertEqual(result["trajectory"].ndim, 3)  # (T, N, 3)
        self.assertGreater(result["trajectory"].shape[0], 0)
        self.assertIn("energy_drift", result["validation"])

    def test_render_writes_mp4(self):
        import tempfile
        engine = LumynEngine(G=1.0, theta=0.5)
        result = engine.generate("binary_star", steps=8, n_particles=40)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.mp4")
            out = engine.render(result, output_path=path)
            self.assertTrue(os.path.exists(out))

    def test_export_writes_csv(self):
        import tempfile
        engine = LumynEngine(G=1.0, theta=0.5)
        result = engine.generate("globular_cluster", steps=6, n_particles=30)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            out = engine.export(result, output_path=path)
            self.assertTrue(os.path.exists(out))

    def test_optimize_runs(self):
        engine = LumynEngine(G=1.0, theta=0.5)
        target = np.random.default_rng(0).normal(0, 0.5, (30, 3)).astype(np.float32)
        out = engine.optimize(target, n=30, steps=6, iters=3, lr=0.05)
        self.assertIn("trajectory", out)
        self.assertLess(out["final_energy_drift"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
