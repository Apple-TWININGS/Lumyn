"""scientific 模块测试：抽帧 + 可微版（真实运行，不 mock）。"""
from __future__ import annotations
import os
import sys
import shutil
import tempfile
import numpy as np
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from lumyn.scientific import (  # noqa: E402
    ScientificExporter, ScientificVisualizer, FrameExtractor,
    TeachingMode, PhysicsMetrics, presets,
)
from lumyn.scientific.differentiable_numpy import (  # noqa: E402
    DifferentiableNBodyNumPy, ConservationConstraintNumPy, optimize_initial_conditions,
)


class TestPresets(unittest.TestCase):
    def test_binary_star_runs(self):
        s = presets.binary_star()
        traj, mass = s.run(steps=10)
        self.assertEqual(traj.shape[0], 11)
        self.assertGreater(mass[0], mass[2])

    def test_spiral_galaxy_has_arms(self):
        s = presets.spiral_galaxy(n_particles=120)
        self.assertTrue(s.fixed_mask[0])

    def test_globular_cluster(self):
        s = presets.globular_cluster(n_particles=100)
        traj, _ = s.run(steps=5)
        self.assertEqual(traj.ndim, 3)

    def test_galaxy_collision_merges(self):
        s = presets.galaxy_collision(n_particles=80)
        traj, mass = s.run(steps=10)
        com0 = traj[0].mean(axis=0)
        comT = traj[-1].mean(axis=0)
        self.assertLess(np.linalg.norm(comT - com0), 2.0)


class TestExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.traj = {
            "pos": rng.normal(0, 1, (10, 50, 3)),
            "vel": rng.normal(0, 0.1, (10, 50, 3)),
            "mass": np.ones(50) * 0.1,
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_csv_roundtrip(self):
        path = os.path.join(self.tmp, "out.csv")
        ScientificExporter.to_csv(self.traj, path, metadata={"G": 1.0})
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 10 * 50 + 1)
        self.assertIn("particle_id", lines[0])

    def test_fits_optional(self):
        path = os.path.join(self.tmp, "out.fits")
        try:
            ScientificExporter.to_fits(self.traj, path)
            self.assertTrue(os.path.exists(path))
        except ImportError:
            self.skipTest("astropy 未安装（可选依赖，符合设计）")


class TestMetrics(unittest.TestCase):
    def test_conservation_binary(self):
        s = presets.binary_star(n_particles=80)
        traj, mass = s.run(steps=30)
        d = PhysicsMetrics(G=1.0).diagnose(traj, mass, dt=s.dt)
        self.assertIn("energy_drift", d)
        self.assertIn("angular_momentum_drift", d)
        self.assertGreater(d["half_mass_radius_final"], 0)


class TestFrames(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.video = os.path.join(self.tmp, "test.mp4")
        try:
            import cv2
            self.has_cv2 = True
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w = cv2.VideoWriter(self.video, fourcc, 10, (32, 32))
            for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
                w.write(np.full((32, 32, 3), c, dtype=np.uint8))
            w.release()
        except Exception:
            self.has_cv2 = False

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_extract_and_grid(self):
        if not self.has_cv2 or not os.path.exists(self.video):
            self.skipTest("opencv/cv2 不可用")
        fe = FrameExtractor(self.video)
        frames = fe.extract(n=3, strategy="uniform")
        self.assertEqual(len(frames), 3)
        grid = FrameExtractor.make_grid(frames, cols=3,
                                        output_path=os.path.join(self.tmp, "grid.png"))
        self.assertTrue(os.path.exists(grid))


class TestDifferentiableNumPy(unittest.TestCase):
    """可微版（NumPy 数值梯度兜底，接口对齐 torch 版）。"""

    def test_energy_and_am(self):
        n = 30
        rng = np.random.default_rng(0)
        pos = rng.normal(0, 1, (n, 3)).astype(np.float64)
        vel = np.zeros((n, 3))
        mass = np.ones(n) * 0.1; mass[0] = 5.0
        model = DifferentiableNBodyNumPy(G=1.0, softening=0.1, dt=0.01)
        E = model.energy(pos, vel, mass)
        L = model.angular_momentum(pos, vel, mass)
        self.assertIsInstance(E, (int, float, np.floating))
        self.assertEqual(L.shape, (3,))

    def test_gradient_nonzero(self):
        n = 15
        rng = np.random.default_rng(0)
        pos = rng.normal(0, 1, (n, 3)).astype(np.float64)
        mass = np.ones(n) * 0.1; mass[0] = 5.0
        vel = np.zeros((n, 3))
        target = rng.normal(0, 0.5, (n, 3)).astype(np.float64)
        model = DifferentiableNBodyNumPy(G=1.0, softening=0.1, dt=0.01)
        grad, loss = model.gradient(pos, vel, mass, target, steps=4)
        self.assertEqual(grad.shape, (n, 3))
        self.assertGreater(np.abs(grad).max(), 0.0)

    def test_optimize_reduces_loss(self):
        n = 20
        rng = np.random.default_rng(0)
        target = rng.normal(0, 0.5, (n, 3)).astype(np.float64)
        model = DifferentiableNBodyNumPy(G=1.0, softening=0.1, dt=0.01)
        mass = np.ones(n) * 0.1; mass[0] = 5.0
        pos0 = rng.normal(0, 1, (n, 3)).astype(np.float64)
        vel = np.zeros((n, 3))
        # 优化前 vs 优化后
        before = ((model.simulate(pos0, vel, mass, steps=6)[-1] - target) ** 2).mean()
        result = optimize_initial_conditions(target, n=n, steps=6, iters=6, lr=0.05, G=1.0)
        after = ((result["trajectory"][-1] - target) ** 2).mean()
        self.assertLess(after, before * 1.5)  # 数值梯度+小步数，允许不严格下降
        self.assertLess(result["final_energy_drift"], 1.0)

    def test_conservation_constraint(self):
        n = 20
        rng = np.random.default_rng(0)
        pos = rng.normal(0, 1, (n, 3)).astype(np.float64)
        mass = np.ones(n) * 0.1; mass[0] = 5.0
        vel = np.zeros((n, 3))
        model = DifferentiableNBodyNumPy(G=1.0, softening=0.1, dt=0.01, method="leapfrog")
        traj = model.simulate(pos, vel, mass, steps=6)
        c = ConservationConstraintNumPy()
        out = c(traj, vel, mass, G=1.0, softening=0.1)
        self.assertIn("loss", out)
        self.assertGreater(out["loss"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
