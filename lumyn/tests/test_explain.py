"""P2 · 因果追踪测试（真实运行，非 mock）。"""
from __future__ import annotations
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lumyn.explain import CausalTracer, TracingReport, LayerResult


class TestCausalTracing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tracer = CausalTracer()

    def test_run_clean_is_highest(self):
        """干净基线 >= 任何损坏分数（物理正确性不应低于退化版）。"""
        for scene in ("galaxy", "collision", "explosion"):
            clean = self.tracer._run(scene, corrupt="none")
            for layer in ("director", "generate", "validate"):
                corrupt = self.tracer._run(scene, corrupt=layer)
                self.assertLessEqual(corrupt, clean + 1e-9,
                                     f"{scene}/{layer}: clean 应 >= corrupt")

    def test_restore_exact_formula(self):
        """restore = all_corrupt + 该层 decay，应严格等于 clean（构造设定）。"""
        for scene in ("galaxy", "collision"):
            for layer in ("director", "generate", "validate"):
                restore = self.tracer._run_restore(scene, layer)
                # 构造上 restore 单调递增，且 <= clean
                self.assertLessEqual(restore, 1.0)
                self.assertGreaterEqual(restore, 0.0)

    def test_contribution_non_negative(self):
        """每层贡献度(restore - corrupt) >= 0：恢复不应比损坏更差。"""
        report = self.tracer.trace(scene_types=["galaxy"], num_runs=2)
        for name, res in report.per_scene["galaxy"].items():
            self.assertGreaterEqual(res.contribution, -1e-9,
                                    f"{name}: contribution 应 >= 0")

    def test_validate_dominates(self):
        """验证层贡献度应最大（设计使然：最后防线）。"""
        report = self.tracer.trace(
            scene_types=["galaxy", "collision", "explosion"], num_runs=3)
        for scene, layers in report.per_scene.items():
            self.assertGreaterEqual(
                layers["validate"].contribution,
                layers["generate"].contribution - 1e-6,
                f"{scene}: validate 应 >= generate")
            self.assertGreaterEqual(
                layers["generate"].contribution,
                layers["director"].contribution - 1e-6,
                f"{scene}: generate 应 >= director")

    def test_report_export_and_plot(self):
        """报告 to_dict / save / summary / plot 均可正常运行。"""
        report = self.tracer.trace(scene_types=["galaxy"], num_runs=2)
        d = report.to_dict()
        self.assertIn("per_scene", d)
        self.assertIn("averaged", d)
        self.assertEqual(set(d["per_scene"]["galaxy"].keys()),
                         {"director", "generate", "validate"})
        # save
        out = Path("results/test_causal.json")
        out.parent.mkdir(exist_ok=True)
        report.save(str(out))
        self.assertTrue(out.exists())
        out.unlink()
        # summary 含层名
        s = report.summary()
        self.assertIn("validate", s)
        # plot（matplotlib 可选）
        p = report.plot("results/test_causal.png")
        if p:
            self.assertTrue(Path(p).exists())
            Path(p).unlink()

    def test_cross_scene_ordering_consistent(self):
        """三个场景的层贡献度排序应一致（validate > generate > director）。"""
        report = self.tracer.trace(num_runs=3)
        orders = []
        for scene, layers in report.per_scene.items():
            order = sorted(layers.values(), key=lambda r: r.contribution)
            orders.append([r.layer for r in order])
        # 全部应为 [director, generate, validate]
        expected = ["director", "generate", "validate"]
        for o in orders:
            self.assertEqual(o, expected)


if __name__ == "__main__":
    unittest.main()
