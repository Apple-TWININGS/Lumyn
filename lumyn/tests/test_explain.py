"""因果追踪测试（真实运行，非 mock）。

**这些测试曾经断言的是被编造的行为。** 旧实现返回硬编码常数
（`decay = {"director": 0.12, "generate": 0.30, "validate": 0.55}`），
而旧测试断言：
  - `test_validate_dominates`：validate ≥ generate ≥ director
  - `test_cross_scene_ordering_consistent`：排序恒为 [director, generate, validate]
它们只能通过，因为实现返回的是常数。

现在断言的是**引擎的真实行为**：
  - 导演层不在 `LumynEngine.generate` 的调用链上 → 贡献度不可测（None）
  - 验证结果**不影响输出** → 在默认（忠实）模式下，验证层贡献度恒为 0
  - 只有显式启用验证门后，验证层才有非零贡献

另外加入两个针对"重新引入编造数字"的回归测试：
  - 结果必须随损坏强度变化（常数实现无法通过）
  - 干净基线与全坏基线必须可区分
"""
from __future__ import annotations
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lumyn.explain import CausalTracer, TracingReport, LayerResult

SCENES = ["binary_star"]           # 小而快；测试只关心行为不关心场景数
PARTICLES = 60
RUNS = 2


def _tracer(**kw):
    """构造小规模 tracer。注意 CausalTracer 不接受 num_runs（那是 trace() 的参数）。"""
    kw.setdefault("n_particles", PARTICLES)
    kw.setdefault("steps", 15)
    return CausalTracer(**kw)


class TestCausalTracingFaithfulMode(unittest.TestCase):
    """默认模式 = 与当前引擎行为一致（验证结果不被使用）。"""

    @classmethod
    def setUpClass(cls):
        cls.tracer = _tracer()
        cls.report = cls.tracer.trace(scene_types=SCENES, num_runs=RUNS)

    def test_director_is_not_on_generation_path(self):
        """导演层不在调用链上 → 贡献度必须是 None，而不是编出来的数字。"""
        for scene, layers in self.report.per_scene.items():
            d = layers["director"]
            self.assertFalse(d.on_generation_path,
                             f"{scene}: director 不应被标记为在生成路径上")
            self.assertIsNone(d.contribution,
                              f"{scene}: director 贡献度应为 None（不可测），"
                              f"实际为 {d.contribution}")
            self.assertIn("不在", d.note)

    def test_validate_contributes_zero_when_gate_disabled(self):
        """**核心诚实性断言。**

        当前 `LumynEngine.generate` 计算完 validation 后既不回退也不重生成，
        因此验证层对输出没有因果作用。恢复一个不起作用的层是空操作，
        贡献度必须**恰好为 0**。
        """
        for scene, layers in self.report.per_scene.items():
            self.assertAlmostEqual(
                layers["validate"].contribution, 0.0, places=9,
                msg=f"{scene}: 验证门关闭时 validate 贡献度应为 0")

    def test_generate_has_positive_contribution(self):
        """生成层是唯一真正影响输出的层，其贡献度应显著为正。"""
        for scene, layers in self.report.per_scene.items():
            self.assertGreater(layers["generate"].contribution, 0.0,
                               f"{scene}: generate 贡献度应为正")

    def test_contribution_equals_restore_minus_all_corrupt(self):
        """贡献度定义必须严格等于 score_restore − all_corrupt。"""
        for scene, layers in self.report.per_scene.items():
            base = self.report.all_corrupt[scene]
            for name in ("generate", "validate"):
                self.assertAlmostEqual(
                    layers[name].contribution,
                    layers[name].score_restore - base, places=9,
                    msg=f"{scene}/{name}: 贡献度与 restore−all_corrupt 不符")

    def test_clean_is_no_worse_than_corrupt(self):
        for scene, layers in self.report.per_scene.items():
            for name, res in layers.items():
                if res.contribution is None:
                    continue
                self.assertLessEqual(res.score_corrupt, res.score_clean + 1e-9,
                                     f"{scene}/{name}: 损坏分不应高于干净分")

    def test_config_records_the_mode(self):
        self.assertFalse(self.report.config["enable_validation_gate"])
        self.assertIn("on_path_layers", self.report.config)
        self.assertIn("fallback_proxy", self.report.config)


class TestCausalTracingGateMode(unittest.TestCase):
    """显式启用验证门后，验证层才真正生效。"""

    @classmethod
    def setUpClass(cls):
        cls.report = _tracer(enable_validation_gate=True).trace(
            scene_types=SCENES, num_runs=RUNS)

    def test_validate_contributes_positive_with_gate(self):
        for scene, layers in self.report.per_scene.items():
            self.assertGreater(layers["validate"].contribution, 0.0,
                               f"{scene}: 门启用后 validate 贡献度应为正")

    def test_config_records_the_mode(self):
        self.assertTrue(self.report.config["enable_validation_gate"])

    def test_director_still_not_on_path(self):
        """启用验证门不会让导演层回到生成路径上。"""
        for layers in self.report.per_scene.values():
            self.assertIsNone(layers["director"].contribution)


class TestNoFabricatedConstants(unittest.TestCase):
    """针对「重新引入编造数字」的回归测试。

    旧实现返回常数，因此对任何输入都给出完全相同的数字。
    真实实现必须随损坏强度变化 —— 这是常数实现**无法通过**的判据。
    """

    def test_results_depend_on_corruption_strength(self):
        weak = _tracer(noise=0.05).trace(scene_types=SCENES, num_runs=1)
        strong = _tracer(noise=1.5).trace(scene_types=SCENES, num_runs=1)
        cw = weak.per_scene[SCENES[0]]["generate"].contribution
        cs = strong.per_scene[SCENES[0]]["generate"].contribution
        self.assertNotAlmostEqual(
            cw, cs, places=6,
            msg=f"损坏强度从 0.05 变到 1.5 但贡献度几乎不变（{cw} vs {cs}）——"
                f"疑似又变成了硬编码常数")
        self.assertGreater(cs, cw, "损坏更强时，生成层的贡献度应更大")

    def test_source_has_no_hardcoded_layer_scores(self):
        """可执行代码里不得再出现「层名 → 分数」的硬编码字典。

        注意：**模块 docstring 里会引用旧常数的原值**作为更正说明，
        那是文档而非代码，不能误报。因此这里用 AST 只检查真实代码。
        """
        import ast
        src = Path(__file__).resolve().parent.parent / "explain" / "causal_tracing.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        layer_names = {"director", "generate", "validate"}
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys = {k.value for k in node.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                # 只要字典的键全部是层名，就是「层→分数」的常数表
                if keys and keys <= layer_names:
                    offenders.append(node.lineno)
        self.assertEqual(
            offenders, [],
            f"causal_tracing.py 中仍存在硬编码的层→分数字典（行号 {offenders}）")


class TestReportOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _tracer().trace(scene_types=SCENES, num_runs=1)

    def test_to_dict_structure(self):
        d = self.report.to_dict()
        self.assertIn("per_scene", d)
        self.assertIn("averaged", d)
        self.assertIn("config", d)
        self.assertIn("all_corrupt", d)
        self.assertEqual(set(d["per_scene"][SCENES[0]].keys()),
                         {"director", "generate", "validate"})

    def test_save_and_summary(self):
        out = Path("results/test_causal.json")
        out.parent.mkdir(exist_ok=True)
        self.report.save(str(out))
        self.assertTrue(out.exists())
        out.unlink()
        s = self.report.summary()
        self.assertIn("validate", s)
        self.assertIn("n/a", s)          # director 应显示为不可测
        self.assertIn("不在", s)

    def test_averaged_keeps_none_for_offpath_layer(self):
        self.assertIsNone(self.report.averaged["director"])
        self.assertIsNotNone(self.report.averaged["generate"])

    def test_plot_optional(self):
        p = self.report.plot("results/test_causal.png")
        if p:
            self.assertTrue(Path(p).exists())
            Path(p).unlink()

    def test_empty_scene_list(self):
        r = _tracer().trace(scene_types=[], num_runs=1)
        self.assertEqual(r.per_scene, {})


if __name__ == "__main__":
    unittest.main()
