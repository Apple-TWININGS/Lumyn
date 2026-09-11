"""eval 模块测试（真实运行，非 mock）：答案判定 + bootstrap CI + 验证层自检。"""
from __future__ import annotations
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lumyn.eval import answer_judge as aj
from lumyn.eval.bootstrap_ci import bootstrap_ci as bootstrap_ci_fn
from lumyn.eval import check_validator as cv


class TestAnswerJudge(unittest.TestCase):
    def test_self_test(self):
        """judge 自测 7/7 通过。"""
        self.assertTrue(aj.self_test())

    def test_judge_numeric(self):
        correct, val = aj.judge("答案是 3.14", "3.14", tol=0.05)
        self.assertTrue(correct)
        self.assertAlmostEqual(val, 3.14, places=4)

    def test_judge_relative_error(self):
        # 相对误差截断：ref=100, pred=106 → 误差 6% > tol=0.05 → 错
        correct, _ = aj.judge("106", "100", tol=0.05)
        self.assertFalse(correct)

    def test_extract_number_scientific(self):
        self.assertEqual(aj.extract_number("10^3"), 1000.0)
        self.assertEqual(aj.extract_number("速度 3.0 m/s^2"), 3.0)


class TestBootstrapCI(unittest.TestCase):
    def test_exact_match(self):
        """[0.7, 0.9] 手算 → 均值 0.8，CI 端点 [0.7, 0.9]。"""
        lo, hi, mean = bootstrap_ci_fn([0.7, 0.8, 0.9], n_iter=1000, seed=42)
        self.assertAlmostEqual(mean, 0.8, places=6)
        self.assertAlmostEqual(lo, 0.7, places=6)
        self.assertAlmostEqual(hi, 0.9, places=6)

    def test_reproducible(self):
        a = bootstrap_ci_fn([0.1] * 10 + [0.9] * 10, n_iter=200, seed=42)
        b = bootstrap_ci_fn([0.1] * 10 + [0.9] * 10, n_iter=200, seed=42)
        self.assertEqual(a, b)  # 同 seed 结果一致

    def test_empty(self):
        lo, hi, mean = bootstrap_ci_fn([])
        self.assertTrue(all(v != v for v in (lo, hi, mean)))  # NaN


class TestCheckValidator(unittest.TestCase):
    def test_null_rate(self):
        detail = [
            {"id": "p1", "pred_raw": "无法解析", "ref": "3.14", "pred_val": None},
            {"id": "p2", "pred_raw": "= 2.0", "ref": "2.0", "pred_val": 2.0},
        ]
        report = cv.check_validation(detail)
        self.assertEqual(report["null_count"], 1)
        self.assertAlmostEqual(report["null_rate"], 0.5)

    def test_bad_ref_warning(self):
        detail = [{"id": "p1", "pred_raw": "x", "ref": "不是数字", "pred_val": None}]
        report = cv.check_validation(detail)
        self.assertIn("bad_refs", report)


if __name__ == "__main__":
    unittest.main()
