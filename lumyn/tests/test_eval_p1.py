"""P1 测试（真实运行）：E1/E2/E3 错误分类 + 卡方对比。"""
from __future__ import annotations
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lumyn.eval import error_classifier as ec


class TestErrorClassifier(unittest.TestCase):
    def test_auto_classify_e1(self):
        """导演层选错定律 → E1（非静态场景却用了 StaticEquilibrium）。"""
        trace = {"laws": ["StaticEquilibrium"], "constraints": [],
                 "subject_motion": "left_to_right", "static": False}
        rec = ec.ErrorClassifier().auto_classify(trace, final_state={"v": 5.0})
        self.assertEqual(rec.category, "E1")

    def test_auto_classify_e2(self):
        """代数/数值精度问题 → E2。"""
        trace = {"laws": ["Newton2", "Kinematics"], "constraints": ["force_balance"],
                 "subject_motion": "idle", "numeric_instability": True}
        rec = ec.ErrorClassifier().auto_classify(trace, final_state={"v": 1.0})
        self.assertEqual(rec.category, "E2")

    def test_auto_classify_e3(self):
        """守恒约束被违背 → E3。"""
        trace = {"laws": ["Newton2"], "constraints": ["energy_conserved"],
                 "subject_motion": "idle", "constraint_violated": True}
        rec = ec.ErrorClassifier().auto_classify(trace, final_state={"v": 1.0})
        self.assertEqual(rec.category, "E3")

    def test_auto_classify_unknown(self):
        """规则覆盖不到 → Unknown。"""
        rec = ec.ErrorClassifier().auto_classify({}, final_state={})
        self.assertEqual(rec.category, "Unknown")

    def test_chi2_real(self):
        """手算验证卡方：2x2 列联表 [[10,5],[3,12]]。

        chi2 = (10-6.76)^2/6.76 + (5-8.24)^2/8.24
             + (3-6.24)^2/6.24 + (12-8.76)^2/8.76
           = 6.65158...
        """
        chi2, p, dof = ec.chi2_contingency([[10, 5], [3, 12]])
        self.assertAlmostEqual(chi2, 6.6516, places=3)
        self.assertEqual(dof, 1)
        self.assertLess(p, 1.0)

    def test_compare_models(self):
        """compare_models 包装可用。"""
        report = ec.compare_models(
            {"director": 0.1, "generate": 0.3, "validate": 0.6},
            {"director": 0.05, "generate": 0.25, "validate": 0.7},
        )
        self.assertIn("chi2", report)
        self.assertIn("p_value", report)


if __name__ == "__main__":
    unittest.main()
