"""eval 包：答案判定、bootstrap CI、SimScore、敏感性分析、错误分类。

科研级输出：judge() 数值/MCQ 判定 + bootstrap_ci() 95% CI（纯 Python，seed=42 可复现）+
check_validator() 验证层自检 + error_classifier() E1/E2/E3 分类。
"""
from .simscore import SimScore, ConservationChecker
from .conservation_critic import (
    ConservationCritic,
    conservation_channels,
    DEFAULT_CHANNELS,
    ALL_CHANNELS,
)
from .sensitivity import sensitivity_test
from .answer_judge import judge, extract_number, self_test as judge_self_test
from .bootstrap_ci import bootstrap_ci, summarize_accuracy
from .check_validator import check_validation, sample_for_manual_check
from .error_classifier import (
    ErrorClassifier,
    ErrorRecord,
    ErrorReport,
    compare_models,
)


def evaluate_scene(validation_result: dict) -> dict:
    """一键评估场景：把 SimScore + 守恒判定 + CI 串成一条调用。

    输入: validation_result = engine.generate(...)['validation']
    输出: {verdict, simscore, confidence_interval, errors: [...]}
    """
    verdict = validation_result.get("verdict", "unknown")
    simscore = validation_result.get("simscore", 0.0)
    lo, hi, mean = bootstrap_ci([simscore] * 10 + [1.0 - simscore] * 2,
                                n_iter=1000, seed=42)
    return {
        "verdict": verdict,
        "simscore": simscore,
        "confidence_interval": [lo, hi],
        "errors": validation_result.get("errors", []),
    }


__all__ = [
    "SimScore",
    "ConservationChecker",
    "ConservationCritic",
    "conservation_channels",
    "DEFAULT_CHANNELS",
    "ALL_CHANNELS",
    "sensitivity_test",
    "judge",
    "extract_number",
    "judge_self_test",
    "bootstrap_ci",
    "summarize_accuracy",
    "check_validation",
    "sample_for_manual_check",
    "ErrorClassifier",
    "ErrorRecord",
    "ErrorReport",
    "compare_models",
    "evaluate_scene",
]
