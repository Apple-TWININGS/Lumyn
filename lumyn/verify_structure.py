"""校验 lumyn 包结构：physics/ 6 文件、examples/、核心模块、测试可发现。"""
from __future__ import annotations
import os, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))

REQUIRED = {
    "physics": [
        "differentiable.py", "losses.py", "guided_generation.py",
        "differentiable_numpy.py", "README.md", "make_figure.py",
    ],
    "examples": ["guided_generation_demo.py", "README.md"],
    "scientific": ["presets.py", "visualization.py", "export.py", "metrics.py",
                   "teaching.py", "frames.py", "differentiable.py",
                   "differentiable_numpy.py", "__init__.py"],
    "eval": ["answer_judge.py", "bootstrap_ci.py", "check_validator.py",
             "error_classifier.py", "__init__.py"],
    "explain": ["causal_tracing.py", "__init__.py"],
    "engine_bridge/protocol": ["deterministic.py"],
    "gameplay": ["moba_rules.py", "wuxia_rules.py", "navmesh_patch.py", "fow_mask.py"],
    "sync": ["lockstep.py"],
    "mobile": ["lod_policy.py"],
}

PYTHON_DIRS = ["core", "physics", "video", "eval", "memory", "integrations",
               "scientific", "gameplay", "sync", "mobile", "explain",
               "engine_bridge", "engine_bridge/protocol"]


def check_files():
    missing = []
    for sub, files in REQUIRED.items():
        d = os.path.join(HERE, sub)
        for f in files:
            path = os.path.join(d, f)
            if not os.path.exists(path):
                missing.append(f"{sub}/{f}")
    return missing


def count_tests():
    tests = os.path.join(HERE, "tests")
    if not os.path.isdir(tests):
        return 0
    n = 0
    for f in os.listdir(tests):
        if f.startswith("test_") and f.endswith(".py"):
            n += 1
    return n


def main():
    print("== Lumyn structure check ==")
    miss = check_files()
    if miss:
        print(f"[FAIL] {len(miss)} missing files:")
        for m in miss:
            print(f"  - {m}")
        sys.exit(1)
    print(f"[OK] all required files present (physics/6, examples/, scientific/8, ...)")
    print(f"[OK] test modules: {count_tests()}")

    # 尝试编译检查（不依赖 torch/astropy）
    bad = []
    for root, _, files in os.walk(HERE):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                try:
                    compile(open(p, encoding="utf-8").read(), p, "exec")
                except SyntaxError as e:
                    bad.append(f"{p}: {e}")
    if bad:
        print("[FAIL] syntax errors:")
        for b in bad:
            print(f"  {b}")
        sys.exit(1)
    print("[OK] all .py compile clean (syntax)")


if __name__ == "__main__":
    main()
