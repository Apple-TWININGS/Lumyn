"""Verify every link, directory and command claimed by the root README.

This repository's honesty policy is that the README states only what has been
verified by running the code. This script is the enforcement of that sentence:
if the README drifts from the tree, it fails.

Run from the repository root:

    python scripts/verify_readme.py
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

README = ROOT / "README.md"
text = README.read_text(encoding="utf-8")
failures = 0


def check(label: str, ok: bool) -> None:
    global failures
    failures += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {label}")


print("=== 1. markdown links ===")
for link in sorted(set(re.findall(r"\]\(([^)#][^)]*)\)", text))):
    if link.startswith("http"):
        print(f"  [skip] {link}")
        continue
    check(link, (ROOT / link).resolve().exists())

print("\n=== 2. directories named in the layout ===")
for d in [
    "lumyn/core", "lumyn/physics", "lumyn/video", "lumyn/eval", "lumyn/scientific",
    "lumyn/memory", "lumyn/explain", "lumyn/integrations", "lumyn/gameplay",
    "lumyn/sync", "lumyn/mobile", "lumyn/engine_bridge", "lumyn/experiments",
    "lumyn/tests", "lumyn/docs", "scripts",
]:
    check(f"{d}/", (ROOT / d).is_dir())

print("\n=== 3. declared test count vs measured ===")
proc = subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "lumyn/tests"],
    cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
summary = [ln for ln in (proc.stderr or "").splitlines()
           if ln.startswith("Ran ") or ln.startswith("OK")]
print("  " + (" | ".join(summary) if summary else "<unittest produced no summary>"))
declared = re.search(r"\*\*(\d+)\*\* \|", text)
measured = re.search(r"Ran (\d+) tests", proc.stderr or "")
if declared and measured:
    check(f"README says {declared.group(1)}, measured {measured.group(1)}",
          declared.group(1) == measured.group(1))
else:
    check("test count could be parsed from both README and unittest", False)

print("\n=== 4. example scripts named in quick start ===")
for s in ["lumyn/examples/galaxy.py", "lumyn/examples/scientific_demo.py"]:
    check(s, (ROOT / s).is_file())

print("\n=== 5. public API returns the documented keys ===")
try:
    from lumyn import HMMPGameEngine

    keys = set(HMMPGameEngine().generate("binary_star", n_particles=50).keys())
    expected = {"mass", "meta", "scene", "trajectory", "validation"}
    check(f"generate() -> {sorted(keys)}", expected <= keys)
except Exception as exc:  # noqa: BLE001
    check(f"import/generate raised {type(exc).__name__}: {exc}", False)

print("\n=== 6. the four documented scene names are accepted ===")
try:
    from lumyn import HMMPGameEngine

    engine = HMMPGameEngine()
    for name in ["binary_star", "spiral_galaxy", "globular_cluster", "galaxy_collision"]:
        try:
            engine.generate(name, n_particles=30)
            check(name, True)
        except Exception as exc:  # noqa: BLE001
            check(f"{name}: {type(exc).__name__}: {exc}", False)
except Exception as exc:  # noqa: BLE001
    check(f"engine construction raised {type(exc).__name__}: {exc}", False)

print("\n" + "=" * 60)
print(f"failures: {failures}")
sys.exit(1 if failures else 0)
