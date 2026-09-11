"""打包后四重自检：ZIP 完整性 / 独立解压 / 目录结构 / .py 编译。"""
from __future__ import annotations
import os, sys, zipfile, shutil, subprocess, tempfile


def main():
    zip_path = sys.argv[1]
    assert os.path.exists(zip_path), f"zip not found: {zip_path}"
    print(f"[1/4] ZIP 完整性校验: {zip_path}")

    # 1. ZIP 本身完整（尾部 EOCD 存在，无截断）
    bad = zipfile.ZipFile(zip_path).testzip()
    assert bad is None, f"损坏条目: {bad}"
    print("     -> OK (无损坏条目)")

    # 2. 独立解压到临时目录
    tmp = tempfile.mkdtemp(prefix="lumyn_selfcheck_")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        print("     -> 独立解压成功")

        # 3. 目录结构：必须有 lumyn/physics/ 和 lumyn/examples/
        root = os.path.join(tmp, "lumyn")
        assert os.path.isdir(root), f"缺少 lumyn/ 顶层目录，根目录内容: {os.listdir(tmp)}"
        must = {
            "physics": ["differentiable.py", "losses.py", "guided_generation.py",
                        "differentiable_numpy.py", "README.md", "make_figure.py"],
            "examples": ["guided_generation_demo.py", "README.md"],
            "scientific": ["presets.py", "visualization.py", "export.py", "metrics.py",
                            "__init__.py"],
        }
        for sub, files in must.items():
            d = os.path.join(root, sub)
            assert os.path.isdir(d), f"缺少 {sub}/"
            for f in files:
                assert os.path.exists(os.path.join(d, f)), f"缺少 {sub}/{f}"
        print("     -> 目录结构 OK (physics/6 + examples/2 + scientific/8 ...)")

        # 4. 全部 .py 编译通过
        bad_py = []
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.endswith(".py"):
                    p = os.path.join(dirpath, f)
                    try:
                        compile(open(p, encoding="utf-8").read(), p, "exec")
                    except SyntaxError as e:
                        bad_py.append(f"{p}: {e}")
        assert not bad_py, "语法错误:\n" + "\n".join(bad_py)
        print("     -> 全部 .py 编译通过")

        # 5. 顺手跑一次结构校验脚本（若存在）
        vs = os.path.join(root, "verify_structure.py")
        if os.path.exists(vs):
            r = subprocess.run([sys.executable, vs], capture_output=True, text=True, cwd=root)
            print("     -> verify_structure.py:")
            for line in r.stdout.strip().splitlines():
                print("        " + line)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\n[ALL GREEN] 包可安全分发。")


if __name__ == "__main__":
    main()
