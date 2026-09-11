"""清理缓存与构建产物（打包前调用）。

- 删除所有 __pycache__ / .pyc
- 删除外层产出目录 lumyn_output/（不打进包内）
- 删除旧 zip 包，强迫重新打包
"""
import os
import shutil
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(ROOT)

# 1. 清理所有 pycache
for dirpath, dirnames, _ in os.walk(ROOT):
    for d in list(dirnames):
        if d == "__pycache__":
            shutil.rmtree(os.path.join(dirpath, d), ignore_errors=True)

# 2. 清理嵌套副本（若存在）
nested = os.path.join(ROOT, "lumyn")
if os.path.isdir(nested):
    shutil.rmtree(nested, ignore_errors=True)

# 3. 清理产出目录（位于 workspace 下，非包内）
for out_dir in [os.path.join(PARENT, "lumyn_output"),
                os.path.join(PARENT, "a800_cosmic_video"), "build", "dist"]:
    if os.path.isdir(out_dir):
        print("[clean]", out_dir)
        shutil.rmtree(out_dir, ignore_errors=True)

# 4. 清理旧 zip
for z in glob.glob(os.path.join(PARENT, "lumyn*.zip")):
    print("[clean]", z)
    os.remove(z)

print("[clean] done")
