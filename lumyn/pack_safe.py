"""安全打包：相对 arcname + 正斜杠 + with 关闭，兼容 Windows 资源管理器/7-Zip/macOS。"""
from __future__ import annotations
import os, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def pack_safe(src_dir: str, out_zip: str, topdir: str = "lumyn",
               exclude=(".git", "__pycache__", ".DS_Store")) -> int:
    """打包并把所有文件放在 `topdir/` 下，解压后得 lumyn/... 标准结构。

    src_dir 可以是 lumyn/ 本身或其父目录；顶层名统一由 topdir 控制，
    避免"文件直接铺在压缩包根目录"导致 Windows 解压器混乱 / 找不到 physics/。
    """
    src_dir = os.path.abspath(src_dir)
    count = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = [d for d in dirs if not any(p in d for p in exclude)]
            for name in sorted(files):
                if name.startswith(".") and name not in ("LICENSE",):
                    continue
                full = os.path.join(root, name)
                rel = os.path.relpath(full, src_dir).replace(os.sep, "/")
                arc = f"{topdir}/{rel}" if rel != "." else f"{topdir}/"
                zf.write(full, arcname=arc)
                count += 1
    return count


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else HERE
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "lumyn.zip")
    topdir = sys.argv[3] if len(sys.argv) > 3 else "lumyn"
    n = pack_safe(src, out, topdir=topdir)
    size = os.path.getsize(out)
    print(f"[pack] {n} files -> {out} ({size/1024:.1f} KB)")
    # 立即自检：写完就测完整性，坏包当场发现
    bad = zipfile.ZipFile(out).testzip()
    print("[pack] OK (no errors)" if bad is None else f"[pack] BAD entries: {bad}")


if __name__ == "__main__":
    main()
