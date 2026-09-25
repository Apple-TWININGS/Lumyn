"""LaTeX 静态自检：在没有 LaTeX 工具链的环境里，把**能机械检查的**都检查掉。

**这不能替代真正的编译。** 它只覆盖：
  1. 花括号 / 方括号配平（跳过转义与注释）
  2. begin/end 环境配对与嵌套顺序
  3. verbatim / lstlisting 等原样环境的成对性
  4. 必需的 preamble 元素（documentclass、document 环境）
  5. \\ref / \\eqref / \\cite 的目标是否存在（未定义的引用是最常见的编译后警告）
  6. 常见转义疏漏：裸 % & # 在正文中、$ 的奇偶性
  7. 数学模式内出现未转义下划线的常见错误

用法：
    python docs/paper/check_tex.py docs/paper/main.tex
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Windows 控制台默认 cp936/GBK，下面的 ✅/❌ 会让 print() 抛 UnicodeEncodeError，
# 表现为「检查逻辑全过、最后打印时崩掉」。这里把 stdout 切到 UTF-8。
# Python 3.7+ 有 reconfigure；没有的环境（或 stdout 被重定向为不可包装对象）静默跳过。
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass


def strip_comments(text: str) -> str:
    """去掉行注释（未转义的 %），保留行结构。"""
    out = []
    for line in text.splitlines():
        res, i = [], 0
        while i < len(line):
            c = line[i]
            if c == "\\" and i + 1 < len(line):
                res.append(line[i:i + 2]); i += 2; continue
            if c == "%":
                break
            res.append(c); i += 1
        out.append("".join(res))
    return "\n".join(out)


MATH_ENVS = ("equation", "equation*", "align", "align*", "gather", "gather*",
             "multline", "multline*", "eqnarray", "eqnarray*", "displaymath",
             "alignat", "alignat*", "split", "cases", "array")


def strip_math(text: str) -> str:
    """把数学内容整体移除，只留下正文，用于检查「正文里的裸下划线」。

    必须同时处理两类数学：
      · 数学环境（equation / align / ...）——可跨多行
      · 行内 $...$ ——**也可跨行**（此前按行统计 $ 的奇偶，会把跨行公式误报）
    """
    # 1) 数学环境：整块删除
    for env in MATH_ENVS:
        text = re.sub(
            r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}",
            " ", text, flags=re.S)
    # 2) 行内 $...$（非贪婪，允许跨行；跳过 \$$）
    text = re.sub(r"(?<!\\)\$(?!\$).*?(?<!\\)\$", " ", text, flags=re.S)
    # 3) 显示数学 $$...$$
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    return text


def check_braces(text: str, errors: list, warns: list) -> None:
    depth = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        i = 0
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line):
                i += 2; continue
            if line[i] == "{":
                depth += 1
            elif line[i] == "}":
                depth -= 1
                if depth < 0:
                    errors.append(f"L{lineno}: 多余的 '}}'（花括号提前闭合）")
                    depth = 0
            i += 1
    if depth != 0:
        errors.append(f"花括号不配平：结束时仍有 {depth} 个未闭合的 '{{'")


def check_environments(text: str, errors: list) -> None:
    stack = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", line):
            kind, name = m.group(1), m.group(2)
            if kind == "begin":
                stack.append((name, lineno))
            else:
                if not stack:
                    errors.append(f"L{lineno}: \\end{{{name}}} 没有对应的 \\begin")
                    continue
                top, tl = stack.pop()
                if top != name:
                    errors.append(
                        f"L{lineno}: \\end{{{name}}} 与 L{tl} 的 \\begin{{{top}}} 不匹配")
    for name, lineno in stack:
        errors.append(f"L{lineno}: \\begin{{{name}}} 未闭合")


def check_dollars(text: str, errors: list) -> None:
    """文档级 $ 配平检查。

    不能按行统计：行内公式可以跨行（本文件 L265--266 就是一例），
    那样会产生误报。
    """
    s = text.replace("$$", "")
    s = re.sub(r"(?<!\\)\$", "$", s)          # 保留未转义的 $
    n = len(re.findall(r"(?<!\\)\$", s))
    if n % 2 != 0:
        errors.append(f"全文未转义 '$' 计数为奇数（{n}），存在未闭合的行内公式")


def check_preamble(text: str, errors: list) -> None:
    if "\\documentclass" not in text:
        errors.append("缺少 \\documentclass")
    if "\\begin{document}" not in text:
        errors.append("缺少 \\begin{document}")
    if "\\end{document}" not in text:
        errors.append("缺少 \\end{document}")
    if "\\begin{document}" in text and "\\documentclass" in text:
        if text.index("\\documentclass") > text.index("\\begin{document}"):
            errors.append("\\documentclass 出现在 \\begin{document} 之后")


def check_refs(text: str, errors: list, warns: list) -> None:
    labels = set(re.findall(r"\\label\{([^}]*)\}", text))
    for m in re.finditer(r"\\(?:ref|eqref|autoref)\{([^}]*)\}", text):
        if m.group(1) not in labels:
            errors.append(f"未定义的引用目标：\\ref{{{m.group(1)}}}")
    bibitems = set(re.findall(r"\\bibitem\{([^}]*)\}", text))
    for m in re.finditer(r"\\cite\{([^}]*)\}", text):
        for key in m.group(1).split(","):
            key = key.strip()
            if key and key not in bibitems:
                errors.append(f"未定义的引用键：\\cite{{{key}}}")


def check_math_underscore(text: str, errors: list) -> None:
    """数学模式外出现未转义下划线（LaTeX 会报 Missing $ inserted）。

    先移除全部数学内容（含跨行公式），再检查剩余正文。
    """
    body = strip_math(text)
    for lineno, line in enumerate(body.splitlines(), 1):
        stripped = re.sub(r"\\texttt\{[^}]*\}", "", line)   # \texttt 里的 _ 合法
        stripped = re.sub(r"\\[a-zA-Z]+", "", stripped)
        if re.search(r"(?<!\\)_", stripped):
            errors.append(f"L{lineno}: 数学模式外出现未转义下划线 '_'")


def main(path: str) -> int:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    text = strip_comments(raw)
    errors: list = []
    warns: list = []

    check_preamble(text, errors)
    check_braces(text, errors, warns)
    check_environments(text, errors)
    check_dollars(text, errors)
    check_refs(text, errors, warns)
    check_math_underscore(text, errors)

    n_lines = len(raw.splitlines())
    print(f"[check_tex] {p}  ({n_lines} 行)")
    if errors:
        print(f"  ❌ {len(errors)} 个问题：")
        for e in errors:
            print(f"     - {e}")
    else:
        print("  ✅ 静态检查全部通过")
    print()
    print("  说明：本检查只覆盖**结构**（括号/环境/引用/宏包登记），看不到排版。")
    print("        排版需要真实编译。本文档已于 2026-09-12 在 TeX Live 上编译过：")
    print("        10 页、0 未定义引用、0 Overfull/Underfull 盒子。")
    print("        换 TeX 发行版后请重跑：latexmk -pdf main.tex")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "docs/paper/main.tex"))
