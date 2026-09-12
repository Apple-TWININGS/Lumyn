# Paper

arXiv preprint source.

```
main.tex        the paper
check_tex.py    static checker (see below)
```

## Building

```bash
pdflatex main
pdflatex main      # second pass resolves \ref / \cite
```

Two passes are required: the paper uses `\ref`, `\eqref` and `\cite`, and the
first pass writes the `.aux` file those resolve against.

Required packages: `geometry`, `amsmath`, `amssymb`, `booktabs`, `graphicx`,
`microtype`, `hyperref`. All ship with any standard TeX Live or MiKTeX install.
No external figure files are needed — every result is a table.

## Verification status

**This document has now been compiled.** It was built on 2026-09-12 with TeX Live
on Ubuntu 22.04 (`latexmk -pdf`), because the authoring environment had no LaTeX
toolchain. Result:

| Check | Status |
|---|---|
| Builds without error | pass (`latexmk` exit 0) |
| Pages | 10 |
| Undefined references / citations | **0** (28 labels, 3 bibitems, all resolved) |
| Overfull `\hbox` | **0** |
| Underfull `\hbox` / Overfull `\vbox` | 0 / 0 |
| Missing characters | 0 |

The **first** compile was not clean: six Overfull `\hbox` warnings, every one of
them in a table header rather than in body text, the worst by 74 pt (about
26 mm — the table ran roughly an inch past the right margin). They were fixed
in two steps, each measured rather than assumed:

1. `\setlength{\tabcolsep}{4pt}` globally (from the 6 pt default) — this alone
   removed one warning and cut the rest by ~16 pt each.
2. `\small` on all 14 tables, then `\footnotesize` on the three widest — drove
   the remaining warnings to zero and shortened the paper from 11 to 10 pages.

`\resizebox{\textwidth}{!}{...}` was deliberately **not** used: it scales each
table by its own factor, so tables that start at different widths end up with
different font sizes.

`check_tex.py` remains useful and is complementary — it checks brace and
environment balance, preamble presence, and undefined-macro references, and runs
without a TeX installation. It caught all six injected faults in a deliberately
broken file, so it is not a checker that always passes. But it cannot see
layout, which is exactly what the first real compile exposed.

### Known remaining caveat

The build host's TeX Live is Ubuntu's packaged 2022 release. A different TeX
distribution (TeX Live 2024+, MiKTeX) may hyphenate differently and shift line
breaks slightly. The zero-Overfull result is therefore specific to this
toolchain; re-check after switching.

## Relationship to `PAPER_DRAFT.md`

`../PAPER_DRAFT.md` is the Markdown working draft. `main.tex` is the same content
rendered for submission. The Markdown version is the one that is easiest to edit
during review; the LaTeX version is the artefact. When the two disagree, `main.tex`
is authoritative for submission and `PAPER_DRAFT.md` should be regenerated.

## Content policy

Every number in the paper is traced to a producing script in Appendix A. Three
claims made by earlier internal drafts are explicitly **retracted** in §7 rather
than quietly dropped; the falsification of those claims is where the paper's
methodological contributions came from.

No result is computed on synthetic corruption sets — §5.4 shows why, with a
calibration procedure that readers can reuse.
