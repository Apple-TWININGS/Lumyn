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

### Local re-check on MiKTeX (2026-09-13, Windows)

Re-compiled from scratch on a second, independent toolchain — MiKTeX on Windows
(`pdflatex` ×3, no `latexmk`) — to test whether the zero-warning result was an
artefact of one distribution. It was not:

| Check | Result |
|---|---|
| Pages | 10 |
| LaTeX `Warning` | **0** |
| Overfull / Underfull boxes | 0 / 0 |
| Undefined references | 0 (all resolved on the 2nd pass) |
| Pass 2 vs pass 3 | byte-identical output, 313196 bytes |

Pass 2 and pass 3 differ only inside the PDF metadata (`/CreationDate` and the
`/ID` hash); the body content is bit-for-bit identical, so the document is fully
converged. No `\hbox` reflow was introduced by the different hyphenation tables,
which is the failure mode this check existed to catch. See the caveat below for
why the checked-in PDF is still the TeX Live one.

### Known remaining caveat

The build host's TeX Live is Ubuntu's packaged 2022 release, and a different TeX
distribution (TeX Live 2024+, MiKTeX) may hyphenate differently and shift line
breaks slightly. The zero-Overfull result is therefore specific to the toolchain
that produced it; re-check after switching.

The MiKTeX run above is a *verification* pass, not a replacement build: the
`main.pdf` in this directory is still the TeX Live output. arXiv's own TeX Live
is likely to reject PDFs that embed MiKTeX's bitmap `pk` fonts, whereas TeX
Live's font embedding is known-clean under `pdffonts`. So the rule for the
submission artefact is: **build it on TeX Live, use MiKTeX only to check.**

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
