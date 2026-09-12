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

**This document has not been compiled.** No LaTeX toolchain (`pdflatex`,
`xelatex`, `latexmk`, `tectonic`, `pandoc`) was available in the authoring
environment, so the rendered output — page breaks, float placement, font
substitution, package compatibility — is **unverified**.

What *has* been verified mechanically, by `check_tex.py`:

| Check | Status |
|---|---|
| Balanced `{` / `}` (escape- and comment-aware) | pass |
| `\begin` / `\end` pairing and nesting order | pass |
| Document-level `$` parity (multi-line inline math aware) | pass |
| Required preamble (`\documentclass`, `document` environment, ordering) | pass |
| Every `\ref` / `\eqref` target has a matching `\label` | pass |
| Every `\cite` key has a matching `\bibitem` | pass |
| No bare `_` outside math mode | pass |

Run it with:

```bash
python lumyn/docs/paper/check_tex.py lumyn/docs/paper/main.tex
```

The checker was itself validated against a deliberately broken file, in which it
caught all six injected faults (unbalanced brace, mismatched environment,
unclosed `document`, undefined reference, undefined citation, bare underscore).
A checker that always passes would be worthless, so this negative control
matters.

**Before submitting, compile on a machine with TeX and read the PDF.** The
static checks cannot catch layout problems, and this paper's claim of
reproducibility should extend to its own typesetting.

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
