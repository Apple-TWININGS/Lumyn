# Contributing

Thanks for looking. This repository has an unusual rule, so it is worth thirty seconds
of your time before you open a PR.

## The one rule: the README may only state what has been verified

`README.md` is not marketing copy. Every number, every link, every directory and every
command in it is checked by a script:

```bash
python scripts/verify_readme.py     # must print "failures: 0"
```

It verifies the six things the README claims:

1. every relative markdown link resolves to a file that exists
2. every directory named in the layout section exists
3. the declared test count equals the measured test count
4. the example scripts in the quick start exist
5. `HMMPGameEngine().generate(...)` returns the documented keys
6. the four documented scene names are all accepted

**So if you change behaviour, change the README in the same commit, or CI-equivalent
verification fails locally.** This is deliberately strict: it is the reason the README
can be trusted. Superseded numbers are *recorded with the reason they were wrong*, not
deleted — see `lumyn/README.md` §6.5 and `lumyn/docs/PAPER_STATUS.md` §七.

## Tests

```bash
python -m unittest discover -s lumyn/tests
```

Two things to know about the output:

- **Run it with `-s lumyn/tests`, not `-s lumyn`.** Discovering from `lumyn/` imports
  package-internal modules under their top-level names (`core`, `explain`, ...) and
  produces spurious collection errors. The documented count assumes the former.
- **`Ran 113 tests` is stable; the verdict line is not.** On a machine where a test's
  internal subprocess cannot be spawned you get the same 113 with `1 error`
  (a `FileNotFoundError` raised inside a test, unrelated to any assertion). Locally the
  line is `OK (skipped=1, expected failures=1)`. Both are the same 113.

The root README documents a per-file test breakdown; keep it in sync if you add or
remove tests.

## Windows notes

Two environment quirks cost real debugging time here, so they are written down:

- **Console encoding.** The default Windows console is GBK/cp936. Any script that prints
  `✅` / `❌` will raise `UnicodeEncodeError` *after* the checks pass — it looks like the
  checker crashed when it actually succeeded. Scripts here therefore call

  ```python
  sys.stdout.reconfigure(encoding="utf-8")
  ```

  `scripts/verify_readme.py` and `lumyn/docs/paper/check_tex.py` both do this; follow the
  pattern in new scripts, or set `PYTHONIOENCODING=utf-8` in the shell.
- **Line endings.** `.gitattributes` pins `text=auto eol=lf`, so the repository and the
  working tree are both LF. Binary formats (`.png .jpg .mp4 .npz .fits .h5 .pdf`) are
  marked `binary` so Git never applies newline conversion to them.

## LaTeX

`lumyn/docs/paper/` holds the paper. `check_tex.py` is a *static* checker (braces,
environments, `\ref`/`\cite` targets, bare underscores) and **does not replace a real
compile**. It catches the mechanical errors; typesetting needs a toolchain.

```bash
python lumyn/docs/paper/check_tex.py lumyn/docs/paper/main.tex
# real build, from lumyn/docs/paper/:
latexmk -pdf main.tex        # or: pdflatex main.tex   (run 2-3x for cross-refs)
```

Measured state: 10 pages, 0 undefined references, 0 Overfull/Underfull boxes. **Formal
artifacts are built with TeX Live**, not MiKTeX — MiKTeX's bitmap (`pk`) font fallback
draws complaints from arXiv's submission checks. A local MiKTeX build is fine for
verification and must not overwrite the committed PDF.

## Scope

The generation path is a physics integrator, not a learned model, and the limitations
section of the README (`§6`) is the honest boundary of what this does. Claims about
improving on learned generative models are out of scope unless measured — and if a
measurement retracts an earlier claim, the retraction goes in the paper and the README
rather than the claim quietly disappearing.

## Repository settings not stored in files

Issues and topics are server-side settings, so they cannot be set by a commit. To change
them: **Settings → General → Features** (Issues) and the **⚙️ next to About** (topics)
on `https://github.com/Apple-TWININGS/Lumyn`.
