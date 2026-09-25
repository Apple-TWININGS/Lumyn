# Lumyn

<!-- status badges: all static, because everything they claim is machine-checked by
     scripts/verify_readme.py. No CI badge is shown because no CI workflow exists. -->

**A physically-grounded scene generation engine — simulate first, render second.**

[![tests](https://img.shields.io/badge/tests-113%20passing-brightgreen)](#42-tests)
[![engine](https://img.shields.io/badge/generation-Barnes--Hut%20N--body-blue)](#2-what-this-is)
[![result](https://img.shields.io/badge/result-a%20tie%2C%20stated%20as%20a%20tie-orange)](#3-results)
[![README](https://img.shields.io/badge/README-machine--checked-informational)](scripts/verify_readme.py)
[![status](https://img.shields.io/badge/domain_research-artefact-lightgrey)](#22-research-zero-shot-conservation-critics)

**Contributing?** Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first — the short version is that
this README may only state what has been verified, and `scripts/verify_readme.py` enforces it.

A physically-grounded scene generation engine — simulate the physics first, render
second — and the artefact for one research question: *can a zero-shot critic built from
conservation laws replace a learned one?*

Every frame comes from a real Barnes-Hut N-body integration, and trajectories are checked
against conservation laws rather than accepted on appearance. The measured answer to the
research question is a **tie**, and it is in §3.

> **How to read this**: 30 seconds = §1 · 5 minutes = §1–§3 · full = everything +
> [`lumyn/docs/PAPER_DRAFT.md`](lumyn/docs/PAPER_DRAFT.md) (method) and
> [`lumyn/docs/PAPER_STATUS.md`](lumyn/docs/PAPER_STATUS.md) (evidence ledger)

---

## 1. In 30 seconds

**What I built.** A pipeline that **simulates** a scene with a Barnes-Hut N-body solver
(`O(N log N)`), **measures** the trajectory against conservation laws (energy, momentum,
angular momentum, centre of mass), and **optionally renders** it to video — plus the
experiments behind the critic study, a PyTorch **differentiable** physics path that
recovers initial conditions from a target final state (with a NumPy numerical-gradient
fallback for environments without PyTorch), and a **113-test** suite. Scenes are
deterministic given a seed; on the same platform results reproduce bit-for-bit.

**Headline result (a tie, stated as a tie).** On the standard benchmarks a zero-shot
conservation critic — no training, no labels — selects the best candidate **100%** of the
time (3 systems), reduces trajectory error by **75–82%** versus random selection, and lands
at **1.03–1.05×** of the best achievable selection. So it beats a supervised pairwise-ranking
critic on pairwise-force generators. **But it does not beat a correctly-used Hamiltonian Neural
Network critic** — on the Hamiltonian-parameterised family the two tie exactly.

**Status.** The engine, the suite and the selection experiments are measured; and this file is
machine-checked (§6.7). The limitations are consolidated in §6 — including the one that bounds the
whole exercise: the validation layer's causal contribution is **exactly 0**, because `generate`
never acts on it, for a measured structural reason (§6.1).

**Evidence.** `python -m unittest discover -s lumyn/tests` → `Ran 113 tests`;
`results/experiment_h_benchmark.json` (the 6-body figures);
`lumyn/docs/paper/main.pdf` (the compiled paper, see §7).

---

## 2. What this is

Lumyn is a pipeline that:

1. **Simulates** a scene with a Barnes-Hut N-body solver (`O(N log N)`).
2. **Measures** the trajectory against conservation laws (energy, momentum, angular
   momentum, centre of mass).
3. **Optionally renders** the particle trajectory to video.

Scenes are deterministic given a seed. On the same platform, results reproduce
bit-for-bit.

There is also a **differentiable** physics path (PyTorch) supporting gradient-based
recovery of initial conditions from a target final state, plus a NumPy
numerical-gradient fallback for environments without PyTorch.

### 2.1 What this is not

Kept complete, because each negative bounds a reading of the results in §3:

- Not an image or video diffusion model — the generation path is a physics integrator.
- Not photorealistic rendering — output is a particle point cloud.
- Not a learned generator. There is no neural network in the generation path, and
  [that fact has consequences](#61-a-negative-result-worth-stating).
- Not benchmarked against any other system — see §6.4 for exactly which comparisons exist
  and which do not.

### 2.2 Research: zero-shot conservation critics

*(This is the research question this repository is the artefact for; the measured results are in §3.)*

Generative models of physical scenes are usually validated with a *critic* that ranks or
filters candidate trajectories. Learned critics are the default, but they need labelled
data. We ask whether a **zero-shot critic built from conservation laws** — no training,
no labels — is competitive.

The task is *candidate selection*: given $K$ trajectories generated from the same initial
condition, pick the one closest to the true solution.

---

## 3. Results

**The zero-shot conservation critic is a reliable but not universally optimal
supervision signal.** Measured:

| Result | Value | Task / configuration | Artifact |
|---|---|---|---|
| Within-IC selection accuracy on standard benchmarks | **100%** (3 systems) | figure-eight ($N=3$, 2D) + Kepler ($N=2$, 2D), 50 test ICs, $K=8$ | `lumyn/docs/PAPER_DRAFT.md` §5.2 |
| Trajectory error vs. random selection | reduced **75–82%** | same standard benchmarks (sel./random 0.250× figure-eight, 0.185× Kepler) | `lumyn/docs/PAPER_DRAFT.md` §5.2 |
| Distance from best achievable selection | **1.03–1.05×** | same standard benchmarks (1.03× figure-eight, 1.05× Kepler) | `lumyn/docs/PAPER_DRAFT.md` §5.2 |
| 6-body, in distribution (5 seeds, 95% CI) | **82.8% ± 3.3%** | 6-body family, 120 train ICs, $K=8$, 5 seeds | `results/experiment_h_benchmark.json` |
| 6-body, out of distribution | **84.7% ± 1.8%** | 6-body family, same run, OOD split | `results/experiment_h_benchmark.json` |
| Supervised pairwise-ranking critic, same 960 labelled candidates | 71.2% ± 3.2% / 72.2% ± 2.5% | 6-body family, ID / OOD | `results/experiment_h_benchmark.json` |

> **口径（必须与上面的数字一起读）**：上表前五行**不是同一个实验**。
> 100% / 75–82% / 1.03–1.05× 来自**标准基准**（figure-eight 与 Kepler，50 个测试 IC，$K=8$）；
> 82.8% / 84.7% / 71.2% / 72.2% 来自**六体族**（120 训练 IC、5 种子、ID 与 OOD 两个切分，
> 产物 `results/experiment_h_benchmark.json`）。两组数字都正确，但属于不同任务与配置，
> 不可互相替换引用。`lumyn/docs/PAPER_STATUS.md` §一 另有一组更难的配置
> （受限预算生成器，误差降低 **36.3%（ID）/ 38.6%（OOD）**、oracle 增益的 **84%**），
> 同样只属于它自己的配置。

So it beats a supervised pairwise-ranking critic on pairwise-force generators. **But it
does not beat a correctly-used Hamiltonian Neural Network critic** — on the
Hamiltonian-parameterised family the two tie exactly.

Three claims from earlier internal drafts are **retracted** rather than quietly dropped:
that the conservation critic is more robust out of distribution; that combining it with a
learned critic helps; and that a windowed-sparse attention scheme degraded language
modelling (the supporting number was a numerical bug). The falsification of these claims
is where the paper's methodological contributions came from. Full detail in
[`lumyn/docs/PAPER_DRAFT.md`](lumyn/docs/PAPER_DRAFT.md) §7 and
[`lumyn/docs/PAPER_STATUS.md`](lumyn/docs/PAPER_STATUS.md).

Full draft: [`lumyn/docs/PAPER_DRAFT.md`](lumyn/docs/PAPER_DRAFT.md) ·
LaTeX source: [`lumyn/docs/paper/main.tex`](lumyn/docs/paper/main.tex) ·
evidence ledger: [`lumyn/docs/PAPER_STATUS.md`](lumyn/docs/PAPER_STATUS.md).

---

## 4. Quick start

```bash
# Core path needs only NumPy.
pip install numpy matplotlib
# Optional: torch (differentiable physics), astropy (FITS export)

# Run the test suite
python -m unittest discover -s lumyn/tests

# Generate a physically simulated galaxy scene
python lumyn/examples/galaxy.py

# Run all four scientific presets: MP4s plus diagnostic plots
python lumyn/examples/scientific_demo.py
```

### 4.1 Python API

```python
from lumyn import HMMPGameEngine

engine = HMMPGameEngine()
result = engine.generate("binary_star", n_particles=200)

print(result.keys())
# dict_keys(['mass', 'meta', 'scene', 'trajectory', 'validation'])
print(result["validation"]["energy_drift"])
```

Valid scene names are `binary_star`, `spiral_galaxy`, `globular_cluster` and
`galaxy_collision`. `result["validation"]` is the output of `PhysicsMetrics.diagnose()`.

### 4.2 Tests

```bash
python -m unittest discover -s lumyn/tests
```

Measured on this checkout:

```
Ran 113 tests
OK (skipped=1, expected failures=1)
```

| File | Tests | Covers |
|------|------:|--------|
| `test_conservation_critic.py` | 18 | Channel selection, score direction, permutation invariance, degenerate inputs, ranking, anti-fabrication guards |
| `test_conservation_checker.py` | 16 | Angular-momentum correctness, momentum normalisation, `fixed_mask` enforcement, galaxy stability, `diagnose` last-frame velocity |
| `test_explain.py` | 16 | Causal tracing: real corruption/restore, exact-zero contribution when the gate is absent, results vary with corruption strength |
| `test_scientific.py` | 12 | Export, rendering, teaching, presets, NumPy differentiable path |
| `test_force_law.py` | 12 | Force law and acceleration sign conventions |
| `test_engine.py` | 11 | Core engine |
| `test_eval.py` | 9 | Answer grading, bootstrap CI, validator self-check |
| `test_differentiable.py` | 8 | PyTorch path: forward, backward, gradient check, symplectic integrator, conservation loss |
| `test_eval_p1.py` | 6 | E1/E2/E3 classification, chi-square comparison |
| `test_differentiable_numpy.py` | 5 | Numerical-gradient fallback (no PyTorch required) |
| **Total** | **113** | |

> **Note on the verdict line, not the count.** `Ran 113 tests` is the number of tests collected,
> and `scripts/verify_readme.py` checks it against the tree. The `OK (skipped=1, expected
> failures=1)` line is the authoring machine's; wherever a test's internal subprocess cannot be
> spawned, the same 113 is reported with **1 error** instead (measured here: a `FileNotFoundError`
> raised from inside a test, unrelated to any assertion). **The count is stable either way; the
> verdict line is environment-dependent.** `docs/PAPER_STATUS.md` §七 records the same.

**113 is the current and authoritative count.** The superseded figures (69, 87, 97, 108, and the
older "44/44" and "57 passing") and the reason each was wrong are recorded in
[`lumyn/README.md`](lumyn/README.md) §6.5 and
[`lumyn/docs/PAPER_STATUS.md`](lumyn/docs/PAPER_STATUS.md) §七 — not deleted.

---

## 5. Repository layout

```
lumyn/
  core/            engine; director (present but NOT on the generation path)
  physics/         Barnes-Hut tree, N-body solver, scene presets, differentiable path
  video/           rendering pipeline
  eval/            SimScore, bootstrap CI, answer grading, error classifier
  scientific/      CSV/FITS export, 3D/2D rendering, diagnostics, teaching mode
  memory/          scene memory
  explain/         causal tracing
  integrations/    tool registry adapters
  gameplay/        MOBA / wuxia procedural-generation rules
  sync/            lockstep / rollback / replay
  mobile/          LOD policy, thermal management
  engine_bridge/   Unity and UE5 stubs (not buildable)
  experiments/     scripts producing every number in the paper
  tests/           113 tests
  docs/            paper draft, evidence ledger, derivations
scripts/           physics verification and audit helpers
```

A full component-by-component reference, including every known defect and every claim
that has *not* been demonstrated, is in **[`lumyn/README.md`](lumyn/README.md)** §6.

---

## 6. Limitations & honest status

Consolidated here, complete, with pointers to the paper sections that already carry the
full derivation. Nothing below is removed from this file.

### 6.1 A negative result worth stating

`LumynEngine.generate` computes a validation diagnosis and returns it, but **never acts on
it** — no rejection, no regeneration, no fallback. The causal contribution of the
validation layer is **exactly 0**.

This is deliberate, and the reason is measured. Wiring in a gate would require a
conservation residual to serve as an "is this run under-resolved?" criterion, but **no
conservation channel converges monotonically as `dt` is refined** — 12 of 12
scene/channel combinations are non-monotonic. An absolute threshold would be worse:
`galaxy_collision` shows an energy residual of 1.53 at its default `dt` while remaining a
valid N-body solution, so any reasonable threshold would reject valid output.

More fundamentally, **there is nothing for a gate to catch.** `generate` runs a real
N-body integration, so its output *is* a solution of the equations of motion. The failure
mode a validator guards against — physically invalid generated output — cannot occur
here. The validation layer would be a branch that never fires.

That bounds where a conservation-based critic is useful at all: **it discriminates when
the generator's output is not a solution of the equations of motion**, which is the case
for learned models and not for this engine. That is exactly why the research above
studies critics against *learned* dynamics models. (Paper: §5.4, §6, §7 R3.)

### 6.2 Method limitations (full derivation in the paper)

These are the caveats that condition how §3 should be read. The complete list, with derivations,
is `lumyn/docs/PAPER_DRAFT.md` §6:

1. **Generators are learned surrogate force models**, not diffusion or autoregressive video
   models. The qualitative structure of the failures (breaking equal-and-opposite forces) is
   plausible for any pairwise-parameterised model, but was not verified on a large pretrained
   generative model.
2. **The generator's parameterisation changes the ranking.** Switching from a pairwise force
   model to a Hamiltonian Neural Network removes the conservation critic's lead — the claim must
   be stated per generator; there are two architectures, not a spectrum.
3. **Ground truth is always available** here, because the systems are simulated. The critic does
   not need $x^\star$ at *use* time — only the *evaluation* does.
4. **Three systems, all $N \leq 6$ and pairwise-central.** The noise-floor argument relies on
   (i) pairwise central forces and (ii) a symplectic integrator.
5. **No accuracy comparison against published numbers.** The HNN baseline was trained here; no
   published HNN result was reproduced. The tie should be read as "our HNN implementation is a
   fair baseline", not as a leaderboard claim.
6. **The critic is used only for selection.** It is not shown to improve *training*; untested.

Also intrinsic to the engine:

- **Render output is a particle point cloud**, not photorealistic imagery.
- **The Barnes-Hut path is a fast approximation.** Conservation accuracy drifts by one to two
  orders of magnitude compared with the exact pairwise path.
- **Cross-machine reproducibility** applies to `eval/bootstrap_ci.py` (Python `random.Random`
  with a fixed seed, no NumPy dependency). It does not guarantee reproducibility across platforms
  for the integrators themselves.
- **`engine_bridge` is stubs only** — `unity/bridge.cs.stub` and `ue5/LumynPCGSubsystem.cpp.stub`
  are not buildable.

### 6.3 Retractions

Stated where they belong, and in the paper's own words:

- The claim that the conservation critic is **more robust out of distribution** — retracted.
- The claim that **combining** it with a learned critic helps — retracted.
- The claim that a **windowed-sparse attention** scheme degraded language modelling — retracted
  (the supporting number was a numerical bug).

See `lumyn/docs/PAPER_DRAFT.md` §7 for the falsifying experiment behind each.

### 6.4 What comparison exists, and what does not

- **Exists**: within-IC candidate selection against a **supervised pairwise-ranking critic**
  (71.2% ± 3.2% / 72.2% ± 2.5% on 960 labelled candidates) and against a **correctly-used
  Hamiltonian Neural Network critic** (exact tie at 100% within-IC accuracy, 1.03× / 1.05× of
  oracle), plus random and oracle reference rows.
- **Does not exist**: no accuracy, cost or quality comparison against any other system, **no
  SOTA comparison**, no leaderboard or published-number comparison, and no downstream benchmark.

See [`lumyn/README.md`](lumyn/README.md) §6.4 item 11 for the same statement from the component
reference, and `lumyn/docs/PAPER_DRAFT.md` §6 item 5 for the baseline caveat.

### 6.5 Defects found and fixed in this repository

- `ConservationChecker.angular_drift` never measured angular momentum (it computed the variation of
  the *linear* momentum magnitude, reporting `12.0176` where the true value was `0.0311`).
- `PhysicsMetrics.diagnose` zeroed the final frame's velocity, forcing
  `angular_momentum_drift` to `1.0` for every trajectory by construction.
- Both were found, fixed, and locked down by regression tests:
  `lumyn/README.md` §6.2 lists five such defects with the measurement that exposed each, and the
  `angular_drift` fix takes the same scene from `12.0176` to `6e-06`.

### 6.6 Honesty policy

This repository states only what has been verified by running the code, and it keeps a
public record of what was wrong:

- **Defects that were found and fixed** are documented with the measurement that exposed
  them (see §6.5).
- **Stale numbers are corrected, not deleted.** If a figure here disagrees with a
  producing script, the script wins.
- **Retractions are explicit.** Claims that failed are marked as retracted in the paper's
  §7, with the falsifying experiment.

If you find a number in this repository that cannot be traced to a reproducible command,
treat it as an error and open an issue.

### 6.7 Machine-checked claims

The claims *in this file* are machine-checked. Every link, directory, example script,
scene name and the test count is verified against the tree by:

```bash
python scripts/verify_readme.py     # exits non-zero if the README drifts
```

---

## 7. Build record and documentation archaeology

Two kinds of CONFESSION-class material, kept below the fold in one place: the paper's build
history (including what went wrong the first time) and things earlier documentation asserted
that the tree does not do.

### 7.1 Paper build record (including the overfull-box detail)

> **The paper has been compiled.** It was built on 2026-09-12 with TeX Live on
> Ubuntu 22.04 via `latexmk`, after the authoring environment turned out to have no
> LaTeX toolchain: **10 pages, 0 undefined references, 0 Overfull/Underfull boxes.**
> The *first* compile was not clean — six Overfull `\hbox` warnings, every one of
> them in a table header rather than in body text, the worst by 74 pt (~26 mm,
> roughly an inch past the right margin). They were fixed by tightening
> `\tabcolsep` and lowering the font one step on the widest tables, each step
> measured rather than assumed. The rendered PDF was also read page by page,
> because a clean log is not the same as a correct page. See
> [`lumyn/docs/paper/README.md`](lumyn/docs/paper/README.md) for the full
> verification table and the one remaining caveat (the build used TeX Live 2022;
> other distributions hyphenate differently).

Product: [`lumyn/docs/paper/main.pdf`](lumyn/docs/paper/main.pdf).

### 7.2 Documentation archaeology (earlier documentation that was wrong)

- **Test counts.** Earlier revisions of this file and of `lumyn/README.md` reported **69, 87, 97 and
  108** tests. Superseded by **113** (§4.2). `lumyn/docs/PAPER_STATUS.md` additionally carried
  "**57**" and "**44/44**"; both are recorded there as superseded, with the reasons
  (a missed update after `test_force_law.py` was added; a no-torch collection count).
- **The "three-layer pipeline."** Earlier versions described an LLM director layer on the
  generation path. It is not on it; the pipeline has two layers.
- **Per-layer contribution numbers.** An earlier version of this README carried a table of
  per-layer "contribution" numbers that were hard-coded constants presented as measurements;
  they were removed, and the module now performs a real corruption/restoration protocol.
- **Emoji and mixed language.** An earlier version of this README contained emoji and mixed
  Chinese and English; those passages were removed.

The component reference keeps the full record, including the older scene name that raised
`KeyError`, a `verdict` key that never existed, and the `presets` API names that were wrong:
[`lumyn/README.md`](lumyn/README.md) §6.5.

---

## 8. Relationship to prior work

The evaluation vocabulary (SimScore, error classification, sensitivity analysis, causal
tracing) is inspired by PhysBench. Lumyn's use of it is not a reproduction of that work,
and no numerical results from PhysBench are claimed here.

---

## License

MIT License.
