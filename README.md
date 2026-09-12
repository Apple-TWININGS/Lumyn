# Lumyn

A physically-grounded scene generation engine: **simulate the physics first, render
second**. Every frame comes from a real Barnes-Hut N-body integration, and trajectories
are checked against conservation laws rather than accepted on appearance.

This repository is also the artefact for a research question: *can a zero-shot critic
built from conservation laws replace a learned one?* The measured answer is
[below](#research-zero-shot-conservation-critics).

---

## What this is

Lumyn is **not** an image or video diffusion model. It is a pipeline that:

1. **Simulates** a scene with a Barnes-Hut N-body solver (`O(N log N)`).
2. **Measures** the trajectory against conservation laws (energy, momentum, angular
   momentum, centre of mass).
3. **Optionally renders** the particle trajectory to video.

Scenes are deterministic given a seed. On the same platform, results reproduce
bit-for-bit.

There is also a **differentiable** physics path (PyTorch) supporting gradient-based
recovery of initial conditions from a target final state, plus a NumPy
numerical-gradient fallback for environments without PyTorch.

### What this is not

- Not photorealistic rendering — output is a particle point cloud.
- Not a learned generator. There is no neural network in the generation path, and
  [that fact has consequences](#a-negative-result-worth-stating).
- Not benchmarked against any other system.

---

## Research: zero-shot conservation critics

Generative models of physical scenes are usually validated with a *critic* that ranks or
filters candidate trajectories. Learned critics are the default, but they need labelled
data. We ask whether a **zero-shot critic built from conservation laws** — no training,
no labels — is competitive.

The task is *candidate selection*: given $K$ trajectories generated from the same initial
condition, pick the one closest to the true solution.

**The zero-shot conservation critic is a reliable but not universally optimal
supervision signal.** Measured:

| Result | Value |
|---|---|
| Within-IC selection accuracy on standard benchmarks | **100%** (3 systems) |
| Trajectory error vs. random selection | reduced **75–82%** |
| Distance from best achievable selection | **1.03–1.05×** |
| 6-body, in distribution (5 seeds, 95% CI) | **82.8% ± 3.3%** |
| 6-body, out of distribution | **84.7% ± 1.8%** |
| Supervised pairwise-ranking critic, same 960 labelled candidates | 71.2% ± 3.2% / 72.2% ± 2.5% |

So it beats a supervised pairwise-ranking critic on pairwise-force generators. **But it
does not beat a correctly-used Hamiltonian Neural Network critic** — on the
Hamiltonian-parameterised family the two tie exactly.

Three claims from earlier internal drafts are **retracted** rather than quietly dropped:
that the conservation critic is more robust out of distribution; that combining it with a
learned critic helps; and that a windowed-sparse attention scheme degraded language
modelling (the supporting number was a numerical bug). The falsification of these claims
is where the paper's methodological contributions came from.

Full draft: [`lumyn/docs/PAPER_DRAFT.md`](lumyn/docs/PAPER_DRAFT.md) ·
LaTeX source: [`lumyn/docs/paper/main.tex`](lumyn/docs/paper/main.tex) ·
evidence ledger: [`lumyn/docs/PAPER_STATUS.md`](lumyn/docs/PAPER_STATUS.md).

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

---

## Quick start

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

### Python API

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

---

## A negative result worth stating

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
studies critics against *learned* dynamics models.

---

## Tests

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

---

## Repository layout

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
that has *not* been demonstrated, is in **[`lumyn/README.md`](lumyn/README.md)**.

---

## Honesty policy

This repository states only what has been verified by running the code, and it keeps a
public record of what was wrong:

- **Defects that were found and fixed** are documented with the measurement that exposed
  them — e.g. `ConservationChecker.angular_drift` never measured angular momentum (it
  computed the variation of the *linear* momentum magnitude, reporting `12.0176` where
  the true value was `0.0311`); and `PhysicsMetrics.diagnose` zeroed the final frame's
  velocity, forcing `angular_momentum_drift` to `1.0` for every trajectory by
  construction.
- **Stale numbers are corrected, not deleted.** If a figure here disagrees with a
  producing script, the script wins.
- **Retractions are explicit.** Claims that failed are marked as retracted in the paper's
  §7, with the falsifying experiment.

If you find a number in this repository that cannot be traced to a reproducible command,
treat it as an error and open an issue.

The claims *in this file* are machine-checked. Every link, directory, example script,
scene name and the test count is verified against the tree by:

```bash
python scripts/verify_readme.py     # exits non-zero if the README drifts
```

---

## License

MIT License.

---

## Relationship to prior work

The evaluation vocabulary (SimScore, error classification, sensitivity analysis, causal
tracing) is inspired by PhysBench. Lumyn's use of it is not a reproduction of that work,
and no numerical results from PhysBench are claimed here.
