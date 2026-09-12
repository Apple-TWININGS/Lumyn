# Lumyn

A physically-grounded scene and video generation engine: simulate the physics first,
render second. Every frame comes from an N-body integration, and every result is
checked against conservation laws before it is accepted.

This README states only what has been verified by running the code. Claims that could
not be verified have been removed; see [Limitations and unverified claims](#limitations-and-unverified-claims).

---

## What this is

Lumyn is not an image or video diffusion model. It is a pipeline that:

1. **Simulates** a scene with a Barnes-Hut N-body solver.
2. **Validates** the resulting trajectory against conservation laws
   (energy, momentum, angular momentum, center of mass).
3. **Optionally renders** the particle trajectory to video.

Scenes are deterministic given a seed. Parameters are explicit and results are
reproducible bit-for-bit on the same platform.

The package also exposes a **differentiable** physics path (PyTorch) that supports
gradient-based recovery of initial conditions from a target final state, plus a
NumPy numerical-gradient fallback for environments without PyTorch.

---

## Architecture

```
  generation layer
    Barnes-Hut N-body solver (O(N log N))
    NBodySimulator / HMMPGameEngine
                 |
                 v
  validation layer
    ConservationChecker + SimScore + E1/E2/E3 classifier
                 |
                 v
  render layer
    scientific/ and video/ (particle point-cloud rendering)
```

The validation layer is intended to reject trajectories that violate conservation and
fall back to regenerating from the previous keyframe.

> **Note on "three layers".** Earlier versions of this document described a three-layer
> pipeline with an LLM "director" layer on top. That layer is **not wired into the
> generation path**: `LumynEngine.generate` dispatches to the scene presets and never
> invokes the director. The pipeline as it runs today has **two** layers (generation and
> validation). The director module exists in `core/director.py` but is not on the
> generation path.

### Directory layout

```
lumyn/
  core/            engine, director (not wired into generate)
  physics/         Barnes-Hut tree, N-body solver, scene presets, differentiable path
  video/           rendering pipeline
  eval/            SimScore, bootstrap CI, answer grading, error classifier
  memory/          scene memory
  explain/         causal tracing (see limitations)
  integrations/    tool registry adapters
  scientific/      CSV/FITS export, 3D/2D rendering, diagnostics
  gameplay/        MOBA / wuxia procedural-generation rules
  engine_bridge/   Unity and UE5 stubs
  sync/            lockstep / rollback / replay
  mobile/          LOD policy, thermal management
  docs/            PAPER_STATUS.md, RESEARCH_DIRECTION.md
  tests/           unit tests (10 files, 113 tests)
```

---

## Install and run

```bash
# Install dependencies. Only numpy is required for the core path.
pip install numpy matplotlib
# Optional extras: torch (differentiable physics), astropy (FITS export)

# Run the test suite
python -m unittest discover -s lumyn/tests -v

# Generate a physically simulated galaxy scene
python lumyn/examples/galaxy.py

# Run all four scientific presets and write MP4s plus diagnostic plots
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
`galaxy_collision`. Earlier documentation used `"galaxy"`, which raises
`KeyError: 'galaxy'`.

`result["validation"]` is the output of `PhysicsMetrics.diagnose()`. It contains
`energy_drift`, `energy_final`, `energies`, `angular_momentum_drift`, `angular_momenta`,
`com_drift`, `half_mass_radius_final` and `half_mass_radius_shrink`. **There is no
`verdict` key**; earlier documentation printed `result["validation"]["verdict"]`, which
does not exist.

> **`validation` is not currently usable as a check.** Two of its fields are broken:
> `diagnose()` computes velocities as `(traj[t+1] - traj[t]) / dt` using
> `min(t+1, T-1)`, so at the final frame the velocity is exactly zero. This forces
> `angular_momenta[-1] = 0`, which makes `angular_momentum_drift` equal to 1.0 for every
> trajectory by construction (measured: `0.9999999829`) and zeroes the kinetic energy in
> `energy_final`. `energy_drift` survives only because the scene's total energy is
> dominated by the potential term.
>
> `diagnose()` also contains a dead `fixed_mask` branch: it reads
> `getattr(mass, "fixed_mask", None)` from a NumPy array (always `None`), and the body
> that follows multiplies by zero.

---

## Scientific module (`scientific/`)

Positions Lumyn as a tool for physics teaching and research: what you see is a real
N-body integration, not a generated image.

Scenes are obtained from `presets.get(name, **kwargs)`, which returns a `SceneParams`.
Calling `.run(steps)` on it returns a `(trajectory, mass)` tuple.

```python
from lumyn.scientific import presets, metrics

params = presets.get("binary_star", n_particles=50)
trajectory, mass = params.run(steps=10)      # trajectory.shape == (11, 50, 3)

mt = metrics.PhysicsMetrics(G=params.G)
print(mt.energy_drift(trajectory, mass))     # 8.685500968250139e-09
```

| File | Class or function | Purpose |
|------|-------------------|---------|
| `presets.py` | `SceneParams`, `get()`, four scene constructors | Scene definition and evolution |
| `metrics.py` | `PhysicsMetrics`, `energy_drift()`, `momentum_conserved()` | Energy, angular momentum, center of mass, half-mass radius |
| `export.py` | `ScientificExporter.to_csv()`, `.to_fits()` | Trajectory export (FITS requires astropy) |
| `visualization.py` | `ScientificVisualizer.render_3d()`, `.render_2d_with_physics()` | Rendering |
| `frames.py` | `FrameExtractor.extract()`, `.make_grid()` | Frame extraction and contact sheets |
| `teaching.py` | `TeachingMode.compare_mass()`, `.compare_angular_velocity()` | Side-by-side parameter comparison |
| `differentiable.py` | `DifferentiableNBody`, `ConservationConstraint` | PyTorch differentiable path |

Two interface details that earlier documentation got wrong: the functions are
`presets.get()` / `presets.binary_star()`, **not** `presets.load()` or
`presets.get_preset()`; and `ScientificExporter.to_csv()` and `ScientificVisualizer()`
both expect a **dictionary** of trajectories, not a bare array.

See `scientific/README.md` for details, including a second class also named
`DifferentiableNBody` that behaves differently from the one in `physics/`.

---

## Tests

```
python -m unittest discover -s lumyn/tests
```

Current result, measured on this checkout:

```
Ran 113 tests
OK (skipped=1, expected failures=1)
```

Per file:

| File | Tests | Covers |
|------|------:|--------|
| `test_conservation_critic.py` | 18 | Conservation critic: channel selection, score direction, permutation invariance, degenerate inputs, ranking, anti-fabrication guards |
| `test_conservation_checker.py` | 16 | Angular-momentum correctness, momentum normalization, `fixed_mask` enforcement, galaxy scene stability, `diagnose` last-frame velocity |
| `test_explain.py` | 16 | Causal tracing: real corruption/restore, exact-zero contribution when the validation gate is absent, results vary with corruption strength |
| `test_scientific.py` | 12 | Export, rendering, teaching, presets, NumPy differentiable path |
| `test_force_law.py` | 12 | Force law and acceleration sign conventions |
| `test_engine.py` | 11 | Core engine |
| `test_eval.py` | 9 | Answer grading, bootstrap CI, validator self-check |
| `test_differentiable.py` | 8 | PyTorch differentiable path: forward, backward, gradient check, symplectic integrator, conservation loss |
| `test_eval_p1.py` | 6 | E1/E2/E3 classification, chi-square comparison |
| `test_differentiable_numpy.py` | 5 | Numerical-gradient fallback (no PyTorch required) |
| **Total** | **113** | |

The suite passes. Earlier revisions of this document reported 69, 87, 97 and 108 tests,
with one persistent failure in `test_galaxy_non_divergent`. That failure is now fixed, and
it was never a threshold problem — see the limitations section for the three real defects
behind it. The move from 108 to 113 is `test_conservation_checker.py` growing from 11 to
16 tests as the four defects below were pinned down by regression tests.

> **On the older "44/44" and "57 passing" figures.** `test_differentiable*.py`
> (13 tests) are guarded by `@unittest.skipUnless(_HAS_TORCH, ...)`. In an environment
> without PyTorch those 13 are skipped and discovery collects 44 tests, which is where
> the "44/44" figure came from. Later, `test_force_law.py` (12 tests) was added and the
> documented total was not updated, giving "57" instead of 69. Neither figure was
> measured on the current tree with PyTorch installed.

---

## Limitations and unverified claims

This section exists to keep the project honest. Everything below is either a known
defect or a claim that has not been demonstrated.

### Defects that have been fixed

1. **`ConservationChecker.angular_drift` did not measure angular momentum.**
   It computed the variation of the magnitude of the *linear* momentum vector,
   `max | ||P(t)|| - ||P(0)|| | / ||P(0)||`, not `L = sum m (r x v)`. On a test galaxy
   it reported 12.0176 while the true relative change in angular momentum was 0.0311.
   **Fixed**: it now computes the real angular-momentum drift, and the same scene
   reports `6e-06`. Covered by `test_conservation_checker.py`.

2. **`galaxy()`'s `fixed_mask` was never enforced.** `galaxy()` returns a boolean mask
   marking a central body, and `ConservationChecker.check` accepts it, but
   `NBodySimulator` had no `fixed_mask` parameter and never pinned any particle, so the
   nominal "fixed" central body was integrated freely.
   **Fixed**: `step()` and `simulate()` now take `fixed_mask` and zero both the
   acceleration and the velocity of masked particles. Verified: a masked particle's
   displacement is `< 1e-12` over the run.

3. **The `galaxy` scene diverged.** Maximum radius grew from about 1.0 to about 550.
   Two independent causes, both fixed:
   - the circular-velocity law used a central mass of `10.0` while the actual central
     body had mass `1000.0`, leaving every disk particle an order of magnitude too slow;
   - the inner orbit was numerically unresolved — about **1.3 integration steps per
     orbit** at `dt=0.005`, which is guaranteed to be unstable.

   Stability is governed by resolution, not by mass. Measured sweep
   (`experiments/probe_galaxy_params.py`):

   | Configuration | r_max growth | Steps per inner orbit | Result |
   |---|---|---|---|
   | M=1000, r∈[0.1, 1] | 46.7x | 1.3 | diverges |
   | M=200, r∈[0.3, 1] | 14.2x | 14.6 | diverges |
   | M=50, r∈[0.5, 2] | 4.84x | 62.8 | diverges |
   | **M=20, r∈[0.8, 3]** | **1.16x** | **201** | **stable** |

   The scene now uses the last row. `test_galaxy_non_divergent` passes; the previous
   failure was a real defect, not a bad threshold.

4. **`PhysicsMetrics.diagnose` computed the final frame's velocity as zero.**
   Velocities were derived as `(traj[min(t+1, T-1)] - traj[t]) / dt`, which at the last
   frame evaluated to `(traj[T-1] - traj[T-1]) / dt = 0`. Consequences:
   `angular_momenta[-1]` was forced to zero, so `angular_momentum_drift` was
   `|0 - L0| / |L0| = 1.0` for **every** trajectory regardless of physics
   (measured: `0.9999999829`); and `energy_final` was computed with all kinetic energy
   removed. **Fixed**: the last frame reuses the previous frame's velocity, and the
   dead `fixed_mask` branch (which read an attribute from a NumPy array and then
   multiplied by zero) has been removed.

5. **`PhysicsMetrics.momentum_conserved` checked the center of mass, not the momentum.**
   It compared `sum(m_i r_i)` at the first and last frames. **Fixed**: it now applies a
   real momentum criterion normalised by `sum(m_i |v_i|)`, and the old behaviour is
   preserved under the honest name `center_of_mass_conserved()`. The distinction matters
   in one direction only — a uniformly translating system conserves momentum while its
   centre of mass keeps moving, and the old implementation rejected exactly that case.
   Covered by `test_conservation_checker.py`.

### Known implementation defects

7. **`DifferentiableNBody.__init__` detaches its inputs.** The constructor stores
   `nn.Parameter(self._to_tensor(pos).clone().detach()...)`. If you construct a new
   system inside an optimization loop, gradients never reach the tensors you are
   updating, and PyTorch silently skips parameters with `grad is None`. Construct the
   system **once** and optimize `system.parameters()`. Measured: the naive pattern
   produced **0 backward passes out of 50** reaching the caller's tensor.

   `examples/guided_generation_demo.py` demonstrates the correct pattern.

8. **`engine_bridge` is stubs only.** `unity/bridge.cs.stub` and
   `ue5/LumynPCGSubsystem.cpp.stub` are marked `TODO` and are not buildable.

### Claims not yet demonstrated

9. **Causal tracing has been rewritten to be real, and it exposes a design gap.**
   Earlier versions returned hard-coded constants
   (`decay = {"director": 0.12, "generate": 0.30, "validate": 0.55}`) and the README
   presented them as measurements. Those numbers have been removed, and the module now
   performs the actual corruption/restoration protocol.

   The measured result is uncomfortable: `LumynEngine.generate` computes
   `validator.diagnose(...)` and returns it, but **never acts on it** — there is no
   rejection, no regeneration, no fallback. Consequently:
   - the **director** layer is not on the generation path at all, so its contribution
     is not measurable and is reported as `None` (not as a number);
   - the **validation** layer has a causal contribution of exactly **0**.

   ```
   python -m lumyn.explain.causal_tracing            # faithful to the engine
   # scene            director   generate   validate
   # spiral_galaxy   n/a          7.5400     0.0000
   ```

   A gate can be enabled with `--gate`, which shows what the validation layer
   *would* contribute if its verdict were acted on:

   ```
   python -m lumyn.explain.causal_tracing --gate
   # spiral_galaxy   n/a          7.5400     7.5400
   ```

   **Wiring that gate into `LumynEngine.generate` is deliberately not done.**
   The reason is a measured negative result: no conservation channel converges
   monotonically as `dt` is refined (12 of 12 scene/channel combinations are
   non-monotonic), so a conservation residual cannot serve as an
   "is this run under-resolved?" criterion. An absolute threshold would be worse:
   `galaxy_collision` already shows an energy residual of 1.53 at its default `dt`
   while still being a valid N-body solution, so any reasonable threshold would
   reject valid output.

   More fundamentally, **there is nothing for a gate to catch.** `generate` runs a
   real N-body integration, so its output *is* a solution of the equations of
   motion; the failure mode a validator would guard against — physically invalid
   generated output — cannot occur here. The validation layer would be a branch
   that never fires.

   The claim that "the validation layer falls back to the previous keyframe when a
   check fails" describes a mechanism that a *learned* generator needs. Lumyn does
   not contain a learned generator, so that mechanism has no object. This also
   bounds where a conservation-based critic is useful at all: it discriminates when
   the generator's output is *not* a solution of the equations of motion, which is
   the case for learned models and not for this engine.

10. **`docs/EXPERIMENT_TABLES.md` does not exist.** It was referenced by earlier versions
    of this document as the source of four result tables. No such file is present.

11. **No benchmark, baseline, or SOTA comparison exists.** No accuracy, cost, or
    quality comparison against any other system has been run.

12. **The E1/E2/E3 classifier is heuristic.** Rules that do not match fall through to
    `Unknown`. It has not been validated against human labels.

13. **Render output is a particle point cloud**, not photorealistic imagery.

14. **The Barnes-Hut path is a fast approximation.** Conservation accuracy drifts by
    one to two orders of magnitude compared with the exact pairwise path.

15. **Cross-machine reproducibility** applies to `eval/bootstrap_ci.py`, which uses
    Python's `random.Random` with a fixed seed and does not depend on NumPy. It does
    not guarantee reproducibility across platforms for the integrators themselves.

### Relationship to prior work

The evaluation vocabulary (SimScore, error classification, sensitivity analysis,
causal tracing) is inspired by PhysBench. Lumyn's use of it is not a reproduction of
that work, and no numerical results from PhysBench are claimed here.

---

## License

MIT License.

---

## Notes on this document

An earlier version of this README contained emoji, mixed Chinese and English, and a
table of per-layer "contribution" numbers that were hard-coded constants presented as
measurements. Those passages have been removed. If you find a number in this repository
that cannot be traced to a reproducible command, treat it as an error and open an issue.
