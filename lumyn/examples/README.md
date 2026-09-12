# Lumyn examples

End-to-end example scripts. All six run to completion on this checkout, and each one
fixes up `sys.path` itself, so plain `python lumyn/.../script.py` works.

| Script | Output | Purpose |
|--------|--------|---------|
| `physics/make_figure.py` | `lumyn_output/gradient_guided_optimization.png` | Loss and energy-drift convergence curves |
| `scientific/examples/diff_figure.py` | figure | Gradient-guided inversion against a target |
| `examples/galaxy.py` | console diagnostics | Generate one physically-simulated scene and print conservation drift |
| `examples/guided_generation_demo.py` | CSV + convergence figure | Differentiable inversion from a target shape (PyTorch, NumPy fallback) |
| `examples/scientific_demo.py` | 4 × MP4 + frame previews + CSVs + diagnostic figures | All four presets end to end, plus teaching and differentiable blocks |
| `scientific/examples/end_to_end.py` | 4 × MP4 + CSVs + summary table | Preset evolution, diagnostics, rendering, export |

All artefacts are written to `<repo>/lumyn_output/`.

---

## What was broken, and why

An earlier revision of this file listed four of these scripts as non-functional. They
were: `examples/galaxy.py`, `examples/guided_generation_demo.py`,
`examples/scientific_demo.py` and `scientific/examples/end_to_end.py`. The causes were
not flaky environment issues but stale API usage, and each is worth recording because the
same patterns recur:

1. **`sys.path` offset by one level.** Scripts did
   `sys.path.insert(0, dirname(dirname(__file__)))`, which resolves to `lumyn/` rather
   than the repository root, so `import lumyn` raised `ModuleNotFoundError`. Three of the
   four had this.
2. **Calls to APIs that do not exist.** `examples/galaxy.py` passed `img_size=` to
   `LumynEngine`, passed a *Chinese sentence* as `scene_type`, and read
   `result["validation"]["conservation"]` and `result["video"]`. None of those exist:
   `generate` dispatches to named presets and returns `trajectory`, `mass`, `scene`,
   `validation`, `meta`; `validation` is the output of `PhysicsMetrics.diagnose()` and has
   no `conservation` or `verdict` key.
3. **Wrong container type.** `ScientificVisualizer` and `ScientificExporter.to_csv` take a
   *dictionary* of trajectories (`{"pos":…, "vel":…, "mass":…}`), not a bare array.
   `end_to_end.py` passed arrays.
4. **A torch/NumPy mix-up.** `scientific_demo.py` documented "use the NumPy version
   throughout" but imported `DifferentiableNBody` from `scientific/differentiable.py`
   whenever torch was installed, then fed it NumPy arrays — producing
   `AttributeError: 'numpy.ndarray' object has no attribute 'unsqueeze'`. The NumPy
   implementation is now imported explicitly under its own name.
5. **Dead names and wrong paths.** `end_to_end.py` referenced `ROOT` while defining
   `_ROOT`; `scientific_demo.py` wrote its output to `<repo>/../lumyn_output`, i.e. outside
   the repository.
6. **Non-existent helpers.** `presets.get_preset()` (the real name is `presets.get()`) and
   the module-level `metrics.center_of_mass_drift()` (the real method is
   `PhysicsMetrics.center_of_mass`).

## Dependencies

- Required: `numpy`, `matplotlib`
- Optional: `torch` (differentiable optimisation), `astropy` (FITS export),
  `Pillow` (frame extraction), `opencv-python` (video writing)

Each script degrades gracefully when an optional dependency is missing: the rendering,
frame-extraction and plotting blocks are wrapped, and the differentiable demo falls back
to central differences.
