"""Audit the SIGN of every acceleration implementation in Lumyn.

Motivation: reading the source is not reliable here. We already found one
comment describing the opposite of what the code does
(`differentiable.py` L111 annotates `d[i,j] = pos[i] - pos[j]` while the
expression actually yields `pos[j] - pos[i]`), and the two differentiable paths
use opposite sign conditions for the same `repulsive` flag.

So we test them all the same way: two masses on the x-axis, and the left one
must be pulled in +x (toward the right one). This is a pure sign check -- no
integrator, no tolerance, nothing to tune.

    python scripts/audit_force_sign.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

POS = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
MASS = np.array([1.0, 1.0])
EXPECTED = "+x  (toward the other mass)"


def verdict(acc_x):
    if acc_x > 1e-9:
        return "ATTRACTIVE", True
    if acc_x < -1e-9:
        return "REPULSIVE", False
    return "ZERO", False


def report(name, acc, path):
    acc = np.asarray(acc, dtype=float)
    if acc.ndim != 2 or acc.shape[0] < 2:
        print("  %-42s SKIP (unexpected shape %s)" % (name, acc.shape))
        return None
    ax = float(acc[0, 0])
    v, ok = verdict(ax)
    print("  %-42s acc[0,0]=%+9.5f  %-11s %s   %s"
          % (name, ax, v, "OK" if ok else "WRONG", path))
    return ok


def main():
    print("=" * 100)
    print("Force-sign audit. Two masses at x=-0.5 and x=+0.5.")
    print("Gravity requires the left mass to accelerate in %s" % EXPECTED)
    print("=" * 100)

    results = []

    # 1. Barnes-Hut tree (the path the scientific presets actually use)
    try:
        from lumyn.physics.barnes_hut import BarnesHutTree
        tree = BarnesHutTree(max_leaf=16, max_depth=24)
        tree.build(POS, MASS, center=np.zeros(3), size=2.0)
        results.append(report("BarnesHutTree.compute_acceleration",
                              tree.compute_acceleration(theta=0.5),
                              "physics/barnes_hut.py"))
    except Exception as e:
        print("  BarnesHutTree: ERROR %s" % e)

    # 2. NBodySimulator (wraps the tree) -- used by scientific/nbody.py and presets.py
    try:
        from lumyn.physics.nbody import NBodySimulator
        sim = NBodySimulator(G=1.0, theta=0.5, softening=1e-4)
        results.append(report("NBodySimulator._acceleration",
                              sim._acceleration(POS.copy(), MASS.copy()),
                              "physics/nbody.py"))
    except Exception as e:
        print("  NBodySimulator: ERROR %s" % e)

    # 3. differentiable (torch)
    try:
        import torch
        from lumyn.physics.differentiable import DifferentiableNBody
        d = DifferentiableNBody()
        acc = d.compute_acceleration(torch.tensor(POS), torch.tensor(MASS))
        results.append(report("DifferentiableNBody.compute_acceleration",
                              acc.detach().cpu().numpy(),
                              "physics/differentiable.py"))
    except Exception as e:
        print("  DifferentiableNBody: ERROR %s" % e)

    # 4. differentiable_numpy
    try:
        from lumyn.physics.differentiable_numpy import DifferentiableNBodyNumpy
        d = DifferentiableNBodyNumpy()
        acc = d.compute_acceleration(POS.copy(), MASS.copy())
        results.append(report("DifferentiableNBodyNumpy.compute_acceleration",
                              acc, "physics/differentiable_numpy.py"))
    except Exception as e:
        print("  DifferentiableNBodyNumpy: ERROR %s" % e)

    # 5. scientific/differentiable_numpy (a second copy)
    try:
        from lumyn.scientific import differentiable_numpy as sdn
        fns = [n for n in dir(sdn) if "acc" in n.lower() or "NBody" in n]
        print("  scientific/differentiable_numpy exports: %s" % fns)
    except Exception as e:
        print("  scientific/differentiable_numpy: ERROR %s" % e)

    # 6. scientific/nbody (thin wrapper over physics/nbody)
    try:
        from lumyn.scientific.nbody import NBodySim
        s = NBodySim("custom", n_steps=1, mass=MASS.copy())
        pos, vel, mass = POS.copy(), np.zeros_like(POS), MASS.copy()
        p2, v2 = s.sim.step(pos, mass, vel, dt=1e-6)
        acc_like = (v2 - vel) / 1e-6
        results.append(report("scientific.NBodySim -> sim.step (dt=1e-6)",
                              acc_like, "scientific/nbody.py"))
    except Exception as e:
        print("  scientific.NBodySim: ERROR %s" % e)

    print()
    print("=" * 100)
    good = [r for r in results if r is True]
    bad = [r for r in results if r is False]
    print("attractive (correct): %d      repulsive (wrong): %d" % (len(good), len(bad)))
    if bad:
        print()
        print("!! The paths the scientific presets depend on are REPULSIVE.")
        print("!! Trajectories from those paths are not gravitational and cannot")
        print("!! be used as physical ground truth.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
