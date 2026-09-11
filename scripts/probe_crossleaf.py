"""Is there a SECOND force bug beyond the sign?

The far-field loop is
    for cell in self._cells:
        if not cell.children:
            continue
so LEAF cells are never treated as far-field sources. Cross-leaf pairs are
therefore accounted for only if some ANCESTOR cell passes the opening criterion.
When the nearest common ancestor is too close to open, that pair's force is
dropped entirely -- it is in neither the exact within-leaf sum nor the far-field
sum.

Two particles always share a leaf (max_leaf=16), which is why the analytic
two-body orbit is perfect while an N=200 cloud shows ~90% error.

Test: force two particles into SEPARATE leaves with max_leaf=1 and check the
force against the known 1/r^2 value.

    python scripts/probe_crossleaf.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lumyn.physics.barnes_hut import BarnesHutTree  # noqa: E402


def exact(pos, mass, eps=1e-8):
    d = pos[None, :, :] - pos[:, None, :]
    r2 = (d * d).sum(axis=2) + eps
    inv = r2 ** (-1.5)
    np.fill_diagonal(inv, 0.0)
    return (mass[None, :, None] * d * inv[:, :, None]).sum(axis=1)


def run(max_leaf, label):
    pos = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    mass = np.array([1.0, 1.0])
    tree = BarnesHutTree(max_leaf=max_leaf, max_depth=24)
    tree.build(pos, mass, center=np.zeros(3), size=2.0)
    acc = tree.compute_acceleration(theta=0.5)
    ex = exact(pos, mass)
    n_leaves = sum(1 for c in tree._cells if not c.children)
    n_cells = len(tree._cells)
    print("  %-34s cells=%3d leaves=%2d" % (label, n_cells, n_leaves))
    print("       tree  acc[0] = %s" % np.array2string(acc[0], precision=6))
    print("       exact acc[0] = %s" % np.array2string(ex[0], precision=6))
    ok = abs(acc[0, 0] - 1.0) < 1e-6
    print("       %s (expected acc[0,0] = +1.0)" % ("OK" if ok else "WRONG"))
    return ok


def main():
    print("=" * 84)
    print("Cross-leaf force: are nearby pairs in different leaves counted at all?")
    print("=" * 84)
    a = run(16, "max_leaf=16 (both in one leaf)")
    b = run(1, "max_leaf=1  (separate leaves)")
    print()
    print("=" * 84)
    if a and not b:
        print("CONFIRMED: the force is correct only while the pair shares a leaf.")
        print("Cross-leaf pairs are dropped -> the far-field loop skips leaves, and the")
        print("near-field loop only sums pairs WITHIN a leaf. Any pair whose nearest")
        print("common ancestor fails the opening criterion contributes nothing.")
        print()
        print("This is a second, independent defect from the sign error. It makes the")
        print("force systematically too weak wherever particles are close but in")
        print("different leaves -- i.e. exactly in the dense regions that matter.")
    elif a and b:
        print("no cross-leaf defect detected")
    print("=" * 84)
    return 0 if (a and b) else 1


if __name__ == "__main__":
    sys.exit(main())
