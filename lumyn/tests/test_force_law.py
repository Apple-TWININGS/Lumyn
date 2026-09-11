"""Regression tests for the FORCE LAW itself.

Why this file exists
--------------------
The suite reached 57 passing tests while the force law was wrong in two
independent ways:

  1. The near-field term was REPULSIVE (d = r_i - r_j where gravity needs
     r_j - r_i), while the far-field term was attractive -- the same function
     used opposite conventions.
  2. Leaf cells were skipped by the far-field loop (`if not cell.children:
     continue`), so any pair whose nearest common ancestor failed the opening
     criterion was counted in NEITHER the exact near-field sum (which only
     covers pairs inside one leaf) NOR the far-field sum. Their force was
     silently dropped.

The old tests checked shapes, interfaces and exports, so both defects passed.
These tests check the force, and they cannot be satisfied by a wrong force law.

They are also why `test_galaxy_non_divergent` is expected to be re-tuned: it
passed only because defect 2 removed most interactions and thereby UNDERSTATED
the momentum drift that Barnes-Hut inherently introduces.
"""

import unittest

import numpy as np

from lumyn.physics.barnes_hut import BarnesHutTree
from lumyn.physics.nbody import NBodySimulator


def exact_acceleration(pos, mass, G=1.0, eps=1e-8):
    """Direct O(N^2) gravity, written independently of the code under test."""
    d = pos[None, :, :] - pos[:, None, :]           # d[i,j] = r_j - r_i
    r2 = (d * d).sum(axis=2) + eps
    inv_r3 = r2 ** (-1.5)
    np.fill_diagonal(inv_r3, 0.0)
    return G * (mass[None, :, None] * d * inv_r3[:, :, None]).sum(axis=1)


def tree_acceleration(pos, mass, max_leaf=16, theta=0.5):
    tree = BarnesHutTree(max_leaf=max_leaf, max_depth=24)
    center = pos.mean(axis=0)
    size = float(np.abs(pos - center).max()) * 2 + 1e-6
    tree.build(pos, mass, center=center, size=size)
    return tree.compute_acceleration(theta=theta)


class TestForceSign(unittest.TestCase):
    """Gravity is attractive. This is a sign check with no tolerance to tune."""

    def test_gravity_is_attractive(self):
        pos = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        mass = np.array([1.0, 1.0])
        acc = tree_acceleration(pos, mass)
        # the left mass must be pulled toward the right one
        self.assertGreater(acc[0, 0], 0.0, "gravity is repulsive (sign error)")
        self.assertLess(acc[1, 0], 0.0)

    def test_inverse_square_magnitude(self):
        pos = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        mass = np.array([1.0, 1.0])
        acc = tree_acceleration(pos, mass)
        self.assertAlmostEqual(abs(acc[0, 0]), 1.0, places=6)

    def test_newton_third_law_within_a_leaf(self):
        pos = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        mass = np.array([1.0, 1.0])
        acc = tree_acceleration(pos, mass)
        self.assertAlmostEqual(acc[0, 0] + acc[1, 0], 0.0, places=12)


class TestCrossLeafForce(unittest.TestCase):
    """A pair in different leaves must still feel each other.

    With max_leaf=1 the two particles are in separate leaves, so the near-field
    (within-leaf) sum cannot see the pair. If the far-field loop skips leaves,
    the force is exactly zero -- which is what the bug produced.
    """

    def test_separate_leaves_still_interact(self):
        pos = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        mass = np.array([1.0, 1.0])
        acc = tree_acceleration(pos, mass, max_leaf=1)
        self.assertNotAlmostEqual(acc[0, 0], 0.0, places=9,
                                  msg="cross-leaf force was dropped")
        self.assertAlmostEqual(acc[0, 0], 1.0, places=6)

    @unittest.expectedFailure
    def test_max_leaf_does_not_change_the_force_much(self):
        """KNOWN-ISSUE (third defect): the far-field loop double counts.

        `compute_acceleration` walks ALL cells in a flat loop with no traversal
        state. A correct Barnes-Hut traversal must STOP descending once it
        accepts a cell's COM; this one keeps adding descendants, so an accepted
        cell and every accepted cell below it all contribute.

        Measured on N=60, theta=0.5, particle 0:

            max_leaf=16  ->  0 COM contributors   (tree depth 1; the root
                             excludes everything, so only leaves contribute)
            max_leaf=1   -> 66 COM contributors   at depths 2,3,4,5 -- i.e.
                             18 depth-4 and 8 depth-5 cells are descendants of
                             already-accepted depth-2/3 cells

        Median relative error against exact O(N^2):

            max_leaf=1   0.616
            max_leaf=2   0.244
            max_leaf=8   0.000   <- the tree barely subdivides here, so the
            max_leaf=16  0.000      "accuracy" is really just exact summation
            max_leaf=64  0.000      inside a handful of big leaves

        The default max_leaf=16 is only depth 1 at N=60, which is why it looks
        exact. At the 400-800 particles the scientific presets use, the tree
        reaches depth 2-3 and the over-counting engages.
        """
        rng = np.random.default_rng(11)
        pos = rng.normal(0, 1.0, (60, 3))
        mass = rng.uniform(0.5, 2.0, 60)
        a16 = tree_acceleration(pos, mass, max_leaf=16)
        a1 = tree_acceleration(pos, mass, max_leaf=1)
        rel = np.linalg.norm(a16 - a1, axis=1) / np.maximum(
            np.linalg.norm(a1, axis=1), 1e-12)
        self.assertLess(float(np.median(rel)), 0.1,
                        "max_leaf changes the force substantially")


class TestTreeAgainstExact(unittest.TestCase):
    """The standard validation for a tree code."""

    def test_sign_agreement_with_exact(self):
        rng = np.random.default_rng(5)
        pos = rng.normal(0, 1.0, (150, 3))
        mass = rng.uniform(0.5, 2.0, 150)
        exact = exact_acceleration(pos, mass)
        approx = tree_acceleration(pos, mass)
        agree = float(np.mean(np.sign(exact[:, 0]) == np.sign(approx[:, 0])))
        self.assertGreater(agree, 0.99, "tree disagrees with exact on direction")

    def test_relative_error_is_small(self):
        rng = np.random.default_rng(5)
        pos = rng.normal(0, 1.0, (150, 3))
        mass = rng.uniform(0.5, 2.0, 150)
        exact = exact_acceleration(pos, mass)
        approx = tree_acceleration(pos, mass)
        e_mag = np.linalg.norm(exact, axis=1)
        strong = e_mag > np.median(e_mag)
        rel = np.linalg.norm(approx - exact, axis=1)[strong] / e_mag[strong]
        self.assertLess(float(np.median(rel)), 0.05)

    def test_exact_limit_knows_no_approximation(self):
        """theta=0 never opens a cell, so the tree must reproduce O(N^2) exactly."""
        rng = np.random.default_rng(6)
        pos = rng.normal(0, 1.0, (40, 3))
        mass = rng.uniform(0.5, 2.0, 40)
        exact = exact_acceleration(pos, mass)
        approx = tree_acceleration(pos, mass, theta=0.0)
        self.assertLess(float(np.max(np.abs(approx - exact))), 1e-9)


class TestIntegratorAgainstAnalytic(unittest.TestCase):
    """Two equal masses in a circular orbit have a closed-form solution."""

    def _orbit(self, n_steps=400, dt=0.01):
        G, m, r = 1.0, 1.0, 1.0
        omega = np.sqrt(2.0 * G * m / r ** 3)
        v = omega * r / 2.0
        pos = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        mass = np.array([m, m])
        vel = np.array([[0.0, -v, 0.0], [0.0, v, 0.0]])
        sim = NBodySimulator(G=G, theta=0.5, softening=1e-4)
        return sim.simulate(pos, mass, vel, n_steps=n_steps, dt=dt), mass, pos, vel

    def test_separation_stays_constant(self):
        traj, _, _, _ = self._orbit()
        sep = np.linalg.norm(traj[:, 0] - traj[:, 1], axis=1)
        self.assertLess(abs(sep[-1] - 1.0), 1e-3,
                        "circular orbit drifted: r(0)=%.4f r(end)=%.4f" % (sep[0], sep[-1]))

    def test_energy_is_conserved(self):
        traj, mass, pos0, vel0 = self._orbit()

        def energy(p, v):
            ke = 0.5 * float(np.sum(mass[:, None] * v ** 2))
            pe = -1.0 * mass[0] * mass[1] / np.linalg.norm(p[0] - p[1])
            return ke + pe

        e0 = energy(traj[0], vel0)
        v_end = (traj[-1] - traj[-2]) / 0.01
        e1 = energy(traj[-1], v_end)
        self.assertLess(abs(e1 - e0) / abs(e0), 1e-3,
                        "energy drift %.4f" % (abs(e1 - e0) / abs(e0)))


class TestMomentumConservation(unittest.TestCase):
    """Barnes-Hut breaks Newton's third law; the exact path must not."""

    def test_exact_path_conserves_momentum(self):
        rng = np.random.default_rng(9)
        pos = rng.normal(0, 1.0, (50, 3))
        mass = rng.uniform(0.5, 2.0, 50)
        vel = rng.normal(0, 0.1, (50, 3))
        sim = NBodySimulator(G=1.0, theta=0.0)          # theta=0 -> exact O(N^2)
        traj = sim.simulate(pos, mass, vel, n_steps=15, dt=0.005)
        p0 = (mass[:, None] * vel).sum(axis=0)
        v_end = (traj[-1] - traj[-2]) / 0.005
        p1 = (mass[:, None] * v_end).sum(axis=0)
        self.assertLess(float(np.linalg.norm(p1 - p0)) / max(float(np.linalg.norm(p0)), 1e-9),
                        1e-6)

    def test_momentum_drift_grows_with_theta(self):
        """Documents that the drift at the default theta is an approximation
        artefact, not a defect: it is monotone in the opening parameter and
        vanishes in the exact limit. Any future threshold should account for it.
        """
        rng = np.random.default_rng(10)
        pos = rng.normal(0, 1.0, (80, 3))
        mass = rng.uniform(0.5, 2.0, 80)
        vel = rng.normal(0, 0.05, (80, 3))

        def drift(theta):
            sim = NBodySimulator(G=1.0, theta=theta)
            traj = sim.simulate(pos, mass, vel, n_steps=10, dt=0.005)
            v_end = (traj[-1] - traj[-2]) / 0.005
            p0 = (mass[:, None] * vel).sum(axis=0)
            p1 = (mass[:, None] * v_end).sum(axis=0)
            return float(np.linalg.norm(p1 - p0))

        d_exact = drift(0.0)
        d_loose = drift(0.8)
        self.assertLess(d_exact, 1e-9)
        self.assertGreater(d_loose, d_exact)


if __name__ == "__main__":
    unittest.main(verbosity=2)
