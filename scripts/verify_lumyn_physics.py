"""Independent verification of Lumyn's physics.

We do not trust the test suite here, because the suite can pass while the force
law is wrong: a galaxy demo still "looks like a galaxy" if the net force happens
to be attractive at large N, even when the near-field and far-field terms have
opposite signs.

Two checks that cannot be satisfied by a wrong force law:

  A. Direct force check, no integrator involved. Two masses on the x-axis; the
     acceleration of the left one must point in +x (toward the right one). This
     is sign-only and has no tolerance to tune.

  B. Analytic two-body circular orbit. For equal masses m separated by r,
     omega = sqrt(2 G m / r^3) and the separation must stay constant. Any sign
     error turns it into an unbounded repulsion, which is impossible to miss.

  C. Large-N sanity: does the net acceleration behave attractively when the
     near-field and far-field terms are both active?

    python scripts/verify_lumyn_physics.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []


def check(name, ok, detail=""):
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILURES.append(name)


def force_sign_check():
    """A: acceleration of the left mass must point toward the right mass."""
    print("\nA. direct force sign (no integrator)")
    from lumyn.physics.barnes_hut import BarnesHutTree
    pos = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    mass = np.array([1.0, 1.0])
    tree = BarnesHutTree(max_leaf=16, max_depth=24)
    tree.build(pos, mass, center=np.zeros(3), size=2.0)
    acc = tree.compute_acceleration(theta=0.5)
    print("     acc = %s" % np.array2string(acc, precision=5))
    # left mass at x=-0.5 pulled by right mass at x=+0.5 => acc_x > 0
    check("near-field: left mass pulled toward right (+x)",
          acc[0, 0] > 0, "acc[0,0] = %.5f" % acc[0, 0])
    check("near-field: right mass pulled toward left (-x)",
          acc[1, 0] < 0, "acc[1,0] = %.5f" % acc[1, 0])
    # Newton's third law: total momentum change must vanish
    check("near-field: equal and opposite (sum ~ 0)",
          abs(acc[0, 0] + acc[1, 0]) < 1e-12,
          "sum = %.3e" % (acc[0, 0] + acc[1, 0]))
    check("near-field: magnitude ~ 1/r^2 = 1.0",
          abs(abs(acc[0, 0]) - 1.0) < 1e-3, "|acc| = %.5f" % abs(acc[0, 0]))
    return acc


def exact_acceleration(pos, mass, G=1.0, eps=1e-8):
    """Direct O(N^2) gravity: a_i = G sum_j m_j (r_j - r_i)/|r_j - r_i|^3."""
    d = pos[None, :, :] - pos[:, None, :]           # d[i,j] = r_j - r_i
    r2 = (d * d).sum(axis=2) + eps
    inv_r3 = r2 ** (-1.5)
    np.fill_diagonal(inv_r3, 0.0)
    return G * (mass[None, :, None] * d * inv_r3[:, :, None]).sum(axis=1)


def tree_vs_exact_check():
    """A2: the standard validation for a tree code -- compare against O(N^2).

    The earlier version of this check placed two near-symmetric clumps and got
    exactly zero, which said nothing about the far field and was a defect in the
    test, not in the code. Comparing against the exact sum exercises near field
    and far field together.
    """
    print("\nA2. Barnes-Hut tree vs exact O(N^2)")
    from lumyn.physics.barnes_hut import BarnesHutTree
    rng = np.random.default_rng(3)
    n = 200
    pos = rng.normal(0.0, 1.0, (n, 3))
    mass = rng.uniform(0.5, 2.0, n)

    exact = exact_acceleration(pos, mass)
    tree = BarnesHutTree(max_leaf=16, max_depth=24)
    tree.build(pos, mass, center=pos.mean(axis=0),
               size=float(np.abs(pos - pos.mean(axis=0)).max()) * 2 + 1e-6)
    approx = tree.compute_acceleration(theta=0.5)

    # sign agreement and relative error on the bulk
    agree = float(np.mean(np.sign(exact[:, 0]) == np.sign(approx[:, 0])))
    e_mag = np.linalg.norm(exact, axis=1)
    a_mag = np.linalg.norm(approx, axis=1)
    strong = e_mag > np.median(e_mag)
    rel = np.linalg.norm(approx - exact, axis=1)[strong] / e_mag[strong]
    print("     sign agreement on x-component : %.3f" % agree)
    print("     median relative error (strong) : %.4f" % float(np.median(rel)))
    check("tree agrees with exact on the sign of the force", agree > 0.99,
          "%.3f" % agree)
    check("tree relative error < 5% on strongly-forced particles",
          float(np.median(rel)) < 0.05, "%.4f" % float(np.median(rel)))
    check("tree and exact agree on the NET force direction",
          float(np.dot(exact.ravel(), approx.ravel())) > 0,
          "dot = %.1f" % float(np.dot(exact.ravel(), approx.ravel())))


def circular_orbit_check():
    """B: analytic two-body circular orbit; separation must stay constant."""
    print("\nB. analytic two-body circular orbit")
    from lumyn.physics.nbody import NBodySimulator
    G, m, r = 1.0, 1.0, 1.0
    omega = np.sqrt(2.0 * G * m / r ** 3)
    v = omega * r / 2.0
    pos = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    mass = np.array([m, m])
    vel = np.array([[0.0, -v, 0.0], [0.0, v, 0.0]])

    sim = NBodySimulator(G=G, theta=0.5, softening=1e-4)
    traj = sim.simulate(pos, mass, vel, n_steps=400, dt=0.01)
    sep = np.linalg.norm(traj[:, 0] - traj[:, 1], axis=1)
    print("     r(0)=%.4f  r(mid)=%.4f  r(end)=%.4f   (exact orbit: r==1)"
          % (sep[0], sep[len(sep) // 2], sep[-1]))
    check("separation stays bounded (< 1.5x initial)", sep[-1] < 1.5,
          "r(end)/r(0) = %.3f" % (sep[-1] / sep[0]))
    check("separation does not run away (< 3x at any time)", sep.max() < 3.0,
          "max r = %.3f" % sep.max())


def energy_check():
    """B2: total energy drift over a circular orbit."""
    print("\nB2. energy conservation on the same orbit")
    from lumyn.physics.nbody import NBodySimulator
    G, m, r = 1.0, 1.0, 1.0
    omega = np.sqrt(2.0 * G * m / r ** 3)
    v = omega * r / 2.0
    pos = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    mass = np.array([m, m])
    vel = np.array([[0.0, -v, 0.0], [0.0, v, 0.0]])

    sim = NBodySimulator(G=G, theta=0.5, softening=1e-4)

    def total_energy(p, vv):
        ke = 0.5 * mass.sum() * 0.0
        ke = 0.5 * float(np.sum(mass[:, None] * vv ** 2))
        d = np.linalg.norm(p[0] - p[1])
        pe = -G * m * m / d
        return ke + pe

    E0 = total_energy(pos, vel)
    p, vv = pos.copy(), vel.copy()
    for _ in range(400):
        p, vv = sim.step(p, mass, vv, dt=0.01)
    E1 = total_energy(p, vv)
    drift = abs(E1 - E0) / max(abs(E0), 1e-12)
    print("     E0=%.6f  E1=%.6f  relative drift=%.4f" % (E0, E1, drift))
    check("relative energy drift < 5%", drift < 0.05, "drift = %.4f" % drift)


def main():
    print("=" * 78)
    print("Independent verification of Lumyn physics")
    print("=" * 78)
    force_sign_check()
    tree_vs_exact_check()
    circular_orbit_check()
    energy_check()

    print("\n" + "=" * 78)
    if FAILURES:
        print("FAILED CHECKS (%d):" % len(FAILURES))
        for f in FAILURES:
            print("   - %s" % f)
        print("\nLumyn's trajectories cannot be used as physical ground truth")
        print("until these are resolved.")
    else:
        print("all checks passed")
    print("=" * 78)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
