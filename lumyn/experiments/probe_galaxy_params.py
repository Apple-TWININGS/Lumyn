"""探测：galaxy 场景在什么参数下真正稳定（有界 + 分辨率充足）？

判据
----
1. 有界：r_max(末) 不超过 r_max(首) 的若干倍
2. 分辨率充足：内圈轨道周期 / dt ≥ 阈值（每个轨道至少若干步）
   T(r) = 2*pi*r^{3/2} / sqrt(G*M_enclosed)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
from lumyn import NBodySimulator

G = 1.0


def make(n, seed, M_center, r_in, r_out, disk_m, dt):
    rng = np.random.default_rng(seed)
    r = rng.uniform(r_in, r_out, n)
    theta = rng.uniform(0, 2 * np.pi, n) + 2.0 * r
    pos = np.stack([r * np.cos(theta), r * np.sin(theta),
                    rng.normal(0, 0.02, n)], axis=1)
    mass = np.full(n, disk_m)
    order = np.argsort(r)
    enc = np.empty_like(r)
    enc[order] = np.cumsum(mass[order])
    v = np.sqrt(G * (M_center + enc) / r)
    vel = np.stack([-v * np.sin(theta), v * np.cos(theta), np.zeros(n)], axis=1)
    pos = np.concatenate([np.zeros((1, 3)), pos], axis=0)
    mass = np.concatenate([[M_center], mass])
    vel = np.concatenate([np.zeros((1, 3)), vel], axis=0)
    fixed = np.zeros(len(mass), dtype=bool); fixed[0] = True
    return pos, mass, vel, fixed, r_in, M_center


def measure(cfg, steps=40, dt=0.005):
    pos, mass, vel, fixed, r_in, Mc = make(**cfg)
    sim = NBodySimulator(G=G, theta=0.5)
    traj = sim.simulate(pos, mass, vel, n_steps=steps, dt=dt, fixed_mask=fixed)
    r = np.linalg.norm(traj[:, 1:, :] - traj[:, :1, :], axis=-1)
    T_in = 2 * np.pi * r_in ** 1.5 / np.sqrt(G * Mc)
    return r[0].max(), r[-1].max(), T_in / dt


CONFIGS = [
    dict(name="当前修正版", n=150, seed=1, M_center=1000.0, r_in=0.1, r_out=1.0,
         disk_m=0.1, dt=0.005),
    dict(name="M=200, r∈[0.3,1]", n=150, seed=1, M_center=200.0, r_in=0.3, r_out=1.0,
         disk_m=0.05, dt=0.005),
    dict(name="M=50, r∈[0.5,2]", n=150, seed=1, M_center=50.0, r_in=0.5, r_out=2.0,
         disk_m=0.05, dt=0.005),
    dict(name="M=20, r∈[0.8,3]", n=150, seed=1, M_center=20.0, r_in=0.8, r_out=3.0,
         disk_m=0.02, dt=0.005),
    dict(name="M=10, r∈[1,3]", n=150, seed=1, M_center=10.0, r_in=1.0, r_out=3.0,
         disk_m=0.01, dt=0.005),
]

print(f"{'配置':<22}{'r_max首':>10}{'r_max末':>10}{'增长':>9}{'内圈步数/轨道':>15}{'判定':>8}")
print("-" * 78)
for cfg in CONFIGS:
    name = cfg.pop("name")
    a, b, steps_per_orbit = measure(cfg)
    ok = (b < 3 * a) and (steps_per_orbit >= 20)
    print(f"{name:<22}{a:>10.3f}{b:>10.3f}{b/a:>8.2f}x{steps_per_orbit:>15.1f}"
          f"{('稳定' if ok else '不稳定'):>8}")
    cfg["name"] = name
