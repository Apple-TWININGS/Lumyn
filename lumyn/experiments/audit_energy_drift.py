"""取证 2：energy_drift 随 dt 减小而爆炸 —— 是检查器的错，还是场景发散？"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
from lumyn.physics.scenes import galaxy
from lumyn import NBodySimulator
from lumyn.eval.simscore import ConservationChecker

ck = ConservationChecker()

print("=" * 76)
print("A. 场景本身是否稳定？（半径随时间的增长）")
print("=" * 76)
pos, mass, vel, fixed = galaxy(150, seed=1)
sim = NBodySimulator(G=1.0, theta=0.5)
for dt_sim, n in ((0.005, 20), (0.0025, 40)):
    tr = sim.simulate(pos, mass, vel, n_steps=n, dt=dt_sim)
    r = np.linalg.norm(tr - tr[:, :1, :], axis=-1)
    print(f"  dt={dt_sim:<8} r_min={r.min():.3e}  r_max={r.max():.3e}  "
          f"r_max(首)={r[0].max():.3f}  r_max(末)={r[-1].max():.3f}")

print()
print("=" * 76)
print("B. 对照实验：已知真解的圆轨道（两体），同样的 checker 口径")
print("=" * 76)
print("  若 checker 的定义正确，能量漂移应随 dt 减小而**下降**。")
sim2 = NBodySimulator(G=1.0, theta=1.0)
p2 = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
m2 = np.array([1.0, 1.0])
v2 = np.array([[0, 0.5, 0], [0, -0.5, 0]])
print(f"  {'dt':<10}{'n':<7}{'总时长':<10}{'energy_drift':>16}")
for dt_sim, n in ((0.004, 25), (0.002, 50), (0.001, 100), (0.0005, 200), (0.00025, 400)):
    tr = sim2.simulate(p2, m2, v2, n_steps=n, dt=dt_sim)
    e = ck.check(tr, m2, dt=dt_sim)["energy_drift"]
    print(f"  {dt_sim:<10}{n:<7}{dt_sim*n:<10.4f}{e:>16.6f}")

print()
print("=" * 76)
print("C. 分离两个误差来源：E(差商速度) vs E(真实速度)")
print("=" * 76)
print("  用圆轨道做：差商速度 v_fd = v_true + 0.5*a*dt，其误差 O(dt)。")
G = 1.0
for dt_sim, n in ((0.004, 25), (0.002, 50), (0.001, 100), (0.0005, 200)):
    tr = sim2.simulate(p2, m2, v2, n_steps=n, dt=dt_sim)
    v_fd = (tr[1] - tr[0]) / dt_sim
    # 真实速度：两体圆轨道解析解（绕质心，角速度 omega = 2*v/r = 0.5）
    theta = 0.5 * dt_sim
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    v_true = v2 @ R.T
    E_fd = ck._energy(tr[0], v_fd, m2)
    E_true = ck._energy(tr[0], v_true, m2)
    print(f"  dt={dt_sim:<8} E_fd-E_true = {abs(E_fd-E_true):.6e}   "
          f"v_fd-v_true 误差 = {np.abs(v_fd-v_true).max():.6e}")

print()
print("=" * 76)
print("D. 结论")
print("=" * 76)
print("  B 中若 energy_drift 随 dt 单调下降 -> 定义本身可收敛，galaxy 的爆炸是场景问题；")
print("  B 中若同样爆炸或非单调 -> 检查器定义有问题。")
