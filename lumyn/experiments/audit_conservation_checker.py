"""取证：ConservationChecker 的实现缺陷（论文核心组件）。

测试 test_galaxy_non_divergent 失败（momentum_drift 8.11 > 2.0），
但真正的问题不是阈值，而是这个检查器本身的定义就不成立。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
from lumyn.physics.scenes import galaxy, explosion
from lumyn import NBodySimulator
from lumyn.eval.simscore import ConservationChecker

print("=" * 76)
print("缺陷 1：momentum_drift 除以接近 0 的 |P_0|")
print("=" * 76)
pos, mass, vel, fixed = galaxy(150, seed=1)
sim = NBodySimulator(G=1.0, theta=0.5)
traj = sim.simulate(pos, mass, vel, n_steps=20, dt=0.005)

print(f"  中心黑洞质量 = {mass[fixed][0]:.3f}（占总量 {mass.sum():.3f}）")
print(f"  初始总动量 P0(全部)     = {np.linalg.norm((mass[:,None]*vel).sum(0)):.6e}")
free = ~np.asarray(fixed, bool)
print(f"  初始总动量 P0(仅自由粒子) = {np.linalg.norm((mass[free,None]*vel[free]).sum(0)):.6e}")
print("  -> 两者都接近 0（旋转星系的净动量天然接近零）")
print("  -> momentum_drift = |P_T-P_0| / (|P_0|+1e-8)：分母是噪声量级，")
print("     结果由数值噪声主导，不反映任何物理。")

# 黑洞是否真的静止？
bh_traj = traj[:, fixed, :][:, 0, :]
print(f"  黑洞位置在各时刻的最大变化 = {np.abs(bh_traj - bh_traj[0]).max():.3e}")
bh_v_fd = (bh_traj[-1] - bh_traj[0]) / (0.005 * 20)
print(f"  黑洞 fd 速度 = {np.linalg.norm(bh_v_fd):.3e}  -> 质量×速度 = "
      f"{mass[fixed][0]*np.linalg.norm(bh_v_fd):.3e}")
print("  -> 即使黑洞静止，free=slice(None) 时 P 里仍多出它的 (近零) 贡献；")
print("     但分子的 P_T-P_0 与分母的 |P_0| 都是噪声，掩码与否会给出完全不同的比值。")

print()
print("=" * 76)
print("缺陷 2：angular_drift 不是角动量")
print("=" * 76)
p_norms = []
for t in range(20):
    v = (traj[t + 1] - traj[t]) / 0.005
    p_norms.append(np.linalg.norm((mass[:, None] * v).sum(axis=0)))
p_norms = np.array(p_norms)
print(f"  angular_drift 实际计算的是 max|‖P(t)‖-‖P(0)‖| / ‖P(0)‖")
print(f"    ‖P(t)‖ 序列 = {np.round(p_norms, 6)}")
print(f"    这正是**线动量的模长**，不是 L = Σ m (r×v)。")
print("  -> 变量名与物理量不符；源码注释也承认这是『近似』。")

# 真角动量
L = []
for t in range(21):
    if t + 1 <= 20:
        v = (traj[t + 1] - traj[t]) / 0.005
    else:
        v = (traj[t] - traj[t - 1]) / 0.005
    L.append((mass[:, None] * np.cross(traj[t], v)).sum(0))
L = np.array(L)
print(f"  真实角动量 ‖L(t)‖ 相对变化 = "
      f"{np.abs(np.linalg.norm(L,axis=1) - np.linalg.norm(L[0]))[1:].max()/np.linalg.norm(L[0]):.6f}")

print()
print("=" * 76)
print("缺陷 3：能量用 O(dt) 的差商速度，误差盖过真实漂移")
print("=" * 76)
print(f"  energy_drift（checker 口径，dt=0.005） = "
      f"{ConservationChecker().check(traj, mass, dt=0.005)['energy_drift']:.4f}")
for dt_sim, n in ((0.005, 20), (0.0025, 40), (0.00125, 80), (0.000625, 160)):
    tr = sim.simulate(pos, mass, vel, n_steps=n, dt=dt_sim)
    e = ConservationChecker().check(tr, mass, dt=dt_sim)["energy_drift"]
    print(f"    dt={dt_sim:<9} (n={n:<4}) energy_drift = {e:.4f}")
print("  -> 若随 dt 减小而收敛，说明是差商伪影；真实漂移应远小于此。")

print()
print("=" * 76)
print("缺陷 4：_energy 的 O(N^2) Python 双重循环")
print("=" * 76)
import time
t0 = time.time()
ck = ConservationChecker()
ck._energy(traj[0], (traj[1] - traj[0]) / 0.005, mass)
print(f"  N={len(pos)} 单次 _energy 耗时 {(time.time()-t0)*1000:.1f} ms（纯 Python 双重循环）")
print("  -> check() 对每步都调用一次；N 更大时不可用。")
print("     model 的『更低成本』主张需要以这个实现为基准重新评估。")
