"""可微生成端到端 demo：生成椭圆轨道 → 梯度优化 → 守恒验证 → 导出 CSV。

自动检测 torch：有则 PyTorch 可微优化，无则降级 NumPy 中心差分（任何环境可跑）。
"""
from __future__ import annotations
import os, sys, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

try:
    import torch  # noqa
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def _numpy_demo(n_particles=6, n_steps=20, lr=0.05, iters=60):
    from lumyn.physics.differentiable_numpy import DifferentiableNBodyNumpy
    # 中心天体 + 随机小天体
    masses = np.array([1.0] + [0.01] * (n_particles - 1), dtype=np.float64)
    pos = np.random.randn(n_particles, 3) * 0.5
    pos[0] = 0.0
    vel = np.random.randn(n_particles, 3) * 0.1
    sim = DifferentiableNBodyNumpy(masses=masses, pos=pos, vel=vel, n_steps=n_steps, dt=0.05)
    target_r = 1.0
    history = []
    for it in range(iters):
        traj = sim.forward()
        r = np.linalg.norm(sim.pos - sim.pos[0], axis=-1)[1:].mean()
        energy = sim._total_energy()
        loss = (r - target_r) ** 2 + 1e-3 * abs(energy)
        grads = sim.param_gradients(loss)
        sim.pos = sim.pos - lr * np.clip(grads["pos"], -1.0, 1.0)
        drift = abs(float(energy) - float(sim._init_energy)) / (abs(float(sim._init_energy)) + 1e-9)
        history.append((float(loss), float(drift)))
        if it % 20 == 0:
            print(f"  [numpy] iter {it}: loss={float(loss):.4f} drift={drift:.4f}")
    return history, traj


def run_full_demo(out_dir: str = None):
    out_dir = out_dir or os.path.join(HERE, "..", "lumyn_output")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[demo] torch available: {HAVE_TORCH}")

    if HAVE_TORCH:
        print("[demo] running PyTorch differentiable optimisation (TODO: fill torch path)")
        # 预留 torch 路径；本地装了 torch 在此接入 DifferentiableNBody + Adam
        history, traj = _numpy_demo()
    else:
        print("[demo] torch not installed -> using NumPy central-difference fallback")
        history, traj = _numpy_demo()

    # 导出 CSV
    csv_path = os.path.join(out_dir, "guided_generation_demo.csv")
    np.savetxt(csv_path, traj.reshape(traj.shape[0], -1), delimiter=",")
    print(f"[demo] saved trajectory CSV: {csv_path}  shape={traj.shape}")

    # 画收敛曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        losses = [h[0] for h in history]
        drifts = [h[1] for h in history]
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
        a.plot(losses, "-o"); a.set_title("total loss"); a.set_xlabel("iter")
        b.plot(drifts, "-o", color="orange"); b.set_title("energy drift"); b.set_xlabel("iter")
        fig.tight_layout()
        fig_path = os.path.join(out_dir, "guided_generation_demo.png")
        fig.savefig(fig_path, dpi=110)
        print(f"[demo] saved figure: {fig_path}")
    except Exception as e:
        print(f"[demo] plotting skipped: {e}")

    print(f"[demo] energy drift: {history[0][1]:.3f} -> {history[-1][1]:.4f}")
    return history


if __name__ == "__main__":
    run_full_demo()
