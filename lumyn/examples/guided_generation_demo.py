"""可微生成端到端 demo：从目标形态反演守恒的初始条件 → 导出 CSV / 收敛图。

自动检测 torch：有则走 PyTorch 可微路径，无则降级 NumPy 中心差分（任何环境可跑）。

运行：
    PYTHONPATH=. python lumyn/examples/guided_generation_demo.py

本脚本此前调用的是**不存在的 API**：`DifferentiableNBodyNumpy(masses=..., n_steps=...)`、
`sim.param_gradients(loss)`、`sim._total_energy()`、`sim._init_energy`，
且 `sys.path` 只上跳一级导致 `import lumyn` 失败。现按真实接口重写。
"""
from __future__ import annotations
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# 包根是 lumyn/ 的父目录；此前只上跳一级到 lumyn/，`import lumyn` 必然失败。
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def _run_numpy(n_particles=8, steps=40, lr=0.02, opt_steps=80, seed=0):
    """NumPy 中心差分路径：反演「目标半径 = 1.0」的初始条件。"""
    np.random.seed(seed)
    from lumyn.physics.differentiable_numpy import DifferentiableNBodyNumpy

    pos = np.random.randn(n_particles, 3) * 0.5
    vel = np.random.randn(n_particles, 3) * 0.1
    mass = np.ones(n_particles, dtype=np.float64)

    sim = DifferentiableNBodyNumpy(pos, vel, mass, steps=steps)
    history = []

    def callback(it, info):
        history.append((float(info.get("loss", np.nan)),
                        float(info.get("energy_drift", np.nan))))

    info = sim.optimize(target_radius=1.0, lr=lr, steps=opt_steps,
                        callback=callback)
    out = sim.forward()
    return out["trajectory"], info, history


def _run_torch(n_particles=8, steps=60, lr=0.05, opt_steps=150, seed=0):
    """PyTorch 路径：同样的目标，用 autograd 优化。"""
    import torch
    from lumyn.physics.differentiable import DifferentiableNBody

    torch.manual_seed(seed)
    pos = torch.randn(n_particles, 3, dtype=torch.float64) * 0.5
    vel = torch.randn(n_particles, 3, dtype=torch.float64) * 0.1
    mass = torch.ones(n_particles, dtype=torch.float64)

    # 关键：**只构造一次**，优化 system.parameters()。
    # 若在循环内反复新建系统，构造函数里的 .detach() 会让梯度永远到不了
    # 外部张量，loss 逐位不变（实测 50 步中 0 次反向传播到达）。
    sim = DifferentiableNBody(pos, vel, mass, steps=steps)
    opt = torch.optim.Adam([sim.pos, sim.vel], lr=lr)
    history = []
    for it in range(opt_steps):
        opt.zero_grad()
        out = sim.forward()
        r = out["final_pos"].norm(dim=-1).mean()
        loss = (r - 1.0) ** 2
        loss.backward()
        opt.step()
        if it % 25 == 0:
            history.append((float(loss.detach()), float("nan")))
    with torch.no_grad():
        traj = sim.forward()["trajectory"].numpy()
    return traj, {"final_loss": float(loss.detach())}, history


def run_full_demo(out_dir: str = None) -> dict:
    out_dir = out_dir or os.path.join(HERE, "..", "lumyn_output")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[demo] torch available: {HAVE_TORCH}")

    if HAVE_TORCH:
        print("[demo] 使用 PyTorch autograd 路径")
        traj, info, history = _run_torch()
    else:
        print("[demo] 未安装 torch → 使用 NumPy 中心差分兜底")
        traj, info, history = _run_numpy()

    csv_path = os.path.join(out_dir, "guided_generation_demo.csv")
    np.savetxt(csv_path, traj.reshape(traj.shape[0], -1), delimiter=",")
    print(f"[demo] 轨迹已导出: {csv_path}  shape={traj.shape}")
    print(f"[demo] 优化信息: {info}")

    if history:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            losses = [h[0] for h in history]
            drifts = [h[1] for h in history]
            fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
            a.plot(losses, "-o"); a.set_title("total loss"); a.set_xlabel("checkpoint")
            if np.isfinite(drifts).any():
                b.plot(drifts, "-o", color="orange")
            else:
                b.text(0.5, 0.5, "drift not reported\nby this path",
                       ha="center", va="center", transform=b.transAxes)
            b.set_title("energy drift"); b.set_xlabel("checkpoint")
            fig.tight_layout()
            fig_path = os.path.join(out_dir, "guided_generation_demo.png")
            fig.savefig(fig_path, dpi=110)
            plt.close(fig)
            print(f"[demo] 收敛图已保存: {fig_path}")
        except Exception as e:      # pragma: no cover - 绘图可选
            print(f"[demo] 跳过绘图: {e}")
        print(f"[demo] loss: {history[0][0]:.4f} -> {history[-1][0]:.4f}")

    return {"trajectory": traj, "info": info, "history": history}


if __name__ == "__main__":
    run_full_demo()
