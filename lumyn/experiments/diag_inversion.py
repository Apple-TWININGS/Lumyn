"""实验 A 反演收敛性诊断（临时脚本，非论文产物）。

目的：找出「优化后 shape error 反而变大」的根因。

已确认的两个问题：
  P1（致命）DifferentiableNBody.__init__ 对输入做了 .detach()。
            若在优化循环里反复新建系统，每次前向的输入都与外部张量无关，
            loss 逐位不变、梯度为零 —— 优化器实际上什么都没在优化。
  P2        invert() 把 vel0 钉死在 rng.normal(0, 0.05)（|v|≈0.07），
            而真值 IC 的速率是 v=sqrt(1/r)（|v|≈1.04），差 15x。
            即使梯度通了，这个初速度也使目标不可达。
"""
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
# 包根是 experiments 的上两级（…/lumyn_v14），不是上一级。
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np
import torch
from lumyn.physics.differentiable import DifferentiableNBody

torch.set_default_dtype(torch.float64)

N, STEPS, DT, SOFT, G = 6, 40, 0.01, 0.05, 1.0
MASS = 0.3


def prior_ic(e, rng):
    """与实验 A 的 make_gt_ic 完全一致的先验采样。"""
    ang = rng.uniform(0, 2 * np.pi, N)
    r = (1 - e ** 2) / (1 + e * np.cos(ang))
    pos = np.stack([r * np.cos(ang), r * np.sin(ang), rng.normal(0, 0.05, N)], axis=1)
    v = np.sqrt(1.0 / np.maximum(r, 1e-6))
    vel = np.stack([-v * np.sin(ang), v * np.cos(ang), rng.normal(0, 0.02, N)], axis=1)
    return pos, vel


def rollout_np(pos, vel):
    s = DifferentiableNBody(pos, vel, np.ones(N) * MASS, G=G,
                            softening=SOFT, dt=DT, steps=STEPS)
    with torch.no_grad():
        return s.forward()


def sq_err(final_pos, target):
    return float((final_pos - target).pow(2).mean())


# ---------------- 旧（错误）用法：循环内反复新建系统 ----------------
def invert_legacy(target, init_pos, init_vel, lr, opt_pos, opt_vel, n_steps):
    pos0 = torch.tensor(np.asarray(init_pos, dtype=np.float64)).clone().detach().requires_grad_(True)
    vel0 = torch.tensor(np.asarray(init_vel, dtype=np.float64)).clone().detach().requires_grad_(True)
    mass = torch.ones(N) * MASS
    with torch.no_grad():
        init_err = sq_err(rollout_np(pos0, vel0)["final_pos"], target)

    params = [p for p, on in ((pos0, opt_pos), (vel0, opt_vel)) if on]
    opt = torch.optim.Adam(params, lr=lr)
    grads_seen = 0
    for _ in range(n_steps):
        opt.zero_grad()
        s = DifferentiableNBody(pos0, vel0, mass, G=G, softening=SOFT, dt=DT, steps=STEPS)
        loss = (s.forward()["final_pos"] - target).pow(2).mean()
        loss.backward()
        if pos0.grad is not None:
            grads_seen += 1
        opt.step()
    with torch.no_grad():
        final_err = sq_err(rollout_np(pos0, vel0)["final_pos"], target)
    return init_err, final_err, grads_seen


# ---------------- 新（正确）用法：构造一次，优化 sysm 自己的参数 ----------------
def invert_fixed(target, init_pos, init_vel, lr, opt_pos, opt_vel, n_steps,
                 trace=False):
    sysm = DifferentiableNBody(init_pos, init_vel, np.ones(N) * MASS,
                               G=G, softening=SOFT, dt=DT, steps=STEPS)
    params = [p for p, on in ((sysm.pos, opt_pos), (sysm.vel, opt_vel)) if on]
    opt = torch.optim.Adam(params, lr=lr)

    with torch.no_grad():
        init_err = sq_err(sysm.forward()["final_pos"], target)

    curve = []
    for _ in range(n_steps):
        opt.zero_grad()
        out = sysm.forward()
        loss = (out["final_pos"] - target).pow(2).mean()
        loss.backward()
        opt.step()
        if trace:
            with torch.no_grad():
                curve.append(sq_err(sysm.forward()["final_pos"], target))

    with torch.no_grad():
        final_err = sq_err(sysm.forward()["final_pos"], target)
    return init_err, final_err, curve


def main():
    t0 = time.time()
    print("=" * 74)
    print("0. 速度尺度对照")
    print("=" * 74)
    r0 = np.random.default_rng(1234)
    r1 = np.random.default_rng(7)
    gt_s = [np.linalg.norm(prior_ic(r0.uniform(0.0, 0.3), r0)[1], axis=1).mean()
            for _ in range(5)]
    cur_s = [np.linalg.norm(r1.normal(0, 0.05, (N, 3)), axis=1).mean() for _ in range(5)]
    print(f" 真值 IC 平均速率 v_true  = {np.mean(gt_s):.4f}")
    print(f" invert() 固定 v_init     = {np.mean(cur_s):.4f}")
    print(f" 比值                     = {np.mean(gt_s)/np.mean(cur_s):.1f}x")

    e = 0.15
    p_gt, v_gt = prior_ic(e, np.random.default_rng(4242))
    target = rollout_np(p_gt, v_gt)["final_pos"]

    p_prior, v_prior = prior_ic(e, np.random.default_rng(99))
    r_rand = np.random.default_rng(5)

    print()
    print("=" * 74)
    print("1. 旧用法 vs 新用法（同分布 e=0.15, lr=0.05, 50 步）")
    print("=" * 74)
    print(f"  {'配置':<28}{'反向传播':>12}{'init':>11}{'final':>11}{'ratio':>9}")
    print("  " + "-" * 70)
    for name, (p, v), op, ov in [
        ("旧: pos-only, v~0.05", (r1.normal(0, 0.3, (N, 3)), r1.normal(0, 0.05, (N, 3))), True, False),
        ("旧: pos+vel, 先验初始化", (p_prior, v_prior), True, True),
    ]:
        ie, fe, gs = invert_legacy(target, p, v, 0.05, op, ov, 50)
        print(f"  {name:<28}{str(gs)+'/50':>12}{ie:>11.5f}{fe:>11.5f}{fe/ie:>8.3f}x")

    for name, (p, v), op, ov in [
        ("新: pos-only, v~先验尺度", (p_prior, v_prior), True, False),
        ("新: pos+vel, 先验初始化", (p_prior, v_prior), True, True),
    ]:
        ie, fe, _ = invert_fixed(target, p, v, 0.05, op, ov, 50)
        flag = "OK" if fe < 0.8 * ie else ("~" if fe < ie else "BAD")
        print(f"  {name:<28}{'50/50':>12}{ie:>11.5f}{fe:>11.5f}{fe/ie:>8.3f}x {flag}")

    print()
    print("=" * 74)
    print("2. 新用法下的学习率 / 步数扫描（pos+vel, 先验初始化）")
    print("=" * 74)
    for lr in (0.05, 0.01, 0.005):
        for n in (50, 200, 600):
            ie, fe, curve = invert_fixed(target, p_prior, v_prior, lr, True, True, n,
                                         trace=True)
            print(f"  lr={lr:<6} steps={n:<4} init={ie:.5f} final={fe:.5f} "
                  f"ratio={fe/ie:.4f}  loss {curve[0]:.5f}->{curve[-1]:.6f}")

    print()
    print("=" * 74)
    print("3. 收敛轨迹形状（lr=0.01, 600 步, pos+vel）")
    print("=" * 74)
    ie, fe, curve = invert_fixed(target, p_prior, v_prior, 0.01, True, True, 600, trace=True)
    idx = [0, 1, 2, 5, 10, 25, 50, 100, 200, 399, 599]
    print("  step: " + "  ".join(f"{i+1:>5}" for i in idx))
    print("  err : " + "  ".join(f"{curve[i]:.4f}" for i in idx))

    print()
    print("=" * 74)
    print("4. 跨分布（e=0.65，critic 训练时未见）")
    print("=" * 74)
    p_gt2, v_gt2 = prior_ic(0.65, np.random.default_rng(4242))
    target2 = rollout_np(p_gt2, v_gt2)["final_pos"]
    p_pr2, v_pr2 = prior_ic(0.65, np.random.default_rng(77))
    for lr in (0.01, 0.005):
        ie, fe, _ = invert_fixed(target2, p_pr2, v_pr2, lr, True, True, 600)
        print(f"  lr={lr:<6} init={ie:.5f} final={fe:.5f} ratio={fe/ie:.4f}")

    print(f"\n耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
