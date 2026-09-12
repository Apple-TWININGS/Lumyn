"""实验 J：在**公认基准**上复现 critic 选择实验（含公平的 HNN 对照）。

为什么换到这两个系统
--------------------
1. **figure-eight 三体编排**（Chenciner & Montgomery, 2000）
   经典的周期解，初始条件与周期众所周知，是数值积分与学习式模拟器的标准测试台。
   真值可用高精度积分得到，且系统有界、不混沌发散。

2. **2D Kepler 二体**
   学物理文献（Hamiltonian / Lagrangian Neural Networks，Greydanus 2019、
   Cranmer 2020）的标准低维测试台。**这正是给 HNN 公平对照所需的设置**：
   实验 H 中 HNN 在 36 维状态空间上没训起来（`diagnose_hnn_fairness.py` 已证），
   而在 12 维上它应当是可训练的。

任务与指标同实验 H
------------------
同一初始条件下由学习式生成器产生 K 个候选，用 critic 挑出最接近真值的那个。
指标用 within-IC 秩相关与 IC 内更优率（pooled 会被 IC 间差异主导，见 §二·九·一）。
"""
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

from lumyn.experiments.experiment_a2_neural_generator import NeuralForce   # noqa: E402
from lumyn.experiments.experiment_a1_critic_quality import (   # noqa: E402
    raw_tensor, score_net, DeepSetsCritic,
)
from lumyn.experiments.experiment_h_benchmark import HNN                   # noqa: E402

K = 8
SIGMAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.6, 1.0)
N_TR, N_TE = 150, 60
DT = 0.002

# ---------------------------------------------------------------- 标准系统
#: figure-eight 三体编排的标准初始条件（G=1, 每个质量 =1，Chenciner & Montgomery）
FIG8_POS = np.array([[0.97000436, -0.24308753, 0.0],
                     [-0.97000436, 0.24308753, 0.0],
                     [0.0, 0.0, 0.0]])
FIG8_VEL = np.array([[-0.46620369, -0.43236573, 0.0],
                     [-0.46620369, -0.43236573, 0.0],
                     [0.93240737, 0.86473146, 0.0]])
#: 标准周期（G=1, m=1）
FIG8_PERIOD = 6.32591398


def system_fig8(rng, jitter=0.0):
    """figure-eight 三体。jitter>0 时对初始条件做小扰动以产生不同样本。"""
    pos = FIG8_POS.copy()
    vel = FIG8_VEL.copy()
    if jitter > 0:
        pos = pos + rng.normal(0, jitter, pos.shape)
        vel = vel + rng.normal(0, jitter, vel.shape)
    return pos, vel, np.ones(3)


def system_kepler(rng, e_lo=0.0, e_hi=0.5):
    """2D 二体：绕质心的椭圆轨道（G=1, m1=m2=1）。"""
    e = rng.uniform(e_lo, e_hi)
    a = 1.0
    # 相对坐标 r 的椭圆；两个质量相等 → 各自绕质心的距离是 r/2
    th = rng.uniform(0, 2 * np.pi)
    r = a * (1 - e ** 2) / (1 + e * np.cos(th))
    mu = 0.5                      # G*(m1+m2)=2 → 相对运动的 mu = G(m1+m2) = 2? 取 1 更稳
    mu = 1.0
    v_r = np.sqrt(mu / a) / (1 - e ** 2) * e * np.sin(th)
    v_t = np.sqrt(mu / a) / (1 - e ** 2) * (1 + e * np.cos(th))
    rhat = np.array([np.cos(th), np.sin(th), 0.0])
    that = np.array([-np.sin(th), np.cos(th), 0.0])
    rrel = r * rhat
    vrel = v_r * rhat + v_t * that
    pos = np.stack([rrel / 2, -rrel / 2])
    vel = np.stack([vrel / 2, -vrel / 2])
    return pos, vel, np.ones(2)


def accel_pairwise(pos, mass, G=1.0, soft=1e-12):
    """成对加速度。soft 必须 > 0：否则对角线上 r2=0 会触发除零警告。

    注意：soft 只作用于**对角线之外**的数值稳定性；对角线的自作用在随后被
    fill_diagonal 置零，因此 soft 的大小不影响物理结果。
    """
    d = pos[None, :, :] - pos[:, None, :]
    r2 = (d * d).sum(-1) + soft ** 2
    inv = r2 ** -1.5
    np.fill_diagonal(inv, 0.0)
    return G * (mass[None, :, None] * d * inv[..., None]).sum(1)


def rollout(pos, vel, mass, steps, dt=DT):
    """辛欧拉（kick-drift），返回 (pos_traj, vel_traj)。"""
    p, v = pos.copy(), vel.copy()
    ps, vs = [p.copy()], [v.copy()]
    for _ in range(steps):
        v = v + accel_pairwise(p, mass) * dt
        p = p + v * dt
        ps.append(p.copy()); vs.append(v.copy())
    return np.stack(ps), np.stack(vs)


def rollout_nn(pos, vel, mass, net, sigma, rng, steps, dt=DT):
    p = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)
    v = torch.tensor(vel, dtype=torch.float32).unsqueeze(0)
    m = torch.tensor(mass, dtype=torch.float32)
    ps = [p[0].numpy().astype(np.float64)]; vs = [v[0].numpy().astype(np.float64)]
    with torch.no_grad():
        for _ in range(steps):
            a = net(p, v, m)
            if sigma > 0:
                a = a + torch.tensor(rng.normal(0, sigma, a.shape)
                                     * float(a.abs().mean() + 1e-8), dtype=torch.float32)
            v = v + a * dt
            p = p + v * dt
            ps.append(p[0].numpy().astype(np.float64))
            vs.append(v[0].numpy().astype(np.float64))
    return np.stack(ps), np.stack(vs)


def train_force(P, V, mass, epochs=60, h=64, lr=2e-3, batch=64, seed=0, dt=DT):
    torch.manual_seed(seed)
    Nn = P.shape[2]
    net = NeuralForce(h=h).to(torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Pt = torch.tensor(P, dtype=torch.float32); Vt = torch.tensor(V, dtype=torch.float32)
    A = ((Vt[:, 1:] - Vt[:, :-1]) / dt).reshape(-1, Nn, 3)
    Xp = Pt[:, :-1].reshape(-1, Nn, 3); Xv = Vt[:, :-1].reshape(-1, Nn, 3)
    m = torch.tensor(mass, dtype=torch.float32)
    n = len(Xp)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            ((net(Xp[idx], Xv[idx], m) - A[idx]) ** 2).mean().backward()
            opt.step()
    net.eval()
    return net


# ---------------------------------------------------------------- 通道与 critic
def cons_channels(pos_t, vel_t, mass, G=1.0, soft=1e-12):
    """四个守恒通道的归一化漂移。

    soft 必须 > 0：此前默认 0.0，对角线上 r=0 → `mm/r` 为 inf，
    再乘 eye=0 得 **nan**，使 energy 通道恒为 nan。
    （本实验的 critic 只用了 angular/momentum，故已报告的数字不受影响，
      但该 bug 必须修。）
    """
    ke = 0.5 * (mass * (vel_t ** 2).sum(-1)).sum(-1)
    d = pos_t[:, None] - pos_t[:, :, None]
    r = np.sqrt((d * d).sum(-1) + soft ** 2)
    Nn = pos_t.shape[1]
    eye = 1.0 - np.eye(Nn)
    mm = mass[:, None] * mass[None, :]
    E = ke - (G * mm[None] / r * eye[None]).sum((1, 2)) / 2.0
    L = (mass[None, :, None] * np.cross(pos_t, vel_t)).sum(1)
    Pv = (mass[None, :, None] * vel_t).sum(1)
    E_s = np.abs(E).mean() + 1e-12
    L_s = np.linalg.norm(L, axis=-1).mean() + 1e-12
    P_s = (mass * np.linalg.norm(vel_t, axis=-1)).sum(-1).mean() + 1e-12
    return dict(energy=float(E.max() - E.min()) / E_s,
                angular=float(np.linalg.norm(L.max(0) - L.min(0))) / L_s,
                momentum=float(np.linalg.norm(Pv.max(0) - Pv.min(0))) / P_s)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    def rk(a):
        o = np.argsort(a, kind="mergesort"); r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
        return r
    rx, ry = rk(x), rk(y); rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx * ry).sum() / (np.sqrt((rx ** 2).sum() * (ry ** 2).sum()) + 1e-30))


def hnn_energy(net, pos_t, vel_t, mass, dim):
    T = len(pos_t)
    q = torch.tensor(pos_t.reshape(T, -1), dtype=torch.float32)
    p = torch.tensor((vel_t * mass[None, :, None]).reshape(T, -1), dtype=torch.float32)
    with torch.no_grad():
        return net.H(q, p).squeeze(-1).numpy().astype(np.float64)


def train_hnn_local(P, V, mass, dim, epochs=600, h=128, lr=1e-3, seed=0, dt=DT):
    torch.manual_seed(seed)
    B, T, Nn, _ = P.shape
    m = np.asarray(mass)
    Q = torch.tensor(P.reshape(B, T, Nn * 3), dtype=torch.float32)
    Pm = torch.tensor((V * m[None, None, :, None]).reshape(B, T, Nn * 3), dtype=torch.float32)
    dq = ((Q[:, 1:] - Q[:, :-1]) / dt).reshape(-1, Nn * 3)
    dp = ((Pm[:, 1:] - Pm[:, :-1]) / dt).reshape(-1, Nn * 3)
    qa = Q[:, :-1].reshape(-1, Nn * 3); pa = Pm[:, :-1].reshape(-1, Nn * 3)
    net = HNN(Nn * 3, h=h).to(torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(qa)
    for _ in range(epochs):
        idx = torch.randperm(n)[:512]
        pq, pp = net.time_deriv(qa[idx], pa[idx])
        loss = (((pq - dq[idx]) ** 2).mean() / (dq ** 2).mean()
                + ((pp - dp[idx]) ** 2).mean() / (dp ** 2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def hnn_eq_residual(net, P, V, mass, dt=DT):
    B, T, Nn, _ = P.shape
    m = np.asarray(mass)
    Q = torch.tensor(P.reshape(B, T, Nn * 3), dtype=torch.float32)
    Pm = torch.tensor((V * m[None, None, :, None]).reshape(B, T, Nn * 3), dtype=torch.float32)
    dq = ((Q[:, 1:] - Q[:, :-1]) / dt).reshape(-1, Nn * 3)
    dp = ((Pm[:, 1:] - Pm[:, :-1]) / dt).reshape(-1, Nn * 3)
    with torch.enable_grad():
        pq, pp = net.time_deriv(Q[:, :-1].reshape(-1, Nn * 3), Pm[:, :-1].reshape(-1, Nn * 3))
    return (float(((pq - dq) ** 2).mean().sqrt() / (dq ** 2).mean().sqrt()),
            float(((pp - dp) ** 2).mean().sqrt() / (dp ** 2).mean().sqrt()))


# ---------------------------------------------------------------- 一个基准
def run_benchmark(name, ic_fn, mass, steps, seed):
    rng = np.random.default_rng(seed)
    PT, VT = [], []
    for _ in range(N_TR + N_TE):
        p, v, m = ic_fn(rng)
        pt, vt = rollout(p, v, m, steps)
        PT.append(pt); VT.append(vt)
    PT, VT = np.stack(PT), np.stack(VT)
    TRp, TRv = PT[:N_TR], VT[:N_TR]

    net = train_force(TRp, TRv, mass, epochs=80, h=64, seed=seed)
    hnet = train_hnn_local(TRp, TRv, mass, mass.size, seed=seed)
    r1, r2 = hnn_eq_residual(hnet, PT[N_TR:], VT[N_TR:], mass)

    def mk_block(lo, hi, rs):
        R = np.random.default_rng(rs)
        P, V, E = [], [], []
        for i in range(lo, hi):
            ref_p, ref_v = PT[i], VT[i]
            for s in SIGMAS:
                cp, cv = rollout_nn(ref_p[0], ref_v[0], mass, net, s, R, steps)
                P.append(cp); V.append(cv)
                E.append(float(np.sqrt(((cp - ref_p) ** 2).mean())))
        return np.stack(P), np.stack(V), np.array(E).reshape(hi - lo, K)

    ID = mk_block(N_TR, N_TR + N_TE, seed + 11)

    def cons(pt, vt):
        ch = cons_channels(pt, vt, mass)
        return -(ch["angular"] + ch["momentum"])

    def hnn_sc(pt, vt):
        h = hnn_energy(hnet, pt, vt, mass, mass.size)
        return -float(np.abs(h - h[0]).max())

    crits = [("守恒 (zero-shot)", cons), ("HNN 能量残差", hnn_sc)]
    S = {cn: np.zeros((N_TE, K)) for cn, _ in crits}
    for i in range(N_TE):
        for j in range(K):
            pt, vt = ID[0][i * K + j], ID[1][i * K + j]
            for cn, fn in crits:
                S[cn][i, j] = fn(pt, vt)
    E = ID[2]
    print(f"\n  ==== {name} ====")
    print(f"   HNN 哈密顿方程残差（真实轨迹，测试集）: dq {r1:.4f} | dp {r2:.4f}")
    print(f"   候选 RMSE 中位 = {np.median(E):.5f}  "
          f"IC 内 max/min 中位 = {np.median(E.max(1) / (E.min(1) + 1e-12)):.2f}x")
    rand = float(np.median(E.mean(1))); orac = float(np.median(E.min(1)))
    print(f"   {'critic':<20}{'within-IC ρ':>14}{'IC内更优率':>12}"
          f"{'选中/随机':>11}{'选中/oracle':>12}")
    print("   " + "-" * 70)
    out = {}
    for cn, _ in crits:
        per = [spearman(S[cn][i], E[i]) for i in range(N_TE)]
        per = [x for x in per if not np.isnan(x)]
        wr = float(np.mean([E[i][int(np.argmax(S[cn][i]))] < np.median(E[i])
                            for i in range(N_TE)]))
        pk = float(np.median([E[i][int(np.argmax(S[cn][i]))] for i in range(N_TE)]))
        out[cn] = (float(np.mean(per)), wr, pk / rand, pk / orac)
        print(f"   {cn:<20}{out[cn][0]:>+14.4f}{wr:>11.1%}"
              f"{pk/rand:>10.3f}x{pk/orac:>11.2f}x")
    print(f"   {'[参照] 随机':<20}{'—':>14}{'50.0%':>12}{1.0:>10.3f}x{rand/orac:>11.2f}x")
    print(f"   {'[参照] oracle':<20}{'—':>14}{'100%':>12}{orac/rand:>10.3f}x{1.0:>11.2f}x")
    return out


def main():
    t0 = time.time()
    print("=" * 78)
    print("标准基准上的 critic 选择实验")
    print("=" * 78)

    # --- 1. figure-eight：周期性检查 ---
    p, v, m = system_fig8(np.random.default_rng(0))
    pt, vt = rollout(p, v, m, int(round(FIG8_PERIOD / DT)))
    dev = float(np.abs(pt[-1] - pt[0]).max())
    print(f"\n  figure-eight 周期性检验：一个周期后位置偏差 = {dev:.4e}")
    print(f"   （标准周期 {FIG8_PERIOD}，dt={DT}，共 {int(round(FIG8_PERIOD/DT))} 步）")

    steps = 300
    run_benchmark("figure-eight 三体（N=3, 2D）",
                  lambda r: system_fig8(r, jitter=0.02), np.ones(3), steps, seed=0)
    run_benchmark("Kepler 二体（N=2, 2D）",
                  lambda r: system_kepler(r), np.ones(2), steps, seed=1)

    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
