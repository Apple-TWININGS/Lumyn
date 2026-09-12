"""实验 I：合成的违规分类**是否代表**真实生成模型的失效模式？

这是论文的承重假设
------------------
若「noise / euler / freeze / splice / warp」这五类人工损坏与真实生成模型的
失效模式**不同构**，那么任何在合成损坏集上测出的 critic 排名都不能外推，
论文不能用那套结论。本实验直接检验这一点。

方法（关键：先对齐严重度）
--------------------------
直接比指纹是不公平的：损坏强度不同会让任何两个分布都「不像」。
因此先用二分法把每种人工损坏的严重度调到使其**相对真实解的 RMSE**
与神经生成器相同（同一分位数），再比较失效**模式**。

指纹（每类样本的分布）
----------------------
  四个守恒通道（归一化漂移）+ 三个非守恒结构量：
    step_jump : 单步最大位移 / 位置尺度   —— 抓结构性不连续（splice 类）
    jerk      : |Δ²v| 的均值 / 加速度尺度 —— 抓平滑度异常
    speed_ratio: 平均速率 / 真实平均速率  —— 抓冻结 / 漂移
"""
import sys, os, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np

from lumyn.experiments.experiment_a1_critic_quality import (
    prior_ic, rollout_verlet, corrupt, CORRUPTIONS, BASE_SEV,
)
from lumyn.experiments.experiment_a2_neural_generator import train_force_model
from lumyn.experiments.experiment_d_channels_and_generators import (
    channels, rollout_net, N, MASS, DT, F_TRAIN,
)

N_SAMPLE = 40
FEATS = ("energy", "angular", "momentum", "com",
         "step_jump", "jerk", "speed_ratio")


def fingerprint(pos_t, vel_t, ref_p, ref_v):
    """返回指纹向量（无量纲）。ref_* 为对应的真实轨迹。"""
    ch = channels(pos_t, vel_t)
    lscale = float(np.linalg.norm(pos_t, axis=-1).mean()) + 1e-12
    ascale = float(np.linalg.norm(np.diff(vel_t, axis=0), axis=-1).mean()) + 1e-12

    step_jump = float(np.linalg.norm(np.diff(pos_t, axis=0), axis=-1).max()) / lscale
    acc = np.diff(vel_t, axis=0)
    jerk = float(np.linalg.norm(np.diff(acc, axis=0), axis=-1).mean()) / ascale
    ref_speed = float(np.linalg.norm(ref_v, axis=-1).mean()) + 1e-12
    speed_ratio = float(np.linalg.norm(vel_t, axis=-1).mean()) / ref_speed

    return np.array([ch["energy"], ch["angular"], ch["momentum"], ch["com"],
                     step_jump, jerk, speed_ratio])


def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


def sample_family(kind, gen, sev, rng):
    """生成一条样本，返回 (指纹, rmse)。kind 为 'gen' 或某个损坏名。"""
    p0, v0 = prior_ic(rng.uniform(*F_TRAIN), rng)
    ref_p, ref_v = rollout_verlet(p0, v0, np.ones(N) * MASS)
    if kind == "gen":
        tp, tv = rollout_net(p0, v0, gen, 0.15, rng)
    else:
        # 先得到一条真实轨迹，再按指定严重度损坏它
        tp, tv = corrupt(ref_p, ref_v, kind, rng, sev)
    return fingerprint(tp, tv, ref_p, ref_v), rmse(tp, ref_p), tp, tv


def calibrate(kind, rng, target_rmse, lo=0.02, hi=8.0, iters=14):
    """二分严重度，使该损坏的中位 RMSE 接近目标。"""
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        vals = []
        for _ in range(8):
            _, e, _, _ = sample_family(kind, None, mid, rng)
            vals.append(e)
        if np.median(vals) < target_rmse:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def main():
    t0 = time.time()
    rng = np.random.default_rng(1)
    Ptr, Vtr = [], []
    for _ in range(200):
        p, v = prior_ic(rng.uniform(*F_TRAIN), rng)
        pt, vt = rollout_verlet(p, v, np.ones(N) * MASS)
        Ptr.append(pt); Vtr.append(vt)
    Ptr, Vtr = np.stack(Ptr), np.stack(Vtr)
    mass = np.ones(N) * MASS

    gens = {}
    gens["生成器-弱"] = train_force_model(Ptr[:60], Vtr[:60], mass, epochs=6, h=32, lr=3e-3)
    gens["生成器-中"] = train_force_model(Ptr[:120], Vtr[:120], mass, epochs=12, h=32, lr=3e-3)
    gens["生成器-强"] = train_force_model(Ptr, Vtr, mass, epochs=120, h=64, lr=2e-3)
    print(f"三个生成器就绪（{time.time()-t0:.0f}s）")

    # ---------------- 1. 各类样本的指纹与 RMSE ----------------
    print()
    print("=" * 88)
    print("1. 各类失效模式的指纹（先对齐严重度：把 RMSE 调到与生成器同级）")
    print("=" * 88)

    ref_rng = np.random.default_rng(500)
    gen_rows = {}
    for gname, g in gens.items():
        F, E = [], []
        for _ in range(N_SAMPLE):
            f, e, _, _ = sample_family("gen", g, 0.0, ref_rng)
            F.append(f); E.append(e)
        gen_rows[gname] = (np.array(F), np.array(E))
        print(f"  {gname}: RMSE 中位 = {np.median(E):.5f}")

    # 以「生成器-中」为目标严重度
    target = float(np.median(gen_rows["生成器-中"][1]))
    print(f"\n  目标严重度（对齐用）= 生成器-中 的 RMSE 中位 = {target:.5f}")

    corr_rows = {}
    for kind in CORRUPTIONS:
        k = "noise"
        sev = calibrate(kind, np.random.default_rng(900), target)
        F, E = [], []
        r2 = np.random.default_rng(901)
        for _ in range(N_SAMPLE):
            f, e, _, _ = sample_family(kind, None, sev, r2)
            F.append(f); E.append(e)
        corr_rows[kind] = (np.array(F), np.array(E), sev)
        print(f"  {kind:<8} 校准严重度 = {sev:8.4f}  → RMSE 中位 = {np.median(E):.5f}")

    # ---------------- 2. 指纹距离 ----------------
    print()
    print("=" * 88)
    print("2. 指纹距离：真实生成器 vs 各人工损坏")
    print("    用**标准化**后的指纹（按真实轨迹的标准差缩放），再做中位数 L1 距离")
    print("=" * 88)

    # 标准化尺度：取「生成器-中」的逐维 IQR，避免量级差异主导
    Fg = gen_rows["生成器-中"][0]
    scale = np.percentile(Fg, 75, axis=0) - np.percentile(Fg, 25, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)

    print(f"  {'人工损坏':<10}{'归一化 L1 距离':>18}{'最接近的生成器':>18}")
    print("  " + "-" * 50)
    dists = {}
    for kind in CORRUPTIONS:
        Fc = corr_rows[kind][0] / scale
        best = None
        for gname, (Fgg, _) in gen_rows.items():
            d = float(np.median(np.abs(Fc - Fgg / scale).sum(axis=1)))
            dists[(kind, gname)] = d
            if best is None or d < best[1]:
                best = (gname, d)
        print(f"  {kind:<10}{best[1]:>18.3f}{best[0]:>18}")
    print("\n  （距离越小越像；若所有距离都很大，说明人工损坏与真实失效模式不同构）")

    # ---------------- 3. 哪个通道在两种集合上都重要？ ----------------
    print()
    print("=" * 88)
    print("3. **决策相关**：conservation critic 的通道排序在两类集合上是否一致？")
    print("=" * 88)
    print("    对每个通道，计算它区分「真实轨迹 vs 该类失效」的分离度（生成/真实比值）")
    print()
    hdr = f"  {'通道':<10}"
    for kind in CORRUPTIONS:
        hdr += f"{kind:>12}"
    hdr += f"{'生成器-中':>12}"
    print(hdr)
    print("  " + "-" * (10 + 12 * (len(CORRUPTIONS) + 1)))

    ref = []
    r3 = np.random.default_rng(700)
    for _ in range(N_SAMPLE):
        p0, v0 = prior_ic(r3.uniform(*F_TRAIN), r3)
        rp, rv = rollout_verlet(p0, v0, mass)
        ref.append(channels(rp, rv))
    ref = {c: np.median([r[c] for r in ref]) for c in
           ("energy", "angular", "momentum", "com")}

    refg = {c: np.median(gen_rows["生成器-中"][0][:, i])
            for i, c in enumerate(("energy", "angular", "momentum", "com"))}

    for ci, c in enumerate(("energy", "angular", "momentum", "com")):
        row = f"  {c:<10}"
        for kind in CORRUPTIONS:
            v = np.median(corr_rows[kind][0][:, ci])
            row += f"{v / (ref[c] + 1e-300):>12.2f}"
        row += f"{refg[c] / (ref[c] + 1e-300):>12.2f}"
        print(row)

    print()
    print("  解读：若「生成器-中」这一列与某一损坏列的**形态**（各通道相对高低）不同，")
    print("        则该人工损坏不能代表真实失效模式。")

    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
