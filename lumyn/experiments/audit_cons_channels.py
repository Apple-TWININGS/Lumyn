"""审计：守恒 critic 的各个通道分别有多少判别力？

背景
----
v1 的 cons_drift 把 4 个量纲不同的量**直接相加**（dE + dL + dP + dC）。
和值由量级最大的通道支配，因此「守恒 critic 强/弱」的结论高度依赖
这个任意选择的组合规则 —— 而组合规则本身是一个**没有标签可依**的
超参数选择。这直接冲击「零样本、免训练」的卖点。

本脚本把每个通道单独拿出来测，回答：
  1. 哪个通道真正携带信号？
  2. 上一版的 0.99 是哪个通道贡献的（以及是不是伪影）？
  3. 各通道的归一化方式是否稳健？
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
from lumyn.experiments.experiment_a1_critic_quality import (
    N, MASS, DT, SOFT, G, CORRUPTIONS, F_TRAIN, F_TEST, prior_ic,
    rollout_verlet, corrupt, auroc, build, N_ID, SEV_SUBTLE,
)


def channels(pos_t, vel_t):
    """四个守恒通道，各自用**尺度归一化**（避免除以接近 0 的量）。"""
    m = np.ones(N) * MASS
    ke = 0.5 * (m * (vel_t * vel_t).sum(-1)).sum(-1)
    d = pos_t[:, None, :, :] - pos_t[:, :, None, :]
    r = np.sqrt((d * d).sum(-1) + SOFT ** 2)
    eye = 1.0 - np.eye(N)
    mm = m[:, None] * m[None, :]
    pe = -(mm[None] / r * eye[None]).sum((1, 2)) / 2.0
    E = ke + pe
    L = (m[None, :, None] * np.cross(pos_t, vel_t)).sum(1)
    P = (m[None, :, None] * vel_t).sum(1)
    com = (m[None, :, None] * pos_t).sum(1) / m.sum()

    E_scale = np.abs(E).mean() + 1e-12
    L_scale = np.linalg.norm(L, axis=-1).mean() + 1e-12
    # 动量：净动量天然接近 0，必须用 Σm|v| 这类固有尺度，不能用自己的均值
    P_scale = (m * np.linalg.norm(vel_t, axis=-1)).sum(-1).mean() + 1e-12
    com_scale = np.linalg.norm(pos_t, axis=-1).mean() + 1e-12

    return {
        "energy":   float((E.max() - E.min()) / E_scale),
        "angular":  float(np.linalg.norm(L.max(0) - L.min(0)) / L_scale),
        "momentum": float(np.linalg.norm(P.max(0) - P.min(0)) / P_scale),
        "com":      float(np.linalg.norm(com.max(0) - com.min(0)) / com_scale),
        # 平滑度：非守恒，但守恒 critic 若声称「零样本」，这是对照
        "jerk":     float(np.abs(np.diff(vel_t, 2, axis=0)).mean()),
    }


CH = ("energy", "angular", "momentum", "com", "jerk")


def main():
    print("=" * 82)
    print("物理轨迹上的通道基准（看是否稳定、是否被伪影污染）")
    print("=" * 82)
    P, V, Y, K = build(F_TRAIN, 300, np.random.default_rng(11))
    phys_idx = np.where(Y == 1)[0]
    print(f"  {'通道':<10}{'中位':>14}{'90分位':>14}{'变异系数':>12}")
    for c in CH:
        v = np.array([channels(P[i], V[i])[c] for i in phys_idx])
        print(f"  {c:<10}{np.median(v):>14.4e}{np.percentile(v,90):>14.4e}"
              f"{v.std()/v.mean():>12.3f}")

    # 均匀扰动下的稳定性：物理轨迹加极小扰动，通道不应剧烈跳变
    print()
    print("  稳定性检验（物理轨迹 + 极小位置扰动 1e-6，通道应几乎不变）：")
    rng = np.random.default_rng(0)
    for c in CH:
        a = np.array([channels(P[i], V[i])[c] for i in phys_idx[:60]])
        b = np.array([channels(P[i] + rng.normal(0, 1e-6, P[i].shape), V[i])[c]
                      for i in phys_idx[:60]])
        rel = np.abs(b - a).mean() / (np.abs(a).mean() + 1e-12)
        print(f"    {c:<10} 相对变化 = {rel:.3e}")

    print()
    print("=" * 82)
    print("逐通道 AUROC（ID, e∈[0,0.3], sev=1.0）")
    print("=" * 82)
    print(f"  {'违规':<9}" + "".join(f"{c:>12}" for c in CH) + f"{'四通道和':>12}")
    for k in CORRUPTIONS:
        m = (K == k) | (Y == 1)
        cells = []
        for c in CH:
            s = -np.array([channels(P[i], V[i])[c] for i in range(len(P))])
            cells.append(auroc(s[m], Y[m]))
        ssum = -np.array([sum(channels(P[i], V[i])[c] for c in CH[:4])
                          for i in range(len(P))])
        print(f"  {k:<9}" + "".join(f"{v:>12.4f}" for v in cells)
              + f"{auroc(ssum[m], Y[m]):>12.4f}")

    print()
    print("=" * 82)
    print("OOD 下的逐通道 AUROC（sev=0.25，更隐蔽）")
    print("=" * 82)
    P2, V2, Y2, K2 = build(F_TRAIN, 300, np.random.default_rng(33), sev=SEV_SUBTLE)
    print(f"  {'违规':<9}" + "".join(f"{c:>12}" for c in CH) + f"{'四通道和':>12}")
    for k in CORRUPTIONS:
        m = (K2 == k) | (Y2 == 1)
        cells = []
        for c in CH:
            s = -np.array([channels(P2[i], V2[i])[c] for i in range(len(P2))])
            cells.append(auroc(s[m], Y2[m]))
        ssum = -np.array([sum(channels(P2[i], V2[i])[c] for c in CH[:4])
                          for i in range(len(P2))])
        print(f"  {k:<9}" + "".join(f"{v:>12.4f}" for v in cells)
              + f"{auroc(ssum[m], Y2[m]):>12.4f}")

    print()
    print("=" * 82)
    print("各通道的物理含义检查：freeze 是否真的破坏守恒？")
    print("=" * 82)
    i = phys_idx[0]
    base = channels(P[i], V[i])
    fp, fv = corrupt(P[i], V[i], "freeze", np.random.default_rng(0), 1.0)
    fr = channels(fp, fv)
    print(f"  {'通道':<10}{'物理':>14}{'freeze 后':>14}{'比值':>10}")
    for c in CH:
        print(f"  {c:<10}{base[c]:>14.4e}{fr[c]:>14.4e}"
              f"{fr[c]/(base[c]+1e-12):>10.2f}")
    print("  freeze 让后半段静止：动能归零 → 能量必然大幅变化，")
    print("  但若 energy 通道的归一化尺度 |E| 本身也随之改变，比值可能反而变小。")


if __name__ == "__main__":
    main()
