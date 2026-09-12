"""调试：学习式 critic 为什么训练 loss=0 但 ID AUROC≈0.5？"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import numpy as np
import torch
import torch.nn as nn
from lumyn.experiments.experiment_a1_critic_quality import (
    N, MASS, CORRUPTIONS, F_TRAIN, F_TEST, raw_features, build_family,
    auroc, train_raw_critic, RawCritic,
)

torch.set_default_dtype(torch.float64)

rng_tr = np.random.default_rng(2024)
rng_id = np.random.default_rng(11)
Xtr, Ytr, Ktr = build_family(F_TRAIN, 300, rng_tr)
Xid, Yid, Kid = build_family(F_TRAIN, 150, rng_id)

Ftr = np.stack([raw_features(t) for t in Xtr])
Fid = np.stack([raw_features(t) for t in Xid])
print(f"特征: train {Ftr.shape} id {Fid.shape}")
print(f"特征逐维 std: min={Ftr.std(0).min():.3e} max={Ftr.std(0).max():.3e} "
      f"median={np.median(Ftr.std(0)):.3e}")
print(f"特征全域 abs max = {np.abs(Ftr).max():.2f}")

# 先看一个「完全无脑」的 baseline：用单个最分离的原始特征
print()
print("=== 单特征 AUROC（诊断信号是否真的在数据里）===")
best = []
for j in range(Ftr.shape[1]):
    a = auroc(Fid[:, j], Yid)
    best.append((abs(a - 0.5), j, a))
best.sort(reverse=True)
print("  top-5 单特征:")
for dev, j, a in best[:5]:
    snap = j // (N * 3)
    part = "pos" if (j % (N * 6)) < N * 3 else "vel"
    print(f"    dim {j:>3} (snap{snap} {part}) AUROC = {a:.4f}")

print()
print("=== 重训 RawCritic，观察 train vs ID ===")
for hidden, epochs, lr, wd in [(256, 400, 1e-3, 1e-5),
                               (256, 100, 1e-3, 1e-5),
                               (64, 200, 1e-3, 1e-4),
                               (32, 200, 1e-3, 1e-3)]:
    torch.manual_seed(0)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    Xn = torch.tensor((Ftr - mu) / sd); Yt = torch.tensor(Ytr.astype(np.float64))
    net = RawCritic(Ftr.shape[1], hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(net(Xn), Yt); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        s_tr = net(Xn).numpy()
        s_id = net(torch.tensor((Fid - mu) / sd)).numpy()
    a_tr, a_id = auroc(s_tr, Ytr), auroc(s_id, Yid)
    # 每个损坏类型
    per = []
    for kind in CORRUPTIONS:
        m = (Kid == kind) | (Yid == 1)
        per.append(auroc(s_id[m], Yid[m]))
    print(f"  hidden={hidden:<4} ep={epochs:<4} lr={lr} wd={wd:<7} "
          f"loss={float(loss):.6f} trainAUROC={a_tr:.4f} ID_AUROC={a_id:.4f}")
    print(f"     各类别: " + "  ".join(f"{k}={v:.3f}" for k, v in zip(CORRUPTIONS, per)))
    print(f"     分数分布: 物理 μ={s_id[Yid==1].mean():+.3f} σ={s_id[Yid==1].std():.3f} | "
          f"非物理 μ={s_id[Yid==0].mean():+.3f} σ={s_id[Yid==0].std():.3f}")
