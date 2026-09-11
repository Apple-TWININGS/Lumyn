"""验证王者/永劫接入层：确定性、公平性、帧同步、移动端调度。

运行：cd <project_root> && PYTHONPATH=. python lumyn/test_new_modules.py
"""
import sys, os, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lumyn.engine_bridge.protocol.deterministic import (
    Fixed16, FixedVec3, sorted_interaction_list, make_frame_packet)
from lumyn.gameplay.moba_rules import (
    make_moba_map, symmetry_score, MobaSymmetryChecker)
from lumyn.gameplay.wuxia_rules import (
    make_arena, height_field, ground_bounce, knockback_distance, WuxiaRuleSet)
from lumyn.gameplay.navmesh_patch import trajectory_to_heightfield
from lumyn.gameplay.fow_mask import compute_fow
from lumyn.sync.lockstep import (
    DeterministicRNG, FrameCommand, LockstepSim, replay_match)
from lumyn.mobile.lod_policy import (
    select_keyframes, interpolate_lod, ThermalGuard, LODPolicy)

PASS = 0
FAIL = 0
ERR = []


def expect(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        ERR.append(msg)
        print(f"  ❌ {msg}")


print("=" * 64)
print("Lumyn —— 王者/永劫接入层验证")
print("=" * 64)

# ---------- 1. 确定性定点数 ----------
print("\n[1] 确定性定点数")
a = Fixed16(1.5)
b = Fixed16(0.5)
expect(abs((a + b).to_float() - 2.0) < 1e-2, "定点加法")
expect(abs((a * b).to_float() - 0.75) < 1e-2, "定点乘法")
v = FixedVec3(3.0, 4.0, 0.0)
expect(abs(v.to_float()[0] - 3.0) < 1e-3, "定点向量 x")
r1 = DeterministicRNG(123)
r2 = DeterministicRNG(123)
f1 = [r1.next() for _ in range(100)]
f2 = [r2.next() for _ in range(100)]
expect(f1 == f2, "同 seed 序列比特一致")

# ---------- 2. 交互列表排序确定性 ----------
print("\n[2] 交互列表确定序")
pairs = [(3, 30), (1, 10), (2, 20)]
sorted_pairs = sorted_interaction_list(pairs)
expect(sorted_pairs == [(1, 10), (2, 20), (3, 30)], f"确定排序 (got {sorted_pairs})")

# ---------- 3. 帧数据包 ----------
print("\n[3] 帧数据包")
pkt = make_frame_packet(frame=7, commands=[FrameCommand(7, 1, 2, 3).encode().decode()], seed_state=0xABC)
expect(pkt["frame"] == 7 and pkt["seed"] == 0xABC, "数据包字段")

# ---------- 4. MOBA 镜像公平 ----------
print("\n[4] MOBA 镜像公平性")
m = make_moba_map(seed=42)
chk = MobaSymmetryChecker(m)
res = chk.check()
expect(res["verdict"] == "fair", f"红蓝镜像对称 (score={res['symmetry_score']:.3f})")
expect(res["blue_camps"] == res["red_camps"], "野区数量差=0")

# ---------- 5. 永劫武侠规则 ----------
print("\n[5] 永劫武侠规则")
wm = make_arena(seed=7)
rule = WuxiaRuleSet(wm)
rv = rule.validate()
expect(rv["verdict"] == "ok", f"锚点可达 (coverage={rv['anchor_coverage']:.2f})")
expect(ground_bounce(-5.0, 0.0) >= 0.0, "地面约束反弹到地面之上")
kb = knockback_distance(force=10.0, mass=2.0)
expect(kb > 0.0, f"击退先验 F=ma 输出正位移 (got {kb:.3f})")

# ---------- 6. NavMesh 高度场 + 视野 ----------
print("\n[6] NavMesh 高度场 / 视野")
traj = np.random.randn(5, 10, 3).cumsum(axis=0)
hf = trajectory_to_heightfield(traj, grid=16)
expect(hf.shape == (16, 16), "高度场形状 16x16")
fow = compute_fow((0.0, 0.0), radius=0.5, blockers=[], grid=32)
expect(fow.shape == (32, 32), "视野掩膜形状 32x32")

# ---------- 7. 帧同步 lockstep ----------
print("\n[7] 帧同步 lockstep（双端比特一致）")
sim_a = LockstepSim(seed=999, n_actors=4)
sim_b = LockstepSim(seed=999, n_actors=4)
for actor in range(4):
    for frame in range(4):
        cmd = FrameCommand(frame, actor, action=1, param=float(actor))
        sim_a.submit(cmd)
        sim_b.submit(cmd)
traj_a, hash_a = sim_a.run(4)
traj_b, hash_b = sim_b.run(4)
expect(np.allclose(traj_a, traj_b), "双端轨迹比特一致")
expect(hash_a == hash_b, "双端状态哈希一致")
replay_h = replay_match(999, 4, 4)
expect(len(replay_h) == 4, "回放哈希序列长度=4")

# ---------- 8. 移动端 LOD / 热控 ----------
print("\n[8] 移动端省算力调度")
kf = select_keyframes(np.array([0.1, 0.9, 0.2, 0.8, 0.3]), budget=2)
expect(sorted(kf.tolist()) == [1, 3], f"QUBO 关键帧 top-2 (got {kf.tolist()})")
lod = LODPolicy(budget=3)
plan = lod.plan(np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6]), temp_c=45.0)
expect(len(plan["keyframes"]) <= 3, "关键帧预算≤3")
tg = ThermalGuard()
expect(tg.update(90.0) == "offline_shadow", "过热→offline_shadow 降级")
expect(tg.update(50.0) == "full", "正常→full")

# ---------- 汇总 ----------
print("\n" + "=" * 64)
print(f"✅ PASS: {PASS}   ❌ FAIL: {FAIL}")
if ERR:
    print("\n错误明细:")
    for e in ERR:
        print(f"  - {e}")
sys.exit(1 if FAIL else 0)
