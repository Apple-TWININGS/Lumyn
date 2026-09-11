"""王者（MOBA）/ 永劫（武侠）端到端管线示例。

演示 Lumyn 如何从「prompt」走到「物理可信 + 竞技公平 + 帧同步确定」的游戏场景。

运行：cd lumyn && python examples/moba_wuxia_e2e.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lumyn import (
    # 导演 + 引擎
    PhysicsDirector, HMMPGameEngine,
    # 帧同步
    LockstepSimulator, InputCommand, verify_determinism,
    # 移动端
    LODPolicy, ThermalGuard, NPUDelegate, select_keyframes, adaptive_step, BarnesHutTree,
    # MOBA
    generate_moba_map, check_gameplay_fairness, physics_path_symmetry,
    # 武侠
    generate_wuxia_map, check_anchor_reachability, apply_ground_constraint, validate_knockback,
    # 协议
    DeterministicRNG, FramePacket,
)


def run_moba():
    print("\n" + "=" * 60)
    print("【王者向 · MOBA】三路对称地图 PCG + 公平校验")
    print("=" * 60)

    director = PhysicsDirector()
    plan = director.parse("生成一张三路对称的王者峡谷，红蓝镜像")
    print(f"导演解析: {plan}")

    # 1. 确定性生成地图（同 seed 必同图）
    m_blue = generate_moba_map(seed=42)
    m_red = m_blue.mirror("red")
    print(f"  蓝方基地: {m_blue.bases['blue']}, 红方基地: {m_blue.bases['red']}")
    print(f"  防御塔数: {len(m_blue.towers)}, 野区营地: {len(m_blue.camps)}")

    # 2. 竞技公平校验（结构对称）
    ok, score = check_gameplay_fairness(m_blue, m_red, tol=1e-6)
    print(f"  结构对称: {ok}  site_error={score['site_error']:.2e}  "
          f"塔差={score['tower_count_diff']}  营差={score['camp_count_diff']}")
    assert ok, "红蓝必须镜像对称"

    # 3. 物理路径对称（HMMP 演化后能量曲线重合）
    n = 8
    rng = np.random.RandomState(0)
    pos_blue = rng.randn(n, 3) * 5
    pos_red = -pos_blue.copy()
    mass = np.ones(n)
    dev = physics_path_symmetry(pos_blue, pos_red, mass, n_steps=10)
    print(f"  路径能量最大偏差: {dev['max_energy_deviation']:.2e}  (应≈0)")
    assert dev["max_energy_deviation"] < 1e-5

    # 4. 帧同步：同 seed 双端比特一致
    from lumyn.sync.lockstep import SimState
    N = 4
    sa = LockstepSimulator(SimState(positions=np.zeros((N, 3)), velocities=np.zeros((N, 3)), frame_id=0), seed=999)
    sb = LockstepSimulator(SimState(positions=np.zeros((N, 3)), velocities=np.zeros((N, 3)), frame_id=0), seed=999)
    cmds = [[InputCommand(i, 0, float(i))] for i in range(4)]
    same, diffs = verify_determinism(sa, sb, cmds, n_frames=4)
    print(f"  帧同步双端一致: {same}  max_diff={max(diffs):.2e}")
    assert same

    print("  ✅ MOBA 管线通过")


def run_wuxia():
    print("\n" + "=" * 60)
    print("【永劫向 · 武侠】擂台高度场 + 钩索 + 地面约束 + 击退先验")
    print("=" * 60)

    wm = generate_wuxia_map(seed=7)
    print(f"  高度场: {wm.heightfield.shape}, 钩索锚点: {len(wm.anchors)}, "
          f"平台: {len(wm.platforms)}")

    # 1. 钩索锚点可达性（起点设在擂台边缘，钩索最远距离按地图对角线上限）
    edge = (wm.size * 0.3, 0.0, wm.sample_height(wm.size * 0.3, 0.0))
    reach = check_anchor_reachability(wm, start=edge, max_range=wm.size * 1.5)
    diag = reach["diagnostics"]
    print(f"  锚点覆盖率: {reach['coverage']:.2%}  "
          f"可达{len(reach['reachable'])}/{len(wm.anchors)}  "
          f"(超出范围={diag['n_out_of_range']}, 低于地面={diag['n_below_ground']}, "
          f"建议max_range={diag['suggested_max_range']:.1f})")
    assert reach["coverage"] > 0, "合理钩索范围下应有可达锚点"

    # 2. 3D 地面约束（粒子低于地形则反弹）
    pos = np.array([[0.0, 0.0, -5.0], [3.0, 2.0, 1.0]], dtype=np.float64)
    vel = np.array([[1.0, 1.0, -2.0], [0.0, 0.0, 0.5]], dtype=np.float64)
    p2, v2 = apply_ground_constraint(pos, vel, wm.sample_height)
    assert p2[0, 2] >= wm.sample_height(0, 0) - 1e-6, "粒子必须贴地"
    print(f"  地面约束: 首粒子 z={p2[0,2]:.3f} (地形高={wm.sample_height(0,0):.3f})")

    # 3. 技能击退先验 F=ma
    kb = validate_knockback(force=10.0, mass=2.0, expected_accel=5.0)
    print(f"  击退校验: a={kb['actual_accel']:.2f} (期望5.0)  rel_err={kb['rel_error']:.3f}")
    assert kb["ok"]

    # 4. 移动端 LOD + 热控 + NPU
    lod = LODPolicy()
    print(f"  LOD 调度: 正常→{lod.update(45,60)}  过热→{lod.update(72,25)}")
    assert lod.update(72, 25) == "LOW"
    tg = ThermalGuard(max_temp=70.0)
    assert tg.check(75.0, 2000.0) in ("CRITICAL", "HIGH")
    npu = NPUDelegate(available=True)
    npu.run_motion_prior(np.zeros((1, 32), dtype=np.float32))
    assert npu.stats["motion_prior"] == 1

    # 5. adaptive_step 三档不报错
    pos3 = np.random.RandomState(0).randn(8, 3)
    mass3 = np.ones(8)
    center, size = pos3.mean(0), float(np.abs(pos3 - pos3.mean(0)).max()) * 2 + 1e-6
    tree = BarnesHutTree(max_leaf=4, max_depth=8)
    tree.build(pos3, mass3, center, size)
    for lv in ["HIGH", "MED", "LOW"]:
        acc = adaptive_step(tree, 0.5, lv)
        assert acc.shape == (8, 3)
    print(f"  物理步进: HIGH/MED/LOW 三档 OK")

    print("  ✅ 武侠管线通过")


def run_frame_sync():
    print("\n" + "=" * 60)
    print("【帧同步】数据包序列化 + lockstep 回放导出")
    print("=" * 60)
    pkt = FramePacket(frame_id=7, seed=0xABC, commands=[(1, 2, 3.14)], state_hash=99)
    data = pkt.encode()
    pkt2 = FramePacket.decode(data)
    assert pkt2.frame_id == 7
    # param 经 float32 序列化，允许 ~1e-6 误差（这是协议精度，非业务误差）
    eid, act, param = pkt2.commands[0]
    assert (eid, act) == (1, 2) and abs(param - 3.14) < 2e-6
    print(f"  数据包往返 OK ({len(data)} bytes, float32 精度)")
    rng = DeterministicRNG(123)
    f1 = [rng.next_float() for _ in range(100)]
    rng2 = DeterministicRNG(123)
    f2 = [rng2.next_float() for _ in range(100)]
    assert f1 == f2
    print(f"  确定性 RNG: 同 seed 序列一致 (100 个浮点)")
    print("  ✅ 帧同步通过")


if __name__ == "__main__":
    run_moba()
    run_wuxia()
    run_frame_sync()
    print("\n" + "=" * 60)
    print("🎉 王者 / 永劫 接入管线全部通过")
    print("=" * 60)
