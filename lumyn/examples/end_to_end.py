"""端到端示例：prompt → 场景 → 视频 + 物理验证。

运行：
    cd <项目根，即 lumyn/ 的父目录>
    python -m lumyn.examples.end_to_end
或：
    cd lumyn && python examples/end_to_end.py
"""
import os
import sys

# 让 'lumyn' 成为可导入的顶层包
_HERE = os.path.dirname(os.path.abspath(__file__))
# examples/ -> lumyn/ -> 项目根
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from lumyn import LumynEngine  # noqa: E402


PROMPTS = [
    "一个星系围绕黑洞缓慢旋转",
    "两颗台球发生弹性碰撞",
    "超新星爆炸，碎片向四周飞散",
]


def main():
    engine = LumynEngine(G=1.0, theta=0.5, img_size=64, simscore_threshold=0.6)

    for prompt in PROMPTS:
        print("=" * 64)
        print(f"Prompt: {prompt}")
        # galaxy 引力梯度陡，用更小 dt 保证守恒精度
        dt = 0.005 if "星系" in prompt or "galaxy" in prompt.lower() else 0.01
        result = engine.generate(prompt, world_id="game_alpha",
                                n_particles=120, n_steps=15, dt=dt, max_retries=3)

        v = result["validation"]
        cons = v["conservation"]
        print(f"  场景类型 : {result['meta']['scene_type']}")
        print(f"  粒子数   : {result['meta']['n_particles']}")
        print(f"  视频帧   : {result['video'].shape}")
        print(f"  能量守恒 : {cons['energy_conserved']}  (drift={cons['energy_drift']:.4f})")
        print(f"  动量守恒 : {cons['momentum_conserved']}  (drift={cons['momentum_drift']:.4f})")
        print(f"  SimScore : {v['simscore']:.3f}")
        print(f"  判定     : {v['verdict']}  (retries={result['meta']['retries']})")

    print("\n🎉 端到端验证完成")


if __name__ == "__main__":
    main()
