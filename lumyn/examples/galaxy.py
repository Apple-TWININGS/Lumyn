"""单场景快速示例：生成星系并保存为 numpy 帧。"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # lumyn/ 的父目录

from lumyn import LumynEngine  # noqa: E402
import numpy as np  # noqa: E402


def main():
    engine = LumynEngine(G=1.0, theta=0.5, img_size=64)
    result = engine.generate("一个星系围绕黑洞旋转", world_id="demo")

    v = result["validation"]["conservation"]
    print(f"场景: {result['meta']['scene_type']}")
    print(f"帧数: {result['video'].shape[0]}  分辨率: {result['video'].shape[1:]}")
    print(f"能量守恒: {v['energy_conserved']}  漂移: {v['energy_drift']:.4f}")
    print(f"动量守恒: {v['momentum_conserved']}  漂移: {v['momentum_drift']:.4f}")
    print(f"判定: {result['validation']['verdict']}")

    # 保存帧供外部查看（如需要）
    import numpy as np
    np.save("/data/workspace/lumyn/_demo_frames.npy", result["video"])
    print("已保存: _demo_frames.npy")


if __name__ == "__main__":
    main()
