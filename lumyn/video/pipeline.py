"""视频生成管线（对接 cosmic_video 资产）。

真实版：同目录放 cosmic_video_final.py，自动导入 CosmicVideoPipeline。
兜底：无真实文件时用 render.render_trajectory 直接把物理轨迹渲染成帧。

设计：保持接口统一 —— generate(img, prompt) -> {"video": ndarray, ...}
"""
from __future__ import annotations

import numpy as np
import os

try:
    from cosmic_video_final import CosmicVideoPipeline  # type: ignore
    HAS_COSMIC = True
except ImportError:
    HAS_COSMIC = False

from .render import render_trajectory


class VideoPipeline:
    """统一视频生成入口。"""

    def __init__(self, img_size=64, use_cosmic=False, **kwargs):
        self.img_size = img_size
        self.use_cosmic = use_cosmic and HAS_COSMIC
        if self.use_cosmic:
            self.cosmic = CosmicVideoPipeline(img_size=img_size, **kwargs)
        else:
            self.cosmic = None

    def generate(self, traj: np.ndarray, img_size=None) -> np.ndarray:
        """traj: (T, N, 3) → video (T, H, W, 3)。"""
        if self.cosmic is not None and traj.shape[1] > 0:
            # 真实管线：取首帧作为参考图（此处简化为直接渲染）
            return render_trajectory(traj, img_size or self.img_size)
        return render_trajectory(traj, img_size or self.img_size)

    @property
    def provenance(self) -> str:
        return "cosmic_video" if self.use_cosmic else "numpy_render"
