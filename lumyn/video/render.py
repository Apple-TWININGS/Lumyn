"""视频渲染：把粒子轨迹 (T,N,3) → 视频帧 (T,H,W,3)。

轻量实现：归一化 + splatting + 小高斯模糊，无 GPU 依赖。
接入真实 cosmic_video 时，可在此替换为 VAE 解码 + 条件扩散。
"""
from __future__ import annotations

import numpy as np


def render_trajectory(traj: np.ndarray, img_size=64, blur=True) -> np.ndarray:
    """把 (T, N, 3) 轨迹渲染成 (T, H, W, 3) 灰度/单通道视频。

    只取 x-y 平面投影；亮度正比于粒子数密度。
    """
    n_steps, n_particles, _ = traj.shape
    frames = np.zeros((n_steps, img_size, img_size, 3), dtype=np.float32)

    pmin = traj.min(axis=(0, 1))
    pmax = traj.max(axis=(0, 1))
    scale = (pmax - pmin) + 1e-8

    for t in range(n_steps):
        pos = (traj[t] - pmin) / scale          # (N,3) ∈ [0,1]
        for i in range(n_particles):
            x = int(pos[i, 0] * (img_size - 1))
            y = int(pos[i, 1] * (img_size - 1))
            if 0 <= x < img_size and 0 <= y < img_size:
                frames[t, y, x] = 1.0
        if blur:
            frames[t] = _gaussian_blur3(frames[t])
    return frames


def _gaussian_blur3(img: np.ndarray) -> np.ndarray:
    """3x3 可分高斯模糊。"""
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        for i in range(3):
            for j in range(3):
                out[:, :, c] += kernel[i, j] * pad[i:i + img.shape[0], j:j + img.shape[1], c]
    return np.clip(out, 0, 1)
