"""FrameExtractor：从动态视频抽取关键帧，拼成网格预览图。

设计动机：科研/教学场景需要在 README、论文、课件里嵌入"一眼看懂演化过程"的
静态预览。抽帧策略对齐 mobile/lod_policy 的 QUBO 关键帧思想——在物理量变化
最大的时刻多抽，静态区间少抽，而不是均匀抽。
"""
from __future__ import annotations
import os
import numpy as np
from typing import List, Optional, Tuple


def _try_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


class FrameExtractor:
    """从 MP4 抽帧，支持均匀 / 变化感知两种策略。"""

    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        self.video_path = video_path
        self._cv2 = _try_cv2()
        if self._cv2 is None:
            raise ImportError("FrameExtractor 需要 opencv-python: pip install opencv-python")

    def extract(self, n: int = 9, strategy: str = "uniform") -> List[np.ndarray]:
        """抽取 n 帧。strategy: 'uniform' 或 'change_aware'。"""
        cv2 = self._cv2
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"无法读取帧数: {self.video_path}")

        if strategy == "change_aware":
            indices = self._change_aware_indices(cap, total, n)
        else:
            indices = np.linspace(0, total - 1, n, dtype=int).tolist()

        frames = []
        prev = None
        idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while cap.isOpened() and len(frames) < n:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in indices:
                frames.append(frame)
            if strategy == "change_aware":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev is not None:
                    change = float(np.abs(gray.astype(np.int16) - prev).mean())
                    if change > 4.0 and len(frames) < n:
                        frames.append(frame)
                prev = gray
            idx += 1
        cap.release()
        return frames

    def _change_aware_indices(self, cap, total: int, n: int) -> List[int]:
        """在帧间差异大的时刻采样（粗扫一遍取 top-k）。"""
        cv2 = self._cv2
        diffs = []
        prev = None
        for _ in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(float(np.abs(gray.astype(np.int16) - prev).mean()))
            prev = gray
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not diffs:
            return np.linspace(0, total - 1, n, dtype=int).tolist()
        diffs = np.array(diffs)
        # 归一化到 [0, total-1]
        norm = (diffs - diffs.min()) / (diffs.max() - diffs.min() + 1e-8)
        scores = norm + np.linspace(0, 0.3, len(norm))  # 略偏向后段（合并更精彩）
        topk = np.argsort(scores)[::-1][:n]
        return sorted(int(i) for i in topk)

    @staticmethod
    def make_grid(frames: List[np.ndarray], cols: int = 3,
                  output_path: str = "preview.png", labels: Optional[List[str]] = None):
        """把帧拼成 cols 列的网格图，带可选标签。"""
        cv2 = _try_cv2()
        n = len(frames)
        rows = (n + cols - 1) // cols
        h = max(f.shape[0] for f in frames)
        w = max(f.shape[1] for f in frames)
        pad = 4
        grid = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 255, dtype=np.uint8)
        for i, frame in enumerate(frames):
            r, c = divmod(i, cols)
            y0 = pad + r * (h + pad)
            x0 = pad + c * (w + pad)
            f = cv2.resize(frame, (w, h))
            grid[y0:y0 + h, x0:x0 + w] = f
            if labels and i < len(labels):
                cv2.putText(grid, labels[i], (x0 + 6, y0 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(output_path, grid)
        return output_path
