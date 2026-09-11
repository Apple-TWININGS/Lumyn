"""导演层：把 prompt 解析成符号执行轨迹 + 物理参数。

- 离线：基于关键词的规则解析（无 API 也能跑）
- 在线：可插拔 DeepSeekPlanner（复用 cosmic_video 的 plan_shot）
"""
from __future__ import annotations

import os
import numpy as np


class Director:
    """prompt → 符号执行轨迹 + 场景生成参数。"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.planner = None  # 懒加载：避免无 cosmic_video 时报错

    def parse(self, prompt: str) -> dict:
        """返回完整轨迹字典（对齐 SimScore 的输入格式）。"""
        trace = self._rule_based(prompt)
        if self.api_key:
            trace = self._enrich_with_deepseek(prompt, trace)
        return trace

    def plan_scene(self, prompt: str) -> dict:
        """返回场景生成参数 (scene_type, n_particles, ...)。"""
        p = prompt.lower()
        if any(k in p for k in ["galaxy", "orbit", "star", "planet", "星系", "轨道"]):
            scene_type = "galaxy"
        elif any(k in p for k in ["explode", "burst", "爆炸", "爆开"]):
            scene_type = "explosion"
        elif any(k in p for k in ["collide", "crash", "碰撞", "撞"]):
            scene_type = "collision"
        else:
            scene_type = "galaxy"  # 默认

        trace = self.parse(prompt)
        n_particles = 300 if scene_type == "collision" else 500

        return {
            "scene_type": scene_type,
            "n_particles": n_particles,
            "G": 1.0,
            "dt": 0.01 if scene_type != "explosion" else 0.005,
            "n_steps": 50,
            "repulsion": 0.8 if scene_type in ("collision", "explosion") else 0.0,
            # 中心势阱（星系）需要更精确的力计算，用更小 theta
            "theta": 0.3 if scene_type == "galaxy" else 0.5,
            "trace": trace,
        }

    def _rule_based(self, prompt: str) -> dict:
        p = prompt.lower()
        trace = {
            "laws": [],
            "variables": {},
            "steps": [],
            "constraints": [],
            "camera_motion": "static",
            "subject_motion": "idle",
            "motion_intensity": 0.3,
            "keyframe_sec": [0.0, 0.5, 1.0],
        }
        if any(k in p for k in ["orbit", "galaxy", "star", "轨道", "星系"]):
            trace["laws"] = ["Newton2", "ConservationOfEnergy"]
            trace["steps"] = [
                {"law": "Newton2", "expr": "F = G*m1*m2/r^2"},
                {"law": "ConservationOfEnergy", "expr": "E = 0.5*m*v^2 - G*M*m/r"},
            ]
            trace["constraints"] = ["energy_conserved", "momentum_conserved"]
            trace["subject_motion"] = "orbit"
            trace["motion_intensity"] = 0.6
        elif any(k in p for k in ["collide", "碰撞"]):
            trace["laws"] = ["ConservationOfMomentum"]
            trace["steps"] = [
                {"law": "ConservationOfMomentum", "expr": "m1*v1 + m2*v2 = const"},
            ]
            trace["constraints"] = ["momentum_conserved"]
        return trace

    def _enrich_with_deepseek(self, prompt: str, trace: dict) -> dict:
        """尝试用 DeepSeek 分镜结果丰富轨迹（可选，失败不影响离线流程）。"""
        try:
            if self.planner is None:
                from cosmic_video_final import DeepSeekPlanner  # type: ignore
                self.planner = DeepSeekPlanner(api_key=self.api_key)
            plan = self.planner.plan_shot(prompt, img=None, num_frames=8)
            if plan:
                trace["camera_motion"] = plan.get("camera_motion", trace["camera_motion"])
                trace["subject_motion"] = plan.get("subject_motion", trace["subject_motion"])
                trace["motion_intensity"] = plan.get("motion_intensity", trace["motion_intensity"])
        except Exception:
            pass
        return trace
