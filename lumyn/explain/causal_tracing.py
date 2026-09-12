"""管线层因果追踪（损坏 / 恢复实验）。

======================================================================
重要更正
======================================================================
本模块此前**返回硬编码常数**：

    base  = {"galaxy": 0.85, "collision": 0.80, "explosion": 0.75}.get(...)
    decay = {"director": 0.12, "generate": 0.30, "validate": 0.55}

这些数字不是测量值。README 曾据此声称「验证层贡献度 >50%，是论文中最有力的证据」。
该说法已被删除。本文件为**真实实现**，不再包含任何常数分数。

同时更正一条设计前提：**导演层当前不在生成路径上**。
`LumynEngine.generate()` 的实现是

    scene = presets.get(scene_type, **kwargs)
    traj, mass = scene.run(steps=steps)
    validation = self.validator.diagnose(traj, mass, dt=scene.dt)

既不调用 director，也不依据 validation 做任何回退或重生成。
因此：
  - `director` 对输出**没有因果路径**，其贡献度记为 `None`（不是 0，是"不可测"）
  - `validate` **只产生诊断信息，不影响输出**，其因果贡献度为 **0**

======================================================================
协议（与原始论文的「逐层恢复」语义一致）
======================================================================
   clean            ：各层均正常
   all_corrupt      ：所有**在路径上**的层均损坏 → 基线
   corrupt(k)       ：只损坏第 k 层
   restore(k)       ：其余层保持损坏，只恢复第 k 层
   贡献度(k) = score(restore k) − score(all_corrupt)

各层的"损坏"含义（均为真实操作）
--------------------------------
   generate ：在生成出的轨迹上逐步注入过程噪声（模拟动力学不准的生成器）
   validate ：**关闭验证门**。门的语义是「不通过则回退到安全网」。
              关闭后，不合格的轨迹会被原样放行。
   director ：不在生成路径上，无法做有意义的损坏（见上）

`enable_validation_gate` 的默认值是 `False`，因为这是**当前引擎的真实行为**
（验证结果不被使用）。设为 `True` 会启用一个真实生效的验证门，
从而可以测量「如果验证层真的起作用，它能贡献多少」。
报告会标明本次运行采用的是哪种模式。

安全网（fallback）
------------------
验证门判定不合格时回退到「上一次已知良好的轨迹」。真实系统中这对应
「回退到上一个关键帧重生成」；本实现中以**同一场景的未损坏轨迹**作为该安全网的
代理（它必然物理合法），并在报告的 `config` 中明确标注该代理。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..eval.conservation_critic import ConservationCritic
from ..scientific import presets

#: 只有在生成路径上的层才可能有非零因果贡献。
ON_PATH_LAYERS = ("generate", "validate")
#: 全部被报告的层。`director` 在路径外，仅为向后兼容保留条目。
ALL_LAYERS = ("director", "generate", "validate")

#: 生成层损坏：每步注入的过程噪声强度（相对当前加速度尺度）。
DEFAULT_NOISE = 0.35
#: 判定"物理合格"的门限：干净轨迹残差中位数的该倍数。
DEFAULT_GATE_FACTOR = 5.0


@dataclass
class LayerResult:
    """单层的损坏 / 恢复分数。"""
    layer: str
    on_generation_path: bool = True
    score_clean: float = 0.0      # 干净基线（参考，不进差值）
    score_corrupt: float = 0.0    # 只损坏该层
    score_restore: float = 0.0    # 其余层损坏、只恢复该层
    contribution: Optional[float] = None   # restore − all_corrupt；不在路径上为 None
    note: str = ""


@dataclass
class TracingReport:
    """跨场景汇总报告。**所有数值均为实测。**"""
    per_scene: Dict[str, Dict[str, LayerResult]] = field(default_factory=dict)
    averaged: Dict[str, Optional[float]] = field(default_factory=dict)
    all_corrupt: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "all_corrupt": self.all_corrupt,
            "per_scene": {s: {n: asdict(r) for n, r in layers.items()}
                          for s, layers in self.per_scene.items()},
            "averaged": self.averaged,
        }

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def summary(self) -> str:
        lines = ["=== 管线层因果追踪（实测）==="]
        mode = ("验证门已启用" if self.config.get("enable_validation_gate")
                else "验证门未启用（与当前引擎行为一致：验证结果不影响输出）")
        lines.append(f"模式：{mode}")
        lines.append(f"场景得分口径：物理性分数（越大越物理），"
                     f"贡献度 = restore − all_corrupt")
        lines.append("")
        lines.append(f"{'scene':<14}{'director':>10}{'generate':>10}{'validate':>10}"
                     f"{'all_corrupt':>14}")
        for scene, layers in self.per_scene.items():
            cells = []
            for n in ALL_LAYERS:
                c = layers[n].contribution if n in layers else None
                cells.append("  n/a（路径外）" if c is None else f"{c:>10.4f}")
            lines.append(f"{scene:<14}" + "".join(cells)
                         + f"{self.all_corrupt.get(scene, float('nan')):>14.4f}")
        if self.averaged:
            lines.append("")
            lines.append("--- 跨场景平均 ---")
            cells = []
            for n in ALL_LAYERS:
                v = self.averaged.get(n)
                cells.append("  n/a（路径外）" if v is None else f"{v:>10.4f}")
            lines.append(f"{'avg':<14}" + "".join(cells))
        lines.append("")
        lines.append("注：director 不在 LumynEngine.generate 的调用链上，"
                     "其贡献度不可测（None），不是 0。")
        return "\n".join(lines)

    def plot(self, path: str = "causal_tracing.png") -> Optional[str]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None
        layers = [l for l in ALL_LAYERS
                  if any(self.per_scene[s][l].contribution is not None
                         for s in self.per_scene)]
        scenes = list(self.per_scene.keys())
        if not layers:
            return None
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(layers))
        w = 0.8 / max(len(scenes), 1)
        for si, scene in enumerate(scenes):
            vals = [self.per_scene[scene][l].contribution for l in layers]
            ax.bar(x + w * (si - len(scenes) / 2 + 0.5), vals, width=w, label=scene)
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.set_ylabel("contribution (restore − all_corrupt)")
        ax.set_title("Causal tracing over Lumyn pipeline layers (measured)")
        ax.axhline(0, color="gray", lw=0.8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


# ---------------------------------------------------------------- 主类
class CausalTracer:
    """对 Lumyn 管线做真实的损坏 / 恢复实验。

    参数
    ----
    engine : 具有 `generate(scene_type, steps=...)` 的引擎；缺省用 LumynEngine。
    enable_validation_gate : 是否启用真实生效的验证门。
        **默认 False** —— 与当前引擎行为一致（验证结果被计算但不影响输出）。
        设为 True 时，门的语义是「守恒残差超阈值 → 回退到安全网」。
    noise : 生成层的损坏强度。
    gate_factor : 门限 = 干净轨迹残差中位数 × gate_factor。
    seed : 决定噪声的随机种子，保证可复现。
    """

    LAYERS = ALL_LAYERS

    def __init__(self, engine=None, *, enable_validation_gate: bool = False,
                 noise: float = DEFAULT_NOISE, gate_factor: float = DEFAULT_GATE_FACTOR,
                 steps: int = 40, seed: int = 0, mass: Optional[np.ndarray] = None,
                 n_particles: int = 150):
        if engine is None:
            from ..core.engine import LumynEngine
            engine = LumynEngine()
        self.engine = engine
        self.enable_validation_gate = bool(enable_validation_gate)
        self.noise = float(noise)
        self.gate_factor = float(gate_factor)
        self.steps = int(steps)
        self.seed = int(seed)
        self.n_particles = int(n_particles)
        self._mass = mass
        self._G = float(getattr(engine, "_G", 1.0))
        # ConservationCritic 的 mass 是**构造参数**而非方法参数，而每个场景的质量
        # 数组不同，因此按场景缓存 critic 实例。
        self._critic_cache: Dict[str, ConservationCritic] = {}
        if mass is not None:
            self._critic_cache["__default__"] = ConservationCritic(mass=mass, G=self._G)
        self._gate_cache: Dict[str, float] = {}
        # 干净轨迹必须缓存：每次 _run 都重算的话，一次 trace 会重复生成几十遍，
        # 在 N=400~800 时既慢又吃内存（ConservationCritic 会构造 (T,N,N,3) 中间数组）。
        self._clean_cache: Dict[str, tuple] = {}

    # ------------------------------------------------------------ 基本构件
    def _clean_trajectory(self, scene_type: str):
        """未经损坏的生成结果（同时充当安全网代理）。**结果被缓存。**"""
        if scene_type not in self._clean_cache:
            scene = presets.get(scene_type, n_particles=self.n_particles)
            traj, mass = scene.run(steps=self.steps)
            self._clean_cache[scene_type] = (traj, mass, scene)
        return self._clean_cache[scene_type]

    @staticmethod
    def _vel(traj: np.ndarray, dt: float) -> np.ndarray:
        """由位置差分得到速度（末帧沿用倒数第二帧，避免零速度伪影）。"""
        vel = np.zeros_like(traj)
        vel[:-1] = (traj[1:] - traj[:-1]) / dt
        vel[-1] = vel[-2] if len(traj) > 1 else 0.0
        return vel

    def _degrade(self, traj: np.ndarray, dt: float, rng) -> np.ndarray:
        """生成层损坏：逐步注入累积过程噪声。

        噪声按**当前位移尺度**缩放，因此与场景规模无关；
        噪声逐步累积 → 破坏的正是长期守恒，而非单帧抖动。
        """
        out = traj.copy()
        scale = float(np.abs(np.diff(traj, axis=0)).mean()) + 1e-12
        drift = np.zeros_like(traj[0])
        for t in range(1, len(out)):
            drift = drift + rng.normal(0.0, self.noise * scale, out[t].shape)
            out[t] = out[t] + drift
        return out

    def _critic_for(self, scene_type: str, mass: np.ndarray) -> ConservationCritic:
        """按场景缓存 critic：mass 是构造参数，而各场景质量不同。"""
        if scene_type not in self._critic_cache:
            self._critic_cache[scene_type] = ConservationCritic(mass=mass, G=self._G)
        return self._critic_cache[scene_type]

    def _residual(self, traj: np.ndarray, mass: np.ndarray, dt: float,
                  scene_type: str) -> float:
        critic = self._critic_for(scene_type, mass)
        return critic.residual(traj, self._vel(traj, dt))

    def _gate_threshold(self, scene_type: str, dt: float) -> float:
        """门限：由**干净轨迹**的残差决定（不接触被损坏的样本）。"""
        if scene_type not in self._gate_cache:
            traj, mass, _ = self._clean_trajectory(scene_type)
            self._gate_cache[scene_type] = \
                self._residual(traj, mass, dt, scene_type) * self.gate_factor
        return self._gate_cache[scene_type]

    # ------------------------------------------------------------ 管线
    def _run(self, scene_type: str, *, degrade: bool, gate: bool, seed: int = 0):
        """跑一次管线，返回 (score, residual, traj, gated)。

        degrade : 是否损坏生成层（注入过程噪声）
        gate    : 验证层是否处于**正常**状态。注意——若全局
                  `enable_validation_gate=False`，则验证门**根本不存在**，
                  该参数无效（恢复一个不存在的层是空操作，贡献度必须为 0）。
                  这一区分很关键：早先版本用 gate_override 强制开门，
                  导致「门关闭」模式下 restore(validate) 也拿到非零贡献。
        score   = −residual（越大越物理）。
        """
        clean_traj, mass, scene = self._clean_trajectory(scene_type)
        dt = scene.dt
        rng = np.random.default_rng(seed)

        traj = self._degrade(clean_traj, dt, rng) if degrade else clean_traj.copy()

        effective_gate = bool(gate) and self.enable_validation_gate
        gated = False
        if effective_gate:
            if self._residual(traj, mass, dt, scene_type) > \
                    self._gate_threshold(scene_type, dt):
                # 验证门的真实作用：拒绝并回退到安全网（此处以干净轨迹为代理）
                traj = clean_traj.copy()
                gated = True

        res = self._residual(traj, mass, dt, scene_type)
        return -res, res, traj, gated

    # ------------------------------------------------------------ 对外入口
    def trace(self, scene_types: Optional[List[str]] = None,
              num_runs: int = 3) -> TracingReport:
        """对多个场景跑完整的损坏 / 恢复协议。返回实测报告。

        scene_types=None    → 使用默认场景列表
        scene_types=[]      → 不跑任何场景（显式空列表不等于"用默认值"）
        """
        if scene_types is None:
            scenes = ["spiral_galaxy", "globular_cluster", "galaxy_collision"]
        else:
            scenes = list(scene_types)
        report = TracingReport(config={
            "enable_validation_gate": self.enable_validation_gate,
            "noise": self.noise,
            "gate_factor": self.gate_factor,
            "steps": self.steps,
            "num_runs": num_runs,
            "on_path_layers": list(ON_PATH_LAYERS),
            "fallback_proxy": "同一场景的未损坏轨迹（真实系统对应「回退到上一个关键帧」）",
            "score": "−归一化守恒残差（angular+momentum 通道），越大越物理",
        })

        for scene in scenes:
            runs = range(num_runs)

            def avg(degrade: bool, gate: bool) -> float:
                vals = [self._run(scene, degrade=degrade, gate=gate,
                                  seed=self.seed + 1000 * i)[0] for i in runs]
                return float(np.mean(vals))

            # 干净基线（各层均正常）
            score_clean = avg(degrade=False, gate=True) if self.enable_validation_gate \
                else avg(degrade=False, gate=False)
            # 全坏基线：所有**在路径上**的层都损坏
            score_all_corrupt = avg(degrade=True, gate=False)
            report.all_corrupt[scene] = score_all_corrupt

            layers: Dict[str, LayerResult] = {}

            # --- director：不在生成路径上 ---
            layers["director"] = LayerResult(
                layer="director", on_generation_path=False,
                contribution=None,
                note="不在 LumynEngine.generate 的调用链上，无法做有意义的损坏，"
                     "贡献度不可测（None，不是 0）")

            # --- generate：其余层损坏（gate 关），只恢复 generate ---
            g = LayerResult(
                layer="generate", on_generation_path=True,
                score_clean=score_clean,
                score_corrupt=avg(degrade=True, gate=True),   # 只损坏 generate
                score_restore=avg(degrade=False, gate=False),  # 只恢复 generate
            )
            g.contribution = g.score_restore - score_all_corrupt
            layers["generate"] = g

            # --- validate：其余层损坏（generate 坏），只恢复 validate（门开）---
            v = LayerResult(
                layer="validate", on_generation_path=True,
                score_clean=score_clean,
                score_corrupt=avg(degrade=False, gate=False),  # 只损坏 validate
                score_restore=avg(degrade=True, gate=True),    # 只恢复 validate
            )
            v.contribution = v.score_restore - score_all_corrupt
            if not self.enable_validation_gate:
                v.note = ("验证门未启用：验证层的输出不影响生成结果，"
                          "因此「恢复验证层」是空操作，贡献度恒为 0。"
                          "这是引擎的真实行为，不是测量失败 —— "
                          "启用 --gate 才会得到一个真正生效的验证层。")
            layers["validate"] = v

            report.per_scene[scene] = layers

        # 跨场景平均（路径外的层保持 None）
        n = len(report.per_scene) or 1
        for name in ALL_LAYERS:
            vals = [report.per_scene[s][name].contribution for s in report.per_scene]
            report.averaged[name] = None if any(v is None for v in vals) \
                else float(sum(vals) / n)
        return report


# ---------------------------------------------------------------- CLI
def main():
    import argparse
    p = argparse.ArgumentParser(description="Lumyn 管线层因果追踪（实测）")
    p.add_argument("--scenes", default="spiral_galaxy,globular_cluster,galaxy_collision")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--gate", action="store_true",
                   help="启用真实生效的验证门（默认关闭，与当前引擎行为一致）")
    p.add_argument("--particles", type=int, default=150,
                   help="每场景粒子数（ConservationCritic 构造 (T,N,N,3) 中间数组，"
                        "N 过大会很吃内存）")
    p.add_argument("--save", action="store_true")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    tracer = CausalTracer(enable_validation_gate=args.gate,
                          n_particles=args.particles)
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    report = tracer.trace(scene_types=scenes, num_runs=args.runs)
    print(report.summary())

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    if args.save:
        report.save(out_dir / "causal_tracing.json")
        print("[save] results/causal_tracing.json")
    if args.plot:
        path = report.plot(out_dir / "causal_tracing.png")
        print(f"[plot] {path}" if path else "[plot] 跳过（matplotlib 未安装）")


if __name__ == "__main__":
    main()
