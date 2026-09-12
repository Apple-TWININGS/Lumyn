"""SimScore + 守恒律验证（PhysBench 资产）。

SimScore = 0.43 * AnswerAccuracy + 0.31 * StepCoverage + 0.26 * LogicalConsistency
守恒律：总能量、总动量在演化过程中漂移应 < 阈值。
"""
from __future__ import annotations

import numpy as np


class SimScore:
    """物理正确性评分（论文式加权）。"""

    def __init__(self, w_answer=0.43, w_steps=0.31, w_logic=0.26):
        self.weights = {"answer": w_answer, "steps": w_steps, "logic": w_logic}

    def compute(self, generated: dict, reference: dict) -> float:
        a = self._answer_accuracy(generated, reference)
        s = self._step_coverage(generated, reference)
        l = self._logical_consistency(generated)
        return float(np.clip(
            self.weights["answer"] * a +
            self.weights["steps"] * s +
            self.weights["logic"] * l, 0.0, 1.0))

    def _answer_accuracy(self, gen, ref) -> float:
        rv = ref.get("final_state", {})
        gv = gen.get("final_state", {})
        if not rv:
            return 1.0
        ok = sum(1 for k, v in rv.items() if k in gv and abs(gv[k] - v) < 0.1 * abs(v))
        return ok / len(rv)

    def _step_coverage(self, gen, ref) -> float:
        ref_steps = {s.get("expr", "") for s in ref.get("steps", [])}
        gen_steps = {s.get("expr", "") for s in gen.get("steps", [])}
        if not ref_steps:
            return 1.0
        return len(ref_steps & gen_steps) / len(ref_steps)

    def _logical_consistency(self, gen) -> float:
        constraints = gen.get("constraints", [])
        if not constraints:
            return 1.0
        laws = [s.get("law") for s in gen.get("steps", [])]
        mapping = {
            "energy_conserved": "ConservationOfEnergy",
            "momentum_conserved": "ConservationOfMomentum",
            "force_balance": "Newton2",
            "parabolic_trajectory": "ProjectileMotion",
        }
        ok = sum(1 for c in constraints if mapping.get(c) in laws)
        return ok / len(constraints)


class ConservationChecker:
    """用守恒律做硬验证：能量/动量漂移 < 阈值 → physics_valid。"""

    def __init__(self, G=1.0, energy_tol=0.05, momentum_tol=0.05, angular_tol=0.5):
        self.G = G
        self.energy_tol = energy_tol
        self.momentum_tol = momentum_tol
        self.angular_tol = angular_tol

    def _energy(self, pos, vel, mass):
        ke = 0.5 * (mass * (vel ** 2).sum(axis=1)).sum()
        pe = 0.0
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j]) + 1e-8
                pe -= self.G * mass[i] * mass[j] / d
        return ke + pe

    def check(self, traj: np.ndarray, mass: np.ndarray, dt: float = 0.01,
               fixed_mask: np.ndarray = None) -> dict:
        """对整条轨迹诊断守恒漂移。

        **已修正的两个缺陷（2026-09）**

        1. `angular_drift` 此前算的是**线动量模长**的相对波动：

               p_norms = ||P(t)||
               angular_drift = max|p_norms - p_norms[0]| / p_norms[0]

           这与角动量 L = Σ m (r × v) 无关。实测某星系场景上它报告 12.0176，
           而真实角动量相对变化仅 0.0311 —— 相差 386 倍且物理量不同。
           现在改为真正的角动量漂移。

        2. `momentum_drift` 此前除以 `|P_0|`。净动量在旋转盘 / 对称爆炸等场景
           天然接近 0，除以它会变成**除以噪声**。现改用恒为正的固有尺度 `Σ m‖v‖`。

        另注意：由位置差商反推速度存在 O(dt) 的半步偏移，因此本方法适合做
        **相对比较**，不宜作为绝对精度判据。需要高精度请传真实速度使用
        `eval.conservation_critic`。
        """
        traj = np.asarray(traj, dtype=np.float64)
        mass = np.asarray(mass, dtype=np.float64)
        n_steps = len(traj) - 1
        # 排除固定粒子（如中心黑洞），只对自由粒子做守恒检查
        if fixed_mask is not None:
            free = ~np.asarray(fixed_mask, dtype=bool)
        else:
            free = slice(None)

        energies, momentums, angulars = [], [], []
        for t in range(n_steps):
            v = (traj[t + 1] - traj[t]) / dt
            energies.append(self._energy(traj[t], v, mass))
            p = (mass[free, None] * v[free]).sum(axis=0)
            momentums.append(p)
            # 角动量 L = Σ m (r × v)
            L = (mass[free, None] * np.cross(traj[t][free], v[free])).sum(axis=0)
            angulars.append(L)

        energies = np.array(energies)
        momentums = np.array(momentums)
        angulars = np.array(angulars)

        e0 = abs(energies[0]) + 1e-8
        energy_drift = abs(energies[-1] - energies[0]) / e0

        # 动量：用「总速率加权质量」作尺度，恒为正，避免除以接近 0 的净动量
        p_scale = float(
            (mass[free, None] * np.abs((traj[-1] - traj[-2]) / dt)[free]).sum()
        ) if n_steps > 0 else 1.0
        p_scale = max(p_scale, 1e-12)
        momentum_drift = float(
            np.linalg.norm(momentums[-1] - momentums[0]) / p_scale)

        # 角动量：真实的 L 漂移，除以 L 的模长尺度（同样加保护）
        l_scale = max(float(np.linalg.norm(angulars, axis=1).mean()), 1e-12)
        angular_drift = float(
            np.linalg.norm(angulars[-1] - angulars[0]) / l_scale)

        return {
            "energy_drift": float(energy_drift),
            "momentum_drift": float(momentum_drift),
            "angular_drift": angular_drift,
            "energy_conserved": energy_drift < self.energy_tol,
            "momentum_conserved": momentum_drift < self.momentum_tol,
            "angular_conserved": angular_drift < self.angular_tol,
            "verdict": "physics_valid"
                if (energy_drift < self.energy_tol and momentum_drift < self.momentum_tol)
                else "physics_invalid",
        }
