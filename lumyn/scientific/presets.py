"""天体预设场景工厂：双星、旋涡星系、球状星团、星系合并。"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class SceneParams:
    """统一场景参数容器。"""
    name: str = "scene"
    n_particles: int = 500
    masses: np.ndarray = None
    positions: np.ndarray = None
    velocities: np.ndarray = None
    G: float = 1.0
    dt: float = 0.01
    steps: int = 60
    omega: float = 1.0
    fixed_mask: np.ndarray = None

    def run(self, steps: int = None):
        """用 NBodySimulator 演化，返回 (traj, mass)。"""
        from ..physics.nbody import NBodySimulator
        steps = steps or self.steps
        sim = NBodySimulator(G=self.G, theta=0.5)
        pos = self.positions.copy()
        vel = self.velocities.copy()
        mass = self.masses.copy()
        traj = [pos.copy()]
        for _ in range(steps):
            pos, vel = sim.step(pos, mass, vel, dt=self.dt)
            traj.append(pos.copy())
        return np.stack(traj, axis=0), mass


def binary_star(n_particles: int = 200, m1: float = 3.0, m2: float = 1.0,
                separation: float = 4.0) -> SceneParams:
    """双星开普勒轨道：验证 Kepler 第三定律 T² ∝ a³。"""
    rng = np.random.default_rng(42)
    # 两主星在焦点，伴星盘绕
    pos = np.zeros((n_particles, 3))
    vel = np.zeros((n_particles, 3))
    masses = np.ones(n_particles) * 0.01
    # 主星
    pos[0] = [-separation / 2, 0, 0]
    pos[1] = [separation / 2, 0, 0]
    masses[0] = m1
    masses[1] = m2
    # 圆轨道速度：v = sqrt(G*(m1+m2)*(1/r1 + 1/r2)/2) 近似
    v_orb = np.sqrt(1.0 * (m1 + m2) / separation)
    vel[0] = [0, v_orb * m2 / (m1 + m2), 0]
    vel[1] = [0, -v_orb * m1 / (m1 + m2), 0]
    # 其余为盘星，随机小半径圆轨道
    r = rng.uniform(0.3, 3.0, n_particles - 2)
    theta = rng.uniform(0, 2 * np.pi, n_particles - 2)
    pos[2:, 0] = r * np.cos(theta)
    pos[2:, 1] = r * np.sin(theta)
    vk = np.sqrt((m1 + m2) / np.maximum(r, 0.2))
    vel[2:, 0] = -vk * np.sin(theta)
    vel[2:, 1] = vk * np.cos(theta)
    return SceneParams(name="binary_star", n_particles=n_particles, masses=masses,
                       positions=pos, velocities=vel, dt=0.02, steps=80)


def spiral_galaxy(n_particles: int = 600, omega: float = 1.0) -> SceneParams:
    """旋涡星系：密度波旋臂。"""
    rng = np.random.default_rng(7)
    r = rng.uniform(0.2, 2.5, n_particles)
    theta = rng.uniform(0, 2 * np.pi, n_particles) + omega * r
    pos = np.stack([r * np.cos(theta), r * np.sin(theta), rng.normal(0, 0.05, n_particles)], axis=1)
    masses = 5.0 / (1.0 + r ** 2)
    masses[0] = 200.0  # 中心超大质量
    vc = np.sqrt(5.0 / np.maximum(r, 0.2))
    vel = np.stack([-vc * np.sin(theta), vc * np.cos(theta), np.zeros(n_particles)], axis=1)
    fixed = np.zeros(n_particles, dtype=bool); fixed[0] = True
    return SceneParams(name="spiral_galaxy", n_particles=n_particles, masses=masses,
                       positions=pos, velocities=vel, dt=0.01, steps=60, omega=omega, fixed_mask=fixed)


def globular_cluster(n_particles: int = 800) -> SceneParams:
    """球状星团：Plummer 模型，核心塌缩演示。"""
    rng = np.random.default_rng(123)
    pos = rng.normal(0, 0.8, (n_particles, 3))
    pos = pos * (1.0 + rng.uniform(0, 1, n_particles))[:, None]
    masses = np.ones(n_particles) * 0.1
    masses[0] = 50.0
    fixed = np.zeros(n_particles, dtype=bool); fixed[0] = True
    # 热速度各向同性
    vel = rng.normal(0, 0.3, (n_particles, 3))
    return SceneParams(name="globular_cluster", n_particles=n_particles, masses=masses,
                       positions=pos, velocities=vel, dt=0.01, steps=50, fixed_mask=fixed)


def galaxy_collision(n_particles: int = 400, center_offset: float = 3.0) -> SceneParams:
    """两旋涡星系对撞 → 潮汐尾 + 桥。"""
    rng = np.random.default_rng(2024)
    half = n_particles // 2
    # 星系 A：中心在 (-offset, 0)，朝 +x 运动
    rA = rng.uniform(0.2, 1.8, half)
    thA = rng.uniform(0, 2 * np.pi, half) + 1.5 * rA
    posA = np.stack([rA * np.cos(thA) - center_offset, rA * np.sin(thA),
                     rng.normal(0, 0.05, half)], axis=1)
    mA = 3.0 / (1.0 + rA ** 2); mA[0] = 80.0
    vcA = np.sqrt(3.0 / np.maximum(rA, 0.2))
    velA = np.stack([-vcA * np.sin(thA) + 0.6, vcA * np.cos(thA), np.zeros(half)], axis=1)

    # 星系 B：中心在 (+offset, 0)，朝 -x 运动，逆向自转
    rB = rng.uniform(0.2, 1.8, n_particles - half)
    thB = rng.uniform(0, 2 * np.pi, n_particles - half) - 1.5 * rB
    posB = np.stack([rB * np.cos(thB) + center_offset, rB * np.sin(thB),
                     rng.normal(0, 0.05, n_particles - half)], axis=1)
    mB = 3.0 / (1.0 + rB ** 2); mB[0] = 80.0
    vcB = np.sqrt(3.0 / np.maximum(rB, 0.2))
    velB = np.stack([-vcB * np.sin(thB) - 0.6, vcB * np.cos(thB), np.zeros(n_particles - half)], axis=1)

    pos = np.concatenate([posA, posB], axis=0)
    vel = np.concatenate([velA, velB], axis=0)
    masses = np.concatenate([mA, mB], axis=0)
    fixed = np.zeros(n_particles, dtype=bool)
    fixed[0] = fixed[half] = True
    return SceneParams(name="galaxy_collision", n_particles=n_particles, masses=masses,
                       positions=pos, velocities=vel, dt=0.01, steps=70, fixed_mask=fixed)


def get(name: str, **kwargs) -> SceneParams:
    return {
        "binary_star": binary_star,
        "spiral_galaxy": spiral_galaxy,
        "globular_cluster": globular_cluster,
        "galaxy_collision": galaxy_collision,
    }[name](**kwargs)
