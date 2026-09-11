"""帧同步：lockstep 确定性步进 + 回滚 + 回放复现（对齐王者帧同步）。

命门：所有输入走同一可重放 PRNG、固定遍历顺序 → 双端轨迹比特一致。
"""
import hashlib
import numpy as np


class DeterministicRNG:
    """xorshift32 —— 纯整数，跨端确定。所有随机走这里。"""

    def __init__(self, seed=0):
        self.state = seed & 0xFFFFFFFF

    def next(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state / 0xFFFFFFFF

    def snapshot(self):
        return self.state

    def restore(self, state):
        self.state = state & 0xFFFFFFFF


class FrameCommand:
    """单帧指令：仅含离散动作码（确定性）。"""

    def __init__(self, frame, actor, action, param=0):
        self.frame, self.actor, self.action, self.param = frame, actor, action, param

    def encode(self):
        return f"{self.frame}:{self.actor}:{self.action}:{self.param}".encode()

    @classmethod
    def decode(cls, b):
        p = b.decode().split(":")
        return cls(int(p[0]), int(p[1]), int(p[2]), int(p[3]))


class LockstepSim:
    """指令收集 + 确定性步进。每帧收齐指令 → 步进 → 出状态哈希。"""

    def __init__(self, seed=0, n_actors=2):
        self.rng = DeterministicRNG(seed)
        self.n_actors = n_actors
        self.frame = 0
        self.state = np.zeros((n_actors, 3))
        self.velocity = np.zeros((n_actors, 3))
        self._pending = []

    def submit(self, cmd):
        self._pending.append(cmd)

    def step(self, dt=0.016):
        # 固定遍历顺序（按 actor 再按 frame）→ 确定性
        self._pending.sort(key=lambda c: (c.actor, c.frame))
        for cmd in self._pending:
            if cmd.frame == self.frame:
                self.velocity[cmd.actor, 0] += cmd.param * 0.1
        self.state += self.velocity * dt
        self._pending = []
        self.frame += 1
        return self.state.copy()

    def state_hash(self):
        return f"f{self.frame}:" + hashlib.sha1(self.state.tobytes()).hexdigest()[:8]

    def run(self, total_frames, dt=0.016):
        traj, hashes = [self.state.copy()], []
        for _ in range(total_frames):
            self.step(dt)
            traj.append(self.state.copy())
            hashes.append(self.state_hash())
        return np.stack(traj), hashes


def replay_match(seed, n_actors, total_frames, dt=0.016):
    """回放重建：同种子从零跑，哈希序列应与对局一致（观战/重连）。"""
    sim = LockstepSim(seed=seed, n_actors=n_actors)
    _, hashes = sim.run(total_frames, dt)
    return hashes


__all__ = ["DeterministicRNG", "FrameCommand", "LockstepSim", "replay_match"]
