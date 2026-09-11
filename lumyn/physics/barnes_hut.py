"""Barnes-Hut 八叉树：O(N log N) 近似 N 体引力。

从 hmmp_brain.py 提取核心思想，纯 NumPy 实现：
- 空间递归细分（八叉树），叶子存粒子索引
- 质心-质量（COM）聚合，theta 打开判据
- 近场：叶子内精确成对 O(K²)；远场：cell COM 单极近似
- 位置签名缓存：固定 snapshot 跨调用不重建

接口对齐 laniakea_mind.barnes_hut.BarnesHutTree（可直接替换）。
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple


class Cell:
    """八叉树节点。__slots__ 降低内存。"""
    __slots__ = ("center", "half", "total_mass", "com", "indices", "children", "cid")

    def __init__(self, center, half, indices, cid):
        self.center = np.asarray(center, dtype=np.float64)
        self.half = float(half)
        self.indices = indices
        self.children: List["Cell"] = []
        self.cid = cid
        self.total_mass = 0.0
        self.com = self.center.copy()


class BarnesHutTree:
    """生产级 Barnes-Hut 树。

    关键方法
    --------
    build(pos, mass, center, size) -> 构建八叉树 (O(N))
    compute_acceleration(theta) -> (N,3) 加速度，近场+远场 (O(N log N))
    cache_key(pos, theta, size) -> 位置签名
    """

    def __init__(self, max_leaf: int = 16, max_depth: int = 24):
        self.max_leaf = max_leaf
        self.max_depth = max_depth
        self._pos: Optional[np.ndarray] = None
        self._mass: Optional[np.ndarray] = None
        self._cells: List[Cell] = []
        # 缓存
        self._cache_key: Optional[Tuple] = None
        self._cache_root: Optional[Cell] = None

    # ---------------- 树构建 ----------------
    def build(self, pos: np.ndarray, mass: np.ndarray, center: np.ndarray, size: float) -> Cell:
        self._pos = np.ascontiguousarray(pos, dtype=np.float64)
        self._mass = np.ascontiguousarray(mass, dtype=np.float64)
        self._cells = []
        half = size * 0.5
        self._root = self._subdivide(np.arange(len(pos), dtype=np.int64), center.copy(), half, 0)
        return self._root

    def _subdivide(self, indices, center, half, depth) -> Cell:
        cell = Cell(center, half, indices, len(self._cells))
        self._cells.append(cell)

        m = self._mass[indices]
        cell.total_mass = float(m.sum())
        if cell.total_mass > 0:
            cell.com = (self._pos[indices] * m[:, None]).sum(axis=0) / cell.total_mass

        if len(indices) <= self.max_leaf or half < 1e-9 or depth >= self.max_depth:
            return cell

        for octant in range(8):
            bits = [(octant >> k) & 1 for k in range(3)]
            child_center = center + (np.array(bits, dtype=np.float64) - 0.5) * half
            lo = child_center - half * 0.5
            hi = child_center + half * 0.5
            mask = np.ones(len(indices), dtype=bool)
            for k in range(3):
                np.logical_and(
                    mask,
                    (self._pos[indices, k] >= lo[k]) & (self._pos[indices, k] < hi[k]),
                    out=mask,
                )
            sub = indices[mask]
            if len(sub) > 0:
                cell.children.append(self._subdivide(sub, child_center, half * 0.5, depth + 1))
        return cell

    # ---------------- 加速度 ----------------
    def compute_acceleration(self, theta: float = 0.5) -> np.ndarray:
        """近场(精确) + 远场(COM 单极) 加速度。O(N log N)。"""
        pos = self._pos
        mass = self._mass
        n = len(pos)
        acc = np.zeros((n, 3), dtype=np.float64)

        # 每个粒子所属叶子 cell
        leaf_of = np.full(n, -1, dtype=np.int64)
        for cell in self._cells:
            if not cell.children:
                leaf_of[cell.indices] = cell.cid

        # 近场：叶子内部精确成对
        for cell in self._cells:
            if cell.children:
                continue
            idx = cell.indices
            k = len(idx)
            if k < 2:
                continue
            pi = pos[idx]
            mj = mass[idx]
            # d[i,j] = r_j - r_i, i.e. pointing from i TOWARD j.
            # The opposite order (r_i - r_j) makes gravity repulsive, which is
            # also inconsistent with the far-field term below, where
            # (cell.com - pos) is attractive. Verified by
            # scripts/audit_force_sign.py and scripts/verify_lumyn_physics.py.
            d = pos[idx][None, :, :] - pi[:, None, :]   # (K,K,3)
            r2 = (d * d).sum(axis=2) + 1e-8
            inv_r3 = r2 ** (-1.5)
            np.fill_diagonal(inv_r3, 0.0)
            acc[idx] = (mj[None, :, None] * d * inv_r3[:, :, None]).sum(axis=1)

        # 远场：内部节点 COM 作用于「非子树」粒子
        #
        # 叶子也必须能作为源。原实现在这里 `if not cell.children: continue`，
        # 于是跨叶粒子对既不在近场（近场只做叶内精确成对），也不在远场（叶子被
        # 跳过），它们的力被**静默丢弃**。两颗粒子永远同叶（max_leaf=16）所以双体
        # 轨道看起来是对的，而 N=200 的云误差约 90%，星系核心这类"近但在不同叶"
        # 的区域受创最重。
        #
        # 正确做法（标准 Barnes-Hut 遍历）：
        #   判据通过      -> 用 COM 单极近似
        #   判据不通过且是内部节点 -> 递归到子节点（下面的循环会自然覆盖子节点）
        #   判据不通过且是叶子     -> 对该叶精确求和
        for cell in self._cells:
            diff_all = cell.com - pos                       # (N,3)
            dist_all = np.sqrt((diff_all * diff_all).sum(axis=1) + 1e-16)
            near = (2.0 * cell.half / np.maximum(dist_all, 1e-8)) < theta

            if cell.children:
                sub_leaves = self._subtree_leaf_list(cell.cid)
                mask = near.copy()
                if sub_leaves:
                    mask &= ~np.isin(leaf_of, sub_leaves)
                inv_r3 = dist_all ** (-3)
                acc += (cell.total_mass * diff_all * inv_r3[:, None]) * mask[:, None].astype(np.float64)
            else:
                # leaf: approximate with its COM where allowed, else sum exactly
                idx = cell.indices
                mj = mass[idx]
                self_mask = leaf_of == cell.cid
                mask = near & ~self_mask
                inv_r3 = dist_all ** (-3)
                acc += (cell.total_mass * diff_all * inv_r3[:, None]) * mask[:, None].astype(np.float64)

                need = (~near) & (~self_mask)
                if np.any(need):
                    rows = np.where(need)[0]
                    pj = pos[idx]
                    for i in rows:
                        dv = pj - pos[i]                     # toward the leaf particles
                        r2 = (dv * dv).sum(axis=1) + 1e-8
                        acc[i] += (mj[:, None] * dv * (r2 ** -1.5)[:, None]).sum(axis=0)

        return acc

    def _subtree_leaf_list(self, root_cid: int) -> List[int]:
        result: List[int] = []
        stack = [self._cells[root_cid]]
        while stack:
            cur = stack.pop()
            if not cur.children:
                result.append(cur.cid)
            else:
                stack.extend(cur.children)
        return result

    # ---------------- 缓存 ----------------
    @staticmethod
    def cache_key(pos: np.ndarray, theta: float, size: float) -> Tuple:
        """位置签名：(shape, theta, size, hash)。"""
        return (tuple(pos.shape), float(theta), float(size), hash(pos.tobytes()))

    def get_cached(self, key: Tuple):
        return self._cache_root if self._cache_key == key else None

    def set_cached(self, key: Tuple, root: Cell) -> None:
        self._cache_key = key
        self._cache_root = root
