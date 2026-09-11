# Lumyn Examples

端到端可复现示例。每个脚本都能**独立运行、产出文件**，用于论文 Figure / 教学演示 / CI 冒烟测试。

## 快速开始

```bash
cd lumyn
python examples/guided_generation_demo.py
# -> lumyn_output/guided_generation_demo.csv   (轨迹数据)
# -> lumyn_output/guided_generation_demo.png   (损失/能量漂移收敛曲线)
```

## 示例清单

| 脚本 | 产出 | 用途 | 依赖 |
|------|------|------|------|
| `guided_generation_demo.py` | CSV + 收敛图 | **可微生成**：从"我要椭圆轨道"反演符合物理的初始条件 | numpy（torch 可选，自动降级） |

## 可微生成 demo 说明

`run_full_demo()` 流程：
1. **生成**：随机初始条件 → N 体模拟前向
2. **优化**：损失 = (半径 - 目标)² + 能量惩罚，用梯度下降（torch）或中心差分（numpy）更新初始位置
3. **验证**：能量漂移 `|E - E₀| / |E₀|` 单调递减
4. **导出**：轨迹 CSV（每帧一行，可喂 `scientific/export.py` 做 FITS/视频）

**实测**（无 torch 环境，NumPy 中心差分）：
```
energy drift: 0.76 -> 0.045   (下降一个量级)
```

**注意**：NumPy 版精度 ~1e-2，PyTorch 辛积分版可达 1e-4（需本地装 torch，代码路径已预留）。

## 依赖

- 必需：`numpy`, `matplotlib`
- 可选：`torch`（可微优化）、`astropy`（FITS 导出）

## 复现论文 Figure

```bash
python examples/guided_generation_demo.py
# 打开 lumyn_output/guided_generation_demo.png
```
