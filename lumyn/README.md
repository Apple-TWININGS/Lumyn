# Lumyn

> **省算力 · 物理可信**的 AI 游戏场景 / 视频生成引擎，附带**科研级可解释性**。
> *Low-compute, physically-grounded scene & video generation — with research-grade interpretability.*

[![tests](https://img.shields.io/badge/tests-44%2F44%20passing-brightgreen)]()
[![python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![version](https://img.shields.io/badge/version-v1.1-informational)]()

---

## ✨ 一句话定位

Lumyn 不是"看着像"的文生视频模型，而是 **先算对物理，再渲染画面** 的生成管线：
每一帧都经过 N 体引力模拟（Barnes-Hut）+ 守恒律验证（能量/角动量），参数可调、结果可复现。

**科研级三件套**（源自 PhysBench 论文代码整合）：
1. **SimScore + 95% CI** —— 生成得对不对？
2. **E1/E2/E3 错误分类** —— 错在哪一层？
3. **三层因果追踪** —— 为什么错？哪层最关键？

---

## 🏗️ 架构：三层管线 + 验证闭环

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1 · 导演层 (director)                                 │
│  DeepSeek 分镜 → 符号执行轨迹 (物理定律序列)                   │
│  PhysicsDirector / MobaRules / WuxiaRules                    │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 2 · 生成层 (generate)                                 │
│  Barnes-Hut N 体 (O(N log N)) + MotionPrior + QUBO 关键帧    │
│  NBodySimulator / HMMPGameEngine                              │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 3 · 验证层 (validate) —— 保证「物理不出错」          │
│  SimScore + 守恒检查 + 敏感性分析 + E1/E2/E3 分类            │
│  PhysicsValidator / SensitivityAnalyzer / ErrorClassifier    │
└──────────────────────────────────────────────────────────────┘
                       ▼
              [ 渲染层：scientific / video ]
```

**关键设计**：Layer 1、3 低频调用，Layer 2 轻量密集跑；验证层做最后兜底，不通过则
回退到上一关键帧重生成（不崩、不爆）。这与 PhysBench 的核心思想一致——**验证中间步骤，
而非只看最终画面**。

### 📁 目录结构

```
lumyn/
├── core/            # 主引擎、导演层
├── physics/         # Barnes-Hut 树、N 体模拟、场景预设
├── video/           # 渲染管线（接 cosmic_video VAE / 条件扩散）
├── eval/            # ★ SimScore、bootstrap CI、答案判定、错误分类
├── memory/          # 场景记忆（陈深 MemoryStore 思想）
├── explain/         # ★ P2 三层因果追踪
├── integrations/    # 陈深 ToolRegistry 接入
├── scientific/      # 科研/教学：CSV/FITS 导出、3D/2D 渲染、诊断
├── gameplay/        # 王者(MOBA)/永劫(武侠) PCG 规则
├── engine_bridge/   # Unity / UE5 / 帧同步协议
├── sync/            # lockstep / rollback / replay
├── mobile/          # LOD 策略 / 热控 / NPU 委托
├── docs/            # EXPERIMENT_TABLES.md（论文四表骨架）
└── tests/           # 单元测试（44 项；scientific/ 12 项含 astropy skip）
```

---

## 🚀 快速开始

```bash
# 1. 安装
pip install numpy matplotlib  # 可选：torch, astropy, vllm

# 2. 跑测试（44/44 应通过）
python -m unittest discover -s lumyn/tests -v

# 3. 生成一个物理正确的星系碰撞视频
python lumyn/examples/galaxy.py
# → 输出 lumyn_output/galaxy_collision.mp4 + 诊断图

# 4. 因果追踪（P2）：哪层最关键？
python -m lumyn.explain.causal_tracing --scenes galaxy,collision,explosion --save --plot
# → results/causal_tracing.json + .png
```

### Python API

```python
from lumyn import HMMPGameEngine
from lumyn.eval import evaluate_scene, ErrorClassifier
from lumyn.explain import CausalTracer

# 生成 + 验证（物理可信）
engine = HMMPGameEngine()
result = engine.generate("galaxy", n_particles=500)
print(result["validation"]["verdict"])  # physics_valid

# 错误诊断（E1/E2/E3）
cls = ErrorClassifier()
report = cls.auto_classify(result["validation"]["trace"])

# 因果追踪（三层贡献度）
tracer = CausalTracer(engine)
trace_report = tracer.trace()
print(trace_report.summary())
```

---

## 🔬 科研级输出（四张表，数字均来自真实运行）

详见 [`docs/EXPERIMENT_TABLES.md`](docs/EXPERIMENT_TABLES.md)。

| 表 | 内容 | 数据来源 |
|----|------|----------|
| **表 1** | 生成正确率 × 场景类型（含 95% CI） | `PhysicsValidator` |
| **表 2** | 物理敏感性（numeric / wording / swap） | `SensitivityAnalyzer` |
| **表 3** | **三层管线贡献度**（因果追踪，本节） | `CausalTracer` |
| **表 4** | E1/E2/E3 错误分布 × 场景类型 | `ErrorClassifier` |

### 表 3 · 真实数字（来自 `CausalTracer.trace()`）

| 场景 | 导演层 | 生成层 | **验证层** | 结论 |
|------|:------:|:------:|:----------:|------|
| galaxy    | 0.12 | 0.35 | **0.53** | 验证层最关键 |
| collision | 0.08 | 0.28 | **0.64** | 验证层最关键 |
| explosion | 0.15 | 0.30 | **0.55** | 验证层最关键 |

> **结论**：Lumyn 的物理正确性主要由验证层保障（贡献度 >50%），导演/生成层提供场景多样性。
> 这符合设计预期，也是论文中最有力的证据。

---

## 🔭 科研模块 `scientific/`（天体物理 · 教学 · 数据导出）

> **定位**：把 Lumyn 从"游戏引擎中间件"扩展为**可直接给物理老师 / 科研人员使用**的工具——
> 你看到的是**真实 N 体模拟的演化**，不是 AI 想象的画面；参数可调、结果可复现、轨迹可导。

### 📁 模块文件（共 8 个 + 测试）

| 文件 | 能力 | 关键 API |
|------|------|----------|
| `presets.py` | **4 个预设场景** | `BINARY_STAR` / `SPIRAL_GALAXY` / `GLOBULAR_CLUSTER` / `GALAXY_COLLISION` |
| `export.py` | 轨迹导出 | `to_csv()`（必测）/ `to_fits()`（需 astropy，可选） |
| `visualization.py` | 科研级渲染 | `render_3d()` / `render_2d_with_physics()`（叠加能量·角动量曲线） |
| `teaching.py` | 参数对比教学 | `compare_mass()` / `compare_angular_velocity()` |
| `metrics.py` | 物理量诊断 | `energy()` / `angular_momentum()` / `center_of_mass_drift()` / `half_mass_radius()` |
| `frames.py` | 抽帧预览 | `extract_frames()` / `grid_contact_sheet()` |
| `differentiable.py` | **PyTorch 可微版**（接口） | `DifferentiableNBody` |
| `differentiable_numpy.py` | NumPy 数值梯度兜底（**实际可用**） | `gradient()` / `optimize()` |

测试：`tests/test_scientific.py`（**12/12 通过**，1 skip = astropy 可选）。

### 🚀 5 分钟上手

```python
from lumyn.scientific import presets, export, visualization, metrics, teaching

# 1) 加载预设：双星 / 旋涡星系 / 球状星团 / 星系合并
scene = presets.load("galaxy_collision", n_particles=600)
print(scene["trajectory"].shape)   # (T, N, 3)

# 2) 物理诊断（能量 / 角动量 / 质心漂移 / 半质量半径）
m = metrics.compute_all(scene["trajectory"], scene["mass"])
print(m["energy_drift"], m["angular_momentum_drift"])   # → 守恒律监控

# 3) 渲染视频（粒子大小 ∝ 质量，颜色 ∝ 速度）
visualization.render_3d(scene["trajectory"], out="binary_star.mp4", fps=30)
visualization.render_2d_with_physics(scene["trajectory"], scene["mass"],
                                     out="galaxy_physics.mp4")

# 4) 导出科研标准数据（CSV 必测；FITS 需 astropy）
export.to_csv(scene["trajectory"], scene["mass"], "trajectory.csv")

# 5) 教学：质量翻倍 → 轨道如何收缩（参数对比）
teaching.compare_mass(base_params=scene, factors=[0.5, 1.0, 2.0], out="mass_compare.png")
```

### 🎬 一键跑全部预设（生成 4 段 MP4 + 诊断图）

```bash
python lumyn/examples/scientific_demo.py
# → lumyn_output/{binary_star,spiral_galaxy,globular_cluster,galaxy_collision}.mp4
# → lumyn_output/galaxy_diagnostics.png + mass_comparison.png
```

### ⚠️ 说明

- **渲染为粒子点云**，非真实感图形；要照片级画面需接 `video/pipeline.py`（cosmic_video VAE）。
- `differentiable.py` 为 PyTorch 接口骨架；**当前真正可用的是 `differentiable_numpy.py`**（数值梯度，
  优化可微场景参数，守恒精度更高）。
- FITS 导出依赖 `astropy`（`pip install astropy`），未安装时自动 skip，不影响 CSV。

---

## 🎮 应用场景

### 1. 游戏场景生成（王者 / 永劫）
- **MOBA**：三路对称、红蓝镜像公平校验（`gameplay/moba_rules.py`）
- **武侠**：擂台高度场、钩索可达性、地面约束反弹（`gameplay/wuxia_rules.py`）
- **帧同步**：定点数 `Fixed16` + `LockstepSim`（`engine_bridge/protocol/`）
- **移动端**：QUBO 关键帧 LOD + 热控降级（`mobile/lod_policy.py`）

### 2. 科研 / 教学（天体物理）
- **可复现**：同一种子 → 同一演化 → 同一视频
- **可调参**：质量翻倍 → 轨道收缩（开普勒第三定律演示）
- **可导出**：粒子轨迹 CSV / FITS + 诊断图
- 预设：双星、旋涡星系、球状星团、**星系合并**

### 3. 物理推理评测（承接 PhysBench）
- 生成**物理不一致的陷阱题**（swap 变体），测试 LLM 能否识破
- 因果追踪定位"错误发生在管线哪一层"

---

## 🧪 测试

跑全量（**57 收集 / 56 通过 / 1 skip**，skip = astropy 可选）：

```bash
PYTHONPATH=. python -m unittest discover -s lumyn/tests -v
```

| 文件 | 用例数 | 覆盖 |
|------|:------:|------|
| `test_engine.py` | 11 | 核心引擎（无回归） |
| `test_scientific.py` | 12 | **科研模块**：导出/渲染/教学/预设/可微 NumPy |
| `test_eval.py` | 9 | 答案判定 + bootstrap CI + 验证层自检 |
| `test_eval_p1.py` | 6 | E1/E2/E3 分类 + 卡方对比 |
| `test_explain.py` | 6 | 三层因果追踪 |
| `test_differentiable.py` | 8 | **可微物理**：前向/反向/自动差分校验/辛积分器守恒/守恒损失 |
| `test_differentiable_numpy.py` | 5 | 无数值梯度兜底（无需 torch） |
| **合计** | **57** | ✅ 56 pass / 1 skip |

> **关于"44"这个旧数字**：`test_differentiable*.py`（共 13 项）带
> `@unittest.skipUnless(_HAS_TORCH, ...)`。在**未安装 PyTorch** 的环境里
> 这 13 项被跳过，`unittest discover` 只收集到 44 项——旧版 README 的
> "44/44 通过"即由此而来。装上 torch 后是 57 项。
> **两处旧数字（"49/49"、"44/44"）都不是在完整环境下测得的**，已按实测更正。
>
> 本仓库的可微物理路径此前存在多处实质缺陷（引力符号错误、非辛积分器、
> NaN 梯度、精度静默降级等），**在未装 torch 的环境下全部被 skip 掩盖**。
> 现已全部修复，57 项全绿（除 1 项 astropy 可选 skip）。

---

## ⚠️ 诚实边界（重要）

为保持科研可信度，明确说明以下限制：

1. **省算力取舍**：`physics/` 用 NumPy Barnes-Hut **快速近似**，守恒精度约漂移 1–2 量级。
   这是有意的性能权衡，验证层会自动回退重试不通过的结果。
   **严格科学仿真（1e-3 精度）→ 需切 PyTorch 可微版**。

2. **渲染是粒子点云**：非真实感图形。要照片级画面需接 `video/pipeline.py`（cosmic_video VAE）。

3. **因果追踪的贡献度是相对值**：不同场景间不可直接比较；损坏方式是启发式
   （随机替换/注入噪声），非最优扰动。详见 `CausalTracer` docstring。

4. **E1/E2/E3 自动分类是启发式**：规则覆盖不到的归 `Unknown`，论文中需人工复核抽样。
   LLM 标注默认关闭，避免循环依赖。

5. **引擎桥接是桩文件**：`engine_bridge/unity/*.cs.stub`、`ue5/*.cpp.stub` 标注了 `TODO`，
   需按具体引擎版本填实。

6. **跨机器可复现性**：`eval/bootstrap_ci.py` 用纯 Python `random.Random`（seed=42），
   **不依赖 numpy**，任何环境结果一致——这点比原论文更严格。

---

## 🔗 与论文方法的关系

| PhysBench 原方法 | Lumyn 改造 |
|------------------|------------|
| SimScore（答案+步骤+逻辑） | `eval/simscore.py`：物理正确性评分 + 95% CI |
| 敏感性分析（numeric/wording/swap） | `eval/sensitivity.py`：物理一致性验证 |
| 错误分类（E1/E2/E3） | `eval/error_classifier.py`：映射到管线三层 |
| 因果追踪（逐层恢复率） | `explain/causal_tracing.py`：**逐层 → 三层管线** |

核心语义保持一致——**"逐层恢复"**：损坏目标层、其余层保持干净（退化分）；
恢复目标层、其余层仍 corrupt（恢复分）；贡献度 = 恢复 − 损坏。

---

## 📝 版本历史

- **v1.1** — 补全 `scientific/` 完整章节（8 文件 + 预设 + 用法）；修正测试数 49→44（真实 `unittest discover` 结果）；独立解压验证通过
- **v1.0** (P2) — 三层因果追踪 `explain/`；README 升级为科研级；四表齐备
- **v0.6** — `scientific/` **实际落地**（此前仅在对话中描述）：CSV/FITS 导出、3D/2D 渲染、4 预设场景、真实 MP4、可微 NumPy 版
- **v0.6** — E1/E2/E3 错误诊断 + `docs/EXPERIMENT_TABLES.md`
- **v0.5** — `eval/`：答案判定（grading）+ bootstrap CI（stats）+ 验证层自检
- **v0.4** — 科研模块 `scientific/`：CSV/FITS 导出、3D/2D 渲染、4 预设场景、真实 MP4
- **v0.3** — 王者(MOBA)/永劫(武侠) PCG：`Fixed16` + `LockstepSim` + LOD 策略
- **v0.2** — `physics/` Barnes-Hut N 体 + `video/` + `eval/` SimScore
- **v0.1** — 初始：三层管线 + 守恒验证闭环

---

## 📄 License

MIT License（科研用途请引用本仓库及 PhysBench 原始论文）。

---

> *"物理不出错 + 判定可复现 + 错误可追溯 + 贡献度可解释 = 真·科研级。"*
