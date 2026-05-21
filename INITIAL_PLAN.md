# Optiver - Trading at the Close: 初始计划 (Initial Plan)

---

## Context

**Why:** 从零搭建 Kaggle 比赛 "Optiver - Trading at the Close" 的可复现 ML 工程项目。项目当前只有 Kaggle 原始数据文件，没有任何代码、配置、文档或版本控制。

**Target:** 在严格时序约束下（Purged K-Fold CV），用 LightGBM 建立 baseline，逐步迭代特征工程，达到可提交的分数。

---

## 1. 目标 (Goals)

- 搭建符合 CLAUDE.md 规则的标准项目骨架（`src/`, `config/`, `outputs/`, `data/`, `analysis/`）
- 完成数据结构分析报告
- 设计严格时序验证方案（Purged K-Fold by date_id）
- 实现并运行 LightGBM baseline（本地 mock API 验证）
- 初始化 Git 仓库并做首次提交

## 2. 当前项目结构

```
D:\学习\Kaggle Project\Optiver - Trading at the Close\
├── example_test_files/
│   ├── revealed_targets.csv        # 已揭示的测试集 target
│   ├── sample_submission.csv       # 提交模板
│   └── test.csv                    # 测试集 (3天, 33000行)
├── optiver2023/
│   ├── __init__.py                 # 导出 make_env
│   └── competition.cpython-310...so # 比赛闭源API (Linux only)
├── public_timeseries_testing_util.py # 本地Mock API
└── train.csv                       # 训练集 (481天, ~524万行, 640MB)
```

**缺失的关键文件/目录：**
- `CLAUDE.md` (项目级)
- `README.md`
- `requirements.txt`
- `.gitignore`
- `.venv/` (Python 虚拟环境)
- `src/` (核心代码)
- `config/` (实验配置)
- `data/` (数据目录)
- `outputs/` (实验输出)
- `analysis/` (分析报告)
- `notebooks/` (探索性 notebook)
- `.git/` (Git 仓库未初始化)

## 3. 关键规则摘要 (from C:\Users\keles\.claude\CLAUDE.md)

| 规则 | 说明 |
|------|------|
| Plan 模式优先 | 复杂项目先设计再写代码 |
| 优先级 | 当前 Prompt > 项目CLAUDE.md > 全局CLAUDE.md |
| Python 3.10+ | 用 `.venv` 虚拟环境 |
| PEP 8 | 代码风格；核心函数需要 type hints + docstring |
| Git | feat/fix/docs/refactor/test 前缀；功能完成或提交版本时才 push |
| 核心库 | pandas, numpy, sklearn, lightgbm, xgboost |
| Kaggle 流程 | EDA → Baseline → 特征工程 → 模型优化 → 提交 |
| 不猜测意图 | 不确定时先问 |

## 4. 竞赛关键信息速览

| 维度 | 详情 |
|------|------|
| **任务** | 预测纳斯达克收盘竞价中每只股票 60 秒后的价格变动 |
| **数据规模** | 200 只股票 × 481 天 × 55 个 10 秒快照 ≈ 524 万行 |
| **特征列** | 17 列： imbalance 相关、订单簿价格/规模、wap、参考价格等 |
| **Target** | `target` (float64, mean≈-0.048, std≈9.45, range≈[-25,25]) |
| **缺失值** | `far_price` / `near_price` ~55% 缺失 (仅 seconds_in_bucket≥300 时可用) |
| **评估方式** | 在线时序 API (iter_test)，按 time_id 逐批预测，不可回看 |
| **评估指标** | MAE (Mean Absolute Error) |
| **CV 策略** | Purged K-Fold by date_id (冠军方案确认) |

## 5. 分工与执行步骤

### Step 0 — 项目脚手架搭建 (Scaffold)
**角色:** ML Engineer
**输入:** 当前空项目目录
**输出:**
- 创建 `CLAUDE.md` (项目级，含竞赛关键信息)
- 创建 `README.md`
- 创建 `requirements.txt`
- 创建 `.gitignore`
- 创建目录结构: `src/`, `config/`, `data/`, `outputs/`, `analysis/`, `notebooks/`
- 初始化 `.venv` + 安装依赖
- 初始化 Git 仓库
- 将 `train.csv` **复制**到 `data/` 目录（保留原始文件不动）
- 创建 `src/mock_api.py` — Windows 兼容的轻量 iter_test 替代实现
**验收标准:** 目录结构完整，`.venv` 可 import lightgbm/pandas，git status 正常

### Step 1 — 数据调查 (Data Investigation)
**角色:** Data Investigator
**输入:** `data/train.csv`, `example_test_files/test.csv`
**输出:**
- `analysis/01_data_overview.md` — 数据报告 (列描述、missing、分布、target 分析)
- `analysis/01_checks.py` — 可复现的数据检查脚本
- `analysis/01_leakage_report.md` — **泄漏检查报告（date_id 连续性、stock_id 完整性、target 未来泄漏检查）**
**验收标准:** 脚本可独立运行，报告覆盖所有列，泄漏风险已标注
**审查门禁:** Step 1 完成后 **自动触发 Reviewer**，输出审查报告后方可进入 Step 2

### Step 2 — 验证方案设计 (Validation Design)
**角色:** Validation Researcher
**输入:** 数据调查报告
**输出:**
- `src/validation/__init__.py`
- `src/validation/splitter.py` — **从 Kaggle 公开 kernel 复制已验证的 PurgedGroupTimeSeriesSplit 实现**
- `src/validation/test_splitter.py` — **单元测试（验证无 date_id 重叠、purge gap 正确）**
- 设计文档 (写入 `analysis/02_cv_design.md`)
**验收标准:** 单元测试通过，确认无时序泄漏；每折 train/valid date 不重叠且有 purge gap

### Step 3 — Baseline 训练 (Training Baseline)
**角色:** Training Engineer
**输入:** 验证方案 + 原始特征
**输出:**
- `config/baseline_001.yaml` — baseline 实验配置
- `src/training/run.py` — 训练脚本
- `src/features/base_features.py` — 基础特征函数（**每个函数含 docstring 解释金融含义**）
- `outputs/baseline_001/` — metrics.json, oof.csv, feature_list.txt
**验收标准:** 脚本在本地 mock API 跑通，输出 OOF 预测和 MAE 指标
**审查门禁:** Step 3 完成后 **自动触发 Reviewer**，输出审查报告后方可进入下一步

### Step 4 — 最终审查与迭代规划 (Final Review)
**角色:** Reviewer
**输入:** Step 1-3 所有产出
**输出:** 审查报告 (泄漏检查、过度工程检查、代码质量、金融含义注释完整性)
**验收标准:** 所有问题有明确修复建议，无 blocking 问题

## 6. 用户修正清单（已纳入计划）

| # | 修正 | 影响步骤 |
|---|------|----------|
| 1 | 不移动原始 train.csv，用 copy | Step 0 |
| 2 | 增加 leakage 检查报告 | Step 1 |
| 3 | 从 Kaggle kernel 复制 CV 实现 + 单元测试 | Step 2 |
| 4 | Windows 兼容的 mock iter_test 替代 | Step 0 + Step 3 |
| 5 | 所有函数 docstring 含金融含义 | 全局 |
| 6 | Step 1、Step 3 后自动触发 Reviewer | Step 1, Step 3 |

## 7. 下一步执行

**当前阶段:** Step 0 — 立即开始执行项目脚手架搭建。

具体操作:
1. 创建所有必需的目录和文件（CLAUDE.md, README.md, requirements.txt, .gitignore）
2. 创建目录结构: `src/`, `config/`, `data/`, `outputs/`, `analysis/`, `notebooks/`
3. 复制 `train.csv` 到 `data/` 目录
4. 创建 `src/mock_api.py`（Windows 兼容）
5. 初始化 `.venv` + 安装依赖
6. 初始化 Git 仓库并首次提交

## 7. 验收标准总览

- [ ] 项目目录结构完整（src/, config/, data/, outputs/, analysis/, notebooks/）
- [ ] `.venv` 可用，核心包可 import
- [ ] Git 仓库初始化，有首次提交
- [ ] 项目 CLAUDE.md 存在且包含竞赛关键信息
- [ ] `analysis/01_data_overview.md` 覆盖所有列和潜在问题
- [ ] `src/validation/splitter.py` 实现 Purged K-Fold，通过逻辑审查
- [ ] `src/training/run.py` 可本地 mock API 运行，产出 OOF 预测
- [ ] `outputs/baseline_001/` 包含 metrics.json, oof.csv, feature_list.txt
