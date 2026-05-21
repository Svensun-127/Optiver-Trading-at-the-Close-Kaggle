# CLAUDE.md — Optiver Trading at the Close

## 竞赛概要

- **任务:** 预测纳斯达克收盘竞价中每只股票 60 秒后的价格变动 (target)
- **数据:** 200 只股票 x 481 天 x 55 个 10 秒快照 = 524 万行
- **评估方式:** 在线时序 API (iter_test)，按 time_id 逐批预测，不可回看
- **评估指标:** MAE (Mean Absolute Error)
- **特征列 (17):** row_id, stock_id, date_id, time_id, seconds_in_bucket, imbalance_size, imbalance_buy_sell_flag, reference_price, matched_size, far_price, near_price, bid_price, bid_size, ask_price, ask_size, wap, target

## 关键约束

- **严格时序 CV:** Purged K-Fold by date_id（双侧 purge gap），禁止任何随机打乱
- **数据泄漏零容忍:** 任何基于 date_id 的聚合/统计必须在 fold 内独立计算；seconds_in_bucket 时间戳对齐
- **缺失值:** far_price/near_price ~55% 缺失 (仅 seconds_in_bucket >= 300 时可用)
- **Windows 兼容:** optiver2023 编译包仅限 Linux，本地用 `src/mock_api.py` 替代

## 项目结构

```
src/            — 核心代码 (features/, training/, validation/)
config/         — 实验 YAML 配置
data/           — 原始数据副本
outputs/        — 实验输出 (models/, predictions/, metrics/)
analysis/       — 分析报告 (EDA, leakage, CV design)
notebooks/      — 探索性 Jupyter notebook
```

## 环境

- Python 3.10+, `.venv` 虚拟环境
- 核心依赖: pandas, numpy, scikit-learn, lightgbm, xgboost, matplotlib, seaborn, jupyter, ruff, black

## 规则

- PEP 8 代码风格，核心函数含 type hints + docstring
- docstring 必须解释金融含义（例如 "imbalance 是当前买一卖一的不平衡度，正表示买压"）
- Git: feat/fix/docs/refactor/test 前缀
- 执行流程: EDA -> Baseline -> 特征工程 -> 模型优化 -> 提交