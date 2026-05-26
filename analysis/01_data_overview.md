# 01 — Data Overview: Optiver Trading at the Close

## 1. Dataset Summary

| Property | Value |
|---|---|
| File | `data/train.csv` |
| Rows | 5,237,980 |
| Columns | 17 |
| Raw memory | 934 MB |
| After type optimisation (float64→float32, int→smallest) | 560 MB (-40.1%) |
| Stocks | 200 (stock_id 0–199) |
| Trading days | 481 (date_id 0–480) |
| Snapshots per stock-day (expected) | 55 (seconds_in_bucket 0–540, step 10) |
| Expected rows (200×481×55) | 5,291,000 |
| Actual / Expected gap | −53,020 rows (~1%) |

**Coverage warning:** Only 185 out of 481 dates have all 200 stocks present. The remaining 296 dates have one or more stocks missing, which means the "55 snapshots per stock-day" assumption does not hold uniformly. This will affect CV design.

---

## 2. Column Inventory

| # | Column | Dtype (optimised) | Description |
|---|--------|-------------------|-------------|
| 1 | `stock_id` | int16 | Stock identifier (0–199) |
| 2 | `date_id` | int16 | Trading day (0–480, sequential) |
| 3 | `seconds_in_bucket` | int16 | Seconds since closing-auction start (0–540, step 10) |
| 4 | `imbalance_size` | float32 | Net order imbalance at current auction price. Positive = buy pressure. |
| 5 | `imbalance_buy_sell_flag` | int8 | Direction: −1 (sell), 0 (neutral), +1 (buy) |
| 6 | `reference_price` | float32 | Baseline price (usually last trade before auction). Normalised ~1.0. |
| 7 | `matched_size` | float32 | Shares matched at current auction price level |
| 8 | `far_price` | float32 | Far-side order-book price. Available only at seconds_in_bucket ≥ 300. |
| 9 | `near_price` | float32 | Near-side order-book price. Available only at seconds_in_bucket ≥ 300. |
| 10 | `bid_price` | float32 | Best bid (highest buy limit order). Normalised ~1.0. |
| 11 | `bid_size` | float32 | Total shares at best bid |
| 12 | `ask_price` | float32 | Best ask (lowest sell limit order). Normalised ~1.0. |
| 13 | `ask_size` | float32 | Total shares at best ask |
| 14 | `wap` | float32 | Weighted Average Price: (bid_p×ask_s + ask_p×bid_s) / (bid_s + ask_s) |
| 15 | `target` | float32 | 60-second forward WAP change, scaled ×10000 |
| 16 | `time_id` | int16 | Unique time-bucket ID (used for iter_test grouping) |
| 17 | `row_id` | str | Row identifier (string, not a feature) |

---

## 3. Missing Values

| Column | Missing Count | Missing % | Status |
|--------|--------------|-----------|--------|
| far_price | 2,894,342 | 55.26% | Expected — only seconds ≥ 300 |
| near_price | 2,857,180 | 54.55% | Expected — only seconds ≥ 300 |
| reference_price | 220 | 0.004% | Minor |
| imbalance_size | 220 | 0.004% | Minor |
| matched_size | 220 | 0.004% | Minor |
| ask_price | 220 | 0.004% | Minor |
| wap | 220 | 0.004% | Minor |
| bid_price | 220 | 0.004% | Minor |
| target | 88 | 0.002% | Minor — rows unusable for training |
| all others | 0 | 0% | Clean |

**Key observations:**
- `far_price` and `near_price` account for essentially all missing data. They are completely absent for `seconds_in_bucket < 300` (the first 30 snapshots) and become available with increasing completeness thereafter.
- `far_price` availability at seconds=300 starts at 95.5% and rises to 99.97% by seconds=540.
- `near_price` availability is near 100% for all seconds ≥ 300.
- 220 rows have a simultaneous gap across 6 price/size columns — likely a single corrupted time slice or a few affected stock-date combinations.
- 88 targets are missing — these rows are unusable for supervised training.

---

## 4. Target Distribution

| Statistic | Value |
|-----------|-------|
| Mean | −0.0476 |
| Std | 9.453 |
| Min | −385.29 |
| 1st percentile | −25.65 |
| 5th percentile | −14.05 |
| Median (50%) | −0.060 |
| 95th percentile | +13.96 |
| 99th percentile | +25.99 |
| Max | +446.07 |
| Skewness | 0.205 |
| Kurtosis | 22.56 |

The target is roughly centred near zero (mean ≈ −0.048), consistent with efficient-market expectations. However, **kurtosis = 22.6** indicates extremely fat tails — the distribution has far more extreme values than a normal distribution. The min/max (±400) are stark outliers that should be investigated per stock or per time_id pattern before training.

---

## 5. Price Columns

The four price columns (`reference_price`, `bid_price`, `ask_price`, `wap`) are normalised such that their means are ≈ 1.0 with std ≈ 0.0025. They are **highly correlated** (r > 0.97 among all pairs), which means using more than one in a linear model would cause multicollinearity. For tree-based models this is less of an issue, but feature importance may be split across them.

---

## 6. Size Columns

| Column | Mean | Std | Median | 95th pct |
|--------|------|-----|--------|----------|
| imbalance_size | 5.7M | 20.5M | 1.1M | 23.5M |
| matched_size | 45.1M | 139.8M | 12.9M | 172.3M |
| bid_size | 51,814 | 111,421 | 21,969 | 202,287 |
| ask_size | 53,576 | 129,355 | 23,018 | 206,554 |

Size columns have **heavy right-skew** — means are far above medians. Log transformation is recommended for linear models but optional for tree-based learners.

---

## 7. Data Quality Flags

| Flag | Severity | Detail |
|------|----------|--------|
| Missing stock-date combinations | Medium | 53,020 fewer rows than expected; CV splits must handle variable coverage |
| 220 rows with multi-column gaps | Low | Only 0.004% of data; safe to drop |
| 88 missing targets | Low | Only 0.002% of data; safe to drop |
| Extreme target outliers (±400) | Medium | Investigate before training — may need clipping per stock |
| Highly correlated price features | Info | reference/bid/ask/wap all r > 0.97 |
| Uneven stock coverage across dates | Medium | Only 185/481 dates have all 200 stocks |
