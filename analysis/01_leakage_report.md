# 01 — Leakage Report: Optiver Trading at the Close

> **Critical constraint (CLAUDE.md):** 严格时序 CV，数据泄漏零容忍。任何基于 date_id 的聚合/统计必须在 fold 内独立计算。

---

## 1. date_id Structure & Continuity

| Property | Value |
|---|---|
| Range | [0, 480] |
| Count | 481 days |
| Maximum gap | 1 |
| Fully contiguous | Yes |

**Verdict: No gaps.** date_id is a dense, sequential integer range. No days are skipped between 0 and 480. This is important because if there were gaps, a "leave one date out" CV could still allow information to leak across non-adjacent days. With contiguous days, a purge gap between train and validation folds is sufficient.

**CV implication (CLAUDE.md purged K-fold):** The purge window must be at least 1 date_id on each side of every validation split to prevent temporal leakage from adjacent trading days.

---

## 2. Stock Coverage Across Dates

| Property | Value |
|---|---|
| Dates with full 200-stock coverage | 185 / 481 (38.5%) |
| Dates with partial coverage | 296 / 481 (61.5%) |
| Least-covered stock | 10,230 rows (vs expected 26,455 for full coverage) |
| Total rows vs expected | −53,020 rows (−1.0%) |

**Key finding:** Stock coverage is not uniform across dates. Early dates (0–294) have partial stock coverage; a block of 25 dates (295–319) has full coverage; later dates also vary. This means:

1. A **stock_id-aware CV split** is necessary — some stocks may have systematically different date coverage.
2. Feature engineering that aggregates per stock across dates (e.g., rolling mean of wap) may produce inconsistent results for stocks with sparse date coverage.

**Leakage risk: LOW.** Non-uniform coverage is a data-completeness issue, not a leakage vector. However, CV folds that happen to contain dates with similar coverage patterns could produce optimistic validation scores.

---

## 3. Target: No Temporal Trend

| Statistic | Value |
|---|---|
| Mean of daily target means | −0.048 |
| Std of daily target means | 0.369 |
| Min daily mean | −1.19 |
| Max daily mean | +1.74 |
| Correlation(day_idx, daily_mean_target) | 0.024 |

**Verdict: No detectable trend.** The daily average target is stable across time. There is no drift, regime change, or concept shift visible at the daily level. A Purged K-Fold by date_id is therefore valid — no day is systematically "easier" or "harder."

**Leakage risk: LOW.** The absence of trend means that even if a model overfits to recent days, the effect on validation is bounded.

---

## 4. far_price / near_price: Time-Dependent Availability

| Bucket range | far_price | near_price |
|---|---|---|
| seconds 0–290 | **Always missing** | **Always missing** |
| seconds 300 | 95.5% present | ~100% present |
| seconds 310–390 | 97.3–97.7% present | ~100% present |
| seconds 400–450 | 97.7–98.9% present | ~100% present |
| seconds 460–540 | 99.6–99.97% present | ~100% present |

**Key observation:** `far_price` has a ramp-up in availability from 95.5% at seconds=300 to 99.97% at seconds=480+. `near_price` is consistently ~100% from seconds=300 onward. The difference between the two columns (~4% gap at seconds=300) suggests `far_price` is mechanically harder to compute or disseminate in the earliest moments of auction-data availability.

**Leakage risk: NONE — but feature-engineering hazard.** If a model uses `far_price` as a feature, predictions for rows where it is missing (seconds < 300) must either:
- Use a model variant trained without far_price
- Impute far_price (but imputation must NOT use future information within the same time_id or date, per CLAUDE.md rules)

**Recommendation:** For the baseline model, use far_price/near_price only when available (flag-based gating) or fill with a per-stock historical median computed within the training fold only.

---

## 5. Correlated Features: No Information Leakage, but Redundancy

| Pair | Pearson r |
|---|---|
| reference_price ↔ bid_price | 0.983 |
| reference_price ↔ ask_price | 0.984 |
| reference_price ↔ wap | 0.988 |
| bid_price ↔ ask_price | 0.972 |
| bid_price ↔ wap | 0.989 |
| ask_price ↔ wap | 0.989 |

**Verdict: Feature redundancy, NOT leakage.** These four price columns are all measuring essentially the same underlying quantity (the current stock price), normalised slightly differently. Correlations do not arise from data pipeline mistakes — this is expected market microstructure behaviour.

**CLAUDE.md note:** These columns are all available at the same timestamp (within the same seconds_in_bucket snapshot). No future information is encoded. This is not leakage.

---

## 6. imbalance_buy_sell_flag Distribution

| Flag | Count | % |
|---|---|---|
| −1 (sell pressure) | 2,084,349 | 39.8% |
| 0 (neutral) | 1,131,594 | 21.6% |
| +1 (buy pressure) | 2,022,037 | 38.6% |

Slightly more sell-side flags than buy-side, consistent with the slightly negative target mean (−0.048).

---

## 7. Summary: Leakage Risk Assessment

| Check | Result | Risk Level | Mitigation |
|---|---|---|---|
| date_id gaps | None (contiguous) | — | Purge gap of ≥1 in CV folds |
| Target time trend | None (r = 0.024) | — | Standard purged K-fold sufficient |
| Stock coverage | Non-uniform across dates | LOW | CV must be stratified or grouped by date_id, not stock_id |
| far_price missing pattern | Time-dependent (seconds threshold) | NONE | Imputation must be fold-internal |
| Feature correlation | Expected price collinearity | NONE | Drop redundant price columns or let tree models handle |
| Target lookahead | None detected | — | No evidence of future information in features |

**Overall verdict: No data leakage detected.** The dataset appears clean and well-constructed for temporal modelling. The primary constraint remains the CLAUDE.md requirement for strict Purged K-Fold CV with fold-internal statistics only.
