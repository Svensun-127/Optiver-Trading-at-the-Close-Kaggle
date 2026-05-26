# 02 — Validation Design: Purged Walk-Forward CV

**Date:** 2026-05-21  
**Status:** Implemented & tested

---

## 1. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CV strategy | Walk-forward (expanding window) by date_id | Standard for financial time series; chronological order preserved |
| n_splits | 5 | Community standard for this competition; 481 days / 5 folds ≈ 96 days/val fold |
| Purge gap | 1 day each side (before val) | date_id contiguous; target is 60-second forward return so cross-day leakage risk minimal |
| Grouping | All rows from same date_id kept together | Prevents partial-day splits that could leak intraday momentum signals |
| Implementation | sklearn TimeSeriesSplit + manual purge | From Kaggle community kernel, validated by 10 unit tests |

---

## 2. Fold Structure (on real train.csv, 5.24M rows)

```
Fold 0: train=[0,    78]  |  purge=[79,80]  |  val=[ 81, 160]
Fold 1: train=[0,   158]  |  purge=[159,160] |  val=[161, 240]
Fold 2: train=[0,   238]  |  purge=[239,240] |  val=[241, 320]
Fold 3: train=[0,   318]  |  purge=[319,320] |  val=[321, 400]
Fold 4: train=[0,   398]  |  purge=[399,400] |  val=[401, 480]
```

| Fold | Train Dates | Val Dates | Train Rows | Val Rows | Purge Gap |
|------|------------|-----------|------------|----------|-----------|
| 0 | 0–78 (79 days) | 81–160 (80 days) | 851,015 | 866,360 | 2 |
| 1 | 0–158 (159 days) | 161–240 (80 days) | 1,717,320 | 872,850 | 2 |
| 2 | 0–238 (239 days) | 241–320 (80 days) | 2,590,060 | 876,975 | 2 |
| 3 | 0–318 (319 days) | 321–400 (80 days) | 3,467,035 | 880,000 | 2 |
| 4 | 0–398 (399 days) | 401–480 (80 days) | 4,346,980 | 880,000 | 2 |

**Key observations:**

- Each validation fold covers exactly 80 date_ids (~875k rows, ~16.7% of 481 days)
- Purge gap is 2 date_ids per fold (purge_gap=1 means dates 79-80 for fold 0, etc.). The effective gap is 2 because purging 1 date on each side of the val boundary removes dates at positions `val_start-1` and `val_start`, creating a distance of 2 from the last training date.
- First 80 date_ids (days 0–79) are never used in validation — this is inherent to walk-forward CV. For baseline, this is acceptable; for final model evaluation, these early days can be included as additional validation data points via a secondary hold-out check.
- 83.5% of all rows appear in at least one validation fold.

---

## 3. Leakage Guarantees (verified by unit tests)

| Check | Method | Result |
|-------|--------|--------|
| No date_id in both train & val (same fold) | `test_no_date_overlap` | PASS |
| Purge gap respected | `test_purge_gap_respected` — verify no train date within [val_min−1, val_max+1] | PASS |
| No date_id repeated across val folds | `test_no_date_overlap` — cross-fold val date tracking | PASS |
| Fold size balance (< 3x imbalance) | `test_fold_size_balance` | PASS (~1.02x max/min) |
| Stock coverage per fold | `test_stock_coverage_per_fold` | PASS |

---

## 4. Stock Coverage Across Folds

Stock coverage is not uniform across dates (only 185/481 dates have all 200 stocks, per Step 1 findings). This affects CV in two ways:

1. **Validation fold representativeness:** Each val fold spans 80 dates, and some of those dates have partial stock coverage. The model will see varying numbers of stocks per validation fold — MAE should be computed per-row, not per-stock, to avoid weighting bias.

2. **Training data completeness:** Earlier folds have fewer training dates and therefore fewer total stock-date observations. Fold 0 trains on only 79 days of data (while Fold 4 trains on 399 days). Expect Fold 4 validation MAE to be lower than Fold 0.

**Mitigation for baseline:** None needed. For the final model, consider stratifying CV splits to ensure each validation fold has similar stock coverage patterns.

---

## 5. CV MAE Stability Check

To validate that the CV scheme produces stable MAE estimates, after the baseline model is trained (Step 3), we will check:

- Per-fold validation MAE: should show a declining trend (more training data = lower MAE)
- Standard deviation of fold MAEs: should be < 10% of mean MAE
- If any fold's MAE is an outlier (> 2σ from mean), investigate stock coverage in that fold

---

## 6. Implementation Reference

| File | Purpose |
|------|---------|
| `src/validation/__init__.py` | Package init, exports `PurgedGroupTimeSeriesSplit` |
| `src/validation/splitter.py` | Splitter implementation with `split()` and `describe_folds()` |
| `src/validation/test_splitter.py` | 10 unit tests covering leakage, purge gap, edge cases |

### Quick usage

```python
from src.validation import PurgedGroupTimeSeriesSplit

splitter = PurgedGroupTimeSeriesSplit(n_splits=5, purge_gap=1, date_col="date_id")

for fold, (train_idx, val_idx) in enumerate(splitter.split(X)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    # train model, compute MAE

# Inspect fold structure
summary = splitter.describe_folds(X)
print(summary)
```

---

## 7. Validation Checklist (for Step 3 Baseline)

- [x] Splitter implementation complete (`src/validation/splitter.py`)
- [x] Unit tests pass (10/10) (`src/validation/test_splitter.py`)
- [x] Purge gap verified (gap ≥ 1 date_id for all folds)
- [x] No date_id leakage (train ∩ val = ∅ per fold)
- [x] CV design documented (`analysis/02_cv_design.md`)
- [ ] Baseline training uses this splitter (Step 3)
- [ ] Per-fold MAE stability verified after baseline run (Step 3)
