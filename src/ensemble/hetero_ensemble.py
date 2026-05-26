"""
Heterogeneous ensemble: LightGBM (baseline_004) + CatBoost (catboost_baseline_001).

Computes weighted-average OOF predictions across the two model families,
grid-searches weights, and saves the best ensemble to disk.

Usage:
    python src/ensemble/hetero_ensemble.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_oof(path: str) -> pd.DataFrame:
    oof = pd.read_csv(path)
    # Use baseline_004 integer row_id format for alignment
    oof = oof.sort_values("row_id").reset_index(drop=True)
    return oof


def evaluate(
    pred: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> float:
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def grid_search_weights(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    steps: int = 21,
) -> Tuple[float, float, float]:
    best_w = 0.5
    best_mae = float("inf")
    results = []
    for i in range(steps):
        w = i / (steps - 1)  # weight for model B (CatBoost)
        ensemble = w * pred_b + (1 - w) * pred_a
        mae = evaluate(ensemble, target, mask)
        results.append((w, mae))
        if mae < best_mae:
            best_mae = mae
            best_w = w
    return best_w, best_mae, results


def main() -> None:
    print("=" * 64)
    print("  Heterogeneous Ensemble: LightGBM + CatBoost")
    print("=" * 64)

    # ── 1. Load OOF predictions ─────────────────────────────────────────
    lgb_path = "outputs/baseline_004/oof.csv"
    cb_path = "outputs/catboost_baseline_001/oof.csv"

    if not Path(cb_path).exists():
        print(f"\n  ERROR: CatBoost OOF not found at {cb_path}")
        print("  Run `python src/training/run_catboost.py` first.")
        sys.exit(1)

    lgb = load_oof(lgb_path)
    cb = load_oof(cb_path)

    # Align by position (both loaded from same train.csv in same order)
    target = lgb["target"].values
    pred_lgb = lgb["prediction"].values
    pred_cb = cb["prediction"].values

    valid = (lgb["fold"] != -1) & (cb["fold"] != -1)
    print(f"  Rows: {len(target):,}  |  Validation: {valid.sum():,}")

    # ── 2. Individual metrics ────────────────────────────────────────────
    mae_lgb = evaluate(pred_lgb, target, valid)
    mae_cb = evaluate(pred_cb, target, valid)
    corr = float(np.corrcoef(pred_lgb[valid], pred_cb[valid])[0, 1])

    print(f"\n  Model              MAE")
    print(f"  ─────              ───")
    print(f"  LightGBM (b004)    {mae_lgb:.6f}")
    print(f"  CatBoost           {mae_cb:.6f}")
    print(f"  Correlation:       {corr:.4f}")

    # ── 3. Grid-search ensemble weights ──────────────────────────────────
    best_w, best_mae, sweep = grid_search_weights(pred_lgb, pred_cb, target, valid)

    print(f"\n  Weight sweep (w = CatBoost weight):")
    print(f"  {'w_cb':>5s}   {'MAE':>8s}")
    for w, mae in sweep:
        marker = " <-- best" if abs(w - best_w) < 1e-6 else ""
        print(f"  {w:5.2f}   {mae:8.6f}{marker}")

    # ── 4. Best ensemble ─────────────────────────────────────────────────
    ensemble_pred = best_w * pred_cb + (1 - best_w) * pred_lgb
    ensemble_mae = evaluate(ensemble_pred, target, valid)
    improvement = mae_lgb - ensemble_mae

    print(f"\n  {'─' * 50}")
    print(f"  Best ensemble MAE:  {ensemble_mae:.6f}")
    print(f"  Improvement:        {improvement:+.6f}  (vs best single)")
    print(f"  Weight:             LightGBM={1-best_w:.2f}  CatBoost={best_w:.2f}")

    # ── 5. Save outputs ──────────────────────────────────────────────────
    out_dir = Path("outputs/ensemble_hetero_001")
    out_dir.mkdir(parents=True, exist_ok=True)

    # OOF
    oof_df = pd.DataFrame({
        "row_id": lgb["row_id"],
        "target": target,
        "prediction": ensemble_pred.astype(np.float32),
        "fold": lgb["fold"],
    })
    oof_df.to_csv(out_dir / "oof.csv", index=False)
    print(f"\n  Saved: {out_dir / 'oof.csv'}")

    # Submission
    sub_df = oof_df[["row_id", "prediction"]].copy()
    sub_df.to_csv(out_dir / "submission.csv", index=False)
    print(f"  Saved: {out_dir / 'submission.csv'}")

    # Metrics JSON
    metrics: Dict[str, Any] = {
        "experiment": "ensemble_hetero_001",
        "method": "weighted_average",
        "models": {
            "lightgbm": {"name": "baseline_004", "mae": mae_lgb, "weight": 1 - best_w},
            "catboost": {"name": "catboost_baseline_001", "mae": mae_cb, "weight": best_w},
        },
        "prediction_correlation": round(corr, 4),
        "ensemble_mae": round(ensemble_mae, 6),
        "best_single_mae": round(min(mae_lgb, mae_cb), 6),
        "improvement": round(improvement, 6),
        "weight_sweep": [
            {"w_catboost": round(w, 2), "mae": round(m, 6)} for w, m in sweep
        ],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {out_dir / 'metrics.json'}")

    print(f"\n{'=' * 64}")
    if improvement > 0.001:
        print(f"  SUCCESS: Ensemble beats best single model by {improvement:.4f} MAE")
    elif improvement > 0:
        print(f"  Marginal gain: {improvement:.4f} MAE — near noise level")
    else:
        print(f"  No improvement — ensemble does not help")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
