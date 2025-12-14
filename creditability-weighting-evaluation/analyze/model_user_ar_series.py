#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit AR(1)/AR(2) to each user's monthly credibility series (w_user),
pick the lower-AIC model, and write a concise metrics table.

Input:
  parquet/user_monthly_weights/    # directory from build_user_monthly_series.py
Output:
  results/ar_models/ar_model_summary.parquet
  results/ar_models/ar_model_summary.csv
"""

from __future__ import annotations
import os
from typing import Dict, Any, List
from pathlib import Path

import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
import pyarrow.dataset as ds

# ==========================================================
#          PATHS RESOLVED RELATIVE TO PROJECT ROOT
# ==========================================================

# BASE_DIR = project root = analyze/.. = julie-creditability-weighting-evaluation/
BASE_DIR = Path(__file__).resolve().parents[1]

PARQUET_USER_SERIES = BASE_DIR / "parquet" / "user_monthly_weights"
OUT_DIR = BASE_DIR / "results" / "ar_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
#                     Helper Function
# ==========================================================
def fit_ar_best(ts: pd.Series, max_p: int = 2, min_obs: int = 24) -> Dict[str, Any]:
    """
    Fit AR(p) for p=1..max_p and pick the lowest AIC.
    Returns dict with fit stats or {ok: False, reason: "..."}.
    """
    # Ensure sorted by time
    ts = ts.sort_index()

    # 🔧 NEW: collapse duplicate months (if any) by averaging
    # This avoids "cannot reindex on an axis with duplicate labels" in asfreq.
    ts = ts.groupby(ts.index).mean()

    # Keep monthly frequency (silences statsmodels date-freq warnings)
    ts = ts.asfreq("MS")
    ts = ts[ts.notna()]  # drop missing but keep freq attribute

    try:
        ts.index.freq = pd.tseries.frequencies.to_offset("MS")
    except Exception:
        pass

    if ts.size < min_obs:
        return {"ok": False, "reason": f"insufficient_obs(<{min_obs})"}

    best = None
    for p in range(1, max_p + 1):
        try:
            model = AutoReg(ts, lags=p, old_names=False)
            res = model.fit()
            aic = res.aic
            phi = res.params
            resid = res.resid
            lb = acorr_ljungbox(resid, lags=[12], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])

            rec = {
                "ok": True,
                "best_order": p,
                "aic": float(aic),
                "n_obs": int(ts.size),
                "phi1": float(phi[1]) if p >= 1 else np.nan,
                "phi2": float(phi[2]) if p >= 2 else np.nan,
                "sigma2_resid": float(np.var(resid, ddof=1)),
                "lb_pvalue_12": lb_p,
            }
            if (best is None) or (rec["aic"] < best["aic"]):
                best = rec
        except Exception:
            continue

    return best if best is not None else {"ok": False, "reason": "all_orders_failed"}


# ==========================================================
#                         Main
# ==========================================================
def main(limit_users: int | None = None):
    print("📥 Loading user_monthly_weights dataset ...")
    dataset = ds.dataset(str(PARQUET_USER_SERIES), format="parquet")

    # NOTE: user_id is not in the schema; we will reconstruct it from __filename
    table = dataset.to_table(
        columns=["year", "month", "w_user", "__filename"]
    )
    df = table.to_pandas()

    # Derive user_id from filename (e.g., ".../user_12345.parquet" -> "user_12345")
    df["user_id"] = df["__filename"].apply(lambda p: Path(p).stem)

    # Create a proper monthly timestamp
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" +
        df["month"].astype(str) + "-01"
    )
    df = df.sort_values(["user_id", "date"])

    # (Optional) limit sample for faster testing
    if limit_users is not None:
        top_users = (
            df.groupby("user_id")["w_user"]
              .size()
              .sort_values(ascending=False)
              .head(limit_users)
              .index
        )
        df = df[df["user_id"].isin(top_users)]

    # ----- Fit AR models -----
    records: List[Dict[str, Any]] = []

    for uid, g in df.groupby("user_id", sort=False):
        ts = g.set_index("date")["w_user"]
        out = fit_ar_best(ts, max_p=2, min_obs=24)

        rec = {
            "user_id": uid,
            "first": ts.index.min(),
            "last": ts.index.max(),
        }
        rec.update(out)
        records.append(rec)

    res = pd.DataFrame(records)

    # ----- Save outputs -----
    out_parquet = OUT_DIR / "ar_model_summary.parquet"
    out_csv = OUT_DIR / "ar_model_summary.csv"

    res.to_parquet(out_parquet, index=False)
    res.to_csv(out_csv, index=False)

    print("✅ Wrote summary:")
    print(f"  - {out_parquet.resolve()}")
    print(f"  - {out_csv.resolve()}")

    # ----- Summary stats -----
    ok = res["ok"].fillna(False)
    ar1 = (res["best_order"] == 1) & ok
    ar2 = (res["best_order"] == 2) & ok

    print(f"📊 Fit success: {ok.sum()}/{len(res)} users")
    print(f"    AR(1): {ar1.sum()}, AR(2): {ar2.sum()}")
    print("    Tip: inspect lowest-AIC users in the CSV; consider GARCH for large sigma2_resid.")


# ==========================================================
#                   Script Entry Point
# ==========================================================
if __name__ == "__main__":
    # Default small sample for quick smoke test
    main(limit_users=300)
