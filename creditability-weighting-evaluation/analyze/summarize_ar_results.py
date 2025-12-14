# analyze/summarize_ar_results.py

import pandas as pd
from pathlib import Path

# Run this from the project root:
#   python3.10 analyze/summarize_ar_results.py
SUMMARY_CSV = Path("results/ar_models/ar_model_summary.csv")

def main():
    # Load AR summary table
    df = pd.read_csv(SUMMARY_CSV, parse_dates=["first", "last"])
    df_ok = df[df["ok"]].copy()

    # Add a single persistence column (phi) no matter AR(1) or AR(2)
    df_ok["phi"] = df_ok.apply(
        lambda r: r["phi2"] if r["best_order"] == 2 and pd.notna(r["phi2"]) else r["phi1"],
        axis=1,
    )

    print("\nTop 10 users by persistence (phi):")
    print(
        df_ok.sort_values("phi", ascending=False)
        .head(10)[["user_id", "best_order", "phi1", "phi2", "phi", "aic", "n_obs"]]
    )

    print("\nUsers that may need GARCH (large residual variance or serial correlation):")
    flag = (
        df_ok["sigma2_resid"] > df_ok["sigma2_resid"].quantile(0.9)
    ) | (df_ok["lb_pvalue_12"] < 0.05)

    print(
        df_ok[flag]
        .sort_values(["sigma2_resid", "lb_pvalue_12"])
        .head(12)[["user_id", "best_order", "sigma2_resid", "lb_pvalue_12", "n_obs"]]
    )

    print("\nDistribution snapshots:")
    print(df_ok[["phi", "aic", "sigma2_resid"]].describe())

if __name__ == "__main__":
    main()
