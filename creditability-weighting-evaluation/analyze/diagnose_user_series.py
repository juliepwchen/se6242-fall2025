from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf

ROOT = Path(__file__).resolve().parents[1]
USER_WEIGHTS = ROOT / "parquet" / "user_monthly_weights"

N_USERS = 12         # how many to inspect
MIN_MONTHS = 24      # ensure enough history

def collect_users() -> pd.DataFrame:
    # scan partition folders; build small meta table
    rows = []
    for p in USER_WEIGHTS.glob("user_id=*"):
        uid = p.name.split("=", 1)[1]
        parts = list(p.glob("*.parquet"))
        if not parts: 
            continue
        df = pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
        df = df.sort_values("date")
        # keep users with enough months
        if len(df) >= MIN_MONTHS:
            rows.append({
                "user_id": uid,
                "n_months": len(df),
                "var_w_user": float(np.var(df["w_user"])),
                "first": df["date"].iloc[0],
                "last": df["date"].iloc[-1],
            })
    return pd.DataFrame(rows).sort_values(["n_months","var_w_user"], ascending=[False, False])

def diagnose_series(df: pd.DataFrame, title: str):
    y = df["w_user"].values
    dates = df["date"]
    fig = plt.figure(figsize=(11, 6))
    # series
    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(dates, y)
    ax1.set_title(title); ax1.set_ylabel("w_user")
    # ADF
    adf_stat, pval, *_ = adfuller(y, autolag="AIC")
    ax1.text(0.01, 0.02, f"ADF p={pval:.3f}", transform=ax1.transAxes)
    # ACF
    ax2 = fig.add_subplot(2,2,2)
    acf_vals = acf(y, nlags=min(24, len(y)//2), fft=True)
    ax2.vlines(range(len(acf_vals)), [0], acf_vals)
    ax2.set_title("ACF")
    # PACF
    ax3 = fig.add_subplot(2,2,3)
    pacf_vals = pacf(y, nlags=min(24, len(y)//2), method="ywm")
    ax3.vlines(range(len(pacf_vals)), [0], pacf_vals)
    ax3.set_title("PACF")
    # rolling mean (stability vis)
    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(dates, pd.Series(y).rolling(12, min_periods=1).mean())
    ax4.set_title("12-mo Rolling Mean")
    fig.tight_layout()
    plt.show()
    return float(pval)

def main():
    meta = collect_users()
    if meta.empty:
        print("No users found meeting MIN_MONTHS criterion.")
        return
    # pick top N by months (ties broken by variance desc)
    pick = meta.head(N_USERS)
    print(pick[["user_id","n_months","var_w_user","first","last"]].to_string(index=False))

    # loop and diagnose
    for _, row in pick.iterrows():
        p = USER_WEIGHTS / f"user_id={row.user_id}"
        parts = list(p.glob("*.parquet"))
        df = pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
        df = df.sort_values("date")
        diagnose_series(df, f"user={row.user_id} (months={len(df)})")

if __name__ == "__main__":
    main()
