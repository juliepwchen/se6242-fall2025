# analyze/plot_ar_examples.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg

# Run from:
#   cd visual-analytics-project/julie-creditability-weighting-evaluation
#   python3.10 analyze/plot_ar_examples.py

USER_TS_PATH = Path("parquet/user_monthly_weights")

def load_all_user_series() -> pd.DataFrame:
    """
    Load the full user_monthly_weights parquet dataset into a single DataFrame.
    Assumes run_build_user_monthly_series.py wrote a partitioned parquet dataset under
    parquet/user_monthly_weights/.
    """
    if not USER_TS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {USER_TS_PATH}. "
            "Make sure run_build_user_monthly_series.py has been executed."
        )
    return pd.read_parquet(USER_TS_PATH)

def ensure_datetime(df_user: pd.DataFrame) -> pd.Series:
    """
    Build a datetime index for a single user's series.
    Tries 'date' column first, then (year, month).
    """
    cols = set(df_user.columns)
    if "date" in cols:
        dates = pd.to_datetime(df_user["date"])
    elif {"year", "month"} <= cols:
        dates = pd.to_datetime(
            df_user["year"].astype(str) + "-" + df_user["month"].astype(str) + "-01"
        )
    else:
        raise ValueError(
            f"Could not infer time index; columns are: {df_user.columns.tolist()}"
        )
    return dates

def get_w_series(df_user: pd.DataFrame) -> pd.Series:
    """
    Return the credibility series as a pandas Series.
    """
    if "w_user" in df_user.columns:
        vals = df_user["w_user"]
    elif "w" in df_user.columns:
        vals = df_user["w"]
    else:
        raise ValueError(
            f"Could not find credibility column (w_user/w). "
            f"Columns are: {df_user.columns.tolist()}"
        )
    return vals.astype(float)

def choose_example_users(df_all: pd.DataFrame, min_obs: int = 36, max_users: int = 4):
    """
    Pick up to max_users users who have at least min_obs monthly observations.
    """
    counts = df_all.groupby("user_id").size()
    candidates = counts[counts >= min_obs].sort_values(ascending=False)
    selected = candidates.head(max_users).index.tolist()
    print("Selected users for Figure 4:")
    for uid in selected:
        print("  -", uid)
    return selected

def plot_ar_examples():
    # 1) Load full user-month dataset
    df_all = load_all_user_series()

    # 2) Pick 4 representative users with enough history
    if "user_id" not in df_all.columns:
        raise ValueError(f"'user_id' column not found. Columns: {df_all.columns.tolist()}")

    selected_users = choose_example_users(df_all, min_obs=36, max_users=4)
    if not selected_users:
        raise RuntimeError("No users found with at least 36 observations.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, uid in zip(axes, selected_users):
        df_user = df_all[df_all["user_id"] == uid].copy()
        dates = ensure_datetime(df_user)
        w = get_w_series(df_user)

        ts = pd.Series(w.values, index=dates).sort_index()
        ts = ts.asfreq("MS")
        ts = ts[ts.notna()]

        if len(ts) < 10:
            print(f"⚠️ User {uid} has too few non-NA points after resampling; skipping.")
            continue

        # Fit AR(1) and AR(2), pick lower AIC
        res1 = AutoReg(ts, lags=1, old_names=False).fit()
        aic1 = res1.aic

        if len(ts) > 2:
            res2 = AutoReg(ts, lags=2, old_names=False).fit()
            aic2 = res2.aic
        else:
            res2 = None
            aic2 = float("inf")

        if aic1 <= aic2:
            best_res = res1
            order = 1
        else:
            best_res = res2
            order = 2

        fitted = best_res.fittedvalues

        ax.plot(ts.index, ts.values, label="Actual", lw=2)
        ax.plot(fitted.index, fitted.values, label="Fitted", ls="--")

        ax.set_title(f"User {uid} — AR({order})")
        ax.set_xlabel("Month")
        ax.set_ylabel("Credibility Weight")
        ax.legend()

    fig.suptitle("Figure 4 — AR Model Fit Examples", fontsize=16)
    fig.tight_layout()
    fig.savefig("fig_ar_model_examples.png", dpi=200)
    print("Saved → fig_ar_model_examples.png")

if __name__ == "__main__":
    plot_ar_examples()
