from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
USER_WEIGHTS = ROOT / "parquet" / "user_monthly_weights"

# choose a few user_ids you care about (e.g. top N by reviews printed earlier)
USER_IDS = [
    "_BcWyKQL16ndpBdggh2kNA",  # replace/extend as you like
    "Xw7ZjaGfr0WNVt6s_5KZfA",
    "0Igx-a1wAstiBDerGxXk2A",
]

def load_user(user_id: str) -> pd.DataFrame:
    # each user is a partition dir: user_id=.../
    paths = list((USER_WEIGHTS / f"user_id={user_id}").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet found for user {user_id}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    df = df.sort_values("date")
    return df

def main():
    n = len(USER_IDS)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.8*n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, uid in zip(axes, USER_IDS):
        df = load_user(uid)
        ax.plot(df["date"], df["w_user"])
        ax.set_title(f"user_id={uid}  (months={len(df)})")
        ax.set_ylabel("w_user")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Per-User Monthly Credibility (w_user)")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
