import pandas as pd
from pathlib import Path

SRC = Path("ui/data/restaurants_clusters_k6.csv")
OUT_1K = Path("ui/data/dev_1k.csv")
OUT_10 = Path("ui/data/dev_10.csv")

def main():
    df = pd.read_csv(SRC)
    # Shuffle and sample with a fixed seed for reproducibility
    df_1k = df.sample(n=min(1000, len(df)), random_state=42)
    df_10 = df.sample(n=min(10, len(df)), random_state=42)

    df_1k.to_csv(OUT_1K, index=False)
    df_10.to_csv(OUT_10, index=False)

    print(f"Wrote {OUT_1K} ({len(df_1k)} rows)")
    print(f"Wrote {OUT_10} ({len(df_10)} rows)")

if __name__ == "__main__":
    main()
