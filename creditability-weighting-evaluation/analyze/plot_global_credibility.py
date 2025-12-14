from pathlib import Path
from pyspark.sql import SparkSession, functions as F

# plotting libs are optional – import lazily
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]   # .../julie-creditability-weighting-evaluation
    spark = SparkSession.builder.appName("global-credibility-analysis").getOrCreate()

    weights_dir = base_dir / "parquet" / "weights"
    df = spark.read.parquet(str(weights_dir))

    # Global monthly trend: aggregate across all users (Stage A)
    monthly = (
        df.groupBy("year", "month")
          .agg(
              F.avg("w_user").alias("avg_w_user"),
              F.avg("var_stars").alias("avg_var_stars")
          )
          .orderBy("year", "month")
    )

    pdf = monthly.toPandas()
    pdf["date"] = pd.to_datetime(pdf["year"].astype(str) + "-" + pdf["month"].astype(str) + "-01")
    pdf = pdf.set_index("date").sort_index()

    # Simple correlation sanity check
    corr = pdf["avg_w_user"].corr(pdf["avg_var_stars"])
    print(f"Correlation(avg_w_user, avg_var_stars) = {corr:.3f} (expect negative)")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(pdf.index, pdf["avg_w_user"], lw=1.6, label="Average Credibility (w_user)")
    ax1.set_ylabel("Average Credibility (w_user)")
    ax1.set_title("Global Reviewer Credibility Over Time")

    ax2 = ax1.twinx()
    ax2.plot(pdf.index, pdf["avg_var_stars"], lw=1.2, linestyle="--", label="Average Rating Variance")
    ax2.set_ylabel("Average Rating Variance")

    fig.tight_layout()
    plt.show()
