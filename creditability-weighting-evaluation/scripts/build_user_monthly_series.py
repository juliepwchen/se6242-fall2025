# scripts/build_user_monthly_series.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, functions as F, Window


def project_root() -> Path:
    """
    Returns the root of the 'julie-creditability-weighting-evaluation' project
    regardless of where this script is executed from.
    (…/julie-creditability-weighting-evaluation/)
    """
    # scripts/ -> project root
    return Path(__file__).resolve().parents[1]


def get_spark(app: str = "BuildUserMonthlySeries") -> SparkSession:
    """
    Create a Spark session tuned for local development on a laptop.
    - Fewer shuffle partitions to reduce task overhead
    - Larger file/partition bytes to cut tiny-file counts
    """
    return (
        SparkSession.builder.appName(app)
        .config("spark.driver.memory", "6g")  # adjust if needed
        .config("spark.sql.shuffle.partitions", "96")
        .config("spark.sql.files.maxPartitionBytes", 256 * 1024 * 1024)  # 256MB
        .getOrCreate()
    )


def load_reviews(spark: SparkSession, root: Path):
    reviews_dir = root / "parquet" / "reviews"
    df = spark.read.parquet(str(reviews_dir))
    # Expect schema from earlier ingest: user_id, stars, review_ts, year, month
    required = {"user_id", "stars", "review_ts", "year", "month"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"reviews missing required columns: {missing}")
    return df.select("user_id", "stars", "review_ts", "year", "month")


def load_top_users(spark: SparkSession, root: Path):
    top_dir = root / "parquet" / "top_users"
    df = spark.read.parquet(str(top_dir))
    # Expect: user_id, n_reviews_total
    required = {"user_id", "n_reviews_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"top_users missing required columns: {missing}")
    return df.select("user_id", "n_reviews_total")


def build_user_monthly_series(
    spark: SparkSession,
    root: Path,
    repartition_users: int = 96,
    persist_intermediate: bool = True,
    min_reviews_per_month: Optional[int] = None,
):
    """
    Create per-user monthly time series with:
      - var_stars: population variance of stars in that user-month
      - w_user  : 1 / (1 + var_stars)
      - n_reviews: monthly review count per user

    Output:
      - parquet/user_monthly_weights/  (partitioned by user_id)
      - parquet/user_monthly_summary/  (one monthly rollup across users)
    """
    reviews = load_reviews(spark, root)
    top_users = load_top_users(spark, root)

    # Restrict to the top cohort (e.g., 5k users)
    reviews_top = reviews.join(top_users, on="user_id", how="inner")

    # Compute monthly stats per user
    grouped = (
        reviews_top.groupBy("user_id", "year", "month")
        .agg(
            F.count(F.lit(1)).alias("n_reviews"),
            F.var_pop("stars").alias("var_stars"),
        )
    )

    # Handle months with a single review: var_pop = NULL -> treat as 0 variance
    grouped = grouped.withColumn("var_stars", F.coalesce(F.col("var_stars"), F.lit(0.0)))
    grouped = grouped.withColumn("w_user", 1 / (1 + F.col("var_stars")))

    if min_reviews_per_month is not None and min_reviews_per_month > 0:
        grouped = grouped.filter(F.col("n_reviews") >= F.lit(min_reviews_per_month))

    # Order columns for readability
    tidy = grouped.select(
        "user_id", "year", "month", "n_reviews", "var_stars", "w_user"
    ).orderBy("user_id", "year", "month")

    if persist_intermediate:
        tidy = tidy.persist()

    # ---- Write per-user series (avoid million tiny dirs) ----
    out_dir = root / "parquet" / "user_monthly_weights"
    (
        tidy.repartition(repartition_users, F.col("user_id"))
        .write.mode("overwrite")
        .option("compression", "snappy")
        .option("maxRecordsPerFile", 250_000)
        .partitionBy("user_id")  # only user_id as a partition key
        .parquet(str(out_dir))
    )

    # ---- Also write a compact global monthly summary ----
    summary = (
        tidy.groupBy("year", "month")
        .agg(
            F.avg("w_user").alias("avg_w_user"),
            F.avg("var_stars").alias("avg_var_stars"),
            F.sum("n_reviews").alias("total_reviews"),
        )
        .orderBy("year", "month")
    )

    summary_dir = root / "parquet" / "user_monthly_summary"
    (
        summary.coalesce(1)
        .write.mode("overwrite")
        .option("compression", "snappy")
        .parquet(str(summary_dir))
    )

    print(f"✅ Per-user monthly series saved under: {out_dir}")
    print(f"✅ Global monthly summary saved under: {summary_dir}")


def main():
    root = project_root()
    spark = get_spark()

    try:
        build_user_monthly_series(
            spark,
            root,
            repartition_users=96,          # tweak for your machine
            persist_intermediate=True,
            min_reviews_per_month=None,    # or set e.g. 2 to reduce noise
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
