# scripts/select_top_reviewers.py
from pathlib import Path
from pyspark.sql import SparkSession, functions as F

def project_root() -> Path:
    # .../visual-analytics-project/julie-creditability-weighting-evaluation
    return Path(__file__).resolve().parents[1]

def get_spark(app="SelectTopReviewers"):
    return (SparkSession.builder
            .appName(app)
            .config("spark.sql.files.maxPartitionBytes", 128 * 1024 * 1024)
            .getOrCreate())

def main(top_k: int = 5000):
    root = project_root()
    spark = get_spark()

    reviews_dir = root / "parquet" / "reviews"   # produced earlier
    out_dir     = root / "parquet" / "top_users"

    df = spark.read.parquet(str(reviews_dir))

    # Count total reviews per user across all time
    by_user = (df.groupBy("user_id")
                 .agg(F.count("*").alias("n_reviews_total"))
                 .orderBy(F.desc("n_reviews_total")))

    top = by_user.limit(top_k)

    # Save both a compact parquet and a CSV (handy to eyeball)
    top.coalesce(1).write.mode("overwrite").parquet(str(out_dir))
    top.coalesce(1).write.mode("overwrite").option("header", True)\
        .csv(str(out_dir.with_name("top_users_csv")))

    top.show(10, truncate=False)
    print(f"✅ Saved top {top_k} users under: {out_dir}")

if __name__ == "__main__":
    main()
