from pathlib import Path
from pyspark.sql import SparkSession, functions as F

class ReviewLoader:
    """
    Ingest Yelp reviews JSON and write Parquet partitioned by year/month.
    Output root: <repo>/julie-creditability-weighting-evaluation/parquet/reviews/
    """

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]  # .../julie-creditability-weighting-evaluation
        self.spark = (
            SparkSession.builder.appName("review-ingest")
            .config("spark.sql.parquet.compression.codec", "snappy")
            .getOrCreate()
        )

    def load_json(self, path: Path):
        return self.spark.read.json(str(path))

    def ingest(self, review_json_path: Path):
        df_raw = self.load_json(review_json_path)

        # Keep minimal columns we need for weights; parse timestamp
        df = (
            df_raw.select("user_id", "stars", "date")
            .withColumn("review_ts", F.to_timestamp("date"))
            .withColumn("year", F.year("review_ts"))
            .withColumn("month", F.month("review_ts"))
            .dropna(subset=["user_id", "stars", "review_ts"])
        )

        out_dir = self.base_dir / "parquet" / "reviews"
        (
            df.repartition("year", "month")
              .write.mode("overwrite")
              .partitionBy("year", "month")
              .parquet(str(out_dir))
        )

        print(f"✅ Parquet written to: {out_dir}")
        df.show(5, truncate=False)
        return df
