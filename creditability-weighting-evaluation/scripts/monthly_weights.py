from pathlib import Path
from pyspark.sql import SparkSession, functions as F

class MonthlyWeightCalculator:
    """
    Compute monthly reviewer credibility weights:
      w_user = 1 / (1 + var_pop(stars) per (user_id, year, month))
    Writes to: <repo>/julie-creditability-weighting-evaluation/parquet/weights/
    """

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]
        self.spark = SparkSession.builder.appName("monthly-weights").getOrCreate()

    def compute(self, reviews_path: Path):
        print(f"📥 Loading reviews from: {reviews_path}")
        df = self.spark.read.parquet(str(reviews_path))

        agg = (
            df.groupBy("user_id", "year", "month")
              .agg(F.var_pop("stars").alias("var_stars"))
              .withColumn("w_user", 1.0 / (1.0 + F.coalesce(F.col("var_stars"), F.lit(0.0))))
              .orderBy("year", "month")
        )
        return agg

    def write(self, df):
        out_dir = self.base_dir / "parquet" / "weights"
        (
            df.repartition("year", "month")
              .write.mode("overwrite")
              .partitionBy("year", "month")
              .parquet(str(out_dir))
        )
        print(f"✅ Saved monthly reviewer weights to: {out_dir}")
