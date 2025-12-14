import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, to_timestamp

spark = (
    SparkSession.builder
    .appName("YelpPrepareABSACorpus")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .getOrCreate()
)

SRC = "parquet/joined/absa_flat"
OUT_SAMPLE = "parquet/absa_input/sample"
OUT_FULL = "parquet/absa_input/full_by_year"

print("\n========== Loading joined ABSA-flat dataset ==========")
df = spark.read.parquet(SRC)
print(f"Loaded rows: {df.count():,}")

# Keep the minimal columns ABSA needs
df_small = df.select(
    "business_id",
    "user_id",
    "text",
    "city",
    "state",
    "categories",
    "date",
    "review_stars",
    "average_stars",
    "biz_stars",
)

# Cast date to timestamp and add year for optional partitioning
df_small = df_small.withColumn("ts", to_timestamp(col("date"))) \
                   .withColumn("year", year(col("ts")))

# ---- Small sample for fast ABSA prototyping ----
sample_n = 110000  # adjust as you like
print(f"\nCreating sample of {sample_n:,} reviews for fast ABSA iteration ...")
sample_df = df_small.limit(sample_n)
sample_df.write.mode("overwrite").parquet(OUT_SAMPLE)
print(f"Wrote sample to {OUT_SAMPLE}")

# ---- (Optional) Full ABSA input partitioned by year ----
# Comment this block out if you don’t want it yet.
print("\nWriting full ABSA input partitioned by year (can be large) ...")
df_small.write.mode("overwrite").partitionBy("year").parquet(OUT_FULL)
print(f"Wrote full input to {OUT_FULL}")

spark.stop()
