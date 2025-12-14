import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, to_date, col

# -----------------------------------
# Initialize Spark Session
# -----------------------------------
spark = (
    SparkSession.builder
    .appName("YelpDatasetToParquet")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.sql.files.maxPartitionBytes", "256m")  # optional tuning for big files
    .getOrCreate()
)

# -----------------------------------
# Relative paths based on your current working directory:
# algorithm-prototype/
# ├── parquet/yelp_user/
# └── ../yelp_dataset/
# -----------------------------------
base_path = "../yelp_dataset/"
out_base = "parquet/yelp_user/"

files = {
    "business": f"{base_path}yelp_academic_dataset_business.json",
    "checkin": f"{base_path}yelp_academic_dataset_checkin.json",
    "review": f"{base_path}yelp_academic_dataset_review.json",
    "tip": f"{base_path}yelp_academic_dataset_tip.json",
    "user": f"{base_path}yelp_academic_dataset_user.json",
}


# -----------------------------------
# Helper: load, inspect, and write to Parquet
# -----------------------------------
def load_and_write(name, path, partition_by=None):
    print(f"\n========== Loading {name.upper()} Dataset ==========")
    start = time.time()
    df = spark.read.json(path)
    duration = time.time() - start
    print(f"Loaded {name} in {duration:.2f}s, total rows: {df.count():,}")

    # Print schema and sample
    print("\n--- Schema ---")
    df.printSchema()
    print("\n--- Sample rows ---")
    df.show(3, truncate=80)

    # Add partition column if applicable
    if partition_by == "year":
        if "date" in df.columns:
            df = df.withColumn("year", year(to_date(col("date"))))
        elif "yelping_since" in df.columns:
            df = df.withColumn("year", year(to_date(col("yelping_since"))))
        else:
            print(f"No date column found for {name}, skipping year partitioning.")
            partition_by = None

    # Write to Parquet
    out_path = f"{out_base}{name}"
    print(f"Writing {name} to {out_path}")
    if partition_by:
        df.write.mode("overwrite").partitionBy("year").parquet(out_path)
    else:
        df.write.mode("overwrite").parquet(out_path)

    print(f"Finished writing {name}.\n")
    return df


# -----------------------------------
# Load and write each dataset
# -----------------------------------
business_df = load_and_write("business", files["business"])
checkin_df = load_and_write("checkin", files["checkin"])
review_df = load_and_write("review", files["review"], partition_by="year")
tip_df = load_and_write("tip", files["tip"], partition_by="year")
user_df = load_and_write("user", files["user"], partition_by="year")

# -----------------------------------
# Print joinable columns
# -----------------------------------
print("\n========== Summary: Key Join Columns ==========")
print("business_df:", [c for c in business_df.columns if "id" in c.lower()])
print("review_df:", [c for c in review_df.columns if "id" in c.lower()])
print("tip_df:", [c for c in tip_df.columns if "id" in c.lower()])
print("checkin_df:", [c for c in checkin_df.columns if "id" in c.lower()])
print("user_df:", [c for c in user_df.columns if "id" in c.lower()])
print("\nAll datasets successfully written to:")
print(f"   {out_base}")

spark.stop()