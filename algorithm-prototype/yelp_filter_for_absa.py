import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# -----------------------------------
# Initialize Spark Session
# -----------------------------------
spark = (
    SparkSession.builder
    .appName("YelpFilterForABSA")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .getOrCreate()
)

# -----------------------------------
# Load Parquet datasets (already written in Step 1)
# -----------------------------------
base_path = "parquet/yelp_user/"
business_df = spark.read.parquet(f"{base_path}business")
review_df = spark.read.parquet(f"{base_path}review")
user_df = spark.read.parquet(f"{base_path}user")

# -----------------------------------
# Step 1: Filter to restaurant-related businesses only
# -----------------------------------
print("\n========== Filtering Businesses ==========")
start = time.time()
restaurant_df = business_df.filter(
    col("categories").isNotNull() &
    col("categories").rlike("(?i)Restaurant|Food|Cafe|Bar|Bakery|Coffee")
)
print(f"Filtered to {restaurant_df.count():,} restaurant-type businesses.")
restaurant_df.select("name", "categories", "city", "state").show(3, truncate=80)

# -----------------------------------
# Step 2: Join with reviews
# -----------------------------------
print("\n========== Joining Business and Review ==========")
review_rest_df = (
    review_df.join(restaurant_df.select("business_id", "name", "city", "state", "categories"), "business_id")
)
print(f"Joined reviews count: {review_rest_df.count():,}")

# Optional: small sanity print
review_rest_df.select("business_id", "city", "stars", "text").show(3, truncate=80)

# -----------------------------------
# Step 3: (Optional) Join with user info
# -----------------------------------
print("\n========== Joining with User Metadata ==========")
review_user_df = review_rest_df.join(
    user_df.select("user_id", "review_count", "average_stars"), "user_id", "left"
)
print(f"Joined with user metadata: {review_user_df.count():,}")

# -----------------------------------
# Step 4: Select only columns relevant for ABSA (Aspect Extraction & Scoring)
# -----------------------------------
absa_ready_df = review_user_df.select(
    "business_id",
    "name",
    "city",
    "state",
    "user_id",
    "review_count",     # from users
    "average_stars",    # from users
    "stars",
    "text",
    "date",
)

# -----------------------------------
# Step 5: Write intermediate output (Parquet)
# -----------------------------------
out_path = "parquet/filtered/absa_ready"
print(f"\n💾 Writing filtered ABSA-ready dataset to {out_path}")
absa_ready_df.write.mode("overwrite").parquet(out_path)
print(f"✅ Done. Saved filtered dataset for Aspect Extraction & Scoring.\n")

spark.stop()
