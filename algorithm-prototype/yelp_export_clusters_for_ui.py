from pyspark.sql import SparkSession
import os, shutil

def main():
    spark = SparkSession.builder.appName("ExportClustersForUI").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # --- Paths (relative to algorithm-prototype directory) ---
    SRC_PARQUET = "parquet/clustering/sample_kmeans_k6"
    OUT_UI_DIR = "ui/sample_kmeans_k6_for_ui_csv"

    # --- Load the clustered restaurant dataset ---
    print("\n========== Loading clustered dataset ==========")
    df = spark.read.parquet(SRC_PARQUET)
    print(f"Rows: {df.count():,}")
    df.select("name", "city", "cluster", "food", "service", "price", "amb").show(5, truncate=80)

    # --- Select columns for the UI ---
    ui_df = df.select(
        "business_id", "name", "city", "state", "categories",
        "food", "service", "price", "amb", "cluster",
        "n_reviews_sample", "n_users_sample",
        "cover_food_frac", "cover_service_frac", "cover_price_frac", "cover_amb_frac",
        "stars", "review_count"
    )

    # --- Write to CSV in dedicated UI folder ---
    print(f"\n💾 Writing to CSV directory: {OUT_UI_DIR}")
    ui_df.coalesce(1).write.mode("overwrite").option("header", True).csv(OUT_UI_DIR)

    # --- Rename CSV part file to a friendly name ---
    csv_dir = OUT_UI_DIR
    target = os.path.join(csv_dir, "restaurants_clusters_k6.csv")
    parts = [f for f in os.listdir(csv_dir) if f.startswith("part-") and f.endswith(".csv")]
    if parts:
        src = os.path.join(csv_dir, parts[0])
        if os.path.exists(target):
            os.remove(target)
        shutil.move(src, target)
        print(f"Single CSV written at: {os.path.abspath(target)}")
    else:
        print("⚠️ No part-*.csv found; check the output directory.")

    # --- Summary ---
    ui_df.groupBy("cluster").count().orderBy("cluster").show()

    spark.stop()

if __name__ == "__main__":
    main()
