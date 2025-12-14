#!/usr/bin/env python3
# yelp_join_absa_ready.py
# Join filtered review data with business metadata and produce a clean, flat table for ABSA.
# Run from: visual-analytics-project/algorithm-prototype/

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower

def main():
    # -----------------------------------
    # Initialize Spark
    # -----------------------------------
    spark = (
        SparkSession.builder
        .appName("YelpJoinABSAReady")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .getOrCreate()
    )

    # -----------------------------------
    # Paths (relative to algorithm-prototype/)
    # -----------------------------------
    filtered_path = "parquet/filtered/absa_ready"   # from previous step
    business_path = "parquet/yelp_user/business"    # written by yelp_load_all_to_parquet.py
    output_path  = "parquet/joined/absa_flat"       # output of this script

    # -----------------------------------
    # Load filtered reviews
    # -----------------------------------
    print("\n========== Loading Filtered ABSA Dataset ==========")
    t0 = time.time()
    review_df = spark.read.parquet(filtered_path)
    print(f"Loaded {review_df.count():,} filtered reviews in {time.time() - t0:.2f}s.")
    review_df.printSchema()

    # -----------------------------------
    # Load business metadata
    # -----------------------------------
    print("\n========== Loading Business Dataset ==========")
    t1 = time.time()
    business_df = spark.read.parquet(business_path)
    print(f"Loaded {business_df.count():,} businesses in {time.time() - t1:.2f}s.")

    # -----------------------------------
    # Disambiguate overlapping columns BEFORE join
    # (business also has 'stars' and 'review_count')
    # -----------------------------------
    biz_cols = [
        "business_id",
        "categories",
        "latitude",
        "longitude",
        col("stars").alias("biz_stars"),
        col("review_count").alias("biz_review_count"),
    ]

    # -----------------------------------
    # Join reviews with business info
    # -----------------------------------
    print("\n========== Joining Review and Business Info ==========")
    t2 = time.time()
    joined_df = (
        review_df.alias("r")
        .join(business_df.select(*biz_cols).alias("b"), on="business_id", how="left")
    )
    print(f"Joined total rows: {joined_df.count():,} (in {time.time() - t2:.2f}s)")

    # Quick sample (explicit columns to avoid ambiguity)
    joined_df.select(
        "name",
        "categories",
        col("r.stars").alias("review_stars"),
        "text"
    ).show(3, truncate=80)

    # -----------------------------------
    # Keep only columns needed downstream for ABSA
    # -----------------------------------
    absa_flat_df = joined_df.select(
        "business_id",
        "name",
        "city",
        "state",
        "categories",
        col("r.stars").alias("review_stars"),            # review-level stars
        col("r.review_count").alias("user_review_count"),# from user metadata joined earlier
        "average_stars",                                 # user avg stars (from previous join)
        "text",
        "date",
        "user_id",
        "latitude",
        "longitude",
        "biz_stars",                                     # business-level avg stars
        "biz_review_count",                              # business-level review count
    )

    # Optional: normalize casing for later NLP steps
    absa_flat_df = absa_flat_df.withColumn("text", lower(col("text")))

    # -----------------------------------
    # Write output
    # -----------------------------------
    print(f"\n💾 Writing final joined ABSA dataset to {output_path}")
    t3 = time.time()
    absa_flat_df.write.mode("overwrite").parquet(output_path)
    print(f"Done. Saved joined ABSA dataset in {time.time() - t3:.2f}s.\n")

    # Print a small summary of key columns for sanity
    print("========== Summary: Columns in absa_flat ==========")
    print(absa_flat_df.columns)

    spark.stop()

if __name__ == "__main__":
    main()
