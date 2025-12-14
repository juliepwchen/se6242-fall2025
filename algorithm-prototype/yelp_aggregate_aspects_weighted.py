#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Part II: Aggregate per-review aspect scores to restaurant-level vectors using reviewer weights.
- Input A: parquet/absa_scored/sample/                  (baseline scored reviews)
- Input B: parquet/weights/sample_reviewer_weights/     (from Part I)
- Output : parquet/aggregated_weighted/sample_restaurant_vectors/
No existing files are modified.
"""

import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum as Fsum, count, countDistinct, avg, coalesce, lit, when
)

def main():
    t0 = time.time()

    spark = (
        SparkSession.builder
        .appName("AggregateAspectsWeighted")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # -----------------------------
    # 1) Load inputs
    # -----------------------------
    scored_in = "parquet/absa_scored/sample"                     # <- baseline scored reviews
    weights_in = "parquet/weights/sample_reviewer_weights"       # <- from Part I

    print("\n========== Load inputs ==========")
    print(f"Scored reviews: {os.path.abspath(scored_in)}")
    print(f"Reviewer weights: {os.path.abspath(weights_in)}")

    df = spark.read.parquet(scored_in)
    w = spark.read.parquet(weights_in)

    # sanity columns
    need_review_cols = {"business_id", "user_id",
                        "aspect_food", "aspect_service", "aspect_price", "aspect_amb"}
    missing = need_review_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in scored reviews: {missing}")

    need_weight_cols = {"user_id", "w_user"}
    missing_w = need_weight_cols - set(w.columns)
    if missing_w:
        raise ValueError(f"Missing required columns in reviewer weights: {missing_w}")

    print(f"✅ Scored rows: {df.count():,}")
    print(f"✅ Weight rows: {w.count():,}")

    # -----------------------------
    # 2) Join weights; default to 1.0 if user missing
    # -----------------------------
    df_w = (
        df.join(w.select("user_id", "w_user"), on="user_id", how="left")
          .withColumn("w_user", coalesce(col("w_user"), lit(1.0)))
          .withColumn("has_food",    (col("aspect_food")    != 0).cast("double"))
          .withColumn("has_service", (col("aspect_service") != 0).cast("double"))
          .withColumn("has_price",   (col("aspect_price")   != 0).cast("double"))
          .withColumn("has_amb",     (col("aspect_amb")     != 0).cast("double"))
    )

    # -----------------------------
    # 3) Weighted aggregation per business
    # -----------------------------
    print("\n========== Aggregating to restaurant-level vectors (weighted) ==========")
    t1 = time.time()
    agg = (
        df_w.groupBy("business_id")
            .agg(
                # weighted numerators
                Fsum(col("w_user") * col("aspect_food")).alias("num_food"),
                Fsum(col("w_user") * col("aspect_service")).alias("num_service"),
                Fsum(col("w_user") * col("aspect_price")).alias("num_price"),
                Fsum(col("w_user") * col("aspect_amb")).alias("num_amb"),
                # denominator
                Fsum(col("w_user")).alias("den"),
                # basic counts
                count(lit(1)).alias("n_reviews_sample"),
                countDistinct("user_id").alias("n_users_sample"),
                # coverage
                avg("has_food").alias("cover_food_frac"),
                avg("has_service").alias("cover_service_frac"),
                avg("has_price").alias("cover_price_frac"),
                avg("has_amb").alias("cover_amb_frac"),
            )
            # weighted means
            .withColumn("food",    when(col("den") > 0, col("num_food")/col("den")).otherwise(lit(0.0)))
            .withColumn("service", when(col("den") > 0, col("num_service")/col("den")).otherwise(lit(0.0)))
            .withColumn("price",   when(col("den") > 0, col("num_price")/col("den")).otherwise(lit(0.0)))
            .withColumn("amb",     when(col("den") > 0, col("num_amb")/col("den")).otherwise(lit(0.0)))
            .select(
                "business_id", "n_reviews_sample", "n_users_sample",
                "food", "service", "price", "amb",
                "cover_food_frac", "cover_service_frac", "cover_price_frac", "cover_amb_frac"
            )
    )
    dur = time.time() - t1
    print(f"✅ Aggregated {agg.count():,} restaurants in {dur:.2f}s")
    agg.show(5, truncate=False)

    # -----------------------------
    # 4) Optional: enforce min review count per restaurant
    # (kept same as your baseline; adjust if needed)
    # -----------------------------
    MIN_REV = 10
    counts = df_w.groupBy("business_id").agg(count(lit(1)).alias("n_reviews_for_cut"))
    agg = agg.join(counts, on="business_id", how="inner").where(col("n_reviews_for_cut") >= lit(MIN_REV)).drop("n_reviews_for_cut")
    print(f"✅ After min {MIN_REV} reviews filter: {agg.count():,} restaurants")

    # -----------------------------
    # 5) Save to a NEW folder with the same schema your clustering expects
    # -----------------------------
    out_dir = "parquet/aggregated_weighted/sample_restaurant_vectors"
    print(f"\n💾 Writing restaurant vectors (weighted) to {out_dir}")
    agg.write.mode("overwrite").parquet(out_dir)
    print(f"✅ Done in {time.time()-t0:.2f}s")

    spark.stop()

if __name__ == "__main__":
    main()
