import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, count, countDistinct, avg, sum as Fsum, var_pop, coalesce, lit
)

def main():
    spark = (
        SparkSession.builder
        .appName("YelpAggregateAspects")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    SRC = "parquet/absa_scored/sample"
    OUT = "parquet/aggregated/sample_restaurant_vectors"
    MIN_REVIEWS_PER_BIZ = 10  # adjust as you like

    print("\n========== Loading scored reviews ==========")
    t0 = time.time()
    df = spark.read.parquet(SRC)
    total = df.count()
    print(f"Loaded {total:,} scored reviews in {time.time() - t0:.2f}s")
    df.select("business_id","user_id","aspect_food","aspect_service","aspect_price","aspect_amb").show(3, truncate=100)

    # ------------------------------------------------------------
    # 1) Reviewer credibility weights: w_j = 1 / (1 + Var_user(review_stars))
    #    If a user has a single review in the sample -> var = 0 -> w=1
    #    If var is null (rare), coalesce to 0.
    # ------------------------------------------------------------
    print("\n========== Computing reviewer credibility weights ==========")
    t1 = time.time()
    reviewer_stats = (
        df.groupBy("user_id")
          .agg(
              var_pop("review_stars").alias("var_review_stars"),
              count(lit(1)).alias("user_review_count_in_sample")
          )
          .withColumn("var_review_stars", coalesce(col("var_review_stars"), lit(0.0)))
          .withColumn("w_user", 1.0 / (1.0 + col("var_review_stars")))
    )
    print(f"Reviewer rows: {reviewer_stats.count():,} (computed in {time.time()-t1:.2f}s)")

    # Join weights back to scored reviews
    scored = (
        df.join(reviewer_stats.select("user_id","w_user"), on="user_id", how="left")
          .withColumn("w_user", coalesce(col("w_user"), lit(1.0)))
    )

    # ------------------------------------------------------------
    # 2) Prepare weighted contributions per aspect
    #    - Only count rows where aspect != 0 for coverage;
    #      still allow zeros to contribute 0.0 to the weighted mean.
    # ------------------------------------------------------------
    def wcol(name):  # weighted value for an aspect
        return (col("w_user") * col(name))

    def icol(name):  # indicator for coverage
        return when(col(name) != 0.0, lit(1)).otherwise(lit(0))

    scored_w = (
        scored
        .withColumn("w_food",    wcol("aspect_food"))
        .withColumn("w_service", wcol("aspect_service"))
        .withColumn("w_price",   wcol("aspect_price"))
        .withColumn("w_amb",     wcol("aspect_amb"))
        .withColumn("i_food",    icol("aspect_food"))
        .withColumn("i_service", icol("aspect_service"))
        .withColumn("i_price",   icol("aspect_price"))
        .withColumn("i_amb",     icol("aspect_amb"))
    )

    # ------------------------------------------------------------
    # 3) Aggregate to restaurant-level vectors
    #    Weighted mean per aspect: sum(w * score) / sum(w)
    #    Also compute coverage and counts.
    # ------------------------------------------------------------
    print("\n========== Aggregating to restaurant-level vectors ==========")
    t2 = time.time()
    agg = (
        scored_w.groupBy("business_id")
        .agg(
            # metadata for convenience (first/avg as needed)
            count(lit(1)).alias("n_reviews_sample"),
            countDistinct("user_id").alias("n_users_sample"),

            # sums of weights
            Fsum("w_user").alias("sum_w"),

            # weighted sums
            Fsum("w_food").alias("sum_w_food"),
            Fsum("w_service").alias("sum_w_service"),
            Fsum("w_price").alias("sum_w_price"),
            Fsum("w_amb").alias("sum_w_amb"),

            # coverage counts (how many reviews had non-zero aspect evidence)
            Fsum("i_food").alias("cover_food"),
            Fsum("i_service").alias("cover_service"),
            Fsum("i_price").alias("cover_price"),
            Fsum("i_amb").alias("cover_amb")
        )
        # weighted means (protect against sum_w == 0)
        .withColumn("food",    when(col("sum_w") > 0, col("sum_w_food")    / col("sum_w")).otherwise(lit(0.0)))
        .withColumn("service", when(col("sum_w") > 0, col("sum_w_service") / col("sum_w")).otherwise(lit(0.0)))
        .withColumn("price",   when(col("sum_w") > 0, col("sum_w_price")   / col("sum_w")).otherwise(lit(0.0)))
        .withColumn("amb",     when(col("sum_w") > 0, col("sum_w_amb")     / col("sum_w")).otherwise(lit(0.0)))
        # coverage fractions
        .withColumn("cover_food_frac",    col("cover_food")    / col("n_reviews_sample"))
        .withColumn("cover_service_frac", col("cover_service") / col("n_reviews_sample"))
        .withColumn("cover_price_frac",   col("cover_price")   / col("n_reviews_sample"))
        .withColumn("cover_amb_frac",     col("cover_amb")     / col("n_reviews_sample"))
    )

    # Optional: filter out very small restaurants to keep vectors stable
    agg_filtered = agg.filter(col("n_reviews_sample") >= lit(MIN_REVIEWS_PER_BIZ))

    print(f"Aggregated {agg_filtered.count():,} restaurants (min {MIN_REVIEWS_PER_BIZ} reviews) in {time.time()-t2:.2f}s")

    # ------------------------------------------------------------
    # 4) Save restaurant vectors
    # ------------------------------------------------------------
    print(f"\n💾 Writing restaurant vectors to {OUT}")
    t3 = time.time()
    (
        agg_filtered
        .select(
            "business_id",
            "n_reviews_sample",
            "n_users_sample",
            "food","service","price","amb",
            "cover_food_frac","cover_service_frac","cover_price_frac","cover_amb_frac"
        )
        .write.mode("overwrite").parquet(OUT)
    )
    print(f"Done in {time.time()-t3:.2f}s")

    # Small preview
    out_df = spark.read.parquet(OUT)
    out_df.show(5, truncate=100)

    spark.stop()

if __name__ == "__main__":
    main()
