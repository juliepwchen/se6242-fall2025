#!/usr/bin/env python3
# yelp_cluster_kmeans.py
# Cluster restaurant-level aspect vectors with KMeans and write labeled output for UI.
# Reads:  parquet/aggregated/sample_restaurant_vectors/
#         parquet/yelp_user/business
# Writes: parquet/clustering/sample_kmeans_k{K}/

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

def main():
    spark = (
        SparkSession.builder
        .appName("YelpClusterKMeans")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    SRC_VECTORS = "parquet/aggregated/sample_restaurant_vectors"
    SRC_BUSINESS = "parquet/yelp_user/business"
    K = 6  # tweak as you like (try 4–8)
    OUT_DIR = f"parquet/clustering/sample_kmeans_k{K}"

    print("\n========== Load restaurant vectors ==========")
    vec_df = spark.read.parquet(SRC_VECTORS)
    print(f"Rows (restaurants): {vec_df.count():,}")
    vec_df.select("business_id","food","service","price","amb","n_reviews_sample").show(5, truncate=80)

    # (Optional) filter to stable vectors (already min reviews=10; you can raise threshold)
    df = vec_df.filter(col("n_reviews_sample") >= 10)

    # Assemble feature vector and standardize
    print("\n========== Build features & standardize ==========")
    assembler = VectorAssembler(inputCols=["food","service","price","amb"], outputCol="features_raw")
    df2 = assembler.transform(df)

    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    scaler_model = scaler.fit(df2)
    df_scaled = scaler_model.transform(df2)

    # KMeans
    print(f"\n========== KMeans (k={K}) ==========")
    kmeans = KMeans(featuresCol="features", predictionCol="cluster", seed=42, k=K, maxIter=50)
    kmodel = kmeans.fit(df_scaled)
    clustered = kmodel.transform(df_scaled)

    # Evaluate (silhouette)
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster", metricName="silhouette")
    sil = evaluator.evaluate(clustered)
    print(f"Silhouette (squared euclidean): {sil:.4f}")

    # Centers (in standardized space)
    centers = kmodel.clusterCenters()
    for i, c in enumerate(centers):
        print(f"Center {i}: {c}")

    # Join business metadata for UI convenience
    biz = spark.read.parquet(SRC_BUSINESS).select("business_id","name","city","state","categories","stars","review_count")
    out = (
        clustered.join(biz, on="business_id", how="left")
                 .select(
                     "business_id","name","city","state","categories",
                     "n_reviews_sample","n_users_sample",
                     "food","service","price","amb",
                     "cover_food_frac","cover_service_frac","cover_price_frac","cover_amb_frac",
                     "cluster","stars","review_count"
                 )
    )

    print(f"\n💾 Writing clusters to {OUT_DIR}")
    out.write.mode("overwrite").parquet(OUT_DIR)
    print("Done.")

    # Small peek by cluster
    out.groupBy("cluster").count().orderBy("cluster").show()
    out.select("name","city","food","service","price","amb","cluster").show(10, truncate=80)

    spark.stop()

if __name__ == "__main__":
    main()
