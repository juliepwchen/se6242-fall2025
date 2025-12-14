#!/usr/bin/env python3
# yelp_export_for_tableau_reweighting.py
# Creates a UI CSV with standardized (z-score) features so Tableau can do slider-weighted clustering.
# Reads:  parquet/clustering/sample_kmeans_k6
# Writes: ui/tableau_reweighting_input/restaurant_vectors_z.csv

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean as Fmean, stddev as Fstd
import os, shutil

spark = SparkSession.builder.appName("ExportForTableauReweighting").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

SRC = "parquet/clustering/sample_kmeans_k6"
OUT_DIR = "ui/tableau_reweighting_input"
OUT_CSV = os.path.join(OUT_DIR, "restaurant_vectors_z.csv")

df = spark.read.parquet(SRC).select(
    "business_id","name","city","state","categories",
    "food","service","price","amb","cluster",
    "n_reviews_sample","n_users_sample",
    "cover_food_frac","cover_service_frac","cover_price_frac","cover_amb_frac",
    "stars","review_count"
)

# compute global means/stds for z-scores
stats = df.select(
    Fmean("food").alias("m_food"),     Fstd("food").alias("s_food"),
    Fmean("service").alias("m_serv"),  Fstd("service").alias("s_serv"),
    Fmean("price").alias("m_price"),   Fstd("price").alias("s_price"),
    Fmean("amb").alias("m_amb"),       Fstd("amb").alias("s_amb"),
).collect()[0]

m_food, s_food = float(stats.m_food), float(stats.s_food)
m_serv, s_serv = float(stats.m_serv), float(stats.s_serv)
m_price, s_price = float(stats.m_price), float(stats.s_price)
m_amb, s_amb = float(stats.m_amb), float(stats.s_amb)

dfz = (
    df
    .withColumn("food_z",   (col("food")   - m_food) / s_food)
    .withColumn("service_z",(col("service")- m_serv) / s_serv)
    .withColumn("price_z",  (col("price")  - m_price)/ s_price)
    .withColumn("amb_z",    (col("amb")    - m_amb) / s_amb)
)

# write single CSV
dfz.coalesce(1).write.mode("overwrite").option("header", True).csv(OUT_DIR)
# rename part file
parts = [f for f in os.listdir(OUT_DIR) if f.startswith("part-") and f.endswith(".csv")]
if parts:
    src = os.path.join(OUT_DIR, parts[0])
    if os.path.exists(OUT_CSV): os.remove(OUT_CSV)
    shutil.move(src, OUT_CSV)
    print("Wrote:", os.path.abspath(OUT_CSV))
else:
    print("⚠️ No CSV part written — check Spark output dir.")

spark.stop()