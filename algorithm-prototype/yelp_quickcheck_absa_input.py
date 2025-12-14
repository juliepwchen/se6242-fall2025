from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PreviewABSASample").getOrCreate()
df = spark.read.parquet("parquet/absa_input/sample")
print(f"Rows: {df.count():,}")
df.show(3, truncate=100)
spark.stop()
