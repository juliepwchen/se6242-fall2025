#!/usr/bin/env python3
# yelp_absa_scoring_baseline.py
# Baseline ABSA: lexicon-based aspect extraction & sentiment scoring (no external models).
# Reads:  parquet/absa_input/sample/
# Writes: parquet/absa_scored/sample/

import time
import re
from typing import List, Dict, Tuple
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, ArrayType
)

# ----------------------------
# 0) Spark session
# ----------------------------
spark = (
    SparkSession.builder
    .appName("YelpABSA_Baseline")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

SRC = "parquet/absa_input/sample"
OUT = "parquet/absa_scored/sample"

print("\n========== Loading ABSA input (sample) ==========")
t0 = time.time()
df = spark.read.parquet(SRC)
print(f"Loaded rows: {df.count():,} in {time.time() - t0:.2f}s")
df.select("business_id","user_id","text","city","state","review_stars","year").show(3, truncate=100)

# ----------------------------
# 1) Lexicons (lightweight, editable)
#    - You can expand these lists as needed.
# ----------------------------
FOOD_TERMS = {
    "food","pizza","burger","taco","sushi","noodle","pasta","steak","salad","sandwich","fries",
    "bread","sauce","soup","dessert","cake","coffee","tea","pho","ramen","bbq","bbq","seafood",
    "chicken","beef","pork","lamb","fish","roll","rice","taste","flavor","fresh","portion"
}
SERVICE_TERMS = {
    "service","staff","waiter","waitress","server","host","hostess","manager","attentive","rude",
    "friendly","slow","fast","cleanliness","attitude","line","queue","mistake","order","tip","bartender"
}
PRICE_TERMS = {
    "price","priced","expensive","cheap","value","deal","cost","pricy","affordable","overpriced",
    "budget","worth","portion","tip","fees","upcharge","charges"
}
AMBIENCE_TERMS = {
    "ambience","atmosphere","noise","noisy","quiet","music","decor","crowded","clean","dirty",
    "lighting","vibe","cozy","romantic","view","seat","seating","smell","odour","odor","space"
}

# General sentiment words (small, interpretable seed lists)
POS_WORDS = {
    "amazing","great","good","tasty","delicious","fresh","excellent","perfect","friendly","quick",
    "nice","clean","cozy","quiet","affordable","worth","fast","attentive","kind","helpful","love","loved"
}
NEG_WORDS = {
    "bad","terrible","awful","cold","stale","bland","rude","slow","dirty","noisy","loud",
    "overpriced","expensive","worst","hate","hated","disgusting","gross","unfriendly","dry","burnt","burned"
}

# Broadcast for performance
bc_food = spark.sparkContext.broadcast(FOOD_TERMS)
bc_service = spark.sparkContext.broadcast(SERVICE_TERMS)
bc_price = spark.sparkContext.broadcast(PRICE_TERMS)
bc_amb = spark.sparkContext.broadcast(AMBIENCE_TERMS)
bc_pos = spark.sparkContext.broadcast(POS_WORDS)
bc_neg = spark.sparkContext.broadcast(NEG_WORDS)

# ----------------------------
# 2) Text utilities
# ----------------------------
_SENT_SPLIT = re.compile(r"[.!?]+")
_TOKEN = re.compile(r"[a-zA-Z']+")

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text.lower()) if s.strip()]

def tokenize(s: str) -> List[str]:
    return _TOKEN.findall(s.lower()) if s else []

# ----------------------------
# 3) Core scoring function
#     - For each sentence: if it contains any aspect term, count pos/neg words in that sentence.
#     - Aspect score = (pos - neg) / (pos + neg + 1e-6)  in [-1, 1]
#     - Also return the matched positive/negative words per aspect (for UI)
# ----------------------------
def score_review(text: str) -> Tuple[float, float, float, float,
                                     List[str], List[str], List[str], List[str],
                                     List[str], List[str], List[str], List[str]]:
    if not text:
        # scores, and empty top-terms buckets
        return (0.0, 0.0, 0.0, 0.0,
                [], [], [], [],
                [], [], [], [])

    food_terms = bc_food.value
    service_terms = bc_service.value
    price_terms = bc_price.value
    amb_terms = bc_amb.value
    pos_words = bc_pos.value
    neg_words = bc_neg.value

    # counts and term captures
    counts = {
        "food": {"pos":0, "neg":0, "pos_terms":[], "neg_terms":[]},
        "service":{"pos":0, "neg":0, "pos_terms":[], "neg_terms":[]},
        "price":{"pos":0, "neg":0, "pos_terms":[], "neg_terms":[]},
        "amb":{"pos":0, "neg":0, "pos_terms":[], "neg_terms":[]}
    }

    for sent in split_sentences(text):
        toks = tokenize(sent)
        if not toks:
            continue
        toks_set = set(toks)

        # Which aspects are active in this sentence?
        active_aspects = []
        if toks_set & food_terms:    active_aspects.append("food")
        if toks_set & service_terms: active_aspects.append("service")
        if toks_set & price_terms:   active_aspects.append("price")
        if toks_set & amb_terms:     active_aspects.append("amb")

        if not active_aspects:
            continue

        # Collect pos/neg words in this sentence
        pos_hits = [w for w in toks if w in pos_words]
        neg_hits = [w for w in toks if w in neg_words]

        for asp in active_aspects:
            counts[asp]["pos"] += len(pos_hits)
            counts[asp]["neg"] += len(neg_hits)
            # store up to a few matched words (kept small to control size)
            counts[asp]["pos_terms"].extend(pos_hits[:5])
            counts[asp]["neg_terms"].extend(neg_hits[:5])

    def ratio(p, n):
        total = p + n
        if total == 0:
            return 0.0
        return float(p - n) / float(total + 1e-6)

    food_score    = ratio(counts["food"]["pos"],    counts["food"]["neg"])
    service_score = ratio(counts["service"]["pos"], counts["service"]["neg"])
    price_score   = ratio(counts["price"]["pos"],   counts["price"]["neg"])
    amb_score     = ratio(counts["amb"]["pos"],     counts["amb"]["neg"])

    # Limit term lists a bit for storage/preview
    def top_terms(lst, k=10):
        return lst[:k] if lst else []

    return (
        food_score, service_score, price_score, amb_score,
        top_terms(counts["food"]["pos_terms"]),    top_terms(counts["food"]["neg_terms"]),
        top_terms(counts["service"]["pos_terms"]), top_terms(counts["service"]["neg_terms"]),
        top_terms(counts["price"]["pos_terms"]),   top_terms(counts["price"]["neg_terms"]),
        top_terms(counts["amb"]["pos_terms"]),     top_terms(counts["amb"]["neg_terms"]),
    )

# ----------------------------
# 4) Register UDF and apply
# ----------------------------
schema = StructType([
    StructField("aspect_food",    DoubleType(), True),
    StructField("aspect_service", DoubleType(), True),
    StructField("aspect_price",   DoubleType(), True),
    StructField("aspect_amb",     DoubleType(), True),

    StructField("food_pos_terms",    ArrayType(StringType()), True),
    StructField("food_neg_terms",    ArrayType(StringType()), True),
    StructField("service_pos_terms", ArrayType(StringType()), True),
    StructField("service_neg_terms", ArrayType(StringType()), True),
    StructField("price_pos_terms",   ArrayType(StringType()), True),
    StructField("price_neg_terms",   ArrayType(StringType()), True),
    StructField("amb_pos_terms",     ArrayType(StringType()), True),
    StructField("amb_neg_terms",     ArrayType(StringType()), True),
])

score_udf = udf(score_review, schema)

print("\n========== Scoring aspects (baseline lexicon) ==========")
t1 = time.time()
scored = df.select(
    "business_id","user_id","city","state","categories","date",
    "review_stars","average_stars","biz_stars","text"
).withColumn("scores", score_udf(col("text")))

# Explode struct into columns
result = scored.select(
    "business_id","user_id","city","state","categories","date",
    "review_stars","average_stars","biz_stars",
    col("scores.aspect_food").alias("aspect_food"),
    col("scores.aspect_service").alias("aspect_service"),
    col("scores.aspect_price").alias("aspect_price"),
    col("scores.aspect_amb").alias("aspect_amb"),
    col("scores.food_pos_terms").alias("food_pos_terms"),
    col("scores.food_neg_terms").alias("food_neg_terms"),
    col("scores.service_pos_terms").alias("service_pos_terms"),
    col("scores.service_neg_terms").alias("service_neg_terms"),
    col("scores.price_pos_terms").alias("price_pos_terms"),
    col("scores.price_neg_terms").alias("price_neg_terms"),
    col("scores.amb_pos_terms").alias("amb_pos_terms"),
    col("scores.amb_neg_terms").alias("amb_neg_terms"),
)

print(f"Finished scoring in {time.time() - t1:.2f}s")
result.show(3, truncate=100)

# ----------------------------
# 5) Write output
# ----------------------------
print(f"\n💾 Writing scored sample to {OUT}")
t2 = time.time()
result.write.mode("overwrite").parquet(OUT)
print(f"Done in {time.time() - t2:.2f}s")

spark.stop()
