#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yelp_absa_scoring_pyabsa.py
- Reads the 110k review sample from parquet/absa_input/sample/ (created by yelp_prepare_absa_corpus.py)
- Runs PyABSA ATEPC (English checkpoint) to extract aspect terms + sentiments
- Maps extracted terms into coarse aspects: food, service, price, ambience (simple dictionary mapping)
- Writes results to parquet/absa_scored_pyabsa/sample/  (kept separate from baseline outputs)
- Prints lightweight progress checkpoints: processed 1k, 2k, ... rows
- Suppresses noisy logging (HF/Transformers/PyABSA/Spark INFO)
"""

import os
import sys
import json
import logging
from typing import Iterator, List, Dict, Any

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Silence third-party loggers *before* imports
for name in [
    "pyabsa",
    "transformers",
    "huggingface_hub",
    "urllib3",
    "httpx",
    "filelock",
    "torch",
    "matplotlib",
]:
    logging.getLogger(name).setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR, format="%(message)s")

from pyspark.sql import SparkSession, functions as F, types as T

# --------- Spark session ----------
spark = (
    SparkSession.builder.appName("YelpABSA_PyABSA_Sample")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.ui.showConsoleProgress", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

print("\n========== Loading ABSA input (sample from parquet/absa_input/sample) ==========")

SAMPLE_INPUT = "parquet/absa_input/sample"
df = spark.read.parquet(SAMPLE_INPUT)

# Keep only columns we need (robust to schema)
wanted_cols = [
    "business_id", "user_id", "text", "city", "state", "categories",
    "date", "review_stars", "average_stars", "biz_stars", "year"
]
existing = [c for c in wanted_cols if c in df.columns]
df = df.select(*existing)

n_rows = df.count()
print(f"✅ Loaded rows: {n_rows:,}")
df.show(3, truncate=100)

# ---------------- Aspect category mapping (very light heuristic) ----------------
FOOD_WORDS = {
    "food","pizza","burger","taco","sushi","noodle","pho","ramen","steak","salad","pasta",
    "sandwich","fries","rice","sauce","salsa","dessert","cake","bread","coffee","tea",
    "beer","wine","fish","chicken","beef","pork","tofu","veggie","vegan","vegetarian"
}
SERVICE_WORDS = {
    "service","waiter","server","waitress","host","hostess","manager","staff",
    "rude","friendly","attentive","slow","fast","polite","ignore","manager"
}
PRICE_WORDS = {
    "price","priced","expensive","cheap","affordable","value","worth","overpriced",
    "deal","cost","costly","pricing"
}
AMBIENCE_WORDS = {
    "ambience","ambiance","atmosphere","vibe","music","decor","noise","noisy","quiet",
    "patio","outdoor","view","seating","crowded","romantic","cozy","clean","dirty"
}

def aspect_bucket(term: str) -> str:
    t = term.lower()
    if t in FOOD_WORDS: return "food"
    if t in SERVICE_WORDS: return "service"
    if t in PRICE_WORDS: return "price"
    if t in AMBIENCE_WORDS: return "amb"
    return ""  # unknown -> ignored for scoring

# ---------- Per-partition scorer (lazy model init per executor) ----------
# Keep module-level globals for lazy reuse inside executors
_EXTRACTOR = None

def _get_extractor():
    """Lazy-load the PyABSA ATEPC extractor on executors."""
    global _EXTRACTOR
    if _EXTRACTOR is not None:
        return _EXTRACTOR

    # Import here so driver import cost stays low
    from pyabsa import ATEPCCheckpointManager

    # Prefer local English checkpoint if it exists; otherwise use the name ("english")
    # Your logs showed an English ATEPC checkpoint path like below:
    local_ckpt = "checkpoints/ATEPC_ENGLISH_CHECKPOINT/fast_lcf_atepc_English_cdw_apcacc_82.36_apcf1_81.89_atef1_75.43"
    checkpoint = local_ckpt if os.path.exists(local_ckpt) else "english"

    # Quiet-ish settings: don't force download, don't print results
    _EXTRACTOR = ATEPCCheckpointManager.get_aspect_extractor(
        checkpoint=checkpoint,
        auto_device=True,            # CPU/MPS on M-series
        calcluate_sentiment=True,    # PyABSA's flag (spelling from lib)
        # Do NOT set force_download here to avoid network delays
    )
    return _EXTRACTOR

def _score_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Run ATEPC on a batch of texts, return list of dicts with extracted aspects + sentiments.
    """
    extractor = _get_extractor()
    # PyABSA API: suppress saving/printing; return in-memory results
    res = extractor.extract_aspect(
        inference_source=texts,
        pred_sentiment=True,
        print_result=False,
        save_result=False
    )
    # Each item generally contains:
    # {
    #   'sentence': ...,
    #   'aspect': ['pizza', 'service', ...],
    #   'sentiment': ['Positive','Negative',...],
    #   'confidence': [...],
    #   ...
    # }
    return res

def _score_partition(rows: Iterator[Dict[str, Any]], batch_size: int = 16) -> Iterator[Dict[str, Any]]:
    """
    Spark mapPartitions: score texts in batches; yield dict rows with aspect scores & lists.
    Prints checkpoint messages every 1000 processed rows (from executors; lightweight).
    """
    buffer = []
    out_rows = []
    processed = 0

    def flush_buffer():
        nonlocal out_rows, processed
        if not buffer:
            return

        texts = [r["text"] for r in buffer]
        results = _score_batch(texts)

        for src, r in zip(buffer, results):
            # Aggregate to our 4 coarse aspects by counting +/- (simple numeric scoring)
            food_score = service_score = price_score = amb_score = 0.0
            food_terms_pos, service_terms_pos, price_terms_pos, amb_terms_pos = [], [], [], []

            aspects = r.get("aspect", []) or []
            sentiments = r.get("sentiment", []) or []

            for term, pol in zip(aspects, sentiments):
                bucket = aspect_bucket(term)
                if not bucket:
                    continue
                val = 1.0 if str(pol).lower().startswith("pos") else (-1.0 if str(pol).lower().startswith("neg") else 0.0)
                if bucket == "food":
                    food_score += val
                    if val > 0: food_terms_pos.append(term)
                elif bucket == "service":
                    service_score += val
                    if val > 0: service_terms_pos.append(term)
                elif bucket == "price":
                    price_score += val
                    if val > 0: price_terms_pos.append(term)
                elif bucket == "amb":
                    amb_score += val
                    if val > 0: amb_terms_pos.append(term)

            out = dict(src)  # keep original columns we passed in
            out.update({
                "aspect_food": float(food_score) if food_score else 0.0,
                "aspect_service": float(service_score) if service_score else 0.0,
                "aspect_price": float(price_score) if price_score else 0.0,
                "aspect_amb": float(amb_score) if amb_score else 0.0,
                "food_pos_terms": food_terms_pos,
                "service_pos_terms": service_terms_pos,
                "price_pos_terms": price_terms_pos,
                "amb_pos_terms": amb_terms_pos,
            })
            out_rows.append(out)

        buffer.clear()

    for row in rows:
        # Pull only fields we keep to minimize serialization
        item = {
            "business_id": row.get("business_id"),
            "user_id": row.get("user_id"),
            "city": row.get("city"),
            "state": row.get("state"),
            "categories": row.get("categories"),
            "date": row.get("date"),
            "review_stars": float(row.get("review_stars")) if row.get("review_stars") is not None else None,
            "average_stars": float(row.get("average_stars")) if row.get("average_stars") is not None else None,
            "biz_stars": float(row.get("biz_stars")) if row.get("biz_stars") is not None else None,
            "year": row.get("year"),
            "text": row.get("text") or "",
        }
        buffer.append(item)
        processed += 1

        if len(buffer) >= batch_size:
            flush_buffer()

        # Light progress signals every 1000 *partition* rows
        if processed % 1000 == 0:
            print(f"[PyABSA partition] processed {processed:,} rows ...")

        # Avoid unbounded memory in out_rows
        # Yield in chunks of 1024
        if len(out_rows) >= 1024:
            for o in out_rows:
                yield o
            out_rows = []

    # Final flush
    flush_buffer()
    for o in out_rows:
        yield o

# ---------- Apply scorer ----------
print("\n========== Scoring aspects with PyABSA (quiet) ==========")

# Convert to RDD of Python dicts for mapPartitions (fast & memory conscious)
rdd = df.select(*existing).rdd.map(lambda r: r.asDict(recursive=True))
scored_rdd = rdd.mapPartitions(lambda it: _score_partition(it, batch_size=16))

# Define output schema to build back a DataFrame
schema = T.StructType([
    T.StructField("business_id", T.StringType(), True),
    T.StructField("user_id", T.StringType(), True),
    T.StructField("city", T.StringType(), True),
    T.StructField("state", T.StringType(), True),
    T.StructField("categories", T.StringType(), True),
    T.StructField("date", T.StringType(), True),
    T.StructField("review_stars", T.DoubleType(), True),
    T.StructField("average_stars", T.DoubleType(), True),
    T.StructField("biz_stars", T.DoubleType(), True),
    T.StructField("year", T.IntegerType(), True),
    T.StructField("text", T.StringType(), True),

    T.StructField("aspect_food", T.DoubleType(), True),
    T.StructField("aspect_service", T.DoubleType(), True),
    T.StructField("aspect_price", T.DoubleType(), True),
    T.StructField("aspect_amb", T.DoubleType(), True),

    T.StructField("food_pos_terms", T.ArrayType(T.StringType(), True), True),
    T.StructField("service_pos_terms", T.ArrayType(T.StringType(), True), True),
    T.StructField("price_pos_terms", T.ArrayType(T.StringType(), True), True),
    T.StructField("amb_pos_terms", T.ArrayType(T.StringType(), True), True),
])

scored_df = spark.createDataFrame(scored_rdd, schema=schema)

# Small sanity peek
scored_count = scored_df.count()
print(f"✅ Finished PyABSA scoring. Rows: {scored_count:,}")
scored_df.select(
    "business_id","user_id","aspect_food","aspect_service","aspect_price","aspect_amb",
    "food_pos_terms","service_pos_terms","price_pos_terms","amb_pos_terms"
).show(5, truncate=False)

# ---------- Write to a *separate* output tree (keeps baseline safe) ----------
OUT_DIR = "parquet/absa_scored_pyabsa/sample"
print(f"\n💾 Writing scored sample to {OUT_DIR}")
(
    scored_df
    .repartition(16)  # reasonable number for local runs; adjust on cluster
    .write.mode("overwrite").parquet(OUT_DIR)
)
print("✅ Done.\n")

spark.stop()
