# yelp_absa_scoring_pyabsa.py
# Use PyABSA (ATEPC) to extract aspects + sentiments from each review,
# map aspect terms to {food, service, price, amb}, and aggregate to scores in [-1, 1].
# at the very top of yelp_absa_scoring_pyabsa.py
import os
import time
from typing import Dict, List, Tuple

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import (
    StructType, StructField, DoubleType, ArrayType, StringType
)

# ----------------------------
# Paths (relative to algorithm-prototype/)
# ----------------------------
ROOT = os.path.dirname(__file__)
INPUT_DIR = os.path.join(ROOT, "parquet", "absa_input", "sample")
OUTPUT_DIR = os.path.join(ROOT, "parquet", "absa_scored", "sample")

# ----------------------------
# Lazy model loading (per Python worker)
# ----------------------------
_ASPECT_EXTRACTOR = None

def _load_model():
    """
    Load the ATEPC (Aspect Term Extraction + Polarity Classification) model once per worker.
    """
    global _ASPECT_EXTRACTOR
    if _ASPECT_EXTRACTOR is None:
        # Import inside function so Spark workers can lazy-load
        from pyabsa import ATEPCCheckpointManager

        # You can pick a checkpoint; 'multilingual' is robust.
        # Other options: 'english' or a specific checkpoint name if you've downloaded one.
        _ASPECT_EXTRACTOR = ATEPCCheckpointManager.get_aspect_extractor(
            checkpoint="multilingual",
            auto_device=True,   # CPU/GPU auto
            cal_perplexity=False
        )
    return _ASPECT_EXTRACTOR

# ----------------------------
# Category mapping (term → aspect)
# You can expand/adjust this list over time.
# ----------------------------
FOOD_TERMS = {
    "food", "pizza", "burger", "sushi", "taco", "noodle", "pasta", "steak", "salad",
    "dish", "menu", "taste", "flavor", "portion", "dessert", "sandwich", "sauce", "fries",
    "meal", "breakfast", "lunch", "dinner"
}
SERVICE_TERMS = {
    "staff", "server", "waiter", "waitress", "service", "manager", "host", "hostess",
    "attentive", "rude", "friendly", "slow", "quick", "bartender"
}
PRICE_TERMS = {
    "price", "expensive", "cheap", "cost", "pricy", "value", "deal", "worth", "overpriced",
    "reasonable"
}
AMBIENCE_TERMS = {
    "ambience", "ambiance", "noise", "noisy", "quiet", "vibe", "atmosphere",
    "music", "crowded", "decor", "seating", "clean", "dirty"
}

def _categorize_aspect_term(term: str) -> str:
    """
    Map an extracted aspect term to one of {food, service, price, amb} (or '' if unknown).
    """
    t = term.lower().strip()
    # simple normalization
    if t.endswith('s'):
        t = t[:-1]
    if t in FOOD_TERMS:
        return "food"
    if t in SERVICE_TERMS:
        return "service"
    if t in PRICE_TERMS:
        return "price"
    if t in AMBIENCE_TERMS:
        return "amb"
    # Heuristic: map some common food sub-terms indirectly
    if any(x in t for x in ["taste", "cook", "flavor"]):
        return "food"
    if any(x in t for x in ["wait", "serve"]):
        return "service"
    if "price" in t or "cost" in t or "value" in t:
        return "price"
    if any(x in t for x in ["noise", "ambien", "vibe", "atmos", "decor", "music"]):
        return "amb"
    return ""  # unknown / ignore

def _sentiment_to_score(label: str) -> float:
    """
    Map PyABSA sentiment label to a numeric score.
    """
    if not label:
        return 0.0
    L = label.lower()
    if "pos" in L:
        return 1.0
    if "neg" in L:
        return -1.0
    return 0.0  # neutral or unknown

def analyze_review_with_pyabsa(text: str) -> Tuple[float, float, float, float,
                                                   List[str], List[str], List[str], List[str]]:
    """
    Run ATEPC on a single review, bucket aspect terms into categories, and average sentiment per category.
    Returns:
      food_score, service_score, price_score, amb_score,
      food_terms, service_terms, price_terms, amb_terms
    """
    if text is None or not text.strip():
        return (0.0, 0.0, 0.0, 0.0, [], [], [], [])

    extractor = _load_model()
    try:
        # PyABSA expects a list of sentences. We pass a list with one review.
        result = extractor.extract_aspect(
            inference_source=[text],
            pred_sentiment=True,
            save_result=False,  # don't write result files
        )
    except Exception:
        # Fail-safe: if model errors on this text, return zeros
        return (0.0, 0.0, 0.0, 0.0, [], [], [], [])

    # PyABSA returns a list (one entry per input sentence/review)
    # Typical structure (simplified):
    # [{'sentence': '...', 'aspect': ['pizza','staff'], 'sentiment': ['Positive','Negative']}]
    aspects = []
    sentiments = []
    if isinstance(result, list) and len(result) > 0:
        entry = result[0]
        aspects = entry.get("aspect", []) or entry.get("aspects", []) or []
        sentiments = entry.get("sentiment", []) or entry.get("sentiments", []) or []

    # Accumulate sums per category
    sums = {"food": 0.0, "service": 0.0, "price": 0.0, "amb": 0.0}
    cnts = {"food": 0, "service": 0, "price": 0, "amb": 0}
    terms = {"food": [], "service": [], "price": [], "amb": []}

    for term, lab in zip(aspects, sentiments):
        cat = _categorize_aspect_term(str(term))
        if not cat:
            continue
        score = _sentiment_to_score(str(lab))
        sums[cat] += score
        cnts[cat] += 1
        terms[cat].append(str(term))

    def avg(cat):
        return (sums[cat] / cnts[cat]) if cnts[cat] > 0 else 0.0

    return (
        float(avg("food")),
        float(avg("service")),
        float(avg("price")),
        float(avg("amb")),
        terms["food"],
        terms["service"],
        terms["price"],
        terms["amb"],
    )

# Spark UDF schema
RET_SCHEMA = StructType([
    StructField("aspect_food", DoubleType(), True),
    StructField("aspect_service", DoubleType(), True),
    StructField("aspect_price", DoubleType(), True),
    StructField("aspect_amb", DoubleType(), True),
    StructField("food_terms", ArrayType(StringType()), True),
    StructField("service_terms", ArrayType(StringType()), True),
    StructField("price_terms", ArrayType(StringType()), True),
    StructField("amb_terms", ArrayType(StringType()), True),
])

@udf(returnType=RET_SCHEMA)
def pyabsa_udf(text: str):
    return analyze_review_with_pyabsa(text)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("Yelp-ABSA-Scoring-PyABSA")
        # You can tweak local parallelism if needed, e.g.: .master("local[4]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("\n========== Loading ABSA input (sample) ==========")
    t0 = time.time()
    df = spark.read.parquet(INPUT_DIR)
    n = df.count()
    print(f"✅ Loaded rows: {n} in {time.time() - t0:.2f}s")
    df.select("business_id","user_id","text","city","state","review_stars","year").show(3, truncate=100)

    print("\n========== Scoring aspects with PyABSA (ATEPC) ==========")
    t1 = time.time()
    scored = df.select(
        "*",
        pyabsa_udf(col("text")).alias("absa")
    ).select(
        "business_id","user_id","city","state","categories","date",
        "review_stars","average_stars","biz_stars",
        col("absa.aspect_food").alias("aspect_food"),
        col("absa.aspect_service").alias("aspect_service"),
        col("absa.aspect_price").alias("aspect_price"),
        col("absa.aspect_amb").alias("aspect_amb"),
        col("absa.food_terms").alias("food_pos_terms"),      # keep name consistent with your baseline columns
        col("absa.service_terms").alias("service_pos_terms"),
        col("absa.price_terms").alias("price_pos_terms"),
        col("absa.amb_terms").alias("amb_pos_terms"),
    )

    # Show a few
    scored.show(3, truncate=100)
    print(f"✅ Finished PyABSA scoring in {time.time() - t1:.2f}s")

    print(f"\n💾 Writing scored sample to {OUTPUT_DIR}")
    scored.write.mode("overwrite").parquet(OUTPUT_DIR)
    print("✅ Done.\n")
