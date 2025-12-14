# start StreamLit app
From visual-analytics-project directory:
- cd algorithm-prototype

On MacOS: 
- chmod +x run_streamlit.sh
- ./run_streamlit.sh

On Windows:
- run_streamlit.bat

# Phase I: Data Pre-processing (ETL)
# Step 1: Load + Convert to Parquet + Partition if needed
    yelp_load_all_to_parquet.py
    - Loads all raw Yelp JSON files (business, review, user, checkin, tip) from ../yelp_dataset/.
    - Converts them into columnar Parquet format for Spark.
    - Each major entity (business, review, etc.) is written under parquet/yelp_user/.
    - reviews/, tips/, user/ - partitioned by yelping_year (derived from the review date). 
    - business/, checkin/ - unpartitioned, size-based chunks that Spark creates.
# Step 2: Domain Selection + Initial Data Cleaning (textual or semantic cleaning)
    yelp_filter_for_absa.py
    - Reads Parquet datasets from parquet/yelp_user/; 
    - Domain selection 
        - (business): keep only business rows whose categories indicate restaurants. 
        - (business + review) INNER JOIN on restaurants (because only reviews with a matching restaurant business survive)
        - (user) LEFT JOIN users on user_id to attach reviewer info (review_count, average_stars).
    - Initial Cleaning: Drop empty/null text fields. Remove irrelevant columns. 
    - (Optionally) Trim review text + collapse whitespace, parse dates → timestamp + year, filter out empty/ultra-short reviews and rows missing critical keys, normalize categories (lower-case, split on comma, trim each token, deduplicate, rejoin.)
    - Writes filtered set to parquet/filtered/absa_ready/.
# Step 3: Integration + Schema Cleaning (structural cleaning: tables are merged properly, names are consistent, and the schema is clean)
    yelp_join_absa_ready.py
    - Loads parquet/filtered/absa_ready/ and parquet/yelp_user/business.
    - Integration
        - Re-joins with business to enrich with business-level fields (e.g., categories, latitude, longitude, biz_stars, biz_review_count)
    - Schema Cleaning:
        - Resolves ambiguous columns (e.g., renames review stars to review_stars: review_stars (review-level rating), biz_stars (business average), biz_review_count (business-level count)).
        - Ensure types (e.g., date as timestamp, numeric columns as double/long).
        - Attach canonical columns used later by ABSA and clustering.
        - (Optionally) drop rows missing critical fields (text, business_id, date), which avoids runtime errors downstream.
        - stable final schema: 
            - ['business_id','name','city','state','categories','review_stars','user_review_count','average_stars','text','date','user_id','latitude','longitude','biz_stars','biz_review_count']
    - Writes "flattened ABSA table" to parquet/joined/absa_flat/ - a final, denormalized dataset that merges all relevant information from multiple source tables into a single, row-based structure suitable for model training or analysis.
    - Prints sanity samples + final column list.
# Phase II: Aspect Extraction (only needed for Baseline Aspect Scoring)
    yelp_prepare_absa_corpus.py
    - Loads your flattened joined dataset from parquet/joined/absa_flat/.
    - Randomly samples 110,000 reviews from the full ~5.26M restaurant reviews.
    - Selects lightweight columns:"business_id", "name", "city", "text", "review_stars", "average_stars".
    - Writes that sample to: → parquet/absa_input/sample/
        - Output Ex for next phase: biz001 -> Bella Pizza -> “Pizza was amazing but staff rude.” -> [review_stars] 5
    - Partitions the full input by year for scalable processing → parquet/absa_input/full_by_year/.

    yelp_quickcheck_absa_input.py
    - Loads parquet/absa_input/sample/ (or full_by_year) for a quick sanity check.
    - Prints row count, head, and key columns (business_id, user_id, text, city, state, review_stars, year).
    - Zero side effects—no writes.
# Aspect Scoring (Baseline ABSA, rule-based, no ML) - Assign sentiment scores per aspect category.
- Uses a simple keyword-based heuristic 
    — counts positive vs. negative words from small dictionaries (or averages review star ratings).
- Extracts aspects (food, service, price, ambience) using keyword/lexicon matching
    - then assigns simple polarity scores per review.
- output - Restaurant-level average scores for food, service, price, amb

    yelp_absa_scoring_baseline.py
    - Loads the 110k sample review from (parquet/absa_input/sample/).
        - Input Ex: biz001 -> Bella Pizza -> “Pizza was amazing but staff rude.” -> [review_stars] 5
    - Tokenizes text and searches for aspect-specific keywords:
        - food: pizza, dish, meal, taste
        - service: waiter, staff, friendly
        - price: cheap, expensive, value
        - ambience: atmosphere, decor, music
    - Runs a lightweight lexicon baseline to produce per-review aspect scores:
        - Uses small sentiment word lists (e.g., “good”, “amazing” = +1; “bad”, “slow” = –1).
        - Columns: aspect_food, aspect_service, aspect_price, aspect_amb, plus lists of matched positive/negative terms per aspect: 
        - { business_id: biz001, name: "Bella Pizza", aspect_food: +0.9, aspect_service: -0.7, aspect_price:0.0, aspect_amb: 0.0 }
    - Writes scored sample to parquet/absa_scored/sample/.
        - Each row = one review, with per-aspect sentiment scores.
    - Designed to be swappable later with a real ABSA model.
# Phase III: Aspect Scoring (PyABSA) - Skips explicit aspect extraction — it reads raw text and uses a transformer model (e.g., DeBERTa-v3-base) to infer both aspects and sentiments directly from language context.
- Hugging Face Hub/Transformers - Provides pretrained deep-learning models (e.g., BERT, DeBERTa)
- Uses a real pretrained deep-learning model (PyABSA’s FAST_LCF_ATEPC) that performs true aspect extraction + sentiment classification using transformers.
- When choose English - uses the Hugging Face model microsoft/deberta-v3-base - This is a transformer architecture by Microsoft for sentence-level and token-level NLP tasksn - Deep transformer, best accuracy for English ABSA
- Other languages use other models (e.g. BERT, bert-base-multilingual-uncased for multilingual)
- Output - Sentence-level aspect terms with individual confidence scores
# Step 1: Environment Check - Download Hugging Face models for Offline processing
- Run these commands first: 
    > python3.10 -m pip install --upgrade pip setuptools wheel
    > python3.10 -m pip install \
    "pyabsa==2.4.3" \
    "torch>=2.1" \
    "transformers>=4.41" \
    "spacy>=3.7" \
    "huggingface_hub==0.35.3" \
    "fsspec==2024.9.0" \
    "filelock>=3.12"
    > python3.10 -m spacy download en_core_web_sm
    > python3.10 - <<'PY'
    from pyabsa import ATEPCCheckpointManager
    extractor = ATEPCCheckpointManager.get_aspect_extractor(
        checkpoint='multilingual',
        force_download=True,
        auto_device=True
    )
    print("PyABSA multilingual ATEPC checkpoint ready.")
    PY
    > python3.10 - <<'PY'
    from pyabsa import ATEPCCheckpointManager
    print("⬇Forcing re-download of ATEPC 'english' checkpoint ...")
    extractor = ATEPCCheckpointManager.get_aspect_extractor(
        checkpoint='english',
        force_download=True,
        auto_device=True
    )
    print("PyABSA english ATEPC checkpoint ready.")
    PY
    > python3.10 - <<'PY'
    from transformers import AutoTokenizer, AutoModel
    m = "microsoft/mdeberta-v3-base"
    AutoTokenizer.from_pretrained(m)
    AutoModel.from_pretrained(m)
    print("Hugging Face DeBERTa-v3-base cached.")
    PY
    > export HF_HUB_OFFLINE=1
    > export TRANSFORMERS_OFFLINE=1
    > export HF_HUB_DISABLE_PROGRESS_BARS=1 # (optional) quiet down HF/urllib3 logging
    - Output - Sentence-level aspect terms with individual confidence scores
# Step 2: Use 110k random sample data from output of yelp_prepare_absa_corpus.py 
    (FINAL PHASE) yelp_absa_scoring_pyabsa.py
    - Input: reads exactly what yelp_prepare_absa_corpus.py wrote to parquet/absa_input/sample/ (your ~110k sample).
    - Output: writes to parquet/absa_scored_pyabsa/sample/ to keep all your baseline outputs intact.
    - Speed: batches of 16 per ATEPC call and one lazy model load per executor/partition (fastest you’ll get locally without GPUs).
    - Heuristic aspect mapping: PyABSA returns open-ended aspect terms; this file keeps your 4-aspect schema by mapping common terms into {food, service, price, ambience}. You can expand the word lists if you like.
# Step 3: Recommended: Scale up to 1~3 million reviews at a time
    (FINAL PHASE) yelp_absa_scoring_pyabsa.py
    - Loads pre-filtered review text (from parquet/filtered/absa_ready/), already joined with business info via yelp_join_absa_ready.py
    - Uses the pretrained PyABSA transformer model (FAST_LCF_ATEPC) to automatically extract aspect terms and assign sentiment polarity (positive/neutral/negative) with confidence scores.
    - Example:
        - Input review: "The pizza was amazing, but the staff was rude."
        - Model output: {food: +0.98, service: –0.95}
    - Outputs:
        - Detailed per-review aspect results saved to Aspect Term Extraction and Polarity Classification.FAST_LCF_ATEPC.result.json
    - Purpose: Replaces the rule-based baseline with a real ML-driven ABSA pipeline that performs both aspect extraction and sentiment scoring in one step.

# Creditability Weighting + Aspect Aggregation
    yelp_compute_reviewer_weights.py (Part I: Credibility weights)
    - (changeable) Reads the baseline-scored reviews from parquet/absa_scored/sample/.
    - Computes reviewer credibility: w_user = 1 / (1 + Var_user(review_stars)).
        - Each user’s variance represents how erratic their ratings are across restaurants.
            - Group reviews by reviewer based on user_id
            - For each user_id, Spark computes:
                - var_review_stars: population variance of all review_stars that user gave.
                - user_review_count_in_sample: number of reviews that user wrote in this dataset.
        - When variance = 0 → w=1/(1+0)=1.0 → fully credible.
        - When variance = 3 → w=1/(1+3)=0.25 → down-weighted.
    - Writes weights to parquet/weights/sample_reviewer_weights/

    yelp_aggregate_aspects_weighted.py (Part II: Weighted aggregation)
    - Reads the same baseline-scored reviews (parquet/absa_scored/sample/) and the weights from Part I (parquet/weights/sample_reviewer_weights/).
    - Joins w_user back to each review (defaults to 1.0 if missing).
    - Computes weighted means per business for food, service, price, amb, plus coverage fractions and counts.
    - Writes restaurant vectors to parquet/aggregated_weighted/sample_restaurant_vectors/.

    (OLD version) yelp_aggregate_aspects.py
    - Reads parquet/absa_scored/sample/ for Baseline or parquet/absa_scored_pyabsa/sample/ for PyABSA real model.
        - Each row = one review, with per-aspect sentiment scores.
    - Computes reviewer credibility weights (inverse variance of each reviewer’s ratings).
        - computes user-level credibility weights — a measure of how consistent and reliable a reviewer’s ratings
    - Aggregates review-level aspect scores → restaurant-level vectors: food, service, price, amb.
    - Adds diagnostics: n_reviews_sample, n_users_sample, cover_*_frac.
    - Filters to restaurants with ≥ N reviews (configurable).
    - Writes restaurant vectors to parquet/aggregated/sample_restaurant_vectors/.

# K-mean Clustering
yelp_cluster_kmeans.py
- Loads parquet/aggregated/sample_restaurant_vectors/.
- Builds standardized features (z-score of food, service, price, amb).
- Runs KMeans (k=6 by default), prints silhouette and standardized centers.
- Joins cluster labels back to metadata (name, city, stars, review_count).
- Writes clustered dataset to parquet/clustering/sample_kmeans_k6/ and prints cluster counts.

# Interactive UI
yelp_export_clusters_for_ui.py
- Loads parquet/clustering/sample_kmeans_k6/.
- Selects UI-friendly columns (ids, name, city/state, categories, 4 aspects, cluster, counts/coverage).
- Exports a single CSV for the UI here: ui/sample_kmeans_k6_for_ui_csv/restaurants_clusters_k6.csv
- Keeps UI exports separate from Parquet pipeline outputs.

yelp_export_for_tableau_reweighting.py
- Loads parquet/clustering/sample_kmeans_k6/
- Computes global z-scores for food, service, price, amb → food_z, service_z, price_z, amb_z.
- Exports a Tableau/Streamlit-friendly CSV with z-scored features: 
    - ui/tableau_reweighting_input/restaurant_vectors_z.csv

app_streamlit.py
- Load the CSV or Parquet cluster output (like ui/sample_kmeans_k6_for_ui_csv/restaurants_clusters_k6.csv)
- View interactive sliders for “food / service / ambiance / price”
- Filter restaurants by cluster or city
- Show example reviews and metrics dynamically

# UI team 
- Frontend clones repo
- run sample streamlit app
- review SCHEMA.md under ui/data (documentation of the data’s structure)
    - Defines exactly what columns exist in your dataset (names, types, meaning)
    - Ensures consistent understanding of what each field represents and how it’s used
- use ui/data/make_sample.py script to generate small samples

# SCHEMA.md (the contract between the backkend & frontend)
- Basic Description 
    - define where the dataset lives: ui/data/restaurants_clusters_k6.csv
    - what each row represents: One row = one restaurant
- Table Schema - The UI team uses this section to build controls like sliders, filters, or legends.
    - Each column is documented with:
        - Column name (exact)
        - Type (string, float, int)
        - Expected range or example value
        - Meaning / purpose
- Data Dictionary section gives practical examples and guidance - describe how to use the data in an interactive app.
    - What each column is used for visually
    - How to interpret ranges
    - Notes for tooltips, charts, filters, etc.
- Main points
    - Each row = one restaurant (not one review)
    - The key numeric columns (food, service, price, amb) are normalized scores
    - The column cluster is categorical, not numeric ranking
    - The data is already aggregated — they don’t need to run Spark or ABSA
