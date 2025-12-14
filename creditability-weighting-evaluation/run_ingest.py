from pathlib import Path
from scripts.review_loader import ReviewLoader

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent  # .../julie-creditability-weighting-evaluation
    # input json relative to repo root: visual-analytics-project/yelp_dataset/...
    review_json_path = base_dir.parent / "yelp_dataset" / "yelp_academic_dataset_review.json"

    loader = ReviewLoader()
    loader.ingest(review_json_path)
