from pathlib import Path
from scripts.monthly_weights import MonthlyWeightCalculator

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent  # .../julie-creditability-weighting-evaluation
    reviews_parquet = base_dir / "parquet" / "reviews"

    calc = MonthlyWeightCalculator()
    weights = calc.compute(reviews_parquet)
    print("🧮 Sample of computed monthly weights:")
    weights.show(5, truncate=False)
    calc.write(weights)

    # Optional peek
    weights.show(10, truncate=False)
