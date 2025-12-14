# Restaurants Clusters – UI Data Schema

# where the dataset lives and what each row represents
**File:** `ui/data/restaurants_clusters_k6.csv`  
**Row grain:** One row = one restaurant (aggregated from many reviews)  
**Encoding:** UTF-8  
**Delimiter:** `,`

## Required columns (UI must rely on these)
| Column | Type | Range / Example | Meaning |
|---|---|---|---|
| `name` | string | `"Sabrina's Café"` | Display name of the restaurant. |
| `city` | string | `"Philadelphia"` | City for labeling / filtering. |
| `state` | string | `"PA"` | State/region code (US-style). |
| `food` | float | `0.0 – 1.0` (e.g., `0.59`) | Normalized aspect score for **food** (higher is better). |
| `service` | float | `0.0 – 1.0` (e.g., `0.43`) | Normalized aspect score for **service**. |
| `price` | float | `0.0 – 1.0` (e.g., `0.10`) | Normalized aspect score for **price/value**. |
| `amb` | float | `0.0 – 1.0` (e.g., `0.10`) | Normalized aspect score for **ambience**. |
| `cluster` | int | `0..5` (for k=6) | Cluster id assigned by k-means over standardized aspect features. |

## Recommended columns (nice for filters, tooltips)
| Column | Type | Example | Meaning |
|---|---|---|---|
| `business_id` | string | `"-6JdVK-DHB4_43PEksbg1A"` | Stable Yelp id (useful for cross-linking). |
| `n_reviews_sample` | int | `90` | #reviews from the sample used to compute this row. |
| `cover_food_frac` | float | `0.57` | Fraction of reviews with a food signal. |
| `cover_service_frac` | float | `0.59` | Fraction with a service signal. |
| `cover_price_frac` | float | `0.09` | Fraction with a price signal. |
| `cover_amb_frac` | float | `0.18` | Fraction with an ambience signal. |
| `latitude` | float | `39.9526` | Map support (optional). |
| `longitude` | float | `-75.1652` | Map support (optional). |

> **Contract:** Required columns and their names should remain stable. If you add fields, append new columns—don’t rename existing ones.

---

## Data Dictionary (examples)

| Column | Example value | Notes for UI |
|---|---|---|
| `name` | `“Royal House”` | Display as primary label. |
| `city` / `state` | `“New Orleans” / “LA”` | Use together for badges or filters. |
| `food` | `0.70` | Feed to sliders / radar charts; treat as 0–1. |
| `service` | `0.21` | Same. |
| `price` | `0.17` | “Value for money” proxy. |
| `amb` | `0.24` | Ambience/atmosphere. |
| `cluster` | `2` | Use as color/group; add legend with 6 clusters. |
| `n_reviews_sample` | `32` | Good for confidence hints (e.g., dim items with very low counts). |
| `cover_*_frac` | `0.47` | Show as coverage bars in tooltips (what % of reviews actually discussed that aspect). |
| `latitude` / `longitude` | `34.0522 / -118.2437` | Enable optional map view. |

### Ranges & interpretations
- All aspect scores are **normalized** to ~[0, 1] per restaurant. Higher = more positive sentiment for that aspect.
- `cluster` is a label; its numeric value has no ordinal meaning—use color categories.

