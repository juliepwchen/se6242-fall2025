# app_streamlit.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Aspect-weighted Yelp Clusters", layout="wide")

ROOT = os.path.dirname(__file__)
CSV_Z = os.path.join(ROOT, "ui", "tableau_reweighting_input", "restaurant_vectors_z.csv")
CSV_K = os.path.join(ROOT, "ui", "sample_kmeans_k6_for_ui_csv", "restaurants_clusters_k6.csv")

st.title("Criteria-Weighted Clustering (Yelp)")
st.caption("Interactive UI over your ABSA + Aggregation + KMeans pipeline")

# -------------------------
# Load data (prefer z-scored CSV)
# -------------------------
if os.path.exists(CSV_Z):
    df = pd.read_csv(CSV_Z)
    has_z = True
else:
    df = pd.read_csv(CSV_K)
    has_z = False

# Ensure needed columns exist
base_cols = ["business_id","name","city","state","categories","food","service","price","amb"]
for c in base_cols:
    if c not in df.columns:
        st.error(f"Missing column in CSV: {c}")
        st.stop()

# If we don’t have z columns, compute them
if not has_z:
    for col in ["food","service","price","amb"]:
        m, s = df[col].mean(), df[col].std(ddof=0)
        df[f"{col}_z"] = (df[col] - m) / (s if s != 0 else 1.0)
else:
    # Ensure z columns present
    z_needed = ["food_z","service_z","price_z","amb_z"]
    if not all(z in df.columns for z in z_needed):
        st.error("Z-scored columns not found. Regenerate with yelp_export_for_tableau_reweighting.py")
        st.stop()

# Original cluster (if present) for comparison
has_orig_cluster = "cluster" in df.columns

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Weights (sliders)")
wf = st.sidebar.slider("Food weight",    0.0, 3.0, 1.0, 0.1)
ws = st.sidebar.slider("Service weight", 0.0, 3.0, 1.0, 0.1)
wp = st.sidebar.slider("Price weight",   0.0, 3.0, 1.0, 0.1)
wa = st.sidebar.slider("Ambience weight",0.0, 3.0, 1.0, 0.1)

st.sidebar.header("Filters")
city_opts = ["(All)"] + sorted(df["city"].dropna().unique().tolist())[:5000]
city_sel = st.sidebar.selectbox("City", city_opts, index=0)
min_reviews = st.sidebar.number_input("Min reviews (sample)", min_value=0, value=int(df.get("n_reviews_sample", pd.Series([0])).median()) if "n_reviews_sample" in df else 0)

# Apply filters
mask = pd.Series(True, index=df.index)
if city_sel != "(All)":
    mask &= (df["city"] == city_sel)
if "n_reviews_sample" in df.columns:
    mask &= (df["n_reviews_sample"] >= min_reviews)

dff = df.loc[mask].copy()

# -------------------------
# Compute weighted assignment (nearest centroid)
# We derive centroids from current data’s original clusters (z-space),
# then compute weighted squared distance to assign new labels.
# -------------------------
# Build z-matrix
Z = dff[["food_z","service_z","price_z","amb_z"]].to_numpy()

# If we have original clusters, compute their centroids in z-space.
# If not, compute a quick KMeans with k=6 just to get centroids (optional).
if has_orig_cluster and dff["cluster"].nunique() >= 2:
    K = sorted(dff["cluster"].dropna().unique())
    centroids = []
    for k in K:
        Zk = dff.loc[dff["cluster"] == k, ["food_z","service_z","price_z","amb_z"]].to_numpy()
        if len(Zk) == 0:
            continue
        centroids.append((int(k), Zk.mean(axis=0)))
    # Fallback if something is empty
    if not centroids:
        # global mean as a single centroid
        centroids = [(0, Z.mean(axis=0))]
else:
    # No original clusters – make a single centroid at global mean
    centroids = [(0, Z.mean(axis=0))]

weights = np.array([wf, ws, wp, wa], dtype=float)

def assign_weighted(Z, centers, w):
    # weighted squared Euclidean distance in z-space
    # d^2 = sum_i (w_i * (x_i - c_i))^2
    labels = np.zeros(len(Z), dtype=int)
    if len(centers) == 1:
        labels[:] = centers[0][0]
        return labels
    C = np.vstack([c for _, c in centers])  # shape: (k, 4)
    ks = [k for k, _ in centers]
    for i, x in enumerate(Z):
        diffs = (w * (x - C)) ** 2
        dists = diffs.sum(axis=1)
        labels[i] = ks[int(np.argmin(dists))]
    return labels

dff["cluster_weighted"] = assign_weighted(Z, centroids, weights)

# -------------------------
# Plot
# -------------------------
st.subheader("Scatter: Weighted Food vs Service (color = weighted cluster)")
fig = px.scatter(
    dff,
    x="food_z", y="service_z",
    color="cluster_weighted",
    hover_data=["name","city","state","categories","food","service","price","amb","n_reviews_sample","review_count"] if "review_count" in dff.columns else ["name","city","state","categories","food","service","price","amb"],
    opacity=0.8
)
fig.update_layout(height=650)
st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top by Food (weighted view)")
    st.dataframe(
        dff.sort_values("food", ascending=False)[["name","city","food","service","price","amb","cluster_weighted"]].head(20)
    )
with col2:
    st.subheader("Top by Service (weighted view)")
    st.dataframe(
        dff.sort_values("service", ascending=False)[["name","city","service","food","price","amb","cluster_weighted"]].head(20)
    )

# Comparison histogram if original cluster exists
if has_orig_cluster:
    st.markdown("---")
    st.subheader("Cluster size comparison")
    c1 = dff["cluster"].value_counts().sort_index()
    c2 = dff["cluster_weighted"].value_counts().sort_index()
    comp = pd.DataFrame({"original": c1, "weighted": c2}).fillna(0).astype(int)
    st.bar_chart(comp)