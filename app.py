import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
import numpy as np
import os
import pickle
import gdown

# Google Drive file IDs (replace with your actual file IDs)
EMBEDDING_FILE_ID = "1Jnf8Yk_T0S01nUrIeDwx87s1AZCOSnEz"
CSV_FILE_ID = "1b0bStdF_PyJHiq9Ss1mw_XIKUre31fRg"

# Download files if not present
if not os.path.exists("fused_embedding.pkl"):
    st.text("Downloading fused_embedding.pkl...")
    gdown.download(f"https://drive.google.com/uc?id={EMBEDDING_FILE_ID}", "fused_embedding.pkl", quiet=False)

if not os.path.exists("dataset_TSMC2014_NYC.csv"):
    st.text("Downloading dataset_TSMC2014_NYC.csv...")
    gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", "dataset_TSMC2014_NYC.csv", quiet=False)

# Load data
fused_embedding = pickle.load(open("fused_embedding.pkl", "rb"))
metadata_df = pd.read_csv("dataset_TSMC2014_NYC.csv")
metadata_df = metadata_df[['venueId', 'venueCategory', 'latitude', 'longitude']]
metadata_df.columns = ['poi_id', 'category', 'lat', 'lon']
metadata = metadata_df.drop_duplicates('poi_id').set_index('poi_id').to_dict(orient='index')

# App UI
st.title("🧭 Explainable POI Recommender")
selected_poi = st.selectbox("Select a POI", list(fused_embedding.keys()))
top_k = st.slider("Number of recommendations", 1, 10, 5)

# Recommendation logic
def recommend_similar_pois(query_poi_id, top_k):
    query_vec = fused_embedding[query_poi_id].reshape(1, -1)
    poi_ids = list(fused_embedding.keys())
    vectors = np.array([fused_embedding[pid] for pid in poi_ids])
    sims = cosine_similarity(query_vec, vectors)[0]
    top_indices = sims.argsort()[::-1][1:top_k+1]

    results = []
    for idx in top_indices:
        poi_id = poi_ids[idx]
        info = metadata.get(poi_id, {})
        reason = []
        if metadata.get(query_poi_id, {}).get("category") == info.get("category"):
            reason.append("same category")
        try:
            dist = geodesic(
                (metadata[query_poi_id]['lat'], metadata[query_poi_id]['lon']),
                (info['lat'], info['lon'])
            ).km
            if dist < 1.0:
                reason.append("nearby")
        except:
            pass
        results.append({
            "poi_id": poi_id,
            "category": info.get('category', 'Unknown'),
            "lat": info.get('lat'),
            "lon": info.get('lon'),
            "score": round(sims[idx], 3),
            "reason": ", ".join(reason) if reason else "embedding similarity"
        })
    return results

# Run recommendations
recommendations = recommend_similar_pois(selected_poi, top_k)

# Show results
st.subheader("📍 Top Recommendations")
for rec in recommendations:
    st.markdown(f"- **{rec['category']}** (Score: {rec['score']}) — _Because {rec['reason']}_")
    maps_url = f"https://www.google.com/maps/dir/?api=1&origin={metadata[selected_poi]['lat']},{metadata[selected_poi]['lon']}&destination={rec['lat']},{rec['lon']}&travelmode=walking"
    st.markdown(f"[🗺 Directions via Google Maps]({maps_url})")

# Map view
st.subheader("🗺 Map View")
map_center = [metadata[selected_poi]['lat'], metadata[selected_poi]['lon']]
m = folium.Map(location=map_center, zoom_start=15)
folium.Marker(location=map_center, popup="You", icon=folium.Icon(color='blue')).add_to(m)

for rec in recommendations:
    folium.Marker(
        location=[rec['lat'], rec['lon']],
        popup=f"{rec['category']} — {rec['reason']}",
        icon=folium.Icon(color='green')
    ).add_to(m)

st_folium(m, width=700, height=500)
