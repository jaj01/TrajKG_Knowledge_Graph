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

# ----------------- Google Drive File IDs ----------------------
EMBEDDING_FILE_ID = "1Jnf8Yk_T0S01nUrIeDwx87s1AZCOSnEz" 
CSV_FILE_ID = "1b0bStdF_PyJHiq9Ss1mw_XIKUre31fRg" 
NAME_FILE_ID = "1RcgTHYXm7vqJdgLkaohDGee-3tlLb-lI"

# ----------------- Download Files If Not Present ----------------------
if not os.path.exists("fused_embedding.pkl"):
    gdown.download(f"https://drive.google.com/uc?id={EMBEDDING_FILE_ID}", "fused_embedding.pkl", quiet=False)

if not os.path.exists("dataset_TSMC2014_NYC.csv"):
    gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", "dataset_TSMC2014_NYC.csv", quiet=False)

if not os.path.exists("poi_names.csv"):
    gdown.download(f"https://drive.google.com/uc?id={NAME_FILE_ID}", "poi_names.csv", quiet=False)

# ----------------- Load Files ----------------------
fused_embedding = pickle.load(open("fused_embedding.pkl", "rb"))
metadata_df = pd.read_csv("dataset_TSMC2014_NYC.csv")
metadata_df = metadata_df[['venueId', 'venueCategory', 'latitude', 'longitude']]
metadata_df.columns = ['poi_id', 'category', 'lat', 'lon']
metadata = metadata_df.drop_duplicates('poi_id').set_index('poi_id').to_dict(orient='index')

# Load POI names
name_df = pd.read_csv("poi_names.csv")
id_to_name = dict(zip(name_df['venueId'], name_df['venueName']))
name_to_id = {v: k for k, v in id_to_name.items()}
valid_names = {pid: id_to_name[pid] for pid in fused_embedding if pid in id_to_name}
valid_name_to_id = {v: k for k, v in valid_names.items()}

# ----------------- Sidebar Settings ----------------------
st.sidebar.title("⚙️ Settings")
tourist_mode = st.sidebar.checkbox("🧳 Tourist Mode", value=True)
use_custom_location = st.sidebar.checkbox("📍 Use My Location")

user_lat, user_lon = None, None
if use_custom_location:
    user_lat = st.sidebar.number_input("Latitude", value=40.758, format="%.6f")
    user_lon = st.sidebar.number_input("Longitude", value=-73.985, format="%.6f")

top_k = st.sidebar.slider("🔢 # POI Recommendations", 1, 10, 5)
tourist_k = st.sidebar.slider("🧳 # Tourist Spots", 1, 10, 5)

# ----------------- UI - POI Selection ----------------------
st.title("🧭 Explainable POI Recommender")
selected_name = st.selectbox("Select a POI", list(valid_name_to_id.keys()))
selected_poi = valid_name_to_id[selected_name]

# ----------------- Similar POI Recommendations ----------------------
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
            "name": id_to_name.get(poi_id, poi_id),
            "category": info.get('category', 'Unknown'),
            "lat": info.get('lat'),
            "lon": info.get('lon'),
            "score": round(sims[idx], 3),
            "reason": ", ".join(reason) if reason else "embedding similarity"
        })
    return results

poi_recs = recommend_similar_pois(selected_poi, top_k)

st.subheader("📍 Similar POI Recommendations")
for rec in poi_recs:
    st.markdown(f"• **{rec['name']}** ({rec['category']}) — Score: {rec['score']} — _Because {rec['reason']}_")
    maps_url = f"https://www.google.com/maps/dir/?api=1&origin={metadata[selected_poi]['lat']},{metadata[selected_poi]['lon']}&destination={rec['lat']},{rec['lon']}&travelmode=walking"
    st.markdown(f"[🗺 Directions via Google Maps]({maps_url})")

# ----------------- Tourist Recommendations ----------------------
tourist_keywords = ["tourist", "museum", "park", "gallery", "zoo", "monument", "memorial", "castle", "temple"]

def get_tourist_spots(top_n=5, user_lat=None, user_lon=None, radius_km=5):
    filtered = metadata_df.copy()
    filtered['lower_cat'] = filtered['category'].str.lower()
    is_tourist = filtered['lower_cat'].apply(lambda x: any(word in x for word in tourist_keywords))
    tourist_df = filtered[is_tourist].drop_duplicates('poi_id')
    tourist_df = tourist_df[tourist_df['poi_id'].isin(fused_embedding)]

    if user_lat is not None and user_lon is not None:
        tourist_df['distance'] = tourist_df.apply(
            lambda row: geodesic((user_lat, user_lon), (row['lat'], row['lon'])).km, axis=1
        )
        tourist_df = tourist_df[tourist_df['distance'] <= radius_km]
        tourist_df = tourist_df.sort_values('distance')

    return [
        {
            "poi_id": row['poi_id'],
            "name": id_to_name.get(row['poi_id'], row['poi_id']),
            "category": row['category'],
            "lat": row['lat'],
            "lon": row['lon'],
            "reason": "Tourist spot" + (f" ({round(row.get('distance', 0), 2)} km away)" if 'distance' in row else "")
        }
        for _, row in tourist_df.head(top_n).iterrows()
    ]

if tourist_mode:
    st.subheader("🧳 Tourist Recommendations")
    tourist_recs = get_tourist_spots(top_n=tourist_k, user_lat=user_lat, user_lon=user_lon)
    if not tourist_recs:
        st.info("No tourist attractions found near your location.")
    for rec in tourist_recs:
        st.markdown(f"• **{rec['name']}** — {rec['category']} (_{rec['reason']}_)")
        url = f"https://www.google.com/maps/search/?api=1&query={rec['lat']},{rec['lon']}"
        st.markdown(f"[📍 View on Map]({url})")

# ----------------- Map View ----------------------
st.subheader("🗺 Map View")
map_center = [metadata[selected_poi]['lat'], metadata[selected_poi]['lon']]
m = folium.Map(location=map_center, zoom_start=15)
selected_name_display = id_to_name.get(selected_poi, selected_poi)
folium.Marker(
    location=map_center,
    popup=f"Selected: {selected_name_display}",
    icon=folium.Icon(color='blue')
).add_to(m)

for rec in poi_recs:
    folium.Marker(
        location=[rec['lat'], rec['lon']],
        popup=f"{rec['name']} — {rec['reason']}",
        icon=folium.Icon(color='green')
    ).add_to(m)

if tourist_mode:
    for rec in tourist_recs:
        folium.Marker(
            location=[rec['lat'], rec['lon']],
            popup=f"{rec['name']} — {rec['category']}",
            icon=folium.Icon(color='orange', icon="info-sign")
        ).add_to(m)

st_folium(m, width=700, height=500)
