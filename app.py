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
EMBEDDING_FILE_ID = "1QewYv0pvlxdF8pYcjfSXMZnTEhAz09wq"
CSV_FILE_ID = "1b0bStdF_PyJHiq9Ss1mw_XIKUre31fRg"
NAME_FILE_ID = "1Wb7C8c-ZUvijet2q-Y82Cmb646_jWkZc"
FAMOUS_POI_FILE_ID = "1d3S0zbggg_viqwkls3CxBZe5Vks66T4O"

# ----------------- Download Files If Not Present ----------------------
if not os.path.exists("fused_embedding.pkl"):
    gdown.download(f"https://drive.google.com/uc?id=" + EMBEDDING_FILE_ID, "fused_embedding.pkl", quiet=False)

if not os.path.exists("dataset_TSMC2014_NYC.csv"):
    gdown.download(f"https://drive.google.com/uc?id=" + CSV_FILE_ID, "dataset_TSMC2014_NYC.csv", quiet=False)

if not os.path.exists("poi_names.csv"):
    gdown.download(f"https://drive.google.com/uc?id=" + NAME_FILE_ID, "poi_names.csv", quiet=False)

if not os.path.exists("poi_names_famous_nyc.csv"):
    gdown.download(f"https://drive.google.com/uc?id=" + FAMOUS_POI_FILE_ID, "poi_names_famous_nyc.csv", quiet=False)

# ----------------- Load Files ----------------------
fused_embedding = pickle.load(open("fused_embedding.pkl", "rb"))
metadata_df = pd.read_csv("dataset_TSMC2014_NYC.csv")[['venueId', 'venueCategory', 'latitude', 'longitude']]
metadata_df.columns = ['poi_id', 'category', 'lat', 'lon']
metadata = metadata_df.drop_duplicates('poi_id').set_index('poi_id').to_dict(orient='index')

# Load POI names
name_df = pd.read_csv("poi_names.csv")
id_to_name = dict(zip(name_df['venueId'], name_df['venueName']))

# Load famous POIs and enrich metadata
famous_ids = []
if os.path.exists("poi_names_famous_nyc.csv"):
    famous_df = pd.read_csv("poi_names_famous_nyc.csv")
    pid_col = [col for col in famous_df.columns if "venueId" in col or "poi_id" in col][0]
    name_col = [col for col in famous_df.columns if "venueName" in col or "name" in col][0]
    
    id_to_name.update(dict(zip(famous_df[pid_col], famous_df[name_col])))

    col_map = {'venueCategory': 'category', 'latitude': 'lat', 'longitude': 'lon'}
    reverse_map = {v: k for k, v in col_map.items()}
    valid_cols = [reverse_map[c] for c in ['category', 'lat', 'lon'] if reverse_map.get(c) in famous_df.columns]

    if valid_cols:
        enriched_df = famous_df[[pid_col] + valid_cols].copy()
        enriched_df.rename(columns={c: col_map[c] for c in valid_cols}, inplace=True)
        enriched_df.rename(columns={pid_col: 'poi_id'}, inplace=True)
        metadata.update(enriched_df.set_index('poi_id').to_dict(orient='index'))

    famous_ids = famous_df[pid_col].tolist()

# ----------------- UI Setup ----------------------
st.title("📍 POI Recommendation System")

name_to_id = {v: k for k, v in id_to_name.items()}
valid_names = {pid: id_to_name[pid] for pid in fused_embedding if pid in id_to_name}
valid_name_to_id = {v: k for k, v in valid_names.items()}

selected_name = st.selectbox("Select a POI", sorted(valid_name_to_id.keys()))

if selected_name:
    poi_id = valid_name_to_id[selected_name]
    user_lat = metadata[poi_id]['lat']
    user_lon = metadata[poi_id]['lon']

    # ----------------- Nearest Famous POI Recommendations ----------------------
    def get_tourist_places(poi_id, top_k=5):
        source = (metadata[poi_id]['lat'], metadata[poi_id]['lon'])
        places = []
        for fid in famous_ids:
            if fid not in metadata:
                continue
            dest = (metadata[fid]['lat'], metadata[fid]['lon'])
            dist = geodesic(source, dest).km
            places.append((fid, dist))
        return sorted(places, key=lambda x: x[1])[:top_k]

    st.markdown("## 🧳 Famous Tourist Places Nearby")
    tourist_recs = get_tourist_places(poi_id)

    for tid, dist in tourist_recs:
        t_name = id_to_name.get(tid, tid)
        maps_url = f"https://www.google.com/maps/dir/{user_lat},{user_lon}/{metadata[tid]['lat']},{metadata[tid]['lon']}"
        st.markdown(f"- **{t_name}** — {metadata[tid]['category']} (~{dist:.2f} km)  \n  [🧭 Directions]({maps_url})")
