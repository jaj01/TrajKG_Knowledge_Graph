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
    gdown.download(f"https://drive.google.com/uc?id={EMBEDDING_FILE_ID}", "fused_embedding.pkl", quiet=False)

if not os.path.exists("dataset_TSMC2014_NYC.csv"):
    gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", "dataset_TSMC2014_NYC.csv", quiet=False)

if not os.path.exists("poi_names.csv"):
    gdown.download(f"https://drive.google.com/uc?id={NAME_FILE_ID}", "poi_names.csv", quiet=False)

if not os.path.exists("poi_names_famous_nyc.csv"):
    gdown.download(f"https://drive.google.com/uc?id={FAMOUS_POI_FILE_ID}", "poi_names_famous_nyc.csv", quiet=False)

# ----------------- Load Files ----------------------
fused_embedding = pickle.load(open("fused_embedding.pkl", "rb"))
metadata_df = pd.read_csv("dataset_TSMC2014_NYC.csv")
metadata_df = metadata_df[['venueId', 'venueCategory', 'latitude', 'longitude']]
metadata_df.columns = ['poi_id', 'category', 'lat', 'lon']
metadata = metadata_df.drop_duplicates('poi_id').set_index('poi_id').to_dict(orient='index')

# Load POI names
name_df = pd.read_csv("poi_names.csv")
id_to_name = dict(zip(name_df['venueId'], name_df['venueName']))

# Load famous POIs and merge
if os.path.exists("poi_names_famous_nyc.csv"):
    famous_df = pd.read_csv("poi_names_famous_nyc.csv")
    col_names = list(famous_df.columns)
    pid_col = [col for col in col_names if "venueId" in col or "poi_id" in col][0]
    name_col = [col for col in col_names if "venueName" in col or "name" in col][0]

    id_to_name.update(dict(zip(famous_df[pid_col], famous_df[name_col])))

    col_map = {
        'venueCategory': 'category',
        'latitude': 'lat',
        'longitude': 'lon'
    }
    reverse_col_map = {v: k for k, v in col_map.items()}
    safe_cols = [reverse_col_map[col] for col in ['category', 'lat', 'lon'] if reverse_col_map.get(col) in famous_df.columns]

    if safe_cols:
        subset_df = famous_df[[pid_col] + safe_cols].copy()
        subset_df = subset_df.rename(columns={col: col_map[col] for col in safe_cols})
        subset_df = subset_df.rename(columns={pid_col: 'poi_id'})
        metadata.update(subset_df.set_index('poi_id').to_dict(orient='index'))

    famous_ids = famous_df[pid_col].tolist()
else:
    famous_ids = []

name_to_id = {v: k for k, v in id_to_name.items()}
valid_names = {pid: id_to_name[pid] for pid in fused_embedding if pid in id_to_name}
valid_name_to_id = {v: k for k, v in valid_names.items()}

# ----------------- Map View ----------------------
st.subheader("🗺 Map View")
map_center = [metadata[selected_poi]['lat'], metadata[selected_poi]['lon']]
m = folium.Map(location=map_center, zoom_start=15)
selected_name_display = id_to_name.get(selected_poi, selected_poi)
folium.Marker(location=map_center, popup=f"Selected: {selected_name_display}", icon=folium.Icon(color='blue')).add_to(m)

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
