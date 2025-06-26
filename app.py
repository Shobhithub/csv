import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import gdown

@st.cache_data
def load_data():
    file_id = "1SOzwVIqiwl1zOtL1ETndsTOXV3NmIeYV"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "data.csv"
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

# Load and preview data
df = load_data()
st.write("📊 Preview of the CSV file:")
st.dataframe(df.head())

# Extract features for clustering
audio_features = df[['danceability', 'energy', 'loudness', 'speechiness',
                     'instrumentalness', 'liveness', 'valence', 'tempo']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(audio_features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# UI
st.title("Haridwar University")
st.title("🎵 Song Recommender Based on Audio Cluster")

selected_cluster = st.selectbox("Select a cluster (0-3):", sorted(df['Cluster'].unique()))
n = st.slider("Number of Songs", min_value=1, max_value=10, value=5)

# Recommend songs
suggested = df[df['Cluster'] == selected_cluster][['track_name', 'artist_name', 'genre']].sample(n)
st.subheader("🎧 Suggested Songs:")
st.dataframe(suggested)
