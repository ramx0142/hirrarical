import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Page configuration
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("📊 Customer Segmentation using Hierarchical Clustering")
st.write("This app reproduces the Machine Learning workflow from your notebook.")

# 1. Load the Scaler (Saved from your notebook)
try:
    scaler = joblib.load("scaler.pkl")
    # Note: We don't load 'hierarchical_model.pkl' for prediction because 
    # Agglomerative Clustering must be fit to the data to generate labels.
except:
    st.error("Error: 'scaler.pkl' not found. Please ensure it is in the same folder.")

# 2. Sidebar for Data Upload
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload 'Mall_customer.csv'", type=["csv"])

if uploaded_file is not None:
    # Load and show initial data
    df = pd.read_csv(uploaded_file)
    st.subheader("1. First 5 rows of dataset")
    st.write(df.head())

    # Data Preprocessing
    # Drop CustomerID
    df_processed = df.drop("CustomerID", axis=1)
    
    # Label Encoding for Genre
    le = LabelEncoder()
    df_processed["Genre"] = le.fit_transform(df_processed["Genre"])
    
    # Scaling
    X_scaled = scaler.transform(df_processed)

    # 3. Dendrogram Section
    st.divider()
    st.subheader("2. Dendrogram for Hierarchical Clustering")
    
    fig_den, ax_den = plt.subplots(figsize=(10, 6))
    linked = linkage(X_scaled, method="ward")
    dendrogram(linked, ax=ax_den)
    plt.title("Dendrogram for Hierarchical Clustering")
    plt.xlabel("Customers")
    plt.ylabel("Euclidean Distance")
    st.pyplot(fig_den)

    # 4. Clustering Section
    st.divider()
    n_clusters = st.slider("Select number of clusters (Based on Dendrogram)", 2, 10, 5)
    
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = hc.fit_predict(X_scaled)
    df["Cluster"] = labels
    
    st.subheader(f"3. Dataset with {n_clusters} Cluster Labels")
    st.write(df.head())

    # 5. Visualization Section
    st.divider()
    st.subheader("4. Cluster Visualization (Income vs Spending)")
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    scatter = ax_scatter.scatter(
        df["Annual Income (k$)"], 
        df["Spending Score (1-100)"], 
        c=df["Cluster"], 
        cmap='viridis'
    )
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segmentation using Hierarchical Clustering")
    plt.colorbar(scatter, label="Cluster ID")
    st.pyplot(fig_scatter)

    # 6. Metrics
    sil_score = silhouette_score(X_scaled, labels)
    st.success(f"Silhouette Score: {sil_score:.4f}")

else:
    st.info("Please upload the 'Mall_Customers.csv' file in the sidebar to begin.")
