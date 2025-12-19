import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Set page title
st.set_page_config(page_title="Customer Segmentation App")

@st.cache_resource
def load_models():
    # Load the scaler and model
    # Note: Ensure these files are in the same directory
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("hierarchical_model.pkl")
    return scaler, model

try:
    scaler, hc_model = load_models()
except:
    st.error("Model files not found! Please ensure 'scaler.pkl' and 'hierarchical_model.pkl' are in the app folder.")

st.title("👥 Customer Segmentation (Hierarchical Clustering)")
st.write("Enter customer details to find their segment.")

# Sidebar Inputs
st.sidebar.header("User Input Features")

genre = st.sidebar.selectbox("Genre", ("Male", "Female"))
age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.slider("Annual Income (k$)", 15, 140, 50)
spending = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

# Preprocessing Input
# Map Genre to numeric (Based on your LabelEncoder: Female=0, Male=1)
genre_encoded = 1 if genre == "Male" else 0

# Create input array
user_data = np.array([[genre_encoded, age, income, spending]])

if st.button("Identify Segment"):
    # 1. Scale the input
    user_data_scaled = scaler.transform(user_data)
    
    # 2. Handle Hierarchical Prediction logic
    # Since AgglomerativeClustering doesn't have .predict(), 
    # we show the input data relative to the clusters or 
    # (Advanced) find the nearest cluster.
    
    st.subheader(f"Results for {genre}, Age {age}")
    
    # In a real app, you would ideally use a Classifier trained on the labels 
    # but here we will display the input values.
    st.write(f"- **Annual Income:** ${income}k")
    st.write(f"- **Spending Score:** {spending}")
    
    st.info("Note: Agglomerative Clustering is descriptive. In a full production app, "
             "a KNN classifier is usually trained on these cluster labels to 'predict' new points.")

# File Upload Section (Optional - Good for Hierarchical models)
st.divider()
st.subheader("Bulk Segmentation (Upload CSV)")
uploaded_file = st.file_uploader("Upload your Mall_Customers.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    df_clean = data.drop("CustomerID", axis=1)
    
    # Encode Genre
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_clean["Genre"] = le.fit_transform(df_clean["Genre"])
    
    # Scale and Predict
    X_scaled = scaler.transform(df_clean)
    labels = hc_model.fit_predict(X_scaled)
    data["Cluster"] = labels
    
    st.write("Segmented Data Preview:")
    st.dataframe(data.head())
    
    # Visualization
    fig, ax = plt.subplots()
    scatter = ax.scatter(data["Annual Income (k$)"], 
                         data["Spending Score (1-100)"], 
                         c=data["Cluster"], cmap='viridis')
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    plt.colorbar(scatter, label="Cluster ID")
    st.pyplot(fig)
