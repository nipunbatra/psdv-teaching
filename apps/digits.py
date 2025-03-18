import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Load dataset
X, y = load_digits(return_X_y=True)
X = X.astype(np.float32)
y = y.astype(np.int64)

# PCA reduction
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Normalize images to [0,255]
X_scaled = (X - X.min()) / (X.max() - X.min()) * 255
X_scaled = X_scaled.astype(np.uint8)

# Function to convert images to base64
def image_to_base64(image_array):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow(image_array.reshape(8, 8), cmap="gray")
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

# Encode all images
image_data = [image_to_base64(X_scaled[i]) for i in range(len(X_scaled))]

# Create DataFrame
df = pd.DataFrame({
    "PCA1": X_reduced[:, 0],
    "PCA2": X_reduced[:, 1],
    "Label": y,
    "Index": np.arange(len(X_scaled)),
    "Image": image_data
})

# Plot using Plotly
fig = px.scatter(
    df, x="PCA1", y="PCA2", color=df["Label"].astype(str),
    hover_data={"Index": True, "Label": True, "Image": False},  # Hide image in hover
)

# Display in Streamlit
st.title("Digit Visualization with PCA")
st.plotly_chart(fig, use_container_width=True)

# Click to show image
st.write("### Click on a point to see the actual digit image")
selected_index = st.number_input("Enter the index of the point:", min_value=0, max_value=len(X_scaled)-1, step=1, value=0)
st.image(X_scaled[selected_index].reshape(8, 8), caption=f"Digit: {y[selected_index]}", use_column_width=False)
