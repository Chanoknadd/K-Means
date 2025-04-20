# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:13:50 2025

@author: LAB
"""

import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    
#set title
st.title("K-Means Clustering Visualizer by Chanoknad Pongjeen")

#set the page config
st.set_page_config(page_title = "K-Means Clustering", layout="centered")


#load dataset
X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

#Predict using the loaded model
y_kmeans = loaded_model.predict(X)

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = loaded_model.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, label='Centroids')

ax.set_title("k-Means Clustering")
ax.legend()

# Show in Streamlit
st.pyplot(fig)