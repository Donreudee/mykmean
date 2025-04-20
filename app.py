# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:09:32 2025

@author: LAB
"""

import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Load modek
with open ('kmeans_model.pkl', 'rb') as f :
    loaded_model = pickle.load(f)
    

# Set title
st.title(" k-Means Clustering Visualizer by Dondi Tonphot")

# Set the page config
st.set_page_config(page_title="k-means Clustering App", layout="centered")

#load dataset
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

y_kmeans = loaded_model.predict(X)

#plotting
fig, ax = plt.subplot()
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red')
plt.title('k-Means Clustering')
ax.legend()
st.pyplot(fig)