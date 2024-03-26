from flask import Flask, render_template, request, jsonify

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from scipy import stats
import re

app = Flask(__name__)

def find_similar_cocktails(cocktail_name, data, n=5):
    if cocktail_name not in data['name'].values:
        return "Cocktail not found in the database."

    # Get the cluster of the given cocktail
    cocktail_cluster = data[data['name'] == cocktail_name]['Cluster'].iloc[0]

    # Filter data for cocktails in the same cluster
    cluster_data = data[data['Cluster'] == cocktail_cluster]

    # Extract the row of the given cocktail
    cocktail_row = cluster_data[cluster_data['name'] == cocktail_name].drop(['name', 'Cluster', 'tsne-2d-one', 'tsne-2d-two'], axis=1)

    # Exclude the selected cocktail from the cluster data before computing distances
    cluster_data = cluster_data[cluster_data['name'] != cocktail_name]

    # Compute distances from the given cocktail to all others in the cluster
    distances = euclidean_distances(cluster_data.drop(['name', 'Cluster', 'tsne-2d-one', 'tsne-2d-two'], axis=1), cocktail_row).flatten()

    # Add distances to the cluster data
    cluster_data['Distance'] = distances

    # Get top n similar cocktails
    similar_cocktails = cluster_data.sort_values('Distance')[:n]

    return similar_cocktails['name'].tolist()

data = "/Users/aaron/Desktop/Code/Cocktail/cleaned_cocktails.csv"
df = pd.read_csv(data)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/find_cocktails', methods=['POST'])
def find_cocktails():
    cocktail_name = request.form['cocktail_name']
    similar_cocktails = find_similar_cocktails(cocktail_name, df)
    return jsonify(similar_cocktails)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

