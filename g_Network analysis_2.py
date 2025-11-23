#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 22:55:28 2025

@author: donggeunhan
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from collections import Counter
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Load your Excel file
df_full = pd.read_excel("/Users/donggeunhan/Library/CloudStorage/OneDrive-OklahomaAandMSystem/Dissertation/Data/Chapter III/Final data/a_META_AgEcon_Journal_papers_list.xlsx") # <-- replace with your file path if needed
#authors_series_full = df_full.iloc[:, 0].astype(str)
authors_series_full = df_full["authors"].astype(str)


# Parse authors and build full author network
parsed_authors = [set(a.strip() for a in entry.split(';') if a.strip()) for entry in authors_series_full]
G_full = nx.Graph()
for authors in parsed_authors:
    for u in authors:
        for v in authors:
            if u != v:
                G_full.add_edge(u, v)

# Compute global author metrics
degree_dict = dict(G_full.degree())
betweenness_dict = nx.betweenness_centrality(G_full)
closeness_dict = nx.closeness_centrality(G_full)
eigenvector_dict = nx.eigenvector_centrality(G_full, max_iter=1000)
global_max_degree = max(degree_dict.values())

# Identify top 400 authors
#top_authors = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:400]
#top_author_nodes = set(author for author, _ in top_authors)

# Identify top all authors
top_n = int(len(degree_dict) * 1)  
top_authors = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
top_author_nodes = set(author for author, _ in top_authors)

# Build subgraph for all authors
G = G_full.subgraph(top_author_nodes).copy()
degrees = dict(G.degree())

# Identify duplicate last names
last_names = [node.split(',')[0].strip() for node in G.nodes()]
last_name_counts = Counter(last_names)

# Build disambiguated labels
#labels = {}
#for node in G.nodes():
#    last, first = node.split(',')[0].strip(), node.split(',')[1].strip()
#    if last_name_counts[last] > 1:
#        abbrev = first[:3] if len(first) >= 3 else first
#        label = f"{last}-{abbrev}"
#    else:
#        label = last
#    labels[node] = label


labels = {}
for node in G.nodes():
    parts = node.split(',')
    if len(parts) == 2:
        last, first = parts[0].strip(), parts[1].strip()
        if last_name_counts[last] > 1:
            abbrev = first[:3] if len(first) >= 3 else first
            label = f"{last}-{abbrev}"
        else:
            label = last
    else:
        # fallback for malformed names (no comma)
        label = node.strip()
    
    labels[node] = label




# Compute positions for network layout
pos = nx.spring_layout(G, seed=42, k=0.5)

# Choose colormap
cmap = cm.get_cmap('nipy_spectral')
norm = mcolors.Normalize(vmin=min(degrees.values()), vmax=global_max_degree)

# Draw network with labels
plt.figure(figsize=(40, 40))
nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.6)

for node, (x, y) in pos.items():
    degree = degrees[node]
    font_size = 18 + (degree / global_max_degree) * 32
    if font_size > 10:
        color = cmap(norm(degree))
        plt.text(x, y, labels[node], fontsize=font_size, color=color, ha='center', va='center')
        

# Set the directory where you want to save the file
import os
print(os.getcwd())
os.chdir("/Users/donggeunhan/Library/CloudStorage/OneDrive-OklahomaAandMSystem/Dissertation/Data/Chapter III/Final data")

#plt.title("AgEcon Collaborative Network, 2018 to 2023 (All Authors)", fontsize=36)
plt.axis('off')
#plt.savefig("a_AgEcon_Collaborative_Network_Top_10percent.pdf", format = 'pdf', dpi=600, bbox_inches='tight')
plt.savefig("a_AgEcon_Collaborative_Network_AllAuthors.pdf", format = 'pdf', dpi=600, bbox_inches='tight')
plt.close()
print("Network figure saved as a_AgEcon_Collaborative_Network_Top_10percent.png")

# Calculate advanced metrics for the whole dataset
metrics = []
for authors in parsed_authors:
    author_list = list(authors)
    valid_authors = [a for a in author_list if a in G_full.nodes()]
   
    if len(valid_authors) == 0:
        metrics.append([0]*10)
        continue

    # Degrees
    degrees = [degree_dict[a] for a in valid_authors]
    mean_deg = np.mean(degrees)
    max_deg = np.max(degrees)
   
    # Betweenness
    betweenness = [betweenness_dict[a] for a in valid_authors]
    mean_betw = np.mean(betweenness)
   
    # Diversity (entropy of prior coauthors)
    prior_counts = []
    for a in valid_authors:
        neighbors = set(G_full.neighbors(a)) - authors
        prior_counts.append(len(neighbors))
    diversity = entropy(np.array(prior_counts) / sum(prior_counts)) if sum(prior_counts) > 0 else 0
   
    # Closeness
    closeness = [closeness_dict[a] for a in valid_authors]
    mean_close = np.mean(closeness)
    max_close = np.max(closeness)
   
    # Eigenvector
    eigen = [eigenvector_dict[a] for a in valid_authors]
    mean_eigen = np.mean(eigen)
    max_eigen = np.max(eigen)
        
        # Inside your for-loop per article
    if len(valid_authors) > 1:
    # Clustering coefficient from full network
        clustering_vals = [nx.clustering(G_full, a) for a in valid_authors]
        mean_cluster = np.mean(clustering_vals)

    # Optional: global density (constant across articles)
        density = nx.density(G_full)
    else:
        density, mean_cluster = np.nan, np.nan
   
    metrics.append([
        mean_deg, max_deg, density, mean_betw, diversity,
        mean_close, max_close, mean_eigen, max_eigen, mean_cluster
    ])


# Save metrics to Excel
metrics_df = pd.DataFrame(metrics, columns=[
    'MeanDegree', 'MaxDegree', 'Density', 'MeanBetweenness', 'DiversityEntropy',
    'MeanCloseness', 'MaxCloseness', 'MeanEigenvector', 'MaxEigenvector',
    'ClusteringCoefficient'
])

# Add the study_id column from df_full as the first column
metrics_df.insert(0, 'study_id', df_full['study_id'])

#metrics_df.to_excel("C:\\Users\\daylamb\\OneDrive - Oklahoma A and M System\\WAEA\\2025\\Presidential Address\\Analysis\\Data\\Article_Collaborative_Metrics.xlsx", index=False)
#print("Collaborative metrics saved to Article_Collaborative_Metrics.xlsx")

metrics_df.to_excel("/Users/donggeunhan/Library/CloudStorage/OneDrive-OklahomaAandMSystem/Dissertation/Data/Chapter III/Final data/g_Article_Collaborative_Metrics_2.xlsx", index = False)
print("Collaborative metrics saved to Article_Collaborative_Metrics.xlsx")