README

TFM from computational biology in Universidad Politecnica de Madrid

Data retrive from work Liu, S. et al. (2023). Breast adipose tissue-derived extracellular
vesicles from obese women alter tumor cell metabolism.  DOI 10.15252/embr.202357339.

We preprocessed the data in order perform different goals, in this repository you can
find the first steps that include (Log2 transformation, imputation, scaling, dimensionality reduction and labeling)


We aim to analyze DDA proteomic data derived extracellular vesicules from breast adipose
tissue from obese and overweigth women in New York. This work implement different 
approach to select the better hyperparamenters in every technique, you can find a summarization
in the figure 1 and the algorithms steps for the selected parameters.
 
The project is made in the develop environment of Jupyter notebook with python3.

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Upload csv
df = pd.read_csv('abundance_ProteinGroups.csv')

# Exclude the <= 20% sequence coverage: 1740, create new dataset
df_exclude = df[df['Sequence coverage [%]'] > 20]
count_max20_coverage = df_exclude.shape[0]
print('Number of proteins with > 20% sequence coverage:', count_max20_coverage)

# subset for exclude dataframe contaiing only important labels
key_keep_cols = ['Protein IDs', 'Protein names', 'Gene names']  
intensity_cols = [col for col in df.columns if col.startswith('Intensity')]  

df_excludesub = df_exclude[key_keep_cols + intensity_cols]

   
# save the preprocessed dataframe for further analysis  
df_excludesub.to_csv('exclusesubset_proteomics.csv', index=False)  
print("excludesubset data saved as 'exclusesubset_proteomics.csv'") 

# Copy and replace 0 per nan in intensity columns
df_log_exc = df_excludesub.copy()
df_log_exc[intensity_cols] = df_log_exc[intensity_cols].replace(0, np.nan)
df_log_exc[intensity_cols] = np.log2(df_log_exc[intensity_cols])

# Visualize distribution with Log2 transformation
sns.set(style="whitegrid", context="paper", font_scale=1.4)
plt.figure(figsize=(10, 6), dpi=300)
plt.hist(df_log_exc[intensity_cols].stack(), bins=50, color='#69b3a2', edgecolor='black', alpha=0.85)
plt.title('Log2 Intensity Distribution (No Imputation)', fontsize=16, weight='bold')
plt.xlabel('Log2 Intensity', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

# Add Iterative Imputer to df_log_exc
df_log_exc_II = df_log_exc.copy()
intensity_cols = [col for col in df_log_exc_II.columns if col.startswith('Intensity')]
imputer = IterativeImputer(random_state=42)

# Fit and transform
df_log_exc_II[intensity_cols] = imputer.fit_transform(df_log_exc_II[intensity_cols])

# Check if there are any missing values left
print("Remaining missing values:", df_log_exc_II[intensity_cols].isnull().sum().sum())

# Visualization of distribution after imputation
sns.set(style="whitegrid", context="paper", font_scale=1.4)
plt.figure(figsize=(10, 6), dpi=300)
plt.hist(df_log_exc_II[intensity_cols].stack(), bins=50, color='#69b3a2', edgecolor='black', alpha=0.85)
plt.title('Log2 Intensity Distribution After Iterative Imputation)', fontsize=16, weight='bold')
plt.xlabel('Log2 Intensity', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

# Scale and Create a copy of just the intensity data for normalization
intensity_data = df_log_exc_II[intensity_cols].copy()
std_scaler = StandardScaler()
df_log_exc_II_std = pd.DataFrame(std_scaler.fit_transform(intensity_data), columns=intensity_cols)

# Visualization of distribution after scaling
sns.set(style="whitegrid", context="paper", font_scale=1.4)
plt.figure(figsize=(10, 6), dpi=300)
plt.hist(df_log_exc_II_std[intensity_cols].stack(), bins=50, color='#69b3a2', edgecolor='black', alpha=0.85)
plt.title('Log2 Intensity Distribution (Imputation + Scaling))', fontsize=16, weight='bold')
plt.xlabel('Log2 Intensity', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tight_layout()
plt.show()

# UMAP installation
!pip install umap-learn
import umap

# Fit and transform df_exc_II_std
umap_tune = umap.UMAP(n_neighbors=50, min_dist=0.0, metric='cosine', n_components=2, random_state=42)
umap_tune.fit(df_log_exc_II_std)
df_log_exc_II_std_umap_tune = umap_tune.transform(df_log_exc_II_std)

# Fit KMeans
clustering_kmeans = KMeans(
    n_clusters=2, random_state=74, max_iter=500,
    init='k-means++', n_init='auto'
).fit(df_log_exc_II_std_umap_tune)

# Get cluster labels and silhouette score
cluster_labels = clustering_kmeans.labels_
score = silhouette_score(df_log_exc_II_std_umap_tune, cluster_labels)

# Plot
sns.set(style="whitegrid", context="paper", font_scale=1.4)
plt.figure(figsize=(10, 6), dpi=300)
palette = sns.color_palette("Set1", n_colors=2)
plt.scatter(
    df_log_exc_II_std_umap_tune[:, 0], df_log_exc_II_std_umap_tune[:, 1],
    s=25, c=[palette[label] for label in cluster_labels], edgecolor='k', alpha=0.85
)

plt.xlabel("UMAP Component 1", fontsize=14)
plt.ylabel("UMAP Component 2", fontsize=14)
plt.title(f"KMeans Clustering (Silhouette = {score:.2f})", fontsize=16, weight='bold')

#Legends
handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {i+1}',
            markerfacecolor=palette[i], markersize=10, markeredgecolor='k') for i in range(2)]
plt.legend(handles=handles, loc='best')

plt.tight_layout()
plt.show()

# Save cluster plot
plt.savefig("umap_kmeans_clusters.png", dpi=600, bbox_inches='tight')

# Label the UMAP 1 and  2  for df_log_exc_II_umap_tune kmeans
cluster_labels_umap = clustering_kmeans.labels_

# Turn UMAP array into a DataFrame
df_umap_clustered = pd.DataFrame(df_log_exc_II_std_umap_tune, columns=['UMAP Component 1', 'UMAP Component 2'])
df_umap_clustered['cluster'] = cluster_labels_umap

# Merging df_log_exc_II_std and df_log_exc_II_std_umap_tune with aligned indices
df_clustered_full = df_log_exc_II_std.copy()
df_clustered_full['cluster'] = cluster_labels_umap

# extract metadata from initial df_excludesub
metadata = df_excludesub[['Protein IDs', 'Protein names', 'Gene names']].copy()

# Combine metadata with df_clustered_full
df_grouped = pd.concat([metadata.reset_index(drop=True), df_clustered_full.reset_index(drop=True)], axis=1)

# save the df_grouped
df_grouped.to_csv('groupedUMAP_proteomics.csv', index=False)  
print("groupedUMAP data saved as 'groupedUMAP_proteomics.csv'")
