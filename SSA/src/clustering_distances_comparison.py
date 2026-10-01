#code to compare clustering using different distances
#the goal is to analyze if sequences in the same cluster using one distance are still in the same 
#cluster using another distance
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from config import SEQUENCE_NAME, CLUSTERING, LINKAGE,NUM_CLUSTERS
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
CLUSTER_FOLDER=ROOT/'SSA'/'results'/'sequences'/'sequence_clustering'/SEQUENCE_NAME/CLUSTERING
DISS_FOLDER=ROOT/'SSA'/'results'/'sequences'/'sequence_dissimilarities'


distances=['OM_CONST', 'OM_TRATE', 'OM_INDELSLOG', 'LCS', 'LCS_NO_WEIGHTED', 'HAM','HAM_WIND_2','HAM_WIND_2_LOG' ]
#distances=['OM_INDELSLOG', 'LCS', 'LCS_WEIGHTED','HAM_WIND_2','HAM_WIND_2_LOG' ]

"""
#compute Adjusted Rand Index (ARI) between two clusterings
from sklearn.metrics import adjusted_rand_score

#define a matrix to store the ARI values between different clustering results
ari_matrix = pd.DataFrame(index=distances, columns=distances)

for dist in distances:
    #load cluster labels for the first distance
    cluster_labels_1 = pd.read_csv(CLUSTER_FOLDER/dist/f'cluster_labels_{dist}.csv', index_col=0).values.flatten()
    
    for dist2 in distances:
        #load cluster labels for the second distance
        cluster_labels_2 = pd.read_csv(CLUSTER_FOLDER/dist2/f'cluster_labels_{dist2}.csv', index_col=0).values.flatten()
        
        #compute ARI between the two clusterings
        ari = adjusted_rand_score(cluster_labels_1, cluster_labels_2)
        
        #store the ARI value in the matrix
        ari_matrix.loc[dist, dist2] = ari

#save the ARI matrix as a figure
plt.figure(figsize=(10,8))
sns.heatmap(ari_matrix.astype(float), annot=True, cmap='coolwarm', center=0)
plt.title('Adjusted Rand Index between Different Distance Metrics')
plt.xlabel('Distance Metric')
plt.ylabel('Distance Metric')
plt.savefig(CLUSTER_FOLDER/'ari_matrix_heatmap.png')
"""

#compute mantel test with Spearman rank coefficint between dissimilarity matrices of different distances
from skbio.stats.distance import mantel
from itertools import combinations

# Choose your two custom colors for the gradient (Hex codes or standard names)
COLOR_START = "#ffffff"  # Light/Neutral color for low/no correlation (e.g., white)
COLOR_END = "#649ABF"    # Strong color for high correlation (e.g., steel blue)

# --- 1. Load Matrices ---
matrices = {}
for dist in distances:
    file_path = DISS_FOLDER / f'{SEQUENCE_NAME}_{dist}.npy'
    matrices[dist] = np.load(file_path)



#extract a subsample of 2000 sequences to speed up the computation of the Mantel test
np.random.seed(42)  # Rende il campionamento perfettamente riproducibile
sample_idx = np.random.choice(matrices[distances[0]].shape[0], size=2000, replace=False)
sub_matrices = {dist: matrices[dist][np.ix_(sample_idx, sample_idx)] for dist in distances}

# --- 2. Initialize the Result Matrix ---
# Create an empty square DataFrame filled with 1.0 on the diagonal (self-correlation)
matrix_size = len(distances)
df_corr = pd.DataFrame(np.eye(matrix_size), index=distances, columns=distances)
df_pvalues = pd.DataFrame(np.ones((matrix_size, matrix_size)), index=distances, columns=distances)

# --- 3. Compute Mantel Tests ---
print("Computing Rank Mantel tests...")
for dist1, dist2 in combinations(distances, 2):
    corr, p_value, n = mantel(
        sub_matrices[dist1], 
        sub_matrices[dist2], 
        method='spearman', 
        permutations=999
    )

    
    # Fill both symmetric sides of the matrix
    df_corr.loc[dist1, dist2] = corr
    df_corr.loc[dist2, dist1] = corr
    df_pvalues.loc[dist1, dist2] = p_value
    df_pvalues.loc[dist2, dist1] = p_value

# Print the raw text matrix to the console
print("\n--- Resulting Correlation Matrix ---")
print(df_corr.round(4))
print("\n--- Resulting P-Value Matrix ---")
print(df_pvalues.round(4))

# --- 4. Plot the Customized Heatmap ---
plt.figure(figsize=(8, 6))

# Create the custom 2-color gradient palette
from matplotlib.colors import LinearSegmentedColormap

custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", [COLOR_START, COLOR_END])

# Build a mask to hide the upper triangle (optional, makes it cleaner since it's symmetric)
mask = np.triu(np.ones_like(df_corr, dtype=bool), k=1)

sns.heatmap(
    df_corr, 
    annot=True,             # Shows the correlation values inside the squares
    fmt=".3f",              # Decimal formatting
    cmap=custom_cmap,       # Your custom 2-color gradient
    mask=mask,              # Hides the redundant upper triangle
    vmin=0, vmax=1,         # Adjust limits depending on your expected r range
    square=True,            # Forces squares to be perfectly square
    linewidths=.5,          # Subtle line between squares
    cbar_kws={"label": "Spearman Mantel Correlation (r)"}
)

plt.title(f"Mantel Correlation Matrix", fontsize=14, pad=15)
plt.tight_layout()

# Save the plot and show it
plt.savefig(DISS_FOLDER / f"{SEQUENCE_NAME}_mantel_heatmap.png", dpi=300)



# ==============================================================================
# Exact Spearman Rank Correlation on Full Cohort (N = 17,386)
# Pre-computes ranks once per distance to avoid O(N^2 * permutations) bottleneck
# ==============================================================================
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# 1. Extract upper-triangle indices (strictly upper triangle, k=1, ~151M pairs)
n_patients = matrices[distances[0]].shape[0]
triu_indices = np.triu_indices(n_patients, k=1)

# 2. Pre-compute and store the ranks for each dissimilarity metric (runs only 7 times)
print(f"Pre-computing ranks on full cohort (N = {n_patients:,})...")
ranked_vectors = {}
for dist in distances:
    print(f"  Ranking distance: {dist}...")
    # Extract compact 1D vector and cast to float32 to conserve RAM
    upper_vals = matrices[dist][triu_indices].astype(np.float32)
    ranked_vectors[dist] = rankdata(upper_vals).astype(np.float32)
    del upper_vals

# 3. Compute pairwise Spearman correlation (Pearson on pre-computed ranks)
matrix_size = len(distances)
df_corr_full = pd.DataFrame(np.eye(matrix_size), index=distances, columns=distances)

print("\nComputing pairwise correlations...")
for dist1, dist2 in combinations(distances, 2):
    # Pearson correlation on ranks is mathematically identical to Spearman's rho
    rho = np.corrcoef(ranked_vectors[dist1], ranked_vectors[dist2])[0, 1]
    df_corr_full.loc[dist1, dist2] = rho
    df_corr_full.loc[dist2, dist1] = rho

# Free up memory
del ranked_vectors

print("\n--- Strategy B: Full Cohort Correlation Matrix ---")
print(df_corr_full.round(4))
#plot the heatmap for the full cohort correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    df_corr_full, 
    annot=True, 
    fmt=".3f", 
    cmap=custom_cmap, 
    mask=mask, 
    vmin=0, vmax=1, 
    square=True, 
    linewidths=.5, 
    cbar_kws={"label": "Spearman Mantel Correlation (r)"}
)
plt.title(f"Correlation Matrix ", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(DISS_FOLDER / f"{SEQUENCE_NAME}_mantel_heatmap_full_cohort.png", dpi=300)

# 4. Compare discrepancy between Strategy A (subsample) and Strategy B (full cohort)
if "df_corr" in locals():
    abs_diff = (df_corr - df_corr_full).abs()
    print("\n--- Absolute Difference (|Strategy A - Strategy B|) ---")
    print(f"Max difference:  {abs_diff.values.max():.4f}")
    print(f"Mean difference: {abs_diff.values[np.triu_indices(matrix_size, k=1)].mean():.4f}")