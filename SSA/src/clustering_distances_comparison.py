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


distances=['LCS_NO', 'HAM_WIND_1', 'HAM_WIND_1_weighted','HAM_WIND_1_eq_weighted', 'HAM_WIND_1_0.3_weighted',
          'OM_INDELSLOG', 'OM_INDELSLOG2','OM_INDELSQRT' ]

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