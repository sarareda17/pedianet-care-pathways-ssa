############### 2 levels clustering ###############
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import SEQUENCE_NAME, CLUSTERING, LINKAGE, NUM_CLUSTERS, DISTANCE_METRIC, STATES_LABELS, CUSTOM_COLORS, COSTS
ROOT=Path(__file__).parent.parent.parent

#define the path to the data folder
PATH_CLUSTER=ROOT/'SSA'/'results'/'sequences'/'sequence_clustering'/SEQUENCE_NAME/CLUSTERING/DISTANCE_METRIC


#load dataset with sequences
df=pd.read_csv(ROOT/'SSA'/'datasets'/ f'{SEQUENCE_NAME}.csv')
time_cols=[df.columns[i] for i in range(len(df.columns)) if df.columns[i].isdigit()]
cov_cols=[col for col in df.columns if col not in time_cols]
df_sequence=df[['pedianet_id'] + time_cols]

#load the clustering results from the first level clustering
cluster_labels=pd.read_csv(PATH_CLUSTER/f'cluster_labels_{DISTANCE_METRIC}.csv',index_col=0)

#define the CLUSTER on which to perform the second level clustering
CLUSTER=1
#define the DISTANCE_METRIC to use for the second level clustering
DISTANCE_METRIC_2='LCS'
#define the SEQUENCE_NAME to use for the second level clustering
SEQUENCE_NAME_2=f'{SEQUENCE_NAME}_cluster_{CLUSTER}'
#DEFINE THE NUMBER OF CLUSTERS 
NUM_CLUSTERS_2=2

#if it does not exist, create a folder in PATH_SAVE called '2_levels_clustering'
PATH_CLUSTER_2=PATH_CLUSTER/'2_levels_clustering'/DISTANCE_METRIC_2
PATH_CLUSTER_2.mkdir(parents=True, exist_ok=True)


#save ids of the sequences belonging to the specified cluster
cluster_ids=cluster_labels[cluster_labels['Cluster']==CLUSTER].index
#select the sequences belonging to the specified cluster
df_cluster=df_sequence[df_sequence['pedianet_id'].isin(cluster_ids)]
#build sequence object for the selected cluster
from sequenzo import SequenceData
sequence=SequenceData(
    data=df_cluster,
    time=time_cols,
    states=list(STATES_LABELS.keys()),
    labels=list(STATES_LABELS.values()),
    id_col='pedianet_id',
    custom_colors=CUSTOM_COLORS
)
sequence_ids=sequence.ids

###### clustering #######
from dissimilarity import get_dissimilarity_matrix
distance_matrix=get_dissimilarity_matrix(DISTANCE_METRIC_2, sequence, df_cluster,SEQUENCE_NAME_2,COSTS)


if CLUSTERING=='hierarchical':    
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform 
    from sklearn.metrics import silhouette_score
    import fastcluster
    condensed_distance = squareform(distance_matrix)
    Z = fastcluster.linkage(condensed_distance, method=LINKAGE)

    # Dendrogram plot
    plt.figure(figsize=(12, 5))
    #dendrogram(Z, truncate_mode='lastp', p=30, show_leaf_counts=True)
    dendrogram(Z,no_labels=True, color_threshold=0)
    plt.title('Dendrogram')
    plt.xlabel('Cluster size')
    plt.ylabel('Distance')
    plt.savefig(PATH_CLUSTER_2/f'dendrogram_{DISTANCE_METRIC_2}.png', dpi=300, bbox_inches='tight')

    #silhouette scores
    # silhouettes = []
    # print(f"Computing silhouette scores...")
    # for k in range(2, MAX_CLUSTERS + 1):
    #     labels = fcluster(Z, t=k, criterion='maxclust')
        
    #     # Silhouette
    #     sil = silhouette_score(distance_matrix, labels)
    #     silhouettes.append(sil)

    # #Plot silhouette scores
    # plt.figure(figsize=(8, 5))
    # plt.plot(range(2, MAX_CLUSTERS + 1), silhouettes, marker='o', linewidth=2)
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Scores for Hierarchical Clustering')
    # plt.xticks(range(2, MAX_CLUSTERS + 1))
    # plt.tight_layout()
    # plt.savefig(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /LINKAGE /DISTANCE_METRIC/ f"elbow_silhouette_{DISTANCE_METRIC}.png", dpi=300, bbox_inches='tight')
        
    cluster_labels = fcluster(Z, t=NUM_CLUSTERS_2, criterion='maxclust')
    cluster_labels_df = pd.DataFrame({'Entity ID': sequence_ids, 'Cluster': cluster_labels})
    cluster_labels_df.to_csv(PATH_CLUSTER_2/f'cluster_labels_{DISTANCE_METRIC_2}.csv', index=False)


    if sequence is not None:
        from sequenzo import plot_sequence_index
        plot_sequence_index(
            sequence,
            group_dataframe=cluster_labels_df,
            group_column_name="Cluster",
            sort_by= "unsorted",
            save_as=str(PATH_CLUSTER_2 / f'index_plot_by_cluster_{DISTANCE_METRIC_2}_{LINKAGE}_{NUM_CLUSTERS_2}.png')
        )

        from sequenzo import plot_state_distribution
        plot_state_distribution(
            sequence,
            group_dataframe=cluster_labels_df,
            group_column_name="Cluster",
            save_as=str(PATH_CLUSTER_2 / f'state_distribution_by_cluster_{DISTANCE_METRIC_2}_{LINKAGE}_{NUM_CLUSTERS_2}.png'),
            show=False
        )


#build clustering summary table 
df = df_cluster.merge(cluster_labels_df, left_on='pedianet_id', right_on='Entity ID', how='inner')
time_cols = [c for c in df.columns if c.isdigit()]
all_states = list(STATES_LABELS.keys())
rows = []

for cluster_id in sorted(df['Cluster'].unique()):
    cluster_df = df[df['Cluster'] == cluster_id][time_cols]

    # --- N sequences ---
    n_seq = len(cluster_df)

    # --- Mean number of non-NO states per sequence ---
    mean_non_no = cluster_df.apply(
        lambda row: (row != 'NO').sum(), axis=1
    ).mean()

    # --- Mean time points spent in each state per sequence ---
    def patient_time(row):
        vc = row.value_counts()
        return pd.Series({s: vc.get(s, 0) for s in all_states})

    mean_time = cluster_df.apply(patient_time, axis=1).mean()

    
    # --- Assemble row ---
    row = {'Cluster': cluster_id, 'N sequences': n_seq, 'Mean non-NO states': round(mean_non_no, 2)}
    for s in all_states:
        row[f'mean t {STATES_LABELS[s]}'] = round(mean_time[s], 1)

    for s in all_states:
        has_state = (cluster_df == s).any(axis=1).mean() * 100
        row[f'seq with {STATES_LABELS[s]}'] = round(has_state, 1)

    rows.append(row)

summary = pd.DataFrame(rows).set_index('Cluster').T

# --- Save LaTeX ---
summary.to_latex(PATH_CLUSTER_2/f'cluster_summary_{DISTANCE_METRIC_2}_{NUM_CLUSTERS_2}.tex', index=True, float_format="%.1f")








