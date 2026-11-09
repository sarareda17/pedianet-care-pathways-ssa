#MULTI CHANNEL CLUSTERING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

from config import SEASON, SEQUENCE_NAME, SEQUENCE_NAME_0, SEQUENCE_NAME_1, SEQUENCE_NAME_2, SEQUENCE_NAME_3, DISTANCE_METRIC,DISTANCE_METRIC_1, DISTANCE_METRIC_2, DISTANCE_METRIC_3, MAX_CLUSTERS, NUM_CLUSTERS, CLUSTERING, LINKAGE,NUM_CHANNELS,YEARS,SEASON
from config import W_0,W_1,W_2,W_3

def running_multichannel_clustering():
    """
    Executes multichannel clustering:
    1. Loads sequence data and instantiates SequenceData object.
    2. Computes sequence-level and cross-sectional characteristics.
    3. Calculates or loads pairwise dissimilarity matrix for each channel.
    4. Normalizes and combines distance matrices from multiple channels.
    5. Performs clustering and generates cluster profiling reports.
    """
    #load patients_sequence datasets to get ids
    if SEASON== True:
        patients_sequence=pd.read_csv(ROOT/'SSA'/'datasets'/ f"selected_patients_diagnoses_season_{YEARS}.csv")
    else:
        patients_sequence=pd.read_csv(ROOT/'SSA'/'datasets'/ f"selected_patients_diagnoses.csv")

    ids=patients_sequence['pedianet_id'].tolist()

    #load numpy distance matrices for the three channels and normalize them before summing them up

    # Normalize the distance matrix to the range [0, 1]
    def normalize_distance_matrix(dist_matrix):
        min_val = np.min(dist_matrix)
        max_val = np.max(dist_matrix)
        normalized_matrix = (dist_matrix - min_val) / (max_val - min_val)
        return normalized_matrix


    #Pooled distance matrix
    for i in range(NUM_CHANNELS):
        if i==0:
            print(f'Loading distance matrix for channel {i} with sequence {SEQUENCE_NAME_0} and distance metric {DISTANCE_METRIC}')
            pooled_dist=np.load(ROOT/'SSA'/'results'/'sequences'/'sequence_dissimilarities'/ f'{SEQUENCE_NAME_0}_{DISTANCE_METRIC}.npy')
            dist0=normalize_distance_matrix(pooled_dist)
            dist=W_0*dist0   
        elif i==1:
            print(f'Loading distance matrix for channel {i} with sequence {SEQUENCE_NAME_1} and distance metric {DISTANCE_METRIC_1}')
            pooled_dist+=np.load(ROOT/'SSA'/'results'/'sequences'/'sequence_dissimilarities'/ f'{SEQUENCE_NAME_1}_{DISTANCE_METRIC_1}.npy')
            dist1=normalize_distance_matrix(pooled_dist)
            dist+=W_1*dist1
        elif i==2:
            print(f'Loading distance matrix for channel {i} with sequence {SEQUENCE_NAME_2} and distance metric {DISTANCE_METRIC_2}')
            pooled_dist+=np.load(ROOT/'SSA'/'results'/'sequences'/'sequence_dissimilarities'/ f'{SEQUENCE_NAME_2}_{DISTANCE_METRIC_2}.npy')
            dist2=normalize_distance_matrix(pooled_dist)
            dist+=W_2*dist2
        elif i==3:
            print(f'Loading distance matrix for channel {i} with sequence {SEQUENCE_NAME_3} and distance metric {DISTANCE_METRIC_3}')
            pooled_dist+=np.load(ROOT/'SSA'/'results'/'sequences'/'sequence_dissimilarities'/ f'{SEQUENCE_NAME_3}_{DISTANCE_METRIC_3}.npy')
            dist3=normalize_distance_matrix(pooled_dist)
            dist+=W_3*dist3
        else:
            raise ValueError("Invalid channel index")


    from utils_SSA import hierarchical_clustering_custom, Kmedoids_clustering
    if CLUSTERING=='multi_channel_hierarchical':
        cluster_labels=hierarchical_clustering_custom(None,ids, pooled_dist)
    elif CLUSTERING=='multi_channel_k_medoids':
        cluster_labels=Kmedoids_clustering(None,ids,  pooled_dist)

    #load dataset (channel 0) with sequences for analysis
    df=pd.read_csv(ROOT/'SSA'/'datasets'/ f'{SEQUENCE_NAME_0}.csv')
    time_cols=[df.columns[i] for i in range(len(df.columns)) if df.columns[i].isdigit()]
    df_sequence=df[['pedianet_id'] + time_cols]

    #cluster analysis
    from utils_SSA import clustering_analysis, cluster_summary_table
    clustering_analysis(df_sequence, cluster_labels, pooled_dist) 
    cluster_summary_table(df_sequence, cluster_labels)

    return cluster_labels

if __name__ == "__main__":
    running_multichannel_clustering()