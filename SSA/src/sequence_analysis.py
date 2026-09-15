import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import SEQUENCE_NAME, STATES_LABELS, TIME_UNIT, SEQ_LENGTH, DISTANCE_METRIC, MAX_CLUSTERS, NUM_CLUSTERS, COSTS,CUSTOM_COLORS, CLUSTERING, LINKAGE


ROOT = Path(__file__).resolve().parent.parent.parent

import sys
sys.setrecursionlimit(20000)

def run_sequence_analysis():
    """
    Executes State Sequence Analysis:
    1. Loads sequence data and instantiates SequenceData object.
    2. Computes sequence-level and cross-sectional characteristics.
    3. Calculates or loads pairwise dissimilarity matrix.
    4. Performs clustering and generates cluster profiling reports.
    """


    #import dataset with sequences
    df=pd.read_csv(ROOT/'SSA'/'datasets'/ f'{SEQUENCE_NAME}.csv')
    time_cols=[df.columns[i] for i in range(len(df.columns)) if df.columns[i].isdigit()]
    cov_cols=[col for col in df.columns if col not in time_cols]
    df_sequence=df[['pedianet_id'] + time_cols]


    #create SequenceData object for sequenzo
    from sequenzo import SequenceData

    sequence=SequenceData(
        data=df_sequence,
        time=time_cols,
        states=list(STATES_LABELS.keys()),
        labels=list(STATES_LABELS.values()),
        id_col='pedianet_id',
        custom_colors=CUSTOM_COLORS
    )
    sequence_ids=sequence.ids

    #print sequence object description
    print(sequence)

    ##### SEQUENCE VISUALIZATION #####
    from utils_SSA import sequence_visualization_tools, custom_visualization_tools
    sequence_visualization_tools(sequence, SEQUENCE_NAME, STATES_LABELS, CUSTOM_COLORS)
    custom_visualization_tools(df, time_cols, SEQUENCE_NAME, STATES_LABELS, CUSTOM_COLORS)


    ##### SEQUENCE CHARACTERISTICS #####
    from utils_SSA import compute_sequence_characteristics
    compute_sequence_characteristics(sequence, SEQUENCE_NAME, STATES_LABELS,CUSTOM_COLORS)


    ##### SEQUENCE DISSIMILARITY #####
    from dissimilarity import get_dissimilarity_matrix
    distance_matrix=get_dissimilarity_matrix(DISTANCE_METRIC, sequence, df_sequence,SEQUENCE_NAME,COSTS)


    ##### SEQUENCE CLUSTERING #####
    from utils_SSA import perform_clustering
    cluster_labels=perform_clustering(sequence, sequence_ids, distance_matrix)


    ##### CLUSTER ANALYSIS #####
    from utils_SSA import clustering_analysis, cluster_summary_table
    clustering_analysis(df_sequence,cluster_labels, distance_matrix)
    summary_table=cluster_summary_table(df_sequence, cluster_labels)


    return cluster_labels, distance_matrix, summary_table

if __name__ == "__main__":
    run_sequence_analysis()