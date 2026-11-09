import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from config import SEQUENCE_NAME, CLUSTERING, LINKAGE,NUM_CLUSTERS,DISTANCE_METRIC
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
CLUSTER_FOLDER=ROOT/'SSA'/'results'/'sequences'/'sequence_clustering'/SEQUENCE_NAME/CLUSTERING
DISS_FOLDER=ROOT/'SSA'/'results'/'sequences'/'sequence_dissimilarities'

#import sequential dataset with diagnoses 
seq_df=pd.read_csv(ROOT/'SSA'/'datasets'/'seq_diagnoses_30_36.csv',index_col=0)
#import cluster labels
cluster_labels=pd.read_csv(CLUSTER_FOLDER/DISTANCE_METRIC/f'cluster_labels_LCS_NO_WEIGHTED.csv', index_col='Entity ID')

# 1. Merge cluster labels with sequential dataset
seq_df = seq_df.merge(
    cluster_labels, left_index=True, right_index=True, how="left"
)

# 2. Select only time columns and Cluster label
time_cols = [c for c in seq_df.columns if str(c).isdigit()]
seq_df = seq_df[time_cols + ["Cluster"]]

states_labels = {
    "URI": "Acute Upper Respiratory Infections",
    "FEV": "Fever",
    "COU": "Cough",
    "RNF": "Rhinopharyngitis",
    "OTI": "Otitis",
    "COV": "Covid",
    "BRN": "Bronchitis",
    "BSP": "Bronchospasm",
    "FLU": "Influenza",
    "LAR": "Laryngitis",
    "PHA": "Pharyngitis",
    "BRC": "Bronchiolitis",
    "PHT": "Pharyngotonsillitis",
    "OTH": "Other",
    "WHZ": "Wheezing",
    "TON": "Tonsillitis",
    "PNE": "Pneumonia",
    "AST": "Asthma",
    "SIN": "Sinusitis",
    "RHN": "Rhinitis",
    "NO": "No Diagnosis",
}

# 3. Compute the number of months spent in each state for each individual sequence
# Counts how many times each state appears across time_cols for each subject
state_counts_per_seq = pd.DataFrame(index=seq_df.index)
for state in states_labels.keys():
    state_counts_per_seq[state] = (seq_df[time_cols] == state).sum(axis=1)

# Attach Cluster label to patient-level counts
state_counts_per_seq["Cluster"] = seq_df["Cluster"]

# 4. Compute mean and standard deviation per cluster
mean_by_cluster = (
    state_counts_per_seq.groupby("Cluster")[list(states_labels.keys())]
    .mean()
    .T
)
std_by_cluster = (
    state_counts_per_seq.groupby("Cluster")[list(states_labels.keys())].std().T
)

# 5. Compute mean and standard deviation for the whole cohort ('All')
mean_all = state_counts_per_seq[list(states_labels.keys())].mean()
std_all = state_counts_per_seq[list(states_labels.keys())].std()

mean_by_cluster.insert(0, "All", mean_all)
std_by_cluster.insert(0, "All", std_all)

# 6. Format each cell as "Mean ± SD"
summary_df = pd.DataFrame(
    index=states_labels.keys(), columns=mean_by_cluster.columns
)
for col in mean_by_cluster.columns:
    summary_df[col] = (
        mean_by_cluster[col].round(2).astype(str)
        + " ± "
        + std_by_cluster[col].round(2).astype(str)
    )

# 7. Map state abbreviations to descriptive labels
summary_df.index = [states_labels[k] for k in summary_df.index]

# Add sequence count to column headers for completeness
n_patients = seq_df["Cluster"].value_counts().to_dict()
n_patients["All"] = len(seq_df)
summary_df.columns = [f"{col} (N={n_patients[col]})" for col in summary_df.columns]

# 8. Export LaTeX table
summary_df.to_latex(
    CLUSTER_FOLDER
    / DISTANCE_METRIC
    / "cluster_diagnosis_summary_LCS_NO_WEIGHTED.tex",
    index=True,
)





#import sequential dataset for drugs
drug_df=pd.read_csv(ROOT/'SSA'/'datasets'/'seq_drugs_30_36.csv',index_col=0)

#1.merge cluster labels with sequential dataset
drug_df = drug_df.merge(
    cluster_labels, left_index=True, right_index=True, how="left"
)

# 2. Select only time columns and Cluster label
time_cols = [c for c in seq_df.columns if str(c).isdigit()]
seq_df = seq_df[time_cols + ["Cluster"]]

states_labels = {
                'ANT': 'Antibiotics',               
                'NAS': 'Nasal preparations',        
                'THR': 'Throat preparations',       
                'OAD': 'Obstructive airway drugs',  
                'COC': 'Cough and cold preparations', 
                'ANH': 'Antihistamines',            
                'NO':  'No prescription',
            }

#3. Compute the number of months spent in each state for each individual sequence
# Counts how many times each state appears across time_cols for each subject
drug_counts_per_seq = pd.DataFrame(index=drug_df.index)
for drug in states_labels.keys():  
    drug_counts_per_seq[drug] = (drug_df[time_cols] == drug).sum(axis=1)

# Attach Cluster label to patient-level counts
drug_counts_per_seq["Cluster"] = drug_df["Cluster"]

#4. Compute mean and standard deviation per cluster
mean_by_cluster_drug = (
    drug_counts_per_seq.groupby("Cluster")[list(states_labels.keys())]
    .mean()
    .T
)
std_by_cluster_drug = (
    drug_counts_per_seq.groupby("Cluster")[list(states_labels.keys())].std().T
)

#5. Compute mean and standard deviation for the whole cohort ('All')
mean_all_drug = drug_counts_per_seq[list(states_labels.keys())].mean()
std_all_drug = drug_counts_per_seq[list(states_labels.keys())].std()

#6. Format each cell as "Mean ± SD"
summary_df_drug = pd.DataFrame(
    index=states_labels.keys(), columns=mean_by_cluster_drug.columns
)
for col in mean_by_cluster_drug.columns:
    summary_df_drug[col] = (
        mean_by_cluster_drug[col].round(2).astype(str)
        + " ± "
        + std_by_cluster_drug[col].round(2).astype(str)
    )   

#7. Map state abbreviations to descriptive labels
summary_df_drug.index = [states_labels[k] for k in summary_df_drug.index]

# Add sequence count to column headers for completeness
n_patients_drug = drug_df["Cluster"].value_counts().to_dict()
n_patients_drug["All"] = len(drug_df)
summary_df_drug.columns = [f"{col} (N={n_patients_drug[col]})" for col in summary_df_drug.columns]

#8. Export LaTeX table
summary_df_drug.to_latex(
    CLUSTER_FOLDER
    / DISTANCE_METRIC
    / "cluster_drug_summary_LCS_NO_WEIGHTED.tex",
    index=True,
)


#import sequential dataset for secondary care 
sec_df=pd.read_csv(ROOT/'SSA'/'datasets'/'seq_secondary_care_30_36.csv',index_col=0)

# 1. Merge cluster labels with sequential dataset
sec_df = sec_df.merge(
    cluster_labels, left_index=True, right_index=True, how="left"
)

# 2. Select only time columns and Cluster label
time_cols = [c for c in sec_df.columns if str(c).isdigit()]
sec_df = sec_df[time_cols + ["Cluster"]]

states_labels = {
    "HOS": "Hospitalization",
    "ER": "Emergency room access",
    "SV": "Specialist outpatient visit",
}

# 3. Indicator at patient level: True (1) if patient experienced >= 1 month in that state
has_event_per_seq = pd.DataFrame(index=sec_df.index)
for state in states_labels.keys():
    has_event_per_seq[state] = (sec_df[time_cols] == state).any(axis=1)

# Overall indicator: patient experienced at least one event in ANY secondary care state
has_event_per_seq["ANY"] = (
    sec_df[time_cols].isin(states_labels.keys())
).any(axis=1)

# Attach Cluster label
has_event_per_seq["Cluster"] = sec_df["Cluster"]

# Add full labels dictionary including composite row
all_labels = {"ANY": "Any Secondary Care Encounter", **states_labels}
state_keys = list(all_labels.keys())

# 4. Compute counts (n) and cluster sizes (N)
counts_by_cluster = (
    has_event_per_seq.groupby("Cluster")[state_keys].sum().T
)
cluster_sizes = sec_df["Cluster"].value_counts().to_dict()

# Compute counts and size for the full cohort ('All')
counts_all = has_event_per_seq[state_keys].sum()
cluster_sizes["All"] = len(sec_df)

counts_by_cluster.insert(0, "All", counts_all)

# 5. Format each cell as "n (perc%)"
summary_sec = pd.DataFrame(index=state_keys, columns=counts_by_cluster.columns)
for col in counts_by_cluster.columns:
    n_cluster = cluster_sizes[col]
    perc = (counts_by_cluster[col] / n_cluster) * 100
    summary_sec[col] = (
        counts_by_cluster[col].astype(int).astype(str)
        + " ("
        + perc.round(1).astype(str)
        + "%)"
    )

# 6. Map to descriptive labels and add cluster size to column names
summary_sec.index = [all_labels[k] for k in summary_sec.index]
summary_sec.columns = [f"{col} (N={cluster_sizes[col]})" for col in summary_sec.columns]

# 7. Export LaTeX table
summary_sec.to_latex(
    CLUSTER_FOLDER
    / DISTANCE_METRIC
    / "cluster_secondary_care_summary.tex",
    index=True,
)

