import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
from config import SEQ_LENGTH

########## MACRO-STATES DATADRIVEN #############
#looking for macro-states in the diagnoses dataset through clustering 

#I will use the selected_patients_diagnoses.csv dataset to select the patients cohort and the
# diagnoses_preprocessed.csv / macrodiagnoses_preprocessed.csv dataset because they contain:
#- patients with follow up of at least 4 years
#- patients with at least 1 diagnosis in the the 4 years of follow up
#- diagnoses within the 4 years follow up period
#- selected breath diagnoses associated to the defined state label 

patients=pd.read_csv(ROOT/'SSA'/'datasets'/'selected_patients_diagnoses.csv')

#diag=pd.read_csv(ROOT/'SSA'/'datasets'/'diagnoses_preprocessed.csv')
#if we want to use macrodiagnoses
#diag=pd.read_csv(ROOT/'SSA'/'datasets'/'macrodiagnoses_preprocessed.csv')
#if you want to use drugs
#diag=pd.read_csv(ROOT/'SSA'/'datasets'/'drugs_preprocessed.csv')
#if you want to use care setting 
diag=pd.read_csv(ROOT/'SSA'/'datasets'/'care_setting_preprocessed.csv')

#In this dataset patients can have multiple diagnoses in the same time unit 
#The  idea is to create combined state for every patient in every time unit considering all 
#the diagnoses in that time unit. The analysis of the combined states will be done through 
#clustering to find macro-states in the dataset

#Dataset creation
dataset_combined_states=pd.DataFrame(columns=['pedianet_id','time_unit','combined_state'])
#For each patient and for every time unit, I will create a vector containing all the diagnoses state in that time unit if 
#there are any, otherwise I will skip that time unit. Then I will add the vector containing the combined states in a new dataset.
for p in patients['pedianet_id']:
    patient_diag = diag[diag['pedianet_id'] == p]
    for t in patient_diag['time_unit'].unique():
        time_unit_diag = patient_diag[patient_diag['time_unit'] == t]
        if time_unit_diag.shape[0] > 1:
            combined_state=time_unit_diag['state'].astype(str).tolist()
            dataset_combined_states = pd.concat([dataset_combined_states, pd.DataFrame({'pedianet_id': [p], 'time_unit': [t], 'combined_state': [combined_state]})], ignore_index=True)

#Before doing clustering, let's simply count the number of unique combined states in the dataset to see how many there are
unique_combined_states = dataset_combined_states['combined_state'].apply(lambda x: tuple(sorted(x))).nunique()
print(f'Number of unique combined states: {unique_combined_states}')
#print the first 20 more frequent combined states in the dataset
print(dataset_combined_states['combined_state'].apply(lambda x: tuple(sorted(x))).value_counts().head(20))

#Now, let's remove duplicates of states (with sets) inside each combined state and count the number of unique combined states again
combined_states_no_duplicates = dataset_combined_states['combined_state'].apply(lambda x: tuple(sorted(set(x)))).tolist()
#Remove rows with one diagnosis in the combined state (i.e. combined states with only one diagnosis)
combined_states_no_duplicates = [s for s in combined_states_no_duplicates if len(s) > 1]
unique_combined_states_no_duplicates = dataset_combined_states['combined_state'].apply(lambda x: tuple(sorted(set(x)))).nunique()
print(f'Number of unique combined states after removing duplicates: {unique_combined_states_no_duplicates}')
#print the first 20 more frequent combined states in the dataset after removing duplicates
print(pd.Series(combined_states_no_duplicates).value_counts().head(20))






#Clustering analysis on the combined states to find macro-states
#Define the distance between the two combined states as the Jaccard distance between the two sets of diagnoses in the two combined states. The 
#Jaccard distance is defined as 1 - (size of intersection / size of union) of the two sets. The Jaccard distance is a measure of dissimilarity 
#between two sets, and it ranges from 0 to 1, where 0 means the two sets are identical and 1 means they have no elements in common.

def jaccard_distance(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - intersection / union if union != 0 else 0

#Create a distance matrix for the combined states

#use this line if you want to include duplicates
#data=dataset_combined_states['combined_state'].apply(lambda x: tuple(sorted(x)))
data=dataset_combined_states['combined_state'].apply(lambda x: tuple(sorted(set(x))))
data = data[data.apply(lambda x: len(x) > 1)] #remove combined states with only one diagnosis
combined_states_list = data.unique().tolist()

n = len(combined_states_list)   

dist = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist[i, j] = jaccard_distance(set(combined_states_list[i]), set(combined_states_list[j]))

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

condensed = squareform(dist, checks=False)
Z = linkage(condensed, method='ward')

plt.figure(figsize=(10, 7))
plt.title("Dendrogram of Combined States")
plt.xlabel("Combined States")
plt.ylabel("Distance")
dendrogram(Z, labels=combined_states_list, leaf_rotation=90, leaf_font_size=10)
plt.show()
plt.savefig(ROOT/'SSA'/'src'/'dendrogram_combined_states.png')

num_clusters = 4
clusters = fcluster(Z, t=num_clusters, criterion='maxclust')

#Create a new dataset with the combined states and their corresponding cluster labels
freq = data.value_counts()

cluster_df = pd.DataFrame({
    'combined_state': combined_states_list,
    'cluster': clusters,
    'frequency': [freq[s] for s in combined_states_list]
})

#Print the first 20 combined states in each cluster
for c in sorted(cluster_df['cluster'].unique()):
    members = cluster_df[cluster_df['cluster'] == c]['combined_state'].tolist()
    print(f"\n--- Cluster {c} ({len(members)} stati unici) ---")
    for m in members[:20]:
        print(m)

from collections import Counter

#Count the frequency of each diagnosis in each cluster (considering the frequency of each combined state in the dataset, 
#which means that the combined state is multiplied by the number of times it appears in the dataset)
for c in sorted(cluster_df['cluster'].unique()):
    subset = cluster_df[cluster_df['cluster'] == c]
    all_diag = []
    for state, f in zip(subset['combined_state'], subset['frequency']):
        all_diag.extend(list(state) * f)  
    counts = Counter(all_diag)
    print(f"\n--- Cluster {c} ---")
    for diag, count in counts.most_common(3):
        print(f"{diag}: {count}")