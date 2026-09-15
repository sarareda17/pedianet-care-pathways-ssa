import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sequenzo
from config import SEQUENCE_NAME, TIME_UNIT, SEQ_LENGTH, COSTS, STATES_LABELS, CUSTOM_COLORS, COL_ID, COL_STATE, COL_DATE,NUM_CHANNELS
from config import DISTANCE_METRIC,CLUSTERING, LINKAGE, MAX_CLUSTERS, NUM_CLUSTERS
from config import SEQUENCE_NAME_0,SEQUENCE_NAME_1,SEQUENCE_NAME_2, SEQUENCE_NAME_3, STATES_LABELS_1, STATES_LABELS_2 , STATES_LABELS_3
from config import CUSTOM_COLORS_1, CUSTOM_COLORS_2, CUSTOM_COLORS_3
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
ROOT_FOLDER_CHAR=ROOT/'SSA'/'results'/'sequences'/'sequence_characteristics'
ROOT_FOLDER_VISUAL=ROOT/'SSA'/'results'/'sequences'/'sequence_visualizations'
ROOT_FOLDER_CLUSTERING = ROOT / "SSA" / "results" /"sequences" / "sequence_clustering"


##### SEQUENCE CHARACTERISTICS #####

def compute_sequence_characteristics(sequence, SEQUENCE_NAME, STATES_LABELS,custom_colors):
    #create a folder in ROOT_FOLDER_CHAR for the sequence characteristics if it doesn't exist
    (ROOT_FOLDER_CHAR / SEQUENCE_NAME).mkdir(parents=True, exist_ok=True)

    #get sequence indexes
    sequence_indexes = sequence.ids

    ###################### NUMBER OF SUBSEQUENCES #######################
    #counts how many distinct subsequences exist in each sequence 
    from sequenzo.sequence_characteristics.simple_characteristics import get_subsequences_all_sequences
    subsequences_per_sequence_df=get_subsequences_all_sequences(sequence)
    #save distribution plot of subsequences per sequence
    plt.figure(figsize=(10,6))
    subsequences_per_sequence_df['Subseq.'].plot(kind='hist', bins=100)
    plt.title(f'Subsequences per Sequence: {SEQUENCE_NAME}')
    plt.xlabel('Number of distinct subsequences')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'subsequences_per_sequence.png')

    
    ###################### NUMBER OF TRANSITIONS ########################
    #counts how many state changes occur in each sequence 
    from sequenzo.sequence_characteristics.simple_characteristics import get_number_of_transitions
    
    transitions_per_sequence_df=get_number_of_transitions(sequence)
    #normalized by sequence length-1, to be between 0 and 1
    norm_transitions_per_sequence_df=get_number_of_transitions(sequence,norm=True)
    #weight each transition by the inverse probability of that transition occuring
    #(rare transitions contribute more to the total)
    #norm_transitions_per_sequence_df=get_number_of_transitions(sequence,pwight=True)

    #save distribution plot of transitions per sequence
    plt.figure(figsize=(10,6))
    transitions_per_sequence_df['Transitions'].value_counts().sort_index().plot(kind='bar')
    plt.title(f'Transitions per Sequence: {SEQUENCE_NAME}')
    plt.xlabel('Number of transitions')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'transitions_per_sequence.png')

    ######################## SPELL DURATION VARIANCE ####################### 
    #measures how evenly the durations of spells (runs of identical states) are distributed in each sequence 
    #variance of spell durations
    from sequenzo.sequence_characteristics.variance_of_spell_durations import get_spell_duration_variance
    spell_duration_variance_per_sequence_dict=get_spell_duration_variance(sequence)
    #it returns a dictionary with: Mean duration (mean duration of spells in each sequence as a df),
    #Variance (variance of spell durations in each sequence as a df) and Maximum variance (theoretical maximum
    #variance given the sequence length)

    #plot of mean duration of spells per sequence
    plt.figure(figsize=(10,6))
    spell_duration_variance_per_sequence_dict['meand'].head(50).plot(kind='scatter',x='ID', y='meand')
    plt.title(f'Mean Duration of Spells per Sequence: {SEQUENCE_NAME}')
    plt.xlabel('Sequence ID')
    plt.ylabel('Mean Duration of Spells')
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'mean_duration_of_spells_per_sequence.png')

    #plot of variance of spell durations per sequence
    plt.figure(figsize=(10,6))
    spell_duration_variance_per_sequence_dict['result'].head(50).plot(kind='scatter', x='ID', y='var_spell_dur')
    plt.title(f'Variance of Spell Durations per Sequence: {SEQUENCE_NAME}') 
    plt.xlabel('Sequence ID')
    plt.ylabel('Variance of Spell Durations')
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'variance_of_spell_durations_per_sequence.png')



    ######################## STATE FREQUENCIES PER SEQUENCE ################
    #calculates how often each state appears in each sequence 
    from sequenzo.sequence_characteristics.state_frequencies_and_entropy_per_sequence import get_state_freq_and_entropy_per_seq
    state_freq_per_sequence_df=get_state_freq_and_entropy_per_seq(sequence, prop=False)
    prop_state_freq_per_sequence_df=get_state_freq_and_entropy_per_seq(sequence, prop=True) #proportions instead of counts


    #example: plot state 'Nessuna diagnosi' counts distribution for all sequences
    plt.figure(figsize=(10,6))
    state_freq_per_sequence_df['No Diagnosis'].value_counts().sort_index().plot(kind='bar')
    plt.title(f'State NO Frequency per Sequence: {SEQUENCE_NAME}')
    plt.xlabel('Number of times state NO appears in sequence')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'state_NO_freq_per_sequence.png')

    ######################### WITHIN SEQUENCE ENTROPY ######################## 
    #measures how diverse the states are within each sequence
    #Entropy measures the uncertainty or diversity in a sequence. Higher entropy means the states are more evenly distributed,
    #while lower entropy means one state dominates (less diversity).
    from sequenzo.sequence_characteristics.within_sequence_entropy import get_within_sequence_entropy
    entropy_per_sequence_df=get_within_sequence_entropy(sequence, norm=False)
    norm_entropy_per_sequence_df=get_within_sequence_entropy(sequence, norm=True)
    #entropy is log(number of states in sequence), if norm=True we normalized by maximum possible entropy (log(number of states)) so that it is between 0 and 1

    #save distribution plot of entropy per sequence
    plt.figure(figsize=(10,6))
    norm_entropy_per_sequence_df['Entropy'].plot(kind='hist', bins=100)
    plt.title(f'Within Sequence Entropy Distribution: {SEQUENCE_NAME}')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'entropy_per_sequence.png')

    ########################## CROSS SECTIONAL ENTROPY ######################## 
    #analyzes how states are distributed across time points in the sequences computing:
    #State frequencies by time: how often each state appears at each time point 
    #Entropy by time: how diverse or concentrated the states are at each time point. Higher entropy 
    #means states are more evenly distributed; lower entropy means one state dominates.
    #Effective number of states: the equivalent number of equally probable states (exp(entropy)),
    #which is easier to interprete than raw entropy values.
    from sequenzo.sequence_characteristics.overall_cross_sectional_entropy import get_cross_sectional_entropy
    cross_sectional_entropy_dict=get_cross_sectional_entropy(sequence, norm=True,return_format='wide')
    #return_format can be 'tidy','wide' or 'dict'

    #state frequencies by time plot
    plt.figure(figsize=(12,8))
    cross_sectional_entropy_dict['Frequencies'].T.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 8),
    color=custom_colors  
    )
    plt.title(f'State Frequencies by Time: {SEQUENCE_NAME}')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.legend(title='States', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'state_frequencies_by_time.png')

    #entropy by time plot
    plt.figure(figsize=(10,6))
    cross_sectional_entropy_dict['per_time_entropy_norm'].plot(kind='line')
    plt.title(f'Cross-Sectional Entropy by Time: {SEQUENCE_NAME}')
    plt.xlabel('Time')
    plt.ylabel('Normalized Entropy')
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'cross_sectional_entropy_by_time.png')

    #effective number of states by time plot
    plt.figure(figsize=(10,6))
    cross_sectional_entropy_dict['Effective States'].plot(kind='line')
    plt.title(f'Effective Number of States by Time: {SEQUENCE_NAME}')
    plt.xlabel('Time')
    plt.ylabel('Effective Number of States')
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'effective_number_of_states_by_time.png')


    ########################### SEQUENCE COMPLEXITY ##########################
    #combines two aspects of a sequence:
    #Transitions: how often the sequence changes from one state to another 
    #Entropy: how diverse the states are within the sequence.
    from sequenzo.sequence_characteristics.complexity_index import get_complexity_index
    complexity_index_per_sequence_df=get_complexity_index(sequence)
    #it computes the normalized transitions and normalized entropy, then multiplies them to get the complexity index and save the sqrt.

    #save distribution plot of complexity index per sequence
    plt.figure(figsize=(10,6))
    complexity_index_per_sequence_df['Complexity Index'].plot(kind='hist', bins=100)
    plt.title(f'Complexity Index Distribution: {SEQUENCE_NAME}')
    plt.xlabel('Complexity Index')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'complexity_index_per_sequence.png')
    
    ######################### TURBOLENCE ##########################
    #measures how turbolent (irregular, unpredictable, or unstable) a sequence is. It captures two main aspects:
    #Diversity of subsequences (φ): how many distinct subsequences appear in the sequence
    #Variance of spell durations (s²): how evenly the lengths of consecutive spells (runs of the 
    #same state) are distributed.
    #Turbulence = log2( φ × (s²_max + 1) / (s² + 1) )
    from sequenzo.sequence_characteristics.turbulence import get_turbulence
    turbulence_per_sequence_df=get_turbulence(sequence, norm=False)
    #when norm= True, turbolence is normalized relative to a theoretical maximum
    norm_turbolence_per_sequence_df=get_turbulence(sequence, norm=True)

    #save distribution plot of turbulence per sequence
    plt.figure(figsize=(10,6))
    turbulence_per_sequence_df['Turbulence'].plot(kind='hist', bins=100)
    plt.title(f'Turbulence Distribution: {SEQUENCE_NAME}')
    plt.xlabel('Turbulence')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'turbulence_per_sequence.png')

    ####################### LONGITUDINAL CHARACTERISTICS  ########################
    #creates a horizontal bar chart showing four key sequence characteristics for selected sequences:
    #Transitions: how many times sequences change from one state to another
    #Entropy: how diverse/varied the sequences are
    #Turbolence: how chaotic or unpredictable the sequences are 
    #Complexity: how complex the overall pattern is 
    from sequenzo.sequence_characteristics.plot_characteristics import plot_longitudinal_characteristics
    plot_longitudinal_characteristics(
        sequence, k=10, save_as=str(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'longitudinal_characteristics.png')
    )
    #other input parameters: pick_ids (list of specific ids), order_by (default 'complexity')

    ####################### CROSS SECTIONAL CHARACTERISTICS #######################
    #visualizes how diverse the population is at each time point by
    #plotting cross-sectional entropy over time.
    from sequenzo.sequence_characteristics.plot_characteristics import plot_cross_sectional_characteristics
    plot_cross_sectional_characteristics(
        sequence, save_as=str(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'cross_sectional_entropy.png')
    )



    #group meaningful characteristics together in a dataframe to save
    #characteristics_by_sequence
    characteristics_by_sequence_df=pd.DataFrame({
        'number of subsequences': subsequences_per_sequence_df['Subseq.'].values,
        'number of transitions': transitions_per_sequence_df['Transitions'].values,
        'normalized number of transitions': norm_transitions_per_sequence_df['Transitions'].values,
        'mean duration of spells': spell_duration_variance_per_sequence_dict['meand']['meand'].values,
        'variance of spell durations': spell_duration_variance_per_sequence_dict['result']['var_spell_dur'].values,
        'entropy': entropy_per_sequence_df['Entropy'].values,
        'normalized entropy': norm_entropy_per_sequence_df['Entropy'].values,
        'complexity index': complexity_index_per_sequence_df['Complexity Index'].values,
        'turbulence': turbulence_per_sequence_df['Turbulence'].values,
        'normalized turbulence': norm_turbolence_per_sequence_df['Turbulence'].values
    })
    #characteristics_by_sequence_df = pd.concat(
    #[characteristics_by_sequence_df, prop_state_freq_per_sequence_df],
    #axis=1)
    characteristics_by_sequence_df.set_index(sequence_indexes, inplace=True)
    characteristics_by_sequence_df.to_csv(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'characteristics_by_sequence.csv', index=True)
    
    #characteristics_by_time
    characteristics_by_time_df=pd.DataFrame({
        'cross-sectional entropy': cross_sectional_entropy_dict['per_time_entropy_norm'],
        'effective number of states': cross_sectional_entropy_dict['Effective States']
    })
    characteristics_by_time_df = pd.concat(
    [characteristics_by_time_df, cross_sectional_entropy_dict['Frequencies']],
    axis=1
    )
    characteristics_by_time_df.to_csv(ROOT_FOLDER_CHAR / SEQUENCE_NAME / 'characteristics_by_time.csv', index=False)

    return




################## SEQUENCE VISUALIZATION TOOLS ###############


def sequence_visualization_tools(sequence, SEQUENCE_NAME, STATES_LABELS,custom_colors):
    #create a folder in ROOT_FOLDER_VISUAL for the sequence visualizations if it doesn't exist
    (ROOT_FOLDER_VISUAL / SEQUENCE_NAME).mkdir(parents=True, exist_ok=True)

    ############### SEQUENCE INDEX PLOT ############
    from sequenzo import plot_sequence_index
    plot_sequence_index(
        sequence,
         sort_by= "unsorted",
        save_as=str(ROOT_FOLDER_VISUAL / SEQUENCE_NAME / 'sequence_index.png'),
    )

    ############## STATE DISTRIBUTION PLOT ############
    from sequenzo import plot_state_distribution
    plot_state_distribution(
        sequence,
        save_as=str(ROOT_FOLDER_VISUAL / SEQUENCE_NAME / 'state_distribution.png'),
        show=False
    )

    ############## MOST FREQUENT SEQUENCES PLOT ############
    from sequenzo import plot_most_frequent_sequences
    plot_most_frequent_sequences(
        sequence,
        save_as=str(ROOT_FOLDER_VISUAL / SEQUENCE_NAME / 'most_frequent_sequences.png'),
    )

    ############## MODAL STATE PLOT ############
    from sequenzo import plot_modal_state
    plot_modal_state(
        sequence,
        save_as=str(ROOT_FOLDER_VISUAL / SEQUENCE_NAME / 'modal_state.png'),
    )   

    ############## MEAN TIME PLOT #############
    #plot_mean_time() draws a horizontal bar chart showing, for each state, the average amount 
    #of time spent in that state across all sequences. Optionally, it adds standard error bars 
    #so you can judge uncertainty.
    from sequenzo import plot_mean_time
    
    plot_mean_time(
        sequence,
        title=f'Mean Time in Each State: {SEQUENCE_NAME}',
        save_as=str(ROOT_FOLDER_VISUAL / SEQUENCE_NAME / 'mean_time.png'),
    )

    ################# TRANSITION MATRIX PLOT ##############
    #draws a heatmap of state-to-state transition rates: for each “from” state (rows), 
    #it shows the probability of moving to each “to” state at the next time step (columns). 
    #The diagonal cells are stay-in-the-same-state probabilities.
    #Under the hood, the transition matrix is row-normalized, so each row sums to 1.0 (i.e., probabilities).
    from sequenzo import plot_transition_matrix
    
    plot_transition_matrix(
        sequence,
        save_as=str(ROOT_FOLDER_VISUAL / SEQUENCE_NAME / 'transition_matrix.png'),
    )

    ################## RELATIVE FREQUENCY PLOT ##############
    #draws a two-panel figure to summarize sequence “typicality” by frequency groups:
    #Left: the medoid (most central) sequence for each frequency group, shown as a sequence-index strip.
    #Right: a box plot of dissimilarities (distance to the group medoid) for sequences in that group. 
    #It also prints a summary line with a pseudo R², F-statistic, and p-value that quantify how well groups are separated.
    from sequenzo import plot_relative_frequency
    # plot_relative_frequency(
    #     sequence,
    #     distance_matrix
    #     num_groups=20, 
    #     save_as=str(ROOT_FOLDER_VISUAL / SEQUENCE_NAME / 'relative_frequency.png'),
    # )
    #num_groups is default equal to 12 

    ################# SINGLE MEDOID PLOT ##############
    #identifies the single medoid sequence (the most central sequence that minimizes total distance to all others) 
    #and plots it as a colored strip across time. The title also reports the medoid’s coverage: the share of sequences 
    #that lie within a small distance threshold of this medoid.
    from sequenzo import plot_single_medoid
    # plot_single_medoid(
    #     sequence,
    #     save_as=str(ROOT_FOLDER_VISUAL / SEQUENCE_NAME / 'single_medoid.png'),
    # )



    return 

############### SEQUENCE CUSTOM VISUALIZATION TOOLS ############
def custom_visualization_tools(df, time_cols, SEQUENCE_NAME, STATES_LABELS,custom_colors):
    #create a folder in ROOT_FOLDER_VISUAL for the custom visualizations if it doesn't exist
    (ROOT_FOLDER_VISUAL / SEQUENCE_NAME).mkdir(parents=True, exist_ok=True)

    ######STATE DISTRIBUTION PLOT without considering NO state######

    # Map each state code to its color, then drop 'NO'
    state_to_color = dict(zip(STATES_LABELS.keys(), custom_colors))
    states_to_plot = [s for s in STATES_LABELS.keys() if s != 'NO']
    colors = [state_to_color[s] for s in states_to_plot]

    seq_df = df[time_cols]

    counts = seq_df.apply(lambda col: col.value_counts()).fillna(0)
    counts = counts.reindex(states_to_plot).fillna(0)        # only non-'NO' states, fixed order
    perc = counts.div(counts.sum(axis=0), axis=1) * 100      # normalize over diagnosed patients only

    # Stacked area plot
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stackplot(
        perc.columns,
        perc.values,
        labels=[STATES_LABELS[s] for s in states_to_plot],
        colors=colors,
        alpha=0.95,
    )
    ax.set_xlabel('Time')
    ax.set_ylabel('State distribution (%) — excluding "No diagnosis"')
    ax.set_ylim(0, 100)
    ax.margins(x=0)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_VISUAL/ SEQUENCE_NAME /'state_distribution_without_no.png', dpi=300)


    ###STATE DISTRIBUTION PLOT WITHOUT NO STATE ORDERED BY AVERAGE SIZE#####
    # Stacked area plot — ordered by average size
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.stackplot(
        perc.columns,
        perc.values,
        labels=[STATES_LABELS[s] for s in states_to_plot],
        colors=colors,
        alpha=0.95,
    )
    ax.set_xlabel('Time')
    ax.set_ylabel('State distribution (%) — excluding "No diagnosis"')
    ax.set_ylim(0, 100)
    ax.margins(x=0)

    # Reverse legend so it matches the visual top-to-bottom order of the stack
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_VISUAL/ SEQUENCE_NAME /'state_distribution_without_no_ordered.png', dpi=300)
    plt.show()
    
    #######LINE PLOT WITHOUT NO STATE ORDERED BY AVERAGE SIZE######
    # Line plot — one line per state, easier to compare magnitudes
    fig, ax = plt.subplots(figsize=(14, 7))
    for state, color in zip(states_to_plot, colors):
        ax.plot(perc.columns, perc.loc[state], label=STATES_LABELS[state], color=color, linewidth=2)

    ax.set_xlabel('Time')
    ax.set_ylabel('State distribution (%) — excluding "No diagnosis"')
    ax.margins(x=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(ROOT_FOLDER_VISUAL/ SEQUENCE_NAME /'state_distribution_lines.png', dpi=300)

    # Reorder states: smallest average at the bottom, largest at the top
    order = perc.mean(axis=1).sort_values(ascending=True).index.tolist()
    states_to_plot = order
    colors = [state_to_color[s] for s in states_to_plot]
    perc = perc.loc[states_to_plot]

    ########## ONE PLOT PER EACH STATE (subplots) ##########
    import math
    # For each state: count of patients in that state at each time point
    all_states = [s for s in STATES_LABELS.keys() if s != 'NO']

    n_states = len(all_states)
    n_cols = 3
    n_rows = (n_states + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    for i, state in enumerate(all_states):
        state_counts = seq_df.apply(lambda col: (col == state).sum())
        ax = axes[i]
        ax.plot(state_counts.index, state_counts.values,
                color=state_to_color[state], linewidth=2)
        ax.fill_between(state_counts.index, state_counts.values, 0,
                        color=state_to_color[state], alpha=0.20)
        ax.set_title(STATES_LABELS[state])
        ax.set_xlabel('Time')
        ax.set_ylabel('Count')
        ax.margins(x=0)
        ax.grid(True, alpha=0.3)

    # Nasconde gli assi vuoti se gli stati non riempiono tutta la griglia
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('State distribution over time', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(
        ROOT_FOLDER_VISUAL / SEQUENCE_NAME / 'state_distribution_per_state.png',
        dpi=300, bbox_inches='tight'
    )

    ########### STATE DISTRIBUTION TABLE BY SEQUENCES ##########
    #latex table where for each state the count and percentage on the total of sequences including the state are displayed
    state_distribution_table = pd.DataFrame(columns=['State', 'Count', 'Percentage'])
    for state in STATES_LABELS.keys():
        count = (seq_df == state).any(axis=1).sum()
        percentage = (count / len(seq_df)) * 100
        state_distribution_table = pd.concat([state_distribution_table, pd.DataFrame({'State': [STATES_LABELS[state]], 'Count': [count], 'Percentage': [percentage]})], ignore_index=True)
    state_distribution_table.to_latex(ROOT_FOLDER_VISUAL / SEQUENCE_NAME / 'state_distribution_table.tex', index=False, float_format="%.2f")

    ########### TOTAL COUNTS of cell in each state over all cells (time points*sequence) #############
    counts_total = seq_df.apply(pd.Series.value_counts).fillna(0).sum(axis=1)
    count_table = pd.DataFrame(columns=['State', 'Total Count', 'Percentage'])
    for state in STATES_LABELS.keys():
        total_count = counts_total.get(state, 0)
        percentage = (total_count / (len(seq_df) * len(time_cols))) * 100
        count_table = pd.concat([count_table, pd.DataFrame({'State': [STATES_LABELS[state]], 'Total Count': [total_count], 'Percentage': [percentage]})], ignore_index=True)
    count_table.to_latex(ROOT_FOLDER_VISUAL / SEQUENCE_NAME / 'state_total_count_table.tex', index=False, float_format="%.2f")



######### CLUSTERING UTILS FUNCTIONS #########

## 1) Function to compute inertia (sum of distances to medoid) for a given clustering, used for elbow method
def compute_inertia(distance_matrix, labels_df):

    dist = np.array(distance_matrix)
    cluster_ids = sorted(labels_df['Cluster'].unique())
    inertia = 0.0

    for cluster_id in cluster_ids:
        mask = labels_df['Cluster'].values == cluster_id
        indices = np.where(mask)[0]
        
        # submatrix of distances within the cluster
        sub_dist = dist[np.ix_(indices, indices)]
        
        # medoid = point with minimum average distance to all others in cluster
        medoid_local_idx = np.argmin(sub_dist.mean(axis=1))
        
        # sum of distances from all points to the medoid
        inertia += sub_dist[:, medoid_local_idx].sum()

    return inertia



######### CLUSTERING IMPLEMENTATION FUNCTIONS #########

## 0) interface function to call different clustering methods
def perform_clustering(sequence,sequence_ids, distance_matrix):
    #if labels are found in the folder then perform only cluster analysis
    if CLUSTERING in ['hierarchical_clustering','hierarchical_sequenzo']:
        label_path=ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f"cluster_labels_{DISTANCE_METRIC}_{NUM_CLUSTERS}_{LINKAGE}.csv"
        if label_path.exists():
            print(f"Cluster labels already exist for {SEQUENCE_NAME} with {CLUSTERING}, linkage {LINKAGE} and {DISTANCE_METRIC}. Performing only cluster analysis.")
            cluster_labels=pd.read_csv(label_path)
            return  cluster_labels
    else: 
        label_path=ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f"cluster_labels_{DISTANCE_METRIC}_{NUM_CLUSTERS}.csv"
        if label_path.exists():
            print(f"Cluster labels already exist for {SEQUENCE_NAME} with {CLUSTERING} and {DISTANCE_METRIC}. Performing only cluster analysis.")  
            cluster_labels=pd.read_csv(label_path)
            return  cluster_labels
    
    #otherwise perform clustering and cluster analysis
    if CLUSTERING=='hierarchical_sequenzo':
        cluster_labels=hierarchical_clustering(sequence, distance_matrix)
        return cluster_labels
    elif CLUSTERING=='hierarchical':
        cluster_labels=hierarchical_clustering_custom(sequence,sequence_ids, distance_matrix)
        return cluster_labels
    elif CLUSTERING=='k_medoids':
        cluster_labels=Kmedoids_clustering(sequence,sequence_ids, distance_matrix)
        return cluster_labels



## 1) Hierarchical clustering with sequenzo's built-in functions

def hierarchical_clustering(sequence,distance_matrix):
    #create a folder in ROOT_FOLDER_CLUSTERING for the sequence clustering if it doesn't exist
    (ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME/CLUSTERING/DISTANCE_METRIC).mkdir(parents=True, exist_ok=True)
    
    from sequenzo.clustering.hierarchical_clustering import Cluster, ClusterQuality, ClusterResults
    
    #Apply hierarchical clustering with Ward linkage method
    sequence_ids=sequence.ids
    cluster = Cluster(matrix=distance_matrix, entity_ids=sequence_ids, clustering_method=LINKAGE)

    #Plot the dendogram 
    cluster.plot_dendrogram(save_as=str(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f"dendrogram_{DISTANCE_METRIC}_{LINKAGE}.png"), title="Dendogram_" + DISTANCE_METRIC)

    #Analyze cluster quality indicators (CQI) to choose the optimal number of clusters
    #TThe function ClusterQuality computes  for each number of clusters k (from 2 to max_clusters) the following metrics:
    # Average Silhouette Width (ASW) : how well points fit within their assigned clusters
    # Average Silhouette Width (weighted) (ASWw): like ASW, but gives more weight to larger clusters
    # Hubert-Gamma(HG): whether clusters are balanced in size
    # Point-Biserial Correlation (PBC): correlation between distances and cluster labels
    # Calinski-Harabasz (CH): compares separation between clusters vs. compactness within clusters
    # R2: proportion of overall variation explained by the clustering
    # Hubert's C coefficient (HC): consistency of cluster splits in the hierarchy


    cluster_quality=ClusterQuality(cluster, max_clusters=MAX_CLUSTERS)
    cluster_quality.compute_cluster_quality_scores()
    #One-line summary of the best k per indicator
    #print(cluster_quality.get_cqi_table())
    #plot the CQI scores
    cluster_quality.plot_cqi_scores(save_as=str(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f"cqi_scores_{DISTANCE_METRIC}_{LINKAGE}.png"), title="Cluster Quality Indicators")
    #with norm the type of normalization can be specified (default is "zscore")
    #with metrics_list the metrics to plot can be specified (default is all)
    cluster_quality.plot_cqi_scores(metrics_list=["ASW","CH","PBC"],save_as=str(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f"cqi_scores_sil_ch_{DISTANCE_METRIC}_{LINKAGE}.png"), title="ASW and CH Cluster Quality Indicators")


    #export cluster results 
    cluster_results=ClusterResults(cluster)

    #compute inertia elbow plot to choose optimal number of clusters
    inertias = []
    
    for n in range(2, MAX_CLUSTERS + 1):
        labels = cluster_results.get_cluster_memberships(num_clusters=n)
        inertia = compute_inertia(distance_matrix, labels)
        inertias.append(inertia)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(2, MAX_CLUSTERS + 1), inertias, marker='o', linewidth=2)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia (sum of distances to medoid)')
    ax.set_title('Elbow plot')
    ax.set_xticks(range(2, MAX_CLUSTERS + 1))
    plt.tight_layout()
    plt.savefig(
        ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME / CLUSTERING / DISTANCE_METRIC / f'elbow_plot_{DISTANCE_METRIC}_{LINKAGE}.png',
        dpi=300, bbox_inches='tight'
    )

    
    #get cluster labels
    cluster_labels=cluster_results.get_cluster_memberships(num_clusters=NUM_CLUSTERS)
    #export cluster labels to csv
    cluster_labels.to_csv(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f"cluster_labels_{DISTANCE_METRIC}.csv", index=False)
    #check distribution of cluster labels 
    cluster_distribution=cluster_results.get_cluster_distribution(num_clusters=NUM_CLUSTERS)
    #plot cluster distribution
    cluster_results.plot_cluster_distribution(num_clusters=NUM_CLUSTERS, save_as=str(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f"cluster_distribution_{DISTANCE_METRIC}_{LINKAGE}_{NUM_CLUSTERS}.png"), title="Cluster Distribution")
    
    
    #plot sequenzo sequence index plot 
    from sequenzo import plot_sequence_index
    plot_sequence_index(
        sequence,
        group_dataframe=cluster_labels,
        group_column_name="Cluster",
        sort_by= "unsorted", #default is lexicographically
        save_as=str(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f'index_plot_by_cluster_{DISTANCE_METRIC}_{LINKAGE}_{NUM_CLUSTERS}.png')
    )
    

    #plot sequenzo state distribution plot for each cluster 
    from sequenzo import plot_state_distribution
    plot_state_distribution(
        sequence,
        group_dataframe=cluster_labels,
        group_column_name="Cluster",
        save_as=str(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f'state_distribution_by_cluster_{DISTANCE_METRIC}_{LINKAGE}_{NUM_CLUSTERS}.png'),
        show=False
    )

    return cluster_labels



## 2) Hierarchical clustering with custom implementation using scipy and fastcluster
def hierarchical_clustering_custom(sequence,sequence_ids, distance_matrix):
    #create a folder in ROOT_FOLDER_CLUSTERING for the sequence clustering if it doesn't exist
    (ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME/CLUSTERING/DISTANCE_METRIC).mkdir(parents=True, exist_ok=True)
    
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
    plt.savefig(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING/DISTANCE_METRIC/ f"dendrogram_{DISTANCE_METRIC}_{LINKAGE}.png", dpi=300, bbox_inches='tight')

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
        
    cluster_labels = fcluster(Z, t=NUM_CLUSTERS, criterion='maxclust')
    cluster_labels_df = pd.DataFrame({'Entity ID': sequence_ids, 'Cluster': cluster_labels})
    cluster_labels_df.to_csv(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f"cluster_labels_{DISTANCE_METRIC}.csv", index=False)


    if sequence is not None:
        from sequenzo import plot_sequence_index
        plot_sequence_index(
            sequence,
            group_dataframe=cluster_labels_df,
            group_column_name="Cluster",
            sort_by= "unsorted",
            save_as=str(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f'index_plot_by_cluster_{DISTANCE_METRIC}_{LINKAGE}_{NUM_CLUSTERS}.png')
        )

        from sequenzo import plot_state_distribution
        plot_state_distribution(
            sequence,
            group_dataframe=cluster_labels_df,
            group_column_name="Cluster",
            save_as=str(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING/DISTANCE_METRIC/ f'state_distribution_by_cluster_{DISTANCE_METRIC}_{LINKAGE}_{NUM_CLUSTERS}.png'),
            show=False
        )
    return cluster_labels_df


def Kmedoids_clustering(sequence, sequence_ids, distance_matrix):
    #create a folder in ROOT_FOLDER_CLUSTERING for the sequence clustering if it doesn't exist
    (ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME/CLUSTERING/DISTANCE_METRIC).mkdir(parents=True, exist_ok=True)


    #implement K-medoids clustering using the distance matrix
    import kmedoids

    result = kmedoids.fasterpam(distance_matrix, NUM_CLUSTERS)
    labels    = result.labels     
    medoids   = result.medoids   

    pd_cluster_labels=pd.DataFrame({'Entity ID': sequence_ids, 'Cluster': labels})
    #export cluster labels to csv
    pd_cluster_labels.to_csv(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f"cluster_labels_{DISTANCE_METRIC}.csv", index=False)


    if sequence is not None:
        #plot sequence index plot
        from sequenzo import plot_sequence_index
        plot_sequence_index(
            sequence,
            group_dataframe=pd_cluster_labels,
            group_column_name="Cluster",
            sort_by= "unsorted", 
            save_as=str(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f'index_plot_by_cluster_{DISTANCE_METRIC}_{NUM_CLUSTERS}.png')
        )

        #plot state distribution plot for each cluster 
        from sequenzo import plot_state_distribution
        plot_state_distribution(
            sequence,
            group_dataframe=pd_cluster_labels,
            group_column_name="Cluster",
            save_as=str(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME /CLUSTERING /DISTANCE_METRIC/ f'state_distribution_by_cluster_{DISTANCE_METRIC}_{NUM_CLUSTERS}.png'),
            show=False
        )


    return pd_cluster_labels
    
    
######################################################################################################
################################### CLUSTERING ANALYSIS ##############################################
######################################################################################################
        
def clustering_analysis(df_sequence,cluster_labels,distance_matrix):
    FOLDER=ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME / CLUSTERING / DISTANCE_METRIC

    #create dataset with sequence of states and cluster labels
    df=df_sequence.merge(cluster_labels, left_on='pedianet_id', right_on='Entity ID', how='inner')
    time_cols=[df.columns[i] for i in range(len(df.columns)) if df.columns[i].isdigit()]


    # 1). STATE DISTRIBUTION PLOT BY CLUSTER (STACKED AREA PLOT) WITHOUT NO STATE
    state_to_color = dict(zip(STATES_LABELS.keys(), CUSTOM_COLORS))
    states_to_plot = [s for s in STATES_LABELS.keys() if s != 'NO']
    colors = [state_to_color[s] for s in states_to_plot]

    n_clusters = df['Cluster'].nunique()  
    n_cols = 3
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, cluster_id in enumerate(sorted(df['Cluster'].unique())):
        ax = axes[i]
        cluster_df = df[df['Cluster'] == cluster_id][time_cols]

        counts = cluster_df.apply(lambda col: col.value_counts()).fillna(0)
        counts = counts.reindex(states_to_plot).fillna(0)
        perc = counts.div(counts.sum(axis=0), axis=1) * 100

        ax.stackplot(
            perc.columns,
            perc.values,
            labels=[STATES_LABELS[s] for s in states_to_plot],
            colors=colors,
            alpha=0.95,
        )
        ax.set_title(f'Cluster {cluster_id} (n={len(cluster_df)})')
        ax.set_xlabel('Time')
        ax.set_ylabel('State distribution (%)')
        ax.set_ylim(0, 100)
        ax.margins(x=0)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc='lower center', ncol=len(states_to_plot),
            bbox_to_anchor=(0.5, -0.02), frameon=False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('State distribution per cluster (excluding "state NO")', fontsize=16)
    plt.tight_layout()
    plt.savefig(FOLDER/ f'state_NO_distribution_{DISTANCE_METRIC}_{NUM_CLUSTERS}.png',
                dpi=300, bbox_inches='tight')

    # 2). HEATMAP WITH COUNTS OF THE EVENTS IN EACH STATE PER CLUSTER 
    #(on each row a state (including "NO"), on each column a cluster and the value is the number of events of that state in the sequences in that cluster)
    #(color intensity is applied by row to highlight which states are more common in each cluster)
    melted = (
        df[['Cluster'] + time_cols]
        .melt(id_vars='Cluster', value_name='state')
        .groupby(['state', 'Cluster'])
        .size()
        .unstack(level='Cluster')
        .reindex(STATES_LABELS.keys())
        .fillna(0)
        .astype(int)
    )

    # Rank each row: 1=max, 2=second, 3=third, rest=0
    rank_matrix = melted.rank(axis=1, method='min', ascending=False)
    color_matrix = rank_matrix.map(lambda r: 1.0 if r == 1 else (0.6 if r == 2 else (0.3 if r == 3 else 0.0)))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        color_matrix,
        annot=melted,
        fmt='d',
        cmap='Blues',
        vmin=0, vmax=1,
        cbar=False,
        ax=ax,
        linewidths=0.5,
    )

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=plt.cm.Blues(1.0), label='1st highest'),
        Patch(facecolor=plt.cm.Blues(0.6), label='2nd highest'),
        Patch(facecolor=plt.cm.Blues(0.3), label='3rd highest'),
        Patch(facecolor=plt.cm.Blues(0.0), label='Other'),
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1), loc='upper left', frameon=False)

    ax.set_title(f'Count of events in each state per cluster: {SEQUENCE_NAME}')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('State')
    plt.tight_layout()
    plt.savefig(FOLDER / f'state_cluster_counts_{DISTANCE_METRIC}_{NUM_CLUSTERS}.png', dpi=300, bbox_inches='tight')

    #3). HEATMAP WITH PROPORTION OF THE EVENTS IN EACH STATE PER CLUSTER
    #(on each row a state (including "NO"), on each column a cluster and the value is the proportion of events of that state in the sequences in that cluster (normalized by row))
    #(color intensity is applied by row to highlight which states are more common in each cluster)
    melted_perc = melted.div(melted.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(10, 6))
    sns.heatmap(melted_perc, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Percentage'})
    plt.title(f'Proportion of events in each state per cluster: {SEQUENCE_NAME}')
    plt.xlabel('Cluster')
    plt.ylabel('State')
    plt.tight_layout()
    plt.savefig(FOLDER / f'state_cluster_perc_{DISTANCE_METRIC}_{NUM_CLUSTERS}.png', dpi=300, bbox_inches='tight')

    #4). SEQUENCE INDEX PLOT BY CLUSTER, but keep only the 20 sequences closer to the medoid of each cluster using the distance 
    #matrix in input 
    import matplotlib.patches as mpatches
    # Find medoid of each cluster and select 20 closest sequences to it
    n_closest = 100
    state_to_color = dict(zip(STATES_LABELS.keys(), CUSTOM_COLORS))

    n_clusters = df['Cluster'].nunique()
    n_cols = 3
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    # distance_matrix is indexed by pedianet_id (same order as df_sequence)
    ids = df['pedianet_id'].values

    for i, cluster_id in enumerate(sorted(df['Cluster'].unique())):
        ax = axes[i]

        cluster_mask = df['Cluster'] == cluster_id
        cluster_ids = df.loc[cluster_mask, 'pedianet_id'].values

        # Get submatrix of distances within the cluster
        cluster_idx = [list(ids).index(pid) for pid in cluster_ids]
        sub_matrix = distance_matrix[np.ix_(cluster_idx, cluster_idx)]

        # Medoid = sequence with minimum sum of distances to all others in the cluster
        medoid_local_idx = np.argmin(sub_matrix.sum(axis=1))

        # Select n_closest sequences to the medoid (including medoid itself)
        distances_from_medoid = sub_matrix[medoid_local_idx]
        closest_local_idx = np.argsort(distances_from_medoid)[:n_closest]
        closest_ids = cluster_ids[closest_local_idx]

        # Get sequences for those ids
        seq_df = df[df['pedianet_id'].isin(closest_ids)][time_cols].reset_index(drop=True)

        # Sequence index plot: one row per sequence, colored by state
        time_points = list(range(len(time_cols)))

        for row_idx, (_, row) in enumerate(seq_df.iterrows()):
            for t, state in enumerate(row):
                color = state_to_color.get(state, 'grey')
                ax.barh(
                    row_idx, 1,
                    left=t,
                    color=color,
                    edgecolor='none',
                    height=0.8,
                )

        medoid_global_id = cluster_ids[medoid_local_idx]
        medoid_row_idx = list(closest_ids).index(medoid_global_id) if medoid_global_id in closest_ids else None
        if medoid_row_idx is not None:
            ax.axhline(medoid_row_idx, color='red', linewidth=1.2, linestyle='--', label='Medoid')

        ax.set_title(f'Cluster {cluster_id} — {n_closest} sequences closest to medoid')
        ax.set_xlabel('Time')
        ax.set_ylabel('Sequence')
        ax.set_xlim(0, len(time_cols))
        ax.set_ylim(-0.5, len(seq_df) - 0.5)
        ax.set_yticks([])

    # Legend for states
    legend_patches = [mpatches.Patch(color=state_to_color[s], label=STATES_LABELS[s]) for s in STATES_LABELS.keys()]
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(STATES_LABELS),
            bbox_to_anchor=(0.5, -0.02), frameon=False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Sequence index plot — {n_closest} closest to medoid per cluster: {SEQUENCE_NAME}', fontsize=16)
    plt.tight_layout()
    plt.savefig(FOLDER / f'top{n_closest}_index_plot_{DISTANCE_METRIC}_{NUM_CLUSTERS}.png', dpi=300, bbox_inches='tight')


    # SEQUENCE INDEX PLOT BY CLUSTERbut for each cluster select 30 random sequences and plot them in the sequence index plot
    n_samples = 30
    random_seed = 42
    state_to_color = dict(zip(STATES_LABELS.keys(), CUSTOM_COLORS))

    n_clusters = df['Cluster'].nunique()
    n_cols = 3
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, cluster_id in enumerate(sorted(df['Cluster'].unique())):
        ax = axes[i]

        # Filter the DataFrame for the current cluster
        cluster_df = df[df['Cluster'] == cluster_id]

        # Sample up to n_samples sequences randomly
        sample_size = min(n_samples, len(cluster_df))
        sampled_df = cluster_df.sample(n=sample_size, random_state=random_seed)

        # Extract only the sequence time columns
        seq_df = sampled_df[time_cols].reset_index(drop=True)

        # Sequence index plot: horizontal bars colored by state
        for row_idx, (_, row) in enumerate(seq_df.iterrows()):
            for t, state in enumerate(row):
                color = state_to_color.get(state, 'grey')
                ax.barh(
                    row_idx,
                    1,
                    left=t,
                    color=color,
                    edgecolor='none',
                    height=0.8,
                )

        ax.set_title(f'Cluster {cluster_id} — {len(seq_df)} random sequences')
        ax.set_xlabel('Time')
        ax.set_ylabel('Sequence')
        ax.set_xlim(0, len(time_cols))
        ax.set_ylim(-0.5, len(seq_df) - 0.5)
        ax.set_yticks([])

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Shared legend for states
    legend_patches = [
        mpatches.Patch(color=state_to_color[s], label=STATES_LABELS[s])
        for s in STATES_LABELS.keys()
    ]
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=len(STATES_LABELS),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.suptitle(
        f'Sequence index plot — {n_samples} random sequences per cluster: {SEQUENCE_NAME}',
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(
        FOLDER / f'random_{n_samples}_index_plot_{DISTANCE_METRIC}_{NUM_CLUSTERS}.png',
        dpi=300,
        bbox_inches='tight',
    ) 

    #SEQUENCE INDEX PLOT BY CLUSTER but for each cluster select 10 sequences
    n_closest = 10
    state_to_color = dict(zip(STATES_LABELS.keys(), CUSTOM_COLORS))

    n_clusters = df['Cluster'].nunique()
    n_cols = 3
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    # distance_matrix is indexed by pedianet_id (same order as df_sequence)
    ids = df['pedianet_id'].values

    for i, cluster_id in enumerate(sorted(df['Cluster'].unique())):
        ax = axes[i]

        cluster_mask = df['Cluster'] == cluster_id
        cluster_ids = df.loc[cluster_mask, 'pedianet_id'].values

        # Get submatrix of distances within the cluster
        cluster_idx = [list(ids).index(pid) for pid in cluster_ids]
        sub_matrix = distance_matrix[np.ix_(cluster_idx, cluster_idx)]

        # Medoid = sequence with minimum sum of distances to all others in the cluster
        medoid_local_idx = np.argmin(sub_matrix.sum(axis=1))

        # Select n_closest sequences to the medoid (including medoid itself)
        distances_from_medoid = sub_matrix[medoid_local_idx]
        closest_local_idx = np.argsort(distances_from_medoid)[:n_closest]
        closest_ids = cluster_ids[closest_local_idx]

        # Get sequences for those ids
        seq_df = df[df['pedianet_id'].isin(closest_ids)][time_cols].reset_index(drop=True)

        # Sequence index plot: one row per sequence, colored by state
        time_points = list(range(len(time_cols)))

        for row_idx, (_, row) in enumerate(seq_df.iterrows()):
            for t, state in enumerate(row):
                color = state_to_color.get(state, 'grey')
                ax.barh(
                    row_idx, 1,
                    left=t,
                    color=color,
                    edgecolor='none',
                    height=0.8,
                )

        medoid_global_id = cluster_ids[medoid_local_idx]
        medoid_row_idx = list(closest_ids).index(medoid_global_id) if medoid_global_id in closest_ids else None
        if medoid_row_idx is not None:
            ax.axhline(medoid_row_idx, color='red', linewidth=1.2, linestyle='--', label='Medoid')

        ax.set_title(f'Cluster {cluster_id} — {n_closest} sequences closest to medoid')
        ax.set_xlabel('Time')
        ax.set_ylabel('Sequence')
        ax.set_xlim(0, len(time_cols))
        ax.set_ylim(-0.5, len(seq_df) - 0.5)
        ax.set_yticks([])

    # Legend for states
    legend_patches = [mpatches.Patch(color=state_to_color[s], label=STATES_LABELS[s]) for s in STATES_LABELS.keys()]
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(STATES_LABELS),
            bbox_to_anchor=(0.5, -0.02), frameon=False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Sequence index plot — {n_closest} closest to medoid per cluster: {SEQUENCE_NAME}', fontsize=16)
    plt.tight_layout()
    plt.savefig(FOLDER / f'top{n_closest}_index_plot_{DISTANCE_METRIC}_{NUM_CLUSTERS}.png', dpi=300, bbox_inches='tight')


    # SEQUENCE INDEX PLOT BY CLUSTER but for each cluster select 30 random sequences and plot them in the sequence index plot
    n_samples = 10
    random_seed = 42
    state_to_color = dict(zip(STATES_LABELS.keys(), CUSTOM_COLORS))

    n_clusters = df['Cluster'].nunique()
    n_cols = 3
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, cluster_id in enumerate(sorted(df['Cluster'].unique())):
        ax = axes[i]

        # Filter the DataFrame for the current cluster
        cluster_df = df[df['Cluster'] == cluster_id]

        # Sample up to n_samples sequences randomly
        sample_size = min(n_samples, len(cluster_df))
        sampled_df = cluster_df.sample(n=sample_size, random_state=random_seed)

        # Extract only the sequence time columns
        seq_df = sampled_df[time_cols].reset_index(drop=True)

        # Sequence index plot: horizontal bars colored by state
        for row_idx, (_, row) in enumerate(seq_df.iterrows()):
            for t, state in enumerate(row):
                color = state_to_color.get(state, 'grey')
                ax.barh(
                    row_idx,
                    1,
                    left=t,
                    color=color,
                    edgecolor='none',
                    height=0.8,
                )

        ax.set_title(f'Cluster {cluster_id} — {len(seq_df)} random sequences')
        ax.set_xlabel('Time')
        ax.set_ylabel('Sequence')
        ax.set_xlim(0, len(time_cols))
        ax.set_ylim(-0.5, len(seq_df) - 0.5)
        ax.set_yticks([])

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Shared legend for states
    legend_patches = [
        mpatches.Patch(color=state_to_color[s], label=STATES_LABELS[s])
        for s in STATES_LABELS.keys()
    ]
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=len(STATES_LABELS),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.suptitle(
        f'Sequence index plot — {n_samples} random sequences per cluster: {SEQUENCE_NAME}',
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(
        FOLDER / f'random_{n_samples}_index_plot_{DISTANCE_METRIC}_{NUM_CLUSTERS}.png',
        dpi=300,
        bbox_inches='tight',
    )            

        

    #5). STATE DISTRIBUTION PLOT ACROSS CHANNELS BY CLUSTER (STACKED AREA PLOT)

    # ── Carica dati e configura per ogni channel ──────────────────────────────────
    channel_configs = []
    for i in range(NUM_CHANNELS):
        seq_name     = [SEQUENCE_NAME_0,   SEQUENCE_NAME_1, SEQUENCE_NAME_2,   SEQUENCE_NAME_3][i]
        states_labels = [STATES_LABELS, STATES_LABELS_1, STATES_LABELS_2, STATES_LABELS_3][i]
        colors        = [CUSTOM_COLORS, CUSTOM_COLORS_1, CUSTOM_COLORS_2, CUSTOM_COLORS_3][i]
        csv_path      = ROOT / 'SSA' / 'datasets' / f'{seq_name}.csv'
        if not csv_path.exists():
            print(f"[WARNING] Channel {i}: file not found ({csv_path}), skipping.")
            continue
        channel_configs.append({
            'df':            pd.read_csv(csv_path),
            'states_all':    list(states_labels.keys()),
            'states_labels': states_labels,
            'colors':        colors,
        })

    # Pre-calcola time_cols per ogni channel
    for ch_idx, cfg in enumerate(channel_configs):
        cfg['time_cols'] = [c for c in cfg['df'].columns if c.isdigit()]
        cfg['colors_all'] = [
            dict(zip(cfg['states_all'], cfg['colors']))[s]
            for s in cfg['states_all']
        ]

    # ── Layout ────────────────────────────────────────────────────────────────────
    cluster_ids  = sorted(df['Cluster'].unique())
    n_clusters   = len(cluster_ids)
    n_cols       = 3
    n_block_rows = (n_clusters + n_cols - 1) // n_cols
    n_grid_rows  = n_block_rows * NUM_CHANNELS  # NUM_CHANNELS righe per cluster block

    fig, axes = plt.subplots(n_grid_rows, n_cols, figsize=(6 * n_cols, 4 * n_grid_rows))
    axes = axes.flatten()

    for idx, cluster_id in enumerate(cluster_ids):
        col_pos   = idx % n_cols
        block_row = idx // n_cols
        base      = block_row * n_cols * NUM_CHANNELS

        patient_ids = df[df['Cluster'] == cluster_id][COL_ID]

        for ch_idx, cfg in enumerate(channel_configs):
            ax = axes[base + n_cols * ch_idx + col_pos]

            # Filtra per cluster: ch0 usa df direttamente, gli altri filtrano per patient_ids
            if ch_idx == 0:
                cluster_df = df[df['Cluster'] == cluster_id][cfg['time_cols']]
            else:
                cluster_df = cfg['df'][cfg['df'][COL_ID].isin(patient_ids)][cfg['time_cols']]

            counts = cluster_df.apply(lambda col: col.value_counts()).fillna(0)
            counts = counts.reindex(cfg['states_all']).fillna(0)
            perc   = counts.div(counts.sum(axis=0), axis=1) * 100

            ax.stackplot(
                perc.columns, perc.values,
                labels=[cfg['states_labels'][s] for s in cfg['states_all']],
                colors=cfg['colors_all'],
                alpha=0.95,
            )

            title_suffix = f'(n={len(cluster_df)}) — Ch.{ch_idx}' if ch_idx == 0 else f'— Ch.{ch_idx}'
            ax.set_title(f'Cluster {cluster_id} {title_suffix}')
            ax.set_ylabel('State (%)')
            ax.set_ylim(0, 100)
            ax.margins(x=0)

            # xlabel solo sull'ultimo channel
            if ch_idx == NUM_CHANNELS - 1:
                ax.set_xlabel('Time')

    # ── Legends ───────────────────────────────────────────────────────────────────
    for ch_idx, cfg in enumerate(channel_configs):
        handles, labels = axes[n_cols * ch_idx].get_legend_handles_labels()
        y_offset = -0.02 - ch_idx * 0.03
        fig.legend(
            handles, labels,
            loc='lower center',
            ncol=len(cfg['states_all']),
            bbox_to_anchor=(0.5, y_offset),
            frameon=False,
            title=f'Ch.{ch_idx}',
        )

    # ── Nascondi subplot vuoti ─────────────────────────────────────────────────────
    for block_row in range(n_block_rows):
        for col_pos in range(n_cols):
            flat_idx = block_row * n_cols + col_pos
            if flat_idx >= n_clusters:
                base = block_row * n_cols * NUM_CHANNELS
                for ch_idx in range(NUM_CHANNELS):
                    axes[base + ch_idx * n_cols + col_pos].set_visible(False)

    fig.suptitle(
        f'State distribution per cluster — {NUM_CHANNELS} channel(s)',
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(
        ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME / CLUSTERING / DISTANCE_METRIC /
        f'cluster_state_distribution_{NUM_CHANNELS}ch_{DISTANCE_METRIC}_{NUM_CLUSTERS}.png',
        dpi=300, bbox_inches='tight',
    )


    #6). SEQUENCE INDEX PLOT ACROSS CHANNELS BY CLUSTER
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    def plot_index(ax, seq_df, color_map, states_order, title, xlabel=False):
        seq_sorted = seq_df
        state_to_int = {s: i for i, s in enumerate(states_order)}
        mat = seq_sorted.apply(lambda row: row.map(state_to_int)).fillna(0).values.astype(int)
        colors_ordered = [color_map.get(s, '#CCCCCC') for s in states_order]
        cmap = mcolors.ListedColormap(colors_ordered)
        bounds = np.arange(-0.5, len(states_order), 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm, interpolation='none', origin='upper')
        ax.set_title(title)
        ax.set_ylabel('Sequences')
        ax.set_yticks([])
        if xlabel:
            ax.set_xlabel('Time')

    def make_legend_patches(states_labels, color_map):
        return [mpatches.Patch(color=color_map[s], label=states_labels[s])
                for s in states_labels.keys()]

    # ── Configure channels ────────────────────────────────────────────────────────
    channel_configs = []
    for i in range(NUM_CHANNELS):
        seq_name     = [SEQUENCE_NAME_0,   SEQUENCE_NAME_1, SEQUENCE_NAME_2,   SEQUENCE_NAME_3][i]
        states_labels = [STATES_LABELS, STATES_LABELS_1, STATES_LABELS_2, STATES_LABELS_3][i]
        colors        = [CUSTOM_COLORS, CUSTOM_COLORS_1, CUSTOM_COLORS_2, CUSTOM_COLORS_3][i]
        csv_path      = ROOT / 'SSA' / 'datasets' / f'{seq_name}.csv'
        if not csv_path.exists():
            print(f"[WARNING] Channel {i}: file not found ({csv_path}), skipping.")
            continue
        channel_configs.append({
            'df':            pd.read_csv(csv_path),
            'states_all':    list(states_labels.keys()),
            'states_labels': states_labels,
            'colors':        colors,
        })

    # Pre-compute time_cols and color_map for each channel
    for cfg in channel_configs:
        cfg['time_cols'] = [c for c in cfg['df'].columns if c.isdigit()]
        cfg['color_map'] = dict(zip(cfg['states_all'], cfg['colors']))

    # ── Layout ────────────────────────────────────────────────────────────────────
    cluster_ids  = sorted(df['Cluster'].unique())
    n_clusters   = len(cluster_ids)
    n_cols       = 3
    n_block_rows = (n_clusters + n_cols - 1) // n_cols
    n_grid_rows  = n_block_rows * NUM_CHANNELS

    fig, axes = plt.subplots(n_grid_rows, n_cols, figsize=(6 * n_cols, 5 * n_grid_rows))
    axes = axes.flatten()

    for idx, cluster_id in enumerate(cluster_ids):
        col_pos   = idx % n_cols
        block_row = idx // n_cols
        base      = block_row * n_cols * NUM_CHANNELS

        patient_ids = df[df['Cluster'] == cluster_id][COL_ID]
        n = len(patient_ids)

        for ch_idx, cfg in enumerate(channel_configs):
            ax = axes[base + n_cols * ch_idx + col_pos]

            seq_df = cfg['df'][cfg['df'][COL_ID].isin(patient_ids)][cfg['time_cols']].reset_index(drop=True)

            title_suffix = f'(n={n}) — Ch.{ch_idx}' if ch_idx == 0 else f'— Ch.{ch_idx}'
            is_last = (ch_idx == NUM_CHANNELS - 1)

            plot_index(ax, seq_df, cfg['color_map'], cfg['states_all'],
                       f'Cluster {cluster_id} {title_suffix}', xlabel=is_last)

    # ── Legends ───────────────────────────────────────────────────────────────────
    for ch_idx, cfg in enumerate(channel_configs):
        y_offset = -0.02 - ch_idx * 0.03
        fig.legend(
            handles=make_legend_patches(cfg['states_labels'], cfg['color_map']),
            loc='lower center',
            ncol=len(cfg['states_all']),
            bbox_to_anchor=(0.5, y_offset),
            frameon=False,
            title=f'Ch.{ch_idx}',
        )

    # ── Hide empty subplots ───────────────────────────────────────────────────────
    for block_row in range(n_block_rows):
        for col_pos in range(n_cols):
            flat_idx = block_row * n_cols + col_pos
            if flat_idx >= n_clusters:
                base = block_row * n_cols * NUM_CHANNELS
                for ch_idx in range(NUM_CHANNELS):
                    axes[base + ch_idx * n_cols + col_pos].set_visible(False)

    fig.suptitle(
        f'Sequence index plot per cluster — {NUM_CHANNELS} channel(s)',
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(
        ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME / CLUSTERING / DISTANCE_METRIC /
        f'index_plot_{NUM_CHANNELS}ch_{DISTANCE_METRIC}_{NUM_CLUSTERS}.png',
        dpi=300, bbox_inches='tight',
    )

    #7). HEATMAP WITH MEAN TIME (%) SPENT IN EACH STATE PER PATIENT, AVERAGED BY CLUSTER
    # For each patient: proportion of time points in each state → then average by cluster
    # This is patient-normalized (not cluster-normalized) → more interpretable

    all_states = list(STATES_LABELS.keys())

    # Compute per-patient proportion of time in each state
    def patient_state_proportions(row, states):
        vc = row.value_counts(normalize=True)
        return pd.Series({s: vc.get(s, 0.0) for s in states})

    seq_data = df[time_cols]
    patient_props = seq_data.apply(lambda row: patient_state_proportions(row, all_states), axis=1)
    patient_props['Cluster'] = df['Cluster'].values

    # Mean number of time points in each state per sequence, averaged by cluster
    mean_props = (
        patient_props
        .groupby('Cluster')[all_states]
        .mean() * len(time_cols)
    ).T  # shape: (states x clusters)

    mean_props.index = [STATES_LABELS[s] for s in mean_props.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        mean_props,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Mean time points per sequence'},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f'Mean time points spent in each state per sequence, by cluster: {SEQUENCE_NAME}')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('State')
    plt.tight_layout()
    plt.savefig(
        ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME / CLUSTERING / DISTANCE_METRIC / f'mean_time_per_state_{DISTANCE_METRIC}_{NUM_CLUSTERS}.png',
        dpi=300, bbox_inches='tight'
    )

def cluster_summary_table(df_sequence, cluster_labels ):
    """
    Build a summary table per cluster with:
    - Number of sequences
    - Mean number of non-NO states per sequence
    - Mean % time spent in each state (patient-normalized)
    - Total count of each state across all sequences in the cluster
    - Percentage of sequences in the cluster that contain at least one occurrence of each state
    """

    df = df_sequence.merge(cluster_labels, left_on='pedianet_id', right_on='Entity ID', how='inner')
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
    summary.to_latex(ROOT_FOLDER_CLUSTERING / SEQUENCE_NAME / CLUSTERING / DISTANCE_METRIC / f'cluster_summary_{DISTANCE_METRIC}_{NUM_CLUSTERS}.tex', index=True, float_format="%.1f")

    return summary