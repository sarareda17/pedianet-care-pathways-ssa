import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sequenzo
ROOT = Path(__file__).resolve().parent.parent.parent
from config import STATES_LABELS,COSTS


from sequenzo.dissimilarity_measures import get_distance_matrix

##################################DISSIMILARITY MEASURE ##################################

def get_dissimilarity_matrix(metric,sequence, df_sequence,SEQUENCE_NAME,costs):
    #if metric is found in the folder of dissimilarities load it, otherwise compute it and save it in the folder
    dissimilarity_path=ROOT / "SSA" / "results" / "sequences"/"sequence_dissimilarities"/f"{SEQUENCE_NAME}_{metric}.npy"
    if dissimilarity_path.exists():
        print(f"Loading precomputed distance matrix for {metric}...")
        dist=np.load(dissimilarity_path)
    else:
        print(f"Computing distance matrix for {metric}...")
        dist=compute_dissimilarity_matrix(metric,sequence, df_sequence,SEQUENCE_NAME,costs)
    return dist

#function to apply the chosen dissimilarity measure 
def compute_dissimilarity_matrix(metric,sequence, df_sequence,SEQUENCE_NAME,costs):
    if metric == 'OM':
        #OPTIMAL MATCHING with transistion rate substitution costs and indel costs=1
        dist=get_distance_matrix(seqdata=sequence, method='OM',sm="TRATE",indel=1,norm="auto",full_matrix=True)
    elif metric == 'OM_CONSTANT':
        #OPTIMAL MATCHING with constant substitution costs and constant indel costs
        dist=get_distance_matrix(seqdata=sequence, method='OM',sm="CONSTANT",indel=1,norm="auto",full_matrix=True)
    elif metric == 'OM_DIAG':
        #OPTIMAL MATCHING with custom substitution cost matrix and indel costs 
        sm_diag = build_om_diag_sm(STATES_LABELS, cost_between=2.0, cost_no=0.5)
        dist=get_distance_matrix(seqdata=sequence, method='OM',sm=sm_diag,indel=0.5,norm="auto",full_matrix=True)
    elif metric == 'OM_DIAG1':
        #OPTIMAL MATCHING with custom substitution cost matrix and indel costs 
        sm_diag = build_om_diag_sm(STATES_LABELS, cost_between=2.0, cost_no=2.5)
        dist=get_distance_matrix(seqdata=sequence, method='OM',sm=sm_diag,indel=0.5,norm="auto",full_matrix=True)
    elif metric == 'OM_EMBED':
        #OPTIMAL MATCHING with custom substitution cost matrix and indel costs based on the embedding of the states from text descriptions 
        sm_embed, indel_embed = build_om_embed(STATES_LABELS, df_sequence)
        dist = get_distance_matrix(seqdata=sequence, method='OM', sm=sm_embed,
                                 indel=indel_embed, norm="auto", full_matrix=True)
    elif metric == 'OM_INDELSLOG':
        #OPTIMAL MATCHING with custom substitution cost matrix and state dependent indel costs based on the log of the inverse of the state frequence 
        sm_log, indel_log = build_om_indelslog(STATES_LABELS, df_sequence)
        dist = get_distance_matrix(seqdata=sequence, method='OM', sm=sm_log,
                                 indel=indel_log, norm="auto", full_matrix=True)
    elif metric == 'OM_INDELSLOG_NO':
        #OPTIMAL MATCHING with custom substitution cost matrix and state dependent indel costs based on the log of the inverse of the state frequence, BUT WITH A VARINAT FOR STATE NO 
        sm_log, indel_log = build_om_indelslog_variant_NO(STATES_LABELS, df_sequence,frac=0.5)
        dist = get_distance_matrix(seqdata=sequence, method='OM', sm=sm_log,
                                 indel=indel_log, norm="auto", full_matrix=True)
    elif metric == 'OM_INDELSLOG2':
        #OPTIMAL MATCHING with custom substitution cost matrix and state dependent indel costs based on the log of the inverse of the state frequence 
        sm_log2, indel_log2 = build_om_indelslog2(STATES_LABELS, df_sequence)
        dist = get_distance_matrix(seqdata=sequence, method='OM', sm=sm_log2,
                                 indel=indel_log2, norm="auto", full_matrix=True)
    elif metric == 'OM_INDELSLOG2_FIXEDINDEL':
        sm_log2, indel_fixed = build_om_indelslog2_fixedindel(STATES_LABELS, df_sequence, indel_scale=0.6)
        dist = get_distance_matrix(seqdata=sequence, method='OM', sm=sm_log2, indel=indel_fixed, norm="auto", full_matrix=True)
    elif metric == 'OM_INDELSQRT':
        sm_sqrt, indel_sqrt = build_om_indelsqrt(STATES_LABELS, df_sequence)
        dist = get_distance_matrix(seqdata=sequence, method='OM', sm=sm_sqrt, indel=indel_sqrt, norm="auto", full_matrix=True)
    elif metric == 'HAM':
        #HAMMING DISTANCE with mismatch cost of 1
        dist=hamming_distance_matrix(df_sequence)
    elif metric == 'HAM_WIND_2':
        #HAMMING DISTANCE with rolling windows 
        dist=hamming_windows_distance_matrix(df_sequence, window_extension=2, batch_size=500)
    elif metric == 'HAM_WIND_2_LOG2':
        _, indel_log2 = build_om_indelslog2(STATES_LABELS, df_sequence)
        rarity_weight = dict(zip(STATES_LABELS.keys(), indel_log2))
        dist = hamming_windows_distance_matrix_weighted(df_sequence, rarity_weight, window_extension=2, batch_size=500)
    elif metric== 'HAM_WIND_2_LOG':
        #HAMMING DISTANCE with rolling windows and costs based on the frequency of the states
        _, indel_log = build_om_indelslog(STATES_LABELS, df_sequence)
        rarity_weight = dict(zip(STATES_LABELS.keys(), indel_log))
        dist = hamming_windows_distance_matrix_weighted(df_sequence, rarity_weight, window_extension=2, batch_size=500)
    elif metric == 'HAM_WIND_1_LOG_NO':
        #HAMMING DISTANCE with rolling windows and costs based on the log of the frequency of the states, excluding the 'NO' state
        _, indel_log = build_om_indelslog_variant_NO(STATES_LABELS, df_sequence)
        rarity_weight = dict(zip(STATES_LABELS.keys(), indel_log))
        dist = hamming_windows_distance_matrix_weighted(df_sequence, rarity_weight, window_extension=2, batch_size=500)
    elif metric == 'HAM_WIND_1_SQRT':
        dist = hamming_windows_distance_matrix_equalized(df_sequence, window_extension=2, batch_size=500, power=0.5)
    elif metric == 'HAM_WIND_1_0.3_WEIGHTED':
        dist = hamming_windows_distance_matrix_equalized(df_sequence, window_extension=2, batch_size=500, power=0.3)
    elif metric == 'HAM_WIND_2_COSTS':
        #HAMMING DISTANCE with rolling windows and costs
        dist=hamming_windows_distance_matrix_with_costs(df_sequence,costs, window_extension=2, batch_size=500)
    elif metric == 'HAM_WIND_2_COSTS_ONEMATCHED':
        #HAMMING DISTANCE with rolling windows and costs and one match only 
        dist=hamming_windows_distance_matrix_with_costs_matched(df_sequence,costs, window_extension=2)
    elif metric == 'DHAM':
        #DYNAMIC HAMMING DISTANCE with transition rate costs
        dist=get_distance_matrix(seqdata=sequence, method='DHD',sm="TRATE",norm="auto",full_matrix=True)
    elif metric == 'LCS':
        #LONGEST COMMON SUBSEQUENCE
        dist=get_distance_matrix(seqdata=sequence, method='LCS',norm="auto",full_matrix=True)
    elif metric == 'LCS_NO':
        #LONGEST COMMON SUBSEQUENCE without considering subsequences with all no (nessuna diagnosi)
        dist=lcs_distance_matrix(df_sequence, no_state='NO')
    else:
        raise ValueError(f"Unknown distance metric: {metric}")
    
    #save dist
    if hasattr(dist, 'values'):
        dist = dist.values
    dist = np.asarray(dist)
    np.save(ROOT / "SSA" / "results" / "sequences"/"sequence_dissimilarities"/f"{SEQUENCE_NAME}_{metric}.npy", dist)
    return dist


def build_om_diag_sm(states_labels: dict,
                     cost_between: float = 2.0,
                     cost_no: float = 0.5) -> np.ndarray:
    """
    Builds a substitution cost matrix for OM_DIAG.
    
    Args:
        states_labels:  dict mapping state codes to labels (e.g. {'PNE': 'Polmonite', ...})
        cost_between:   substitution cost between any two different diagnoses (default 2.0)
        cost_no:        substitution cost between any diagnosis and NO (default 0.5)
    
    Returns:
        np.ndarray of shape (n_states, n_states)
        
    Cost structure:
        0.0          → same state (diagonal)
        cost_no      → any state ↔ NO
        cost_between → any two different diagnoses
    """
    states = list(states_labels.keys())
    n = len(states)
    no_idx = states.index('NO')
    
    arr = np.full((n, n), cost_between, dtype=float)
    np.fill_diagonal(arr, 0.0)
    
    arr[no_idx, :] = cost_no
    arr[:, no_idx] = cost_no
    arr[no_idx, no_idx] = 0.0
    
    return arr

#function to compute state dependentsubstitution costs based on the distance between the embeddings of the states from text descriptions
#the indel cost are half the average cost of the substitution costs
def build_om_embed(states_labels, df_sequence):
    states = list(states_labels.keys())
    n = len(states)

    sm = np.zeros((n, n), dtype=float)
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if a != b:
                sm[i, j] = COSTS[a][b]

    indel = np.full(n, sm.mean() / 2.0, dtype=float)

    return sm, indel
    
    


#function to compute state dependent indel costs based on the log of 
#the inverse of the state frequence (log (1/(freq))) where freq is the state overall frequency
#and the substitution cost between state i ang j is the sum of the indel cost of state i and the indel cost of state j
def build_om_indelslog(states_labels, df_sequence):
    """
    Builds state-dependent indel costs and the associated substitution cost
    matrix for OM_INDELSLOG.

    Cost structure:
        indel(s) = log( 2 / (1 + freq(s)) )
                   where freq(s) = overall relative frequency of state s,
                   computed across ALL cells (individuals x timepoints)
        sm(i, j) = indel(i) + indel(j)   for i != j
        sm(i, i) = 0.0

    Returns:
        sm:    np.ndarray (n_states, n_states) substitution cost matrix
        indel: np.ndarray (n_states,) state-dependent indel costs, in the
               same order as states_labels.keys() -> passare come argomento
               'indel' di get_distance_matrix
    """
    states = list(states_labels.keys())
    n = len(states)

    seq_matrix = df_sequence.drop(columns=['pedianet_id']).values
    total_cells = seq_matrix.size

    freq = {}
    for s in states:
        freq[s] = np.sum(seq_matrix == s) / total_cells

    indel_per_state = {s: np.log(1 / (freq[s])) for s in states}

    sm = np.zeros((n, n), dtype=float)
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if a != b:
                sm[i, j] = indel_per_state[a] + indel_per_state[b]

    indel = np.array([indel_per_state[s] for s in states])

    return sm, indel

#function to compute state dependent indel costs based on the log of 
#the inverse of the state frequence (log (1/(freq))) where freq is the state overall frequency
#and the substitution cost between state i ang j is the sum of the indel cost of state i and the indel cost of state j
def build_om_indelslog_variant_NO(states_labels, df_sequence, frac=0.5):
    """
    Builds state-dependent indel costs and the associated substitution cost
    matrix for OM_INDELSLOG.

    Cost structure:
        indel(s) = log( 2 / (1 + freq(s)) )
                   where freq(s) = overall relative frequency of state s,
                   computed across ALL cells (individuals x timepoints)
        sm(i, j) = indel(i) + indel(j)   for i != j
        sm(i, i) = 0.0

    Returns:
        sm:    np.ndarray (n_states, n_states) substitution cost matrix
        indel: np.ndarray (n_states,) state-dependent indel costs, in the
               same order as states_labels.keys() -> passare come argomento
               'indel' di get_distance_matrix
    """
    states = list(states_labels.keys())
    n = len(states)

    seq_matrix = df_sequence.drop(columns=['pedianet_id']).values
    total_cells = seq_matrix.size

    freq = {}
    for s in states:
        freq[s] = np.sum(seq_matrix == s) / total_cells

    indel_per_state = {s: np.log(1 / (freq[s])) for s in states}

    #assign state NO the frac of the minimum indel cost of the other states
    min_indel = min(indel_per_state[s] for s in states if s != 'NO')
    indel_per_state['NO'] = frac * min_indel

    sm = np.zeros((n, n), dtype=float)
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if a != b:
                sm[i, j] = indel_per_state[a] + indel_per_state[b]

    indel = np.array([indel_per_state[s] for s in states])

    return sm, indel


#function to compute state dependent indel costs based on the the log of
#the inverse of the state frequence (log (N/F))) where N is the total number of sequences and F is the 
#number of sequences which include the state 
def build_om_indelslog2(states_labels, df_sequence):
    """
    Builds state-dependent indel costs and the associated substitution cost
    matrix for OM_INDELSLOG2 (variante IDF-style, frequenza a livello di
    sequenza invece che di singola cella).

    Cost structure:
        indel(s) = log( N / F(s) )
                   dove N    = numero totale di sequenze (bambini)
                         F(s) = numero di sequenze in cui lo stato s compare
                                almeno una volta
        sm(i, j) = indel(i) + indel(j)   per i != j
        sm(i, i) = 0.0

    Returns:
        sm:    np.ndarray (n_states, n_states) substitution cost matrix
        indel: np.ndarray (n_states,) state-dependent indel costs, stesso
               ordine di states_labels.keys()
    """
    states = list(states_labels.keys())
    n = len(states)

    seq_matrix = df_sequence.drop(columns=['pedianet_id']).values
    N = seq_matrix.shape[0]

    F = {}
    for s in states:
        F[s] = np.sum(np.any(seq_matrix == s, axis=1))
        F[s] = max(F[s], 1)  # evita log(inf) se lo stato non compare mai

    indel_per_state = {s: np.log(N / F[s]) for s in states}

    sm = np.zeros((n, n), dtype=float)
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if a != b:
                sm[i, j] = indel_per_state[a] + indel_per_state[b]

    indel = np.array([indel_per_state[s] for s in states])

    return sm, indel

#function to compute OM substitution costs identical to build_om_indelslog2
#(rarity-based, IDF-style), but paired with a FIXED and HIGH indel cost instead
#of the state-dependent indel vector. A high fixed indel discourages the OM
#algorithm from using insertions/deletions to "shift" events in time, favouring
#substitutions instead -- so temporal alignment (positioning) is preserved,
#while the substitution costs still reflect state rarity as in OM_INDELSLOG2.
def build_om_indelslog2_fixedindel(states_labels, df_sequence, indel_scale=1.0):
    """
    Args:
        states_labels: dict mapping state codes to labels (e.g. STATES_LABELS)
        df_sequence:   wide dataframe (rows=individuals, columns=timepoints,
                       plus a 'pedianet_id' column)
        indel_scale:   multiplier applied to max(sm) to set the fixed indel cost.
                       indel_scale=0.5 -> indel = max(sm)/2 (indifference point:
                       substitution and indel-pair are equally costly in the
                       worst case)
                       indel_scale=1.0 -> indel = max(sm) (substitution is
                       always at least as convenient as an indel pair, strongly
                       discourages time shifts)
                       Values above 1.0 push this further.

    Returns:
        sm:           np.ndarray (n_states, n_states), same substitution cost
                       matrix as build_om_indelslog2 (rarity-based, IDF-style)
        indel_fixed:  float, single fixed indel cost to use for ALL states
    """
    sm, _ = build_om_indelslog2(states_labels, df_sequence)
    indel_fixed = float(sm.max() * indel_scale)
    return sm, indel_fixed

#function to compute state dependent indel costs based on the SQUARE ROOT of
#the inverse of the state overall frequency (1/sqrt(f(s))), where f(s) is
#computed at cell level (across all individuals and time points), following
#the same principle as build_om_indelslog but using sqrt instead of log as the
#monotonic dampening function (both explicitly mentioned as valid options in
#Studer & Ritschard 2016, sec. 3.3.3.2). Compared to the log version, sqrt
#differentiates much more clearly between rare and common diagnoses on this
#dataset, since the log is nearly flat across the observed frequency range.
#The substitution cost between state i and j is the sum of their indel costs.
def build_om_indelsqrt(states_labels, df_sequence):
    """
    Cost structure:
        indel(s) = 1 / sqrt(f(s))
                   where f(s) = overall relative frequency of state s,
                   computed across ALL cells (individuals x timepoints)
        sm(i, j) = indel(i) + indel(j)   for i != j
        sm(i, i) = 0.0

    Returns:
        sm:    np.ndarray (n_states, n_states), substitution cost matrix
        indel: np.ndarray (n_states,), state-dependent indel costs, same
               order as states_labels.keys()
    """
    states = list(states_labels.keys())
    n = len(states)

    seq_matrix = df_sequence.drop(columns=['pedianet_id']).values
    total_cells = seq_matrix.size

    freq = {s: np.sum(seq_matrix == s) / total_cells for s in states}
    indel_per_state = {s: 1.0 / np.sqrt(freq[s]) for s in states}

    sm = np.zeros((n, n), dtype=float)
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if a != b:
                sm[i, j] = indel_per_state[a] + indel_per_state[b]

    indel = np.array([indel_per_state[s] for s in states])

    return sm, indel

#distance matrix based on hamming distance 
def hamming_distance_matrix( df_sequence):
    from scipy.spatial.distance import hamming
    sequence_data = df_sequence.drop(columns=['pedianet_id']).values
    n = sequence_data.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = hamming(sequence_data[i], sequence_data[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix

#distance matrix based on custom hamming distance with rolling window
#compute mismatch between two sequences at time t only if the states at time t do not match and there is no match in a window 
# of size window_extension around t (i.e. from t-window_extension to t+window_extension)
# def hamming_windows_distance_matrix(df_sequence, window_extension=1):
#     from scipy.spatial.distance import hamming
#     sequence_data = df_sequence.drop(columns=['pedianet_id']).values
#     m=sequence_data.shape[1]
#     n = sequence_data.shape[0]
#     distance_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             dist=0
#             if i != j:
#                 for t in range(m):
#                     flag=False
#                     if t < window_extension:
#                         for w in range(0, t+window_extension):
#                             if sequence_data[i, t] == sequence_data[j, w]:
#                                 flag=True
#                                 break
#                     elif t >= m - window_extension:
#                         for w in range(t-window_extension, m):
#                             if sequence_data[i, t] == sequence_data[j, w]:
#                                 flag=True
#                                 break
#                     else:
#                         for w in range(t-window_extension, t+window_extension):
#                             if sequence_data[i, t] == sequence_data[j, w]:
#                                 flag=True
#                                 break
#                     if not flag:
#                         dist+=1
#             distance_matrix[i, j] = dist
    
#     for i in range(n):
#         for j in range(n):
#             distance_matrix[i,j]+=distance_matrix[j,i]
#             distance_matrix[i,j]=distance_matrix[i,j]/(2*m)

#     return distance_matrix


#batch version 
def hamming_windows_distance_matrix(df_sequence, window_extension=1, batch_size=500):
    sequence_data = df_sequence.drop(columns=['pedianet_id']).values  
    
    n, m = sequence_data.shape
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    for t in range(m):
        w_start = max(0, t - window_extension)
        w_end   = min(m, t + window_extension + 1)

        seq_t  = sequence_data[:, t]
        window = sequence_data[:, w_start:w_end]

        for i_start in range(0, n, batch_size):
            i_end = min(n, i_start + batch_size)

            batch_seq  = seq_t[i_start:i_end, None, None]
            all_window = window[None, :, :]

            match = np.any(batch_seq == all_window, axis=2)
            distance_matrix[i_start:i_end, :] += (~match).astype(np.float32)

    distance_matrix = (distance_matrix + distance_matrix.T) / (2 * m)
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix

#function to compute a rolling-window Hamming distance (same window-match logic
#as hamming_windows_distance_matrix) where a mismatch is weighted by the rarity
#of the state observed in sequence i at time t, instead of contributing a fixed
#cost of 1. Rarity does not affect whether a match is found in the window (that
#is still a binary presence/absence check), it only rescales how much a "no
#match found" contributes to the final distance: missing a rare state in the
#window costs more than missing a common one (e.g. NO).
def hamming_windows_distance_matrix_weighted(df_sequence, rarity_weight, window_extension=1, batch_size=500):
    """
    Args:
        df_sequence:      wide dataframe (rows=individuals, columns=timepoints,
                           plus a 'pedianet_id' column)
        rarity_weight:    dict, state -> float, e.g. the indel costs from
                           build_om_indelslog / build_om_indelslog2 (higher
                           for rarer states)
        window_extension: half-width of the rolling window
        batch_size:       batch size for the vectorized loop over individuals

    Returns:
        distance_matrix: np.ndarray (n, n), symmetric, normalized by (2*m)
    """
    sequence_data = df_sequence.drop(columns=['pedianet_id']).values

    n, m = sequence_data.shape
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    # rarity weight vector, mapped onto the actual state values in the sequence
    states = np.unique(sequence_data)
    weight_lookup = {s: rarity_weight[s] for s in states}

    for t in range(m):
        w_start = max(0, t - window_extension)
        w_end   = min(m, t + window_extension + 1)

        seq_t  = sequence_data[:, t]
        window = sequence_data[:, w_start:w_end]

        # rarity weight for the state observed in each sequence i at time t
        weight_t = np.array([weight_lookup[s] for s in seq_t], dtype=np.float32)

        for i_start in range(0, n, batch_size):
            i_end = min(n, i_start + batch_size)

            batch_seq  = seq_t[i_start:i_end, None, None]
            all_window = window[None, :, :]

            match = np.any(batch_seq == all_window, axis=2)  # (batch, n)

            mismatch = (~match).astype(np.float32)
            weight_i = weight_t[i_start:i_end][:, None]  # (batch, 1)

            distance_matrix[i_start:i_end, :] += mismatch * weight_i

    distance_matrix = (distance_matrix + distance_matrix.T) / (2 * m)
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix

def hamming_windows_distance_matrix_equalized(df_sequence, window_extension=1, 
                                                 batch_size=500, power=0.5):
    """
    power=1.0 -> 1/f(s)      (some weights might explode)
    power=0.5 -> 1/sqrt(f(s)) (trade-off)
    """
    sequence_data = df_sequence.drop(columns=['pedianet_id']).values
    n, m = sequence_data.shape
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    states = np.unique(sequence_data)
    total_cells = sequence_data.size
    freq = {s: np.sum(sequence_data == s) / total_cells for s in states}
    weight_lookup = {s: 1.0 / (freq[s] ** power) for s in states}

    for t in range(m):
        w_start = max(0, t - window_extension)
        w_end   = min(m, t + window_extension + 1)
        seq_t  = sequence_data[:, t]
        window = sequence_data[:, w_start:w_end]
        weight_t = np.array([weight_lookup[s] for s in seq_t], dtype=np.float32)

        for i_start in range(0, n, batch_size):
            i_end = min(n, i_start + batch_size)
            batch_seq  = seq_t[i_start:i_end, None, None]
            all_window = window[None, :, :]
            match = np.any(batch_seq == all_window, axis=2)
            mismatch = (~match).astype(np.float32)
            weight_i = weight_t[i_start:i_end][:, None]
            distance_matrix[i_start:i_end, :] += mismatch * weight_i

    distance_matrix = (distance_matrix + distance_matrix.T) / (2 * m)
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

#distance matrix based on custom hamming distance with rolling window with cost 
#compute mismatch between two sequences at time t only if the states at time t do not match and there is no match in a window 
# of size window_extension around t (i.e. from t-window_extension to t+window_extension), but this time  with costs of mismatch
# def hamming_windows_distance_matrix_with_costs(df_sequence, window_extension=1,costs):
#     from scipy.spatial.distance import hamming
#     sequence_data = df_sequence.drop(columns=['pedianet_id']).values
#     m=sequence_data.shape[1]
#     n = sequence_data.shape[0]
#     distance_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             dist=0
#             if i != j:
#                 for t in range(m):
#                     cost=[]
#                     if t < window_extension:
#                         for w in range(0, t+window_extension):
#                             cost.append(costs[sequence_data[i, t]][sequence_data[j, w]])
#                     elif t >= m - window_extension:
#                         for w in range(t-window_extension, m):
#                             cost.append(costs[sequence_data[i, t]][sequence_data[j, w]])
#                     else:
#                         for w in range(t-window_extension, t+window_extension):
#                             cost.append(costs[sequence_data[i, t]][sequence_data[j, w]])
#                     dist += min(cost) if cost else 0
#             distance_matrix[i, j] = dist

#     for i in range(n):
#         for j in range(n):
#             distance_matrix[i,j]+=distance_matrix[j,i]
#             distance_matrix[i,j]=distance_matrix[i,j]/(2*m)

#     return distance_matrix


def hamming_windows_distance_matrix_with_costs(df_sequence, costs, window_extension=1, batch_size=500):
    sequence_data = df_sequence.drop(columns=['pedianet_id']).values  # dtype=object (strings)
    n, m = sequence_data.shape

    # convert costs dict-of-dicts to a numpy matrix for fast lookup
    states = list(costs.keys())
    state_to_idx = {s: i for i, s in enumerate(states)}
    n_states = len(states)
    
    cost_matrix = np.zeros((n_states, n_states), dtype=np.float32)
    for s1 in states:
        for s2 in states:
            cost_matrix[state_to_idx[s1], state_to_idx[s2]] = costs[s1][s2]

    # convert string sequences to integer indices
    sequence_idx = np.vectorize(state_to_idx.get)(sequence_data).astype(np.int32)  # (n, m)

    distance_matrix = np.zeros((n, n), dtype=np.float32)

    for t in range(m):
        w_start = max(0, t - window_extension)
        w_end   = min(m, t + window_extension + 1)

        seq_t  = sequence_idx[:, t]             # (n,)
        window = sequence_idx[:, w_start:w_end]  # (n, w)

        for i_start in range(0, n, batch_size):
            i_end = min(n, i_start + batch_size)

            batch_seq = seq_t[i_start:i_end]  # (batch,)

            # vectorized cost lookup for all (i, j, w) combinations
            # batch_seq: (batch,) → (batch, 1, 1)
            # window:    (n, w)   → (1, n, w)
            # result:             → (batch, n, w)
            batch_costs = cost_matrix[
                batch_seq[:, None, None],  # (batch, 1, 1)
                window[None, :, :]         # (1, n, w)
            ]

            # take the minimum cost across the window
            min_costs = batch_costs.min(axis=2)  # (batch, n)

            distance_matrix[i_start:i_end, :] += min_costs

    # symmetrize and normalize
    distance_matrix = (distance_matrix + distance_matrix.T) / (2 * m)
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix

#distance matrix based on custom hamming distance with rolling window with cost and one match only 
#compute mismatch between two sequences at time t only if the states at time t do not match and there is no match in a window 
# of size window_extension around t (i.e. from t-window_extension to t+window_extension), but this time  with costs of mismatch
# def hamming_windows_distance_matrix_with_costs(df_sequence, window_extension=1,costs):
#     from scipy.spatial.distance import hamming
#     sequence_data = df_sequence.drop(columns=['pedianet_id']).values
#     m=sequence_data.shape[1]
#     n = sequence_data.shape[0]
#     distance_matrix = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             dist=0
#             if i != j:
#                 for t in range(m):
#                     cost=[]
#                     if t < window_extension:
#                         for w in range(0, t+window_extension):
#                             if sequence_data[j,w] is None:
#                                 cost.append(None)
#                             else:
#                                 cost.append(costs[sequence_data[i, t]][sequence_data[j, w]])
#                     elif t >= m - window_extension:
#                         for w in range(t-window_extension, m):
#                             if sequence_data[j,w] is None:
#                                 cost.append(None)
#                             else:   
#                                 cost.append(costs[sequence_data[i, t]][sequence_data[j, w]])
#                     else:
#                         for w in range(t-window_extension, t+window_extension):
#                             if sequence_data[j,w] is None:
#                                 cost.append(None)
#                             else:
#                                 cost.append(costs[sequence_data[i, t]][sequence_data[j, w]])
#                     dist += min(cost) if cost else 0
#                     #save index of mimum cost match
#                     min_cost_idx = np.argmin(cost) if cost else None
#                     #the matched state is removed from the window for the next iterations
#                     if min_cost_idx is not None:
#                         if t < window_extension:
#                             w = min_cost_idx
#                         elif t >= m - window_extension:
#                             w = t - window_extension + min_cost_idx
#                         else:
#                             w = t - window_extension + min_cost_idx
#                         sequence_data[j, w] = None  # mark as matched (or remove from consideration)
#             distance_matrix[i, j] = dist

#     for i in range(n):
#         for j in range(n):
#             distance_matrix[i,j]+=distance_matrix[j,i]
#             distance_matrix[i,j]=distance_matrix[i,j]/(2*m)

#     return distance_matrix


def hamming_windows_distance_matrix_with_costs_matched(df_sequence, costs, window_extension=1):
    sequence_data = df_sequence.drop(columns=['pedianet_id']).values
    n, m = sequence_data.shape
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i+1, n):  # upper triangle only, then symmetrize
            window_j = sequence_data[j].copy()  # local copy to track consumed states

            dist = 0.0
            for t in range(m):
                w_start = max(0, t - window_extension)
                w_end   = min(m, t + window_extension + 1)

                state_i = sequence_data[i, t]

                # find minimum cost among non-consumed states in window
                best_cost = None
                best_w    = None
                for w in range(w_start, w_end):
                    if window_j[w] is None:
                        continue
                    c = costs[state_i][window_j[w]]
                    if best_cost is None or c < best_cost:
                        best_cost = c
                        best_w    = w

                if best_cost is not None:
                    dist += best_cost
                    window_j[best_w] = None  # mark as consumed

            distance_matrix[i, j] = dist

    # symmetrize and normalize
    distance_matrix = distance_matrix + distance_matrix.T
    distance_matrix /= (2 * m)
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix






#LCS swithout considering subsequences with all no (nessuna diagnosi)
from numba import njit, prange

@njit
def _lcs_with_constraint(seq_a, seq_b, no_idx):
    """
    Standard LCS DP, but returns 0 if the longest common subsequence
    consists entirely of no_idx (e.g. 'NO' / 'nessuna diagnosi').
    """
    la, lb = len(seq_a), len(seq_b)

    dp_len  = np.zeros((la + 1, lb + 1), dtype=np.int32)
    dp_flag = np.zeros((la + 1, lb + 1), dtype=np.int8)

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            if seq_a[i-1] == seq_b[j-1]:
                dp_len[i, j]  = dp_len[i-1, j-1] + 1
                dp_flag[i, j] = dp_flag[i-1, j-1] or (seq_a[i-1] != no_idx)
            else:
                len_up   = dp_len[i-1, j]
                len_left = dp_len[i, j-1]
                if len_up > len_left:
                    dp_len[i, j]  = len_up
                    dp_flag[i, j] = dp_flag[i-1, j]
                elif len_left > len_up:
                    dp_len[i, j]  = len_left
                    dp_flag[i, j] = dp_flag[i, j-1]
                else:
                    dp_len[i, j]  = len_up
                    dp_flag[i, j] = dp_flag[i-1, j] or dp_flag[i, j-1]

    return np.int32(dp_len[la, lb]) if dp_flag[la, lb] == 1 else np.int32(0)


@njit(parallel=True)
def _compute_lcs_matrix(sequence_idx, no_idx, m, n):
    distance_matrix = np.zeros((n, n), dtype=np.float32)

    for i in prange(n):
        for j in range(i + 1, n):
            lcs_len = _lcs_with_constraint(sequence_idx[i], sequence_idx[j], no_idx)
            dist = 1.0 - (2.0 * lcs_len) / (2.0 * m)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def lcs_distance_matrix(df_sequence, no_state='NO'):
    sequence_data = df_sequence.drop(columns=['pedianet_id']).values
    n, m = sequence_data.shape

    states       = list(dict.fromkeys(sequence_data.flatten()))
    state_to_idx = {s: i for i, s in enumerate(states)}
    no_idx       = np.int32(state_to_idx[no_state])

    sequence_idx = np.vectorize(state_to_idx.get)(sequence_data).astype(np.int32)

    print("Compiling Numba functions (first run only)...")
    distance_matrix = _compute_lcs_matrix(sequence_idx, no_idx, m, n)

    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix




