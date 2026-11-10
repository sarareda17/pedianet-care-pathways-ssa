############################################################################################
################################## DISSIMILARITY MEASURES ##################################
############################################################################################
#This file contains the functions to compute the dissimilarity matrix between sequences using 
#different dissimilarity measures.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import STATES_LABELS,COSTS
import sequenzo
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

################################## MAIN FUNCTIONS ##########################################
#main functions to get the dissimilarity matrix, either loading it from a file if it exists 
#or computing it if it doesn't

def get_dissimilarity_matrix(metric,sequence, df_sequence,SEQUENCE_NAME):
    dissimilarity_path=ROOT / "SSA" / "results" / "sequences"/"sequence_dissimilarities"/f"{SEQUENCE_NAME}_{metric}.npy"
    if dissimilarity_path.exists():
        print(f"Loading precomputed distance matrix for {metric}...")
        dist=np.load(dissimilarity_path)
    else:
        print(f"Computing distance matrix for {metric}...")
        dist=compute_dissimilarity_matrix(metric,sequence, df_sequence,SEQUENCE_NAME)
    return dist

#load Sequenzo package to compute the dissimilarity matrix 
from sequenzo.dissimilarity_measures import get_distance_matrix

def compute_dissimilarity_matrix(metric,sequence, df_sequence,SEQUENCE_NAME):

    ############################### OMPTIMAL MATCHING #####################################

    if metric == 'OM_TRATE':
        #OPTIMAL MATCHING with TRANSITION RATE substitution costs and indel costs=1
        dist=get_distance_matrix(seqdata=sequence, method='OM',sm="TRATE",indel=1,norm="auto",full_matrix=True)

    elif metric == 'OM_CONST':
        #OPTIMAL MATCHING with constant substitution costs (default=2) and constant indel costs =1.5 
        dist=get_distance_matrix(seqdata=sequence, method='OM',sm="CONSTANT",indel=1.5,norm="auto",full_matrix=True)

    elif metric == 'OM_EMBED':
        #OPTIMAL MATCHING with custom substitution cost matrix and indel costs based on the embedding of the states from text descriptions 
        sm_embed, indel_embed = build_om_embed(STATES_LABELS, df_sequence)
        dist = get_distance_matrix(seqdata=sequence, method='OM', sm=sm_embed, indel=indel_embed, norm="auto", full_matrix=True)
        
    elif metric == 'OM_INDELSLOG':
        #OPTIMAL MATCHING with state dependent indel costs based on the log of the inverse of the global state frequence and
        #substitution costs defined as the sum of the indel costs of the two states
        sm_log, indel_log = build_om_indelslog(STATES_LABELS, df_sequence)
        dist = get_distance_matrix(seqdata=sequence, method='OM', sm=sm_log, indel=indel_log, norm="auto", full_matrix=True)
    
    elif metric == 'OM_INDELSLOG_F':
        #OPTIMAL MATCHING with state dependent indel costs based on the log of the inverse of the patient-level state frequence and
        #substitution costs defined as the sum of the indel costs of the two states
        sm_log2, indel_log2 = build_om_indelslog2(STATES_LABELS, df_sequence)
        dist = get_distance_matrix(seqdata=sequence, method='OM', sm=sm_log2, indel=indel_log2, norm="auto", full_matrix=True)
        
    elif metric == 'OM_INDELSQRT':
        #OPTIMAL MATCHING with state dependent indel costs based on the sqrt of the inverse of the global state frequence and
        #substitution costs defined as the sum of the indel costs of the two states
        sm_sqrt, indel_sqrt = build_om_indelsqrt(STATES_LABELS, df_sequence)
        dist = get_distance_matrix(seqdata=sequence, method='OM', sm=sm_sqrt, indel=indel_sqrt, norm="auto", full_matrix=True)
    
    elif metric == 'OM_INDELSQRT_F':
         #OPTIMAL MATCHING with state dependent indel costs based on the sqrt of the inverse of the patient-level state frequence and
        #substitution costs defined as the sum of the indel costs of the two states   
        sm_sqrt, indel_sqrt = build_om_indelsqrt2(STATES_LABELS, df_sequence)
        dist = get_distance_matrix(seqdata=sequence, method='OM', sm=sm_sqrt, indel=indel_sqrt, norm="auto", full_matrix=True)   


    ################################# HAMMING DISTANCE #####################################
    
    elif metric == 'HAM':
        #HAMMING DISTANCE (standard)
        dist=hamming_distance_matrix(df_sequence)

    elif metric == 'HAM_WIND':
        #HAMMING DISTANCE with rolling windows with window_extension=w (i.e. from t-w to t+w)
        dist=hamming_windows_distance_matrix(df_sequence, window_extension=2, batch_size=500)

    elif metric== 'HAM_WIND_LOG':
        #HAMMING DISTANCE with rolling windows with window_extension=w (i.e. from t-w to t+w)
        #and costs based on OM_INDELSLOG indel costs
        _, indel_log = build_om_indelslog(STATES_LABELS, df_sequence)
        rarity_weight = dict(zip(STATES_LABELS.keys(), indel_log))
        dist = hamming_windows_distance_matrix_weighted(df_sequence, rarity_weight, window_extension=2, batch_size=500)

    elif metric == 'DHAM':
        #DYNAMIC HAMMING DISTANCE with transition rate costs by Sequenzo
        dist=get_distance_matrix(seqdata=sequence, method='DHD',sm="TRATE",norm="auto",full_matrix=True)


    ###################################### LONGEST COMMON SUBSEQUENCE #####################################

    elif metric == 'LCS':
        #LONGEST COMMON SUBSEQUENCE by Sequenzo
        dist=get_distance_matrix(seqdata=sequence, method='LCS',norm="auto",full_matrix=True)

    elif metric == 'LCS_NO':
        #LONGEST COMMON SUBSEQUENCE considering longest subsequences with at least one non-no state (at least one diagnosis)
        dist=lcs_constrained_distance_matrix(df_sequence, no_state='NO')

    elif metric == 'LCS_NO_WEIGHTED':
        #LONGEST COMMON SUBSEQUENCE with weighted cost for state NO w_no fixed in data-driven way apriori
        dist=lcs_weighted_distance_matrix(df_sequence, no_state='NO', w_no=0.15)


    ###################################### TEMPORAL EDIT DISTANCE #####################################

    elif metric == 'TED':
        #TEMPORAL EDIT DISTANCE with fixed costs
        dist=TED(df_sequence, c_sub=2.0, c_ins=1.0, c_del=2.0, c_time=0.5)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")
    
    #save distance matrix for future use
    if hasattr(dist, 'values'):
        dist = dist.values
    dist = np.asarray(dist)
    np.save(ROOT / "SSA" / "results" / "sequences"/"sequence_dissimilarities"/f"{SEQUENCE_NAME}_{metric}.npy", dist)
    return dist



##################################### UTILS FUNCTIONS #######################################
#functions to build the indel and substitution cost matrices in optimal matching or to implement
#tailored distance algotithms not defined in Sequenzo


############################### OMPTIMAL MATCHING ###########################################

#function to compute state dependent substitution costs based on the distance (1-cosine similarity) 
#between the BGE-M3 embeddings of the states text descriptions. The indel cost are set as  half the 
#average cost of all substitution costs. Substitution costs are defined in the COSTS variable in config.py
def build_om_embed(states_labels):
    """
    Builds a substitution cost matrix and indel costs for OM_EMBED based on the
    distance (1-cosine similarity) between the BGE-M3 embeddings of the states text descriptions.
    """
    states = list(states_labels.keys())
    n = len(states)

    sm = np.zeros((n, n), dtype=float)
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if a != b:
                sm[i, j] = COSTS[a][b]

    indel = np.full(n, sm.mean() / 2.0, dtype=float)

    return sm, indel
    
    


#function to compute state dependent indel costs based on the log of the inverse of the state frequence 
#(log (1/(f))) where f is the global state frequence across the sequence cohort. The substitution cost 
#between state i ang j is the sum of the indel cost of state i and the indel cost of state j
def build_om_indelslog(states_labels, df_sequence):
    """
    Builds state-dependent indel costs and the associated substitution cost
    matrix for OM_INDELSLOG based on global state frequencies.
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
    print("indel costs:", indel)

    return sm, indel



#function to compute state dependent indel costs based on the the log of the inverse of the state frequence (log (N/F)))
#where N is the total number of sequences and F is the number of sequences which includes the state 
def build_om_indelslog2(states_labels, df_sequence):
    """
    Builds state-dependent indel costs and the associated substitution cost
    matrix for OM_INDELSLOG_F based on the patient-level state frequencies.
    """
    states = list(states_labels.keys())
    n = len(states)

    seq_matrix = df_sequence.drop(columns=['pedianet_id']).values
    N = seq_matrix.shape[0]

    F = {}
    for s in states:
        F[s] = np.sum(np.any(seq_matrix == s, axis=1))
        F[s] = max(F[s], 1)  #avoid log(inf) if the state never appears

    print("Frequencies (number of sequences containing each state):", F)

    indel_per_state = {s: np.log(N / F[s]) for s in states}
    print("Indel costs (log(N/F)):", indel_per_state)

    sm = np.zeros((n, n), dtype=float)
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if a != b:
                sm[i, j] = indel_per_state[a] + indel_per_state[b]

    indel = np.array([indel_per_state[s] for s in states])

    return sm, indel



#function to compute state dependent indel costs based on the log of the inverse of the state frequence 
#(sqrt (1/(f))) where f is the global state frequence across the sequence cohort. The substitution cost 
#between state i ang j is the sum of the indel cost of state i and the indel cost of state j
def build_om_indelsqrt(states_labels, df_sequence):
    """
    Builds state-dependent indel costs and the associated substitution cost
    matrix for OM_INDELSQRT based on global state frequencies.
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

#function to compute state dependent indel costs based on the log of the inverse of the state frequence 
#(sqrt (N/F)) where F is the patient-level state frequence. The substitution cost 
#between state i ang j is the sum of the indel cost of state i and the indel cost of state j
def build_om_indelsqrt2(states_labels, df_sequence):
    """
    Builds state-dependent indel costs and the associated substitution cost
    matrix for OM_INDELSSQRT_F based on patient-level state frequencies.
    """
    states = list(states_labels.keys())
    n = len(states)

    seq_matrix = df_sequence.drop(columns=['pedianet_id']).values
    N = seq_matrix.shape[0]

    F = {}
    for s in states:
        F[s] = np.sum(np.any(seq_matrix == s, axis=1))
        F[s] = max(F[s], 1)  # avoid division by zero if the state never appears

    print("Frequencies (number of sequences containing each state):", F)

    indel_per_state = {s: np.sqrt(N / F[s]) for s in states}
    print("Indel costs (sqrt(N/F)):", indel_per_state)

    sm = np.zeros((n, n), dtype=float)
    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if a != b:
                sm[i, j] = indel_per_state[a] + indel_per_state[b]

    indel = np.array([indel_per_state[s] for s in states])

    return sm, indel



################################## HAMMING DISTANCE ########################################

#function to compute a distance matrix based on hamming distance 
def hamming_distance_matrix( df_sequence):
    """
    Computes the Hamming distance matrix for a given sequence dataframe.
    """
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





#function to compute a distance matrix based on custom hamming distance with rolling window
#assign mismatch between two sequences at time t only if the states at time t do not match 
#and there is no match in a window of size window_extension around t (i.e. from t-window_extension to t+window_extension)

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
    """
    Computes a rolling-window Hamming distance matrix for a given sequence dataframe.
    """
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
#as hamming_windows_distance_matrix) but a mismatch is weighted by the rarity
#of the state observed in sequence i at time t, instead of contributing a fixed
#cost of 1. Rarity does not affect whether a match is found in the window (that
#is still a binary presence/absence check), it only rescales how much a "no
#match found" contributes to the final distance: missing a rare state in the
#window costs more than missing a common one (e.g. NO). A mismatch observe in sequence i 
#at time t contributes a cost equal to the indel cost of the state used in OM_INDELSLOG
def hamming_windows_distance_matrix_weighted(df_sequence, rarity_weight, window_extension=1, batch_size=500):
    """
    Computes a rolling-window Hamming distance matrix for a given sequence dataframe,
    where mismatches are weighted by the rarity of the state observed in sequence i at time t.
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


##############################LONGEST COMMON SUBSEQUENCE #####################################

#functions to compute LCS considering longest common subsequence with at least one event (not all NO)
@njit
def _lcs_event_constrained(seq_a, seq_b, no_idx):
  la, lb = len(seq_a), len(seq_b)
  # dp0: paths containing only NO states
  dp0 = np.zeros((la + 1, lb + 1), dtype=np.int32)
  # dp1: paths including at least one genuine clinical event (initialized to -1)
  dp1 = np.full((la + 1, lb + 1), -1, dtype=np.int32)

  for i in range(1, la + 1):
    for j in range(1, lb + 1):
      a = seq_a[i - 1]
      b = seq_b[j - 1]

      # Propagation from top or left
      best_0 = max(dp0[i - 1, j], dp0[i, j - 1])
      best_1 = max(dp1[i - 1, j], dp1[i, j - 1])

      if a == b:
        if a == no_idx:
          # NO match: extends both branches
          best_0 = max(best_0, dp0[i - 1, j - 1] + 1)
          if dp1[i - 1, j - 1] != -1:
            best_1 = max(best_1, dp1[i - 1, j - 1] + 1)
        else:
          # Clinical match: triggers a valid dp1 branch
          best_1 = max(best_1, dp0[i - 1, j - 1] + 1)
          if dp1[i - 1, j - 1] != -1:
            best_1 = max(best_1, dp1[i - 1, j - 1] + 1)

      dp0[i, j] = best_0
      dp1[i, j] = best_1

  res = dp1[la, lb]
  return np.int32(res) if res > 0 else np.int32(0)


@njit(parallel=True)
def _compute_lcs_constrained_matrix(sequence_idx, no_idx, m, n):
  distance_matrix = np.zeros((n, n), dtype=np.float32)

  for i in prange(n):
    for j in range(i + 1, n):
      lcs_val = _lcs_event_constrained(sequence_idx[i], sequence_idx[j], no_idx)
      dist = 1.0 - (np.float32(lcs_val) / np.float32(m))
      distance_matrix[i, j] = dist
      distance_matrix[j, i] = dist

  return distance_matrix


def lcs_constrained_distance_matrix(df_sequence, no_state='NO'):
  """Computes the constrained LCS distance matrix, ensuring that the
  longest common subsequence contains at least one common clinical event.
  """
  if 'pedianet_id' in df_sequence.columns:
    sequence_data = df_sequence.drop(columns=['pedianet_id']).values
  else:
    sequence_data = df_sequence.values

  n, m = sequence_data.shape

  states = list(dict.fromkeys(sequence_data.flatten()))
  state_to_idx = {s: i for i, s in enumerate(states)}
  no_idx = np.int32(state_to_idx[no_state])

  sequence_idx = np.vectorize(state_to_idx.get)(sequence_data).astype(np.int32)

  print('Compiling Numba functions and computing distance matrix...')
  distance_matrix = _compute_lcs_constrained_matrix(sequence_idx, no_idx, m, n)

  np.fill_diagonal(distance_matrix, 0.0)
  return distance_matrix





#functions to compute LCS with weighted cost for state NO w_no fixed in data-driven way apriori

@njit
def _lcs_weighted(seq_a, seq_b, no_idx, w_no):
  la, lb = len(seq_a), len(seq_b)
  dp = np.zeros((la + 1, lb + 1), dtype=np.float32)

  for i in range(1, la + 1):
    for j in range(1, lb + 1):
      if seq_a[i - 1] == seq_b[j - 1]:
        # Assign full weight (1.0) to clinical matches and damped weight to NO matches
        match_weight = np.float32(w_no) if seq_a[i - 1] == no_idx else np.float32(1.0)
        dp[i, j] = dp[i - 1, j - 1] + match_weight
      else:
        # Standard LCS propagation
        dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])

  return dp[la, lb]


@njit(parallel=True)
def _compute_lcs_weighted_matrix(sequence_idx, no_idx, w_no, n):
  distance_matrix = np.zeros((n, n), dtype=np.float32)

  # Precompute maximum self-alignment scores for each sequence
  self_scores = np.zeros(n, dtype=np.float32)
  for i in range(n):
    self_scores[i] = _lcs_weighted(sequence_idx[i], sequence_idx[i], no_idx, w_no)

  # Compute pairwise dissimilarities with standard normalization
  for i in prange(n):
    for j in range(i + 1, n):
      score_ab = _lcs_weighted(sequence_idx[i], sequence_idx[j], no_idx, w_no)
      norm_factor = np.sqrt(self_scores[i] * self_scores[j])
      if norm_factor > 0.0:
        dist = 1.0 - (score_ab / norm_factor)
      else:
        dist = 1.0
      distance_matrix[i, j] = dist
      distance_matrix[j, i] = dist

  return distance_matrix


def lcs_weighted_distance_matrix(df_sequence, no_state='NO', w_no=0.1):
  """Computes the state-weighted LCS distance matrix, assigning a full score
  (1.0) to active clinical diagnoses and an fractional score (w_no) to the
  baseline NO state.
  """
  if 'pedianet_id' in df_sequence.columns:
    sequence_data = df_sequence.drop(columns=['pedianet_id']).values
  else:
    sequence_data = df_sequence.values

  n, m = sequence_data.shape

  states = list(dict.fromkeys(sequence_data.flatten()))
  state_to_idx = {s: i for i, s in enumerate(states)}
  no_idx = np.int32(state_to_idx[no_state])

  sequence_idx = np.vectorize(state_to_idx.get)(sequence_data).astype(np.int32)

  print('Compiling Numba functions and computing weighted LCS distance matrix...')
  distance_matrix = _compute_lcs_weighted_matrix(
      sequence_idx, no_idx, np.float32(w_no), n
  )

  np.fill_diagonal(distance_matrix, 0.0)
  return distance_matrix


######################## TEMPORAL EDIT DISTANCE (TED) ##################################### 
from numba import njit

@njit
def compute_all_pairs_single_thread(states_flat, times_flat, offsets, lengths, c_sub, c_ins, c_del, c_time):
    n_samples = len(lengths)
    dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
    
    # Pre-allocate a static DP table (maximum size 37x37 since there are 36 time steps)
    # Reusing this buffer avoids 144.5M memory allocations
    D = np.zeros((37, 37), dtype=np.float32)
    
    for i in range(n_samples):
        len_a = lengths[i]
        off_a = offsets[i]
        
        for j in range(i + 1, n_samples):
            len_b = lengths[j]
            off_b = offsets[j]
            
            # 1. Base conditions for the current pair
            for r in range(1, len_a + 1):
                D[r, 0] = r * c_del
            for c in range(1, len_b + 1):
                D[0, c] = c * c_ins
                
            # 2. Dynamic programming recurrence
            for r in range(1, len_a + 1):
                idx_a = off_a + r - 1
                sa = states_flat[idx_a]
                ta = times_flat[idx_a]
                
                for c in range(1, len_b + 1):
                    idx_b = off_b + c - 1
                    sb = states_flat[idx_b]
                    tb = times_flat[idx_b]
                    
                    if sa == sb:
                        diff = ta - tb
                        if diff < 0:
                            diff = -diff
                        cost_sub = c_time * diff
                    else:
                        cost_sub = c_sub
                        
                    c_diag = D[r - 1, c - 1] + cost_sub
                    c_up = D[r - 1, c] + c_del
                    c_left = D[r, c - 1] + c_ins
                    
                    # Compute min(c_diag, c_up, c_left)
                    min_val = c_diag
                    if c_up < min_val:
                        min_val = c_up
                    if c_left < min_val:
                        min_val = c_left
                        
                    D[r, c] = min_val
                    
            val = D[len_a, len_b]
            dist_matrix[i, j] = val
            dist_matrix[j, i] = val
            
    return dist_matrix



def TED(df_sequence, c_sub=2.0, c_ins=1.0, c_del=2.0, c_time=0.5):
    """
    Computes the pairwise TED dissimilarity matrix using single-threaded Numba.
    Ignores 'NO' states while preserving step indices (0 to 35).
    """
    # Drop identifier column if present
    if 'pedianet_id' in df_sequence.columns:
        seq_df = df_sequence.drop(columns=['pedianet_id'])
    else:
        seq_df = df_sequence.copy()
        
    values = seq_df.values
    n_samples, n_steps = values.shape
    
    # 1. Map categorical states to unique integers, excluding 'NO'
    unique_states = [s for s in np.unique(values) if pd.notna(s) and str(s).strip() != 'NO']
    state_to_id = {state: idx + 1 for idx, state in enumerate(unique_states)}
    
    # 2. Flatten data into contiguous 1D arrays for Numba
    all_states = []
    all_times = []
    offsets = np.zeros(n_samples, dtype=np.int32)
    lengths = np.zeros(n_samples, dtype=np.int32)
    
    current_offset = 0
    for i in range(n_samples):
        row = values[i]
        cnt = 0
        for t in range(n_steps):
            val = row[t]
            if pd.notna(val):
                val_str = str(val).strip()
                if val_str != 'NO' and val_str in state_to_id:
                    all_states.append(state_to_id[val_str])
                    all_times.append(t)
                    cnt += 1
                    
        offsets[i] = current_offset
        lengths[i] = cnt
        current_offset += cnt
        
    states_flat = np.array(all_states, dtype=np.int16)
    times_flat = np.array(all_times, dtype=np.int16)
    
    print(f"Starting computation for {n_samples} sequences (~144.5M pairs) on a single core...")
    
    dist_matrix = compute_all_pairs_single_thread(
        states_flat, times_flat, offsets, lengths,
        np.float32(c_sub), np.float32(c_ins), np.float32(c_del), np.float32(c_time)
    )
    
    print("Computation completed successfully.")
    return dist_matrix