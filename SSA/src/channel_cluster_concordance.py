import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from config import SEQUENCE_NAME, CLUSTERING, LINKAGE,NUM_CLUSTERS,DISTANCE_METRIC
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
ROOT_SEQUENCE=ROOT/'SSA'/'results'/'sequences'/'sequence_clustering'

clust_1 = pd.read_csv(
    ROOT_SEQUENCE
    / 'seq_macrodiagnoses_30_36'
    / CLUSTERING
    / 'LCS_NO_WEIGHTED'
    / 'cluster_labels_LCS_NO_WEIGHTED.csv',
    index_col='Entity ID',
)
clust_2 = pd.read_csv(
    ROOT_SEQUENCE
    / 'seq_drugs_30_36'
    / CLUSTERING
    / 'OM_INDELSLOG'
    / 'cluster_labels_OM_INDELSLOG.csv',
    index_col='Entity ID',
)
clust_3 = pd.read_csv(
    ROOT_SEQUENCE
    / 'seq_secondary_care_30_36'
    / CLUSTERING
    / 'LCS'
    / 'cluster_labels_LCS.csv',
    index_col='Entity ID',
)

# Merge datasets
df = (
    clust_1.iloc[:, 0]
    .to_frame(name='ch1')
    .join(clust_2.iloc[:, 0].to_frame(name='ch2'))
    .join(clust_3.iloc[:, 0].to_frame(name='ch3'))
    .dropna()
)

for col in ['ch1', 'ch2', 'ch3']:
  df[col] = df[col].astype(int)

# Cluster labels
map_ch1 = {
    1: 'C1: Low morbidity',
    2: 'C2: Mixed resp. events',
    3: 'C3: Late SYM',
    4: 'C4: Transient AURI',
    5: 'C5: Severe LRT/OBS',
    6: 'C6: Recurrent SYM',
    7: 'C7: Recurrent AURI',
}

map_ch2 = {
    1: 'C1: Minimal use',
    2: 'C2: Frequent ANT',
    3: 'C3: Frequent OAD',
    4: 'C4: Antihistamines & mix',
    5: 'C5: Nasal prep & mix',
}

map_ch3 = {
    1: 'C1: No utilization',
    2: 'C2: Sporadic utilization',
    3: 'C3: Frequent utilization',
}

df['ch1_lbl'] = df['ch1'].map(map_ch1)
df['ch2_lbl'] = df['ch2'].map(map_ch2)
df['ch3_lbl'] = df['ch3'].map(map_ch3)

order_ch1 = [map_ch1[i] for i in sorted(map_ch1.keys())]
order_ch2 = [map_ch2[i] for i in sorted(map_ch2.keys())]
order_ch3 = [map_ch3[i] for i in sorted(map_ch3.keys())]

# -------------------------------------------------------------
# 2. Gradient Palette & Row-wise Proportion Computation
# -------------------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap
COLOR_START = '#ffffff'
COLOR_END = '#649ABF'
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_gradient', [COLOR_START, COLOR_END]
)


def compute_percentage_matrix(row_col, col_col, row_order, col_order):
  # Normalizzazione per riga: somma riga = 100%
  counts = pd.crosstab(df[row_col], df[col_col]).reindex(
      index=row_order, columns=col_order, fill_value=0
  )
  row_pct = counts.div(counts.sum(axis=1), axis=0) * 100
  return row_pct


pct_1_2 = compute_percentage_matrix('ch1_lbl', 'ch2_lbl', order_ch1, order_ch2)
pct_1_3 = compute_percentage_matrix('ch1_lbl', 'ch3_lbl', order_ch1, order_ch3)

# -------------------------------------------------------------
# 3. Side-by-Side Matrices Plot
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(14, 6.2),
    gridspec_kw={'width_ratios': [5, 3.4], 'wspace': 0.12},
)

# --- Matrice 1: Diagnoses vs Drugs ---
sns.heatmap(
    pct_1_2,
    annot=True,
    fmt='.1f',
    cmap=custom_cmap,
    vmin=0,
    vmax=100,
    cbar=False,
    ax=ax1,
    annot_kws={'fontsize': 10.5, 'color': 'black', 'weight': 'medium'},
    linewidths=0.5,
    linecolor='#e0e0e0',
)

ax1.set_ylabel('Clusters diagnoses channel', fontsize=11, fontweight='bold', labelpad=8)
ax1.set_xlabel('Clusters Drug Prescriptions channel', fontsize=11, fontweight='bold', labelpad=8)
ax1.tick_params(axis='x', rotation=30, labelsize=9.5)
ax1.tick_params(axis='y', rotation=0, labelsize=9.5)

for t in ax1.texts:
  t.set_text(t.get_text() + '%')

# --- Matrice 2: Diagnoses vs Secondary Care ---
sns.heatmap(
    pct_1_3,
    annot=True,
    fmt='.1f',
    cmap=custom_cmap,
    vmin=0,
    vmax=100,
    cbar=True,
    cbar_kws={'label': 'Row Concordance (%)', 'shrink': 0.85, 'pad': 0.08},
    ax=ax2,
    annot_kws={'fontsize': 10.5, 'color': 'black', 'weight': 'medium'},
    linewidths=0.5,
    linecolor='#e0e0e0',
)


ax2.set_ylabel('')
ax2.set_yticks([])
ax2.set_xlabel(
    'Clusters Secondary Care  channel', fontsize=11, fontweight='bold', labelpad=8
)
ax2.tick_params(axis='x', rotation=25, labelsize=9.5)

for t in ax2.texts:
  t.set_text(t.get_text() + '%')

# -------------------------------------------------------------
# 4. Save and Export
# -------------------------------------------------------------
out_dir = ROOT_SEQUENCE
out_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    out_dir / 'cluster_concordance_diagnoses_pairs.png',
    bbox_inches='tight',
    dpi=300,
)


# =============================================================
# 5. Figure: Drugs on Rows vs. Diagnoses & Secondary Care
# =============================================================
# Compute row-normalized percentage matrices
pct_2_1 = compute_percentage_matrix('ch2_lbl', 'ch1_lbl', order_ch2, order_ch1)
pct_2_3 = compute_percentage_matrix('ch2_lbl', 'ch3_lbl', order_ch2, order_ch3)

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(15.5, 5.5),
    gridspec_kw={'width_ratios': [7, 3.4], 'wspace': 0.14},
)

# --- Matrix 1: Drugs vs. Diagnoses ---
sns.heatmap(
    pct_2_1,
    annot=True,
    fmt='.1f',
    cmap=custom_cmap,
    vmin=0,
    vmax=100,
    cbar=False,
    ax=ax1,
    annot_kws={'fontsize': 10.0, 'color': 'black', 'weight': 'medium'},
    linewidths=0.5,
    linecolor='#e0e0e0',
)
ax1.set_ylabel(
    'Clusters Drug Prescriptions channel',
    fontsize=11,
    fontweight='bold',
    labelpad=8,
)
ax1.set_xlabel(
    'Clusters Diagnoses channel', fontsize=11, fontweight='bold', labelpad=8
)
ax1.tick_params(axis='x', rotation=30, labelsize=9.0)
ax1.tick_params(axis='y', rotation=0, labelsize=9.5)

for t in ax1.texts:
  t.set_text(t.get_text() + '%')

# --- Matrix 2: Drugs vs. Secondary Care ---
sns.heatmap(
    pct_2_3,
    annot=True,
    fmt='.1f',
    cmap=custom_cmap,
    vmin=0,
    vmax=100,
    cbar=True,
    cbar_kws={'label': 'Row Concordance (%)', 'shrink': 0.85, 'pad': 0.08},
    ax=ax2,
    annot_kws={'fontsize': 10.0, 'color': 'black', 'weight': 'medium'},
    linewidths=0.5,
    linecolor='#e0e0e0',
)
ax2.set_ylabel('')
ax2.set_yticks([])
ax2.set_xlabel(
    'Clusters Secondary Care channel',
    fontsize=11,
    fontweight='bold',
    labelpad=8,
)
ax2.tick_params(axis='x', rotation=25, labelsize=9.5)

for t in ax2.texts:
  t.set_text(t.get_text() + '%')

plt.savefig(
    out_dir / 'cluster_concordance_drugs_pairs.png', bbox_inches='tight', dpi=300
)

# =============================================================
# 6. Figure: Secondary Care on Rows vs. Diagnoses & Drugs
# =============================================================
# Compute row-normalized percentage matrices
pct_3_1 = compute_percentage_matrix('ch3_lbl', 'ch1_lbl', order_ch3, order_ch1)
pct_3_2 = compute_percentage_matrix('ch3_lbl', 'ch2_lbl', order_ch3, order_ch2)

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(15.5, 4.2),
    gridspec_kw={'width_ratios': [7, 5], 'wspace': 0.14},
)

# --- Matrix 1: Secondary Care vs. Diagnoses ---
sns.heatmap(
    pct_3_1,
    annot=True,
    fmt='.1f',
    cmap=custom_cmap,
    vmin=0,
    vmax=100,
    cbar=False,
    ax=ax1,
    annot_kws={'fontsize': 10.0, 'color': 'black', 'weight': 'medium'},
    linewidths=0.5,
    linecolor='#e0e0e0',
)
ax1.set_ylabel(
    'Clusters Secondary Care channel',
    fontsize=11,
    fontweight='bold',
    labelpad=8,
)
ax1.set_xlabel(
    'Clusters Diagnoses channel', fontsize=11, fontweight='bold', labelpad=8
)
ax1.tick_params(axis='x', rotation=30, labelsize=9.0)
ax1.tick_params(axis='y', rotation=0, labelsize=9.5)

for t in ax1.texts:
  t.set_text(t.get_text() + '%')

# --- Matrix 2: Secondary Care vs. Drugs ---
sns.heatmap(
    pct_3_2,
    annot=True,
    fmt='.1f',
    cmap=custom_cmap,
    vmin=0,
    vmax=100,
    cbar=True,
    cbar_kws={'label': 'Row Concordance (%)', 'shrink': 0.85, 'pad': 0.08},
    ax=ax2,
    annot_kws={'fontsize': 10.0, 'color': 'black', 'weight': 'medium'},
    linewidths=0.5,
    linecolor='#e0e0e0',
)
ax2.set_ylabel('')
ax2.set_yticks([])
ax2.set_xlabel(
    'Clusters Drug Prescriptions channel',
    fontsize=11,
    fontweight='bold',
    labelpad=8,
)
ax2.tick_params(axis='x', rotation=30, labelsize=9.5)

for t in ax2.texts:
  t.set_text(t.get_text() + '%')

plt.savefig(
    out_dir / 'cluster_concordance_secondary_care_pairs.png',
    bbox_inches='tight',
    dpi=300,
)