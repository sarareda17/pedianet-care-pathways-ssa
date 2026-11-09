import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import SEQUENCE_NAME, SEQUENCE_NAME_1,SEQUENCE_NAME_2, CLUSTERING, LINKAGE,NUM_CLUSTERS,DISTANCE_METRIC
from config import STATES_LABELS,STATES_LABELS_1, STATES_LABELS_2, CUSTOM_COLORS, CUSTOM_COLORS_1, CUSTOM_COLORS_2
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
ROOT_SEQUENCE=ROOT/'SSA'/'results'/'sequences'/'sequence_clustering'

seq1=pd.read_csv(ROOT/'SSA'/'datasets'/f'{SEQUENCE_NAME}.csv',index_col='pedianet_id')
seq2=pd.read_csv(ROOT/'SSA'/'datasets'/f'{SEQUENCE_NAME_1}.csv',index_col='pedianet_id')
seq3=pd.read_csv(ROOT/'SSA'/'datasets'/f'{SEQUENCE_NAME_2}.csv',index_col='pedianet_id')

#keep only time cols with states 
seq1=seq1.iloc[:,6:]
seq2=seq2.iloc[:,6:]
seq3=seq3.iloc[:,6:]




# ── Configurazione canali e palette colori ─────────────────────────────────────
# Mappatura stato -> colore per ciascun canale
palette1 = dict(zip(STATES_LABELS.keys(), CUSTOM_COLORS))
palette2 = dict(zip(STATES_LABELS_1.keys(), CUSTOM_COLORS_1))
palette3 = dict(zip(STATES_LABELS_2.keys(), CUSTOM_COLORS_2))

channels_data = [
    {'name': 'Ch.1 (Diagnoses)', 'seq': seq1, 'palette': palette1},
    {'name': 'Ch.2 (Drugs)', 'seq': seq2, 'palette': palette2},
    {'name': 'Ch.3 (Secondary Care)', 'seq': seq3, 'palette': palette3},
]

# ── Selezione paziente casuale ────────────────────────────────────────────────
# Selezioniamo un paziente presente in tutti e tre i dataset
common_ids = seq1.index.intersection(seq2.index).intersection(seq3.index)
random_pos = np.random.randint(0, min(1000, len(common_ids)))
patient_id = common_ids[random_pos]

# ── Plotting delle sequenze a rettangoli ───────────────────────────────────────
n_months = seq1.shape[1]
fig, ax = plt.subplots(figsize=(10, 2.8), dpi=300)

bar_height = 0.7  # Spessore di ciascun canale

for y_idx, ch in enumerate(channels_data):
    # Estrai la sequenza del paziente per il canale corrente
    patient_seq = ch['seq'].loc[patient_id].values
    
    # y_idx invertito per avere Ch.1 in alto, Ch.2 in mezzo e Ch.3 in basso
    y_pos = len(channels_data) - 1 - y_idx

    for month_idx, state in enumerate(patient_seq):
        color = ch['palette'].get(state, '#D3D3D3')  # Grigio di fallback se non trovato
        
        # Rettangolo per il singolo mese con leggero bordo per separare i blocchi
        rect = patches.Rectangle(
            (month_idx, y_pos - bar_height / 2),
            width=1.0,
            height=bar_height,
            facecolor=color,
            edgecolor='white',
            linewidth=0.5,
        )
        ax.add_patch(rect)

# ── Configurazione assi ───────────────────────────────────────────────────────
ax.set_xlim(0, n_months)
ax.set_ylim(-0.6, len(channels_data) - 0.4)

# Asse Y: etichette dei canali (dall'alto verso il basso: Ch.1, Ch.2, Ch.3)
ax.set_yticks(range(len(channels_data)))
ax.set_yticklabels(['Ch.3', 'Ch.2', 'Ch.1'], fontsize=12, fontweight='bold')

# Asse X: tick ogni 4 o 6 mesi per evitare sovrapposizioni (0, 3, 7, 11... o passi regolari)
# Mostriamo 0, poi a passi di 4 e infine l'ultimo mese
xtick_positions = [0] + [m for m in range(3, n_months, 4)]
if (n_months - 1) not in xtick_positions:
    xtick_positions.append(n_months - 1)

ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_positions, fontsize=11)
ax.set_xlabel('Time (Months)', fontsize=12, labelpad=8)

# Rimuovi i bordi del grafico superflui per renderlo pulito da inserire nel diagramma
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)  # Nasconde le tacchette verticali mantenendo il testo

plt.title(f'Multichannel Sequence Representation (Patient ID: {patient_id})', fontsize=12, pad=12)
plt.tight_layout()

# Salvataggio dell'immagine ad alta risoluzione con sfondo trasparente (ideale per diagrammi)
output_path = ROOT_SEQUENCE/f'patient_multichannel_sequence.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)




# ── Generazione Legenda Verticale Unica (1 Colonna, No Sovrapposizioni) ────────
labels_list = [STATES_LABELS, STATES_LABELS_1, STATES_LABELS_2]

# Calcola il numero totale di righe (voci + titoli + separatori)
total_rows = sum(len(d) for d in labels_list) + len(channels_data) * 1.8

# Crea un asse unico trasparente
fig_leg, ax_leg = plt.subplots(figsize=(3.8, total_rows * 0.28), dpi=300)
ax_leg.axis('off')
ax_leg.set_xlim(0, 1)
ax_leg.set_ylim(0, total_rows)

y_cursor = total_rows - 0.5  # Punto di partenza dall'alto

for ch, labels_dict in zip(channels_data, labels_list):
    # Titolo del Canale
    ax_leg.text(
        0.02,
        y_cursor,
        ch['name'],
        fontsize=11,
        fontweight='bold',
        va='center',
        ha='left',
    )
    y_cursor -= 1.1

    # Voci del canale (quadratino + etichetta estesa)
    for key, label_text in labels_dict.items():
        if key in ch['palette']:
            color = ch['palette'][key]

            # Quadratino colorato
            rect = patches.Rectangle(
                (0.04, y_cursor - 0.25),
                0.05,
                0.5,
                facecolor=color,
                edgecolor='none',
            )
            ax_leg.add_patch(rect)

            # Testo dello stato
            ax_leg.text(
                0.12,
                y_cursor,
                label_text,
                fontsize=9.5,
                va='center',
                ha='left',
            )
            y_cursor -= 0.85

    # Spazio extra tra un canale e il successivo
    y_cursor -= 0.4

output_leg_path = ROOT_SEQUENCE / 'multichannel_states_legend_vertical.png'
plt.savefig(output_leg_path, dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig_leg)



import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Load and crop distance matrix (10x10) ──────────────────────────────────────
dist_file = (
    ROOT
    / 'SSA'
    / 'results'
    / 'sequences'
    / 'sequence_dissimilarities'
    / f'{SEQUENCE_NAME}_{DISTANCE_METRIC}.npy'
)
dist = np.load(dist_file)
dist_mini = dist[:10, :10]

# Normalizzazione su [0, 1] se non già normalizzata
if dist_mini.max() > 1.0:
    dist_mini = (dist_mini - dist_mini.min()) / (dist_mini.max() - dist_mini.min())

# Colormap da bianco (0) a #3b99cf (1)
cmap_custom = mcolors.LinearSegmentedColormap.from_list(
    'custom_blue', ['#FFFFFF', '#3b99cf']
)

# ── Mini Heatmap Plot con numeri ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.4, 3.4), dpi=300)

im = ax.imshow(dist_mini, cmap=cmap_custom, vmin=0, vmax=1, interpolation='nearest')

# Griglia bianca sottile per separare chiaramente le celle
ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
ax.grid(which='minor', color='white', linestyle='-', linewidth=1.0)
ax.tick_params(which='minor', bottom=False, left=False)

# Rimuovi etichette e tacche degli assi
ax.set_xticks([])
ax.set_yticks([])

# Scrittura dei valori dentro ogni cella (adatta il colore del testo per leggibilità)
for i in range(10):
    for j in range(10):
        val = dist_mini[i, j]
        # Testo bianco sulle celle più scure, scuro su quelle chiare
        text_color = 'white' if val > 0.55 else '#1f2937'
        ax.text(
            j,
            i,
            f'{val:.2f}'.replace('0.', '.'),  # Formato compatto es. .12 invece di 0.12
            ha='center',
            va='center',
            color=text_color,
            fontsize=6,
            fontweight='medium',
        )

# Bordo attorno alla matrice
for spine in ax.spines.values():
    spine.set_color('#3b99cf')
    spine.set_linewidth(1.0)

# Salvataggio trasparente ad alta risoluzione
output_matrix_path = ROOT_SEQUENCE / 'mini_dissimilarity_matrix.png'
plt.savefig(output_matrix_path, dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig)