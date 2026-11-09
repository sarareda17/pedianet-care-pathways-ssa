import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from config import SEQUENCE_NAME, CLUSTERING, LINKAGE,NUM_CLUSTERS,DISTANCE_METRIC
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
ROOT_SEQUENCE=ROOT/'SSA'/'results'/'sequences'/'sequence_clustering'


#Sankey diagram for cluster transitions
#cluster labels for first channels
clust_1=pd.read_csv(ROOT_SEQUENCE/'seq_macrodiagnoses_30_36'/CLUSTERING/'LCS_NO_WEIGHTED'/f'cluster_labels_LCS_NO_WEIGHTED.csv', index_col='Entity ID')
#cluster labels for second channels
clust_2=pd.read_csv(ROOT_SEQUENCE/'seq_drugs_30_36'/CLUSTERING/'OM_INDELSLOG'/f'cluster_labels_OM_INDELSLOG.csv', index_col='Entity ID')
#cluster labels for third channels 
clust_3=pd.read_csv(ROOT_SEQUENCE/'seq_secondary_care_30_36'/CLUSTERING/'LCS'/f'cluster_labels_LCS.csv', index_col='Entity ID')

from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Align patient trajectories across channels
df = (
    clust_1.iloc[:, 0]
    .to_frame(name="ch1")
    .join(clust_2.iloc[:, 0].to_frame(name="ch2"))
    .join(clust_3.iloc[:, 0].to_frame(name="ch3"))
    .dropna()
)

for col in ["ch1", "ch2", "ch3"]:
    df[col] = df[col].astype(int)

map_ch1 = {
    1: "C1: Recurrent AURI",
    2: "C2: Recurrent SYM",
    3: "C3: Severe LRT and OBS",
    4: "C4: Transient AURI",
    5: "C5: Mixed respiratory events",
    6: "C6: Late SYM",
    7: "C7: Low morbidity",
}

map_ch2 = {
    1: "C1: Antihistamines & mixed",
    2: "C2: Nasal prep & mixed",
    3: "C3: Minimal use",
    4: "C4: Frequent OAD",
    5: "C5: Frequent ANT",
}

map_ch3 = {
    1: "C1: No utilization",
    2: "C2: Frequent utilization",
    3: "C3: Sporadic utilization",
}

df["ch1_lbl"] = df["ch1"].apply(lambda x: map_ch1.get(x, f"Ch1_C{x}"))
df["ch2_lbl"] = df["ch2"].apply(lambda x: map_ch2.get(x, f"Ch2_C{x}"))
df["ch3_lbl"] = df["ch3"].apply(lambda x: map_ch3.get(x, f"Ch3_C{x}"))

layers = [
    sorted(df["ch1_lbl"].unique().tolist()),
    sorted(df["ch2_lbl"].unique().tolist()),
    sorted(df["ch3_lbl"].unique().tolist()),
]

# Set layout dimensions
fig, ax = plt.subplots(figsize=(14, 8))
layer_x = [0.0, 0.5, 1.0]
node_width = 0.03
total_height = 1.0
gap = 0.02
total_patients = len(df)

# Compute vertical node positions
node_positions = {}
for l_idx, (col_name, nodes) in enumerate(
    zip(["ch1_lbl", "ch2_lbl", "ch3_lbl"], layers)
):
    counts = df[col_name].value_counts()
    n_nodes = len(nodes)
    usable_h = total_height - (n_nodes - 1) * gap
    curr_y = total_height

    for node in nodes:
        node_h = (counts.get(node, 0) / total_patients) * usable_h
        node_positions[node] = {
            "x": layer_x[l_idx],
            "y_top": curr_y,
            "y_bottom": curr_y - node_h,
            "height": node_h,
            "curr_out_y": curr_y,
            "curr_in_y": curr_y,
        }
        curr_y -= node_h + gap

# Academic color palette per layer
layer_colors = ["#D72626", "#28c430", "#1e11b2"]


# Bezier ribbon drawing function
def draw_ribbon(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot, color):
    verts = [
        (x0, y0_top),
        (x0 + (x1 - x0) / 2, y0_top),
        (x1 - (x1 - x0) / 2, y1_top),
        (x1, y1_top),
        (x1, y1_bot),
        (x1 - (x1 - x0) / 2, y1_bot),
        (x0 + (x1 - x0) / 2, y0_bot),
        (x0, y0_bot),
        (x0, y0_top),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    path = MPath(verts, codes)
    patch = mpatches.PathPatch(
        path, facecolor=color, edgecolor="none", alpha=0.35
    )
    ax.add_patch(patch)


# Draw flows: Layer 1 -> Layer 2
flow_1_2 = df.groupby(["ch1_lbl", "ch2_lbl"]).size().reset_index(name="count")
for _, row in flow_1_2.iterrows():
    src, tgt, cnt = row["ch1_lbl"], row["ch2_lbl"], row["count"]
    src_h = (
        cnt / total_patients
    ) * node_positions[src]["height"] / (df["ch1_lbl"] == src).mean()
    tgt_h = (
        cnt / total_patients
    ) * node_positions[tgt]["height"] / (df["ch2_lbl"] == tgt).mean()

    s_top = node_positions[src]["curr_out_y"]
    s_bot = s_top - src_h
    node_positions[src]["curr_out_y"] = s_bot

    t_top = node_positions[tgt]["curr_in_y"]
    t_bot = t_top - tgt_h
    node_positions[tgt]["curr_in_y"] = t_bot

    draw_ribbon(
        ax,
        node_positions[src]["x"] + node_width,
        s_top,
        s_bot,
        node_positions[tgt]["x"],
        t_top,
        t_bot,
        "#6097c9",
    )

# Draw flows: Layer 2 -> Layer 3
flow_2_3 = df.groupby(["ch2_lbl", "ch3_lbl"]).size().reset_index(name="count")
for _, row in flow_2_3.iterrows():
    src, tgt, cnt = row["ch2_lbl"], row["ch3_lbl"], row["count"]
    src_h = (
        cnt / total_patients
    ) * node_positions[src]["height"] / (df["ch2_lbl"] == src).mean()
    tgt_h = (
        cnt / total_patients
    ) * node_positions[tgt]["height"] / (df["ch3_lbl"] == tgt).mean()

    s_top = node_positions[src]["curr_out_y"]
    s_bot = s_top - src_h
    node_positions[src]["curr_out_y"] = s_bot

    t_top = node_positions[tgt]["curr_in_y"]
    t_bot = t_top - tgt_h
    node_positions[tgt]["curr_in_y"] = t_bot

    draw_ribbon(
        ax,
        node_positions[src]["x"] + node_width,
        s_top,
        s_bot,
        node_positions[tgt]["x"],
        t_top,
        t_bot,
        "#6097c9",
    )

# Draw rectangular nodes and labels
for l_idx, nodes in enumerate(layers):
    color = layer_colors[l_idx]
    for node in nodes:
        info = node_positions[node]
        rect = mpatches.Rectangle(
            (info["x"], info["y_bottom"]),
            node_width,
            info["height"],
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.add_patch(rect)

        # Vertical center of the block
        y_center = info["y_bottom"] + info["height"] / 2

        if l_idx == 0:
            # Channel 1: aligned to the left of the block, centered vertically
            ax.text(
                info["x"] - 0.015,
                y_center,
                node,
                va="center",
                ha="right",
                fontsize=9,
                fontweight="bold",
            )
        elif l_idx == len(layers) - 1:
            # Channel 3: aligned to the right of the block, centered vertically
            ax.text(
                info["x"] + node_width + 0.015,
                y_center,
                node,
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )
        else:
            # Channel 2 (middle): centered vertically at the mid-point of the block
            ax.text(
                info["x"] - 0.012,
                y_center,
                node,
                va="center",
                ha="right",
                fontsize=8.5,
                fontweight="bold",
            )

# Titles for columns
col_titles = [
    "Clinical Phenotypes",
    "Pharmacological Profiles",
    "Secondary Care Utilization",
]
for idx, (x, title) in enumerate(zip(layer_x, col_titles)):
    offset = node_width / 2
    ax.text(
        x + offset,
        total_height + 0.05,
        title,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.set_xlim(-0.25, 1.25)
ax.set_ylim(-0.05, total_height + 0.1)
ax.axis("off")

# Save PDF directly without external browsers
out_dir = ROOT_SEQUENCE
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(
    out_dir / "sankey_multichannel_transitions.pdf",
    bbox_inches="tight",
    dpi=300,
)
plt.close()









from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -------------------------------------------------------------
# 1. Data Alignment and Label Mapping
# -------------------------------------------------------------
df = (
    clust_1.iloc[:, 0]
    .to_frame(name="ch1")
    .join(clust_2.iloc[:, 0].to_frame(name="ch2"))
    .join(clust_3.iloc[:, 0].to_frame(name="ch3"))
    .dropna()
)

for col in ["ch1", "ch2", "ch3"]:
    df[col] = df[col].astype(int)

map_ch1 = {
    1: "C1: Recurrent AURI",
    2: "C2: Recurrent SYM",
    3: "C3: Severe LRT and OBS",
    4: "C4: Transient AURI",
    5: "C5: Mixed resp. events",
    6: "C6: Late SYM",
    7: "C7: Low morbidity",
}

map_ch2 = {
    1: "C1: Antihistamines & mixed",
    2: "C2: Nasal prep & mixed",
    3: "C3: Minimal use",
    4: "C4: Frequent OAD",
    5: "C5: Frequent ANT",
}

map_ch3 = {
    1: "C1: No utilization",
    2: "C2: Frequent utilization",
    3: "C3: Sporadic utilization",
}

df["ch1_lbl"] = df["ch1"].apply(lambda x: map_ch1.get(x, f"Ch1_C{x}"))
df["ch2_lbl"] = df["ch2"].apply(lambda x: map_ch2.get(x, f"Ch2_C{x}"))
df["ch3_lbl"] = df["ch3"].apply(lambda x: map_ch3.get(x, f"Ch3_C{x}"))

layers = [
    [map_ch1[i] for i in sorted(map_ch1.keys()) if map_ch1[i] in df["ch1_lbl"].values],
    [map_ch2[i] for i in sorted(map_ch2.keys()) if map_ch2[i] in df["ch2_lbl"].values],
    [map_ch3[i] for i in sorted(map_ch3.keys()) if map_ch3[i] in df["ch3_lbl"].values],
]

# -------------------------------------------------------------
# 2. Dimensions and Absolute Node Coordinate Calculation
# -------------------------------------------------------------
layer_x = [0.0, 0.45, 0.90]
node_width = 0.025
total_height = 1.0
gap = 0.02
total_patients = len(df)
layer_colors = ["#D72626", "#108A55", "#1B2A80"]

base_node_positions = {}
usable_heights = {}

for l_idx, (col_name, nodes) in enumerate(
    zip(["ch1_lbl", "ch2_lbl", "ch3_lbl"], layers)
):
    counts = df[col_name].value_counts()
    n_nodes = len(nodes)
    usable_h = total_height - (n_nodes - 1) * gap
    usable_heights[col_name] = usable_h
    curr_y = total_height

    for node in nodes:
        node_h = (counts.get(node, 0) / total_patients) * usable_h
        base_node_positions[node] = {
            "x": layer_x[l_idx],
            "y_top": curr_y,
            "y_bottom": curr_y - node_h,
            "height": node_h,
            "col": col_name,
        }
        curr_y -= node_h + gap

alluvial_flows = (
    df.groupby(["ch1_lbl", "ch2_lbl", "ch3_lbl"])
    .size()
    .reset_index(name="count")
)

order_c1 = {val: i for i, val in enumerate(layers[0])}
order_c2 = {val: i for i, val in enumerate(layers[1])}
order_c3 = {val: i for i, val in enumerate(layers[2])}

alluvial_flows["rank1"] = alluvial_flows["ch1_lbl"].map(order_c1)
alluvial_flows["rank2"] = alluvial_flows["ch2_lbl"].map(order_c2)
alluvial_flows["rank3"] = alluvial_flows["ch3_lbl"].map(order_c3)

# -------------------------------------------------------------
# 3. Bézier Ribbon Generation
# -------------------------------------------------------------
def draw_ribbon(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot, color, edge, alpha, zorder):
    verts = [
        (x0, y0_top),
        (x0 + (x1 - x0) / 2, y0_top),
        (x1 - (x1 - x0) / 2, y1_top),
        (x1, y1_top),
        (x1, y1_bot),
        (x1 - (x1 - x0) / 2, y1_bot),
        (x0 + (x1 - x0) / 2, y0_bot),
        (x0, y0_bot),
        (x0, y0_top),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    patch = mpatches.PathPatch(
        MPath(verts, codes),
        facecolor=color,
        edgecolor=edge,
        linewidth=0.7 if edge != "none" else 0.0,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)

# -------------------------------------------------------------
# 4. Subplot Rendering Function
# -------------------------------------------------------------
def render_alluvial_subplot(ax, highlight_c1, show_labels=True):
    df_step1 = alluvial_flows.sort_values(by=["rank1", "rank2", "rank3"]).copy()
    y_out1 = {k: v["y_top"] for k, v in base_node_positions.items() if v["col"] == "ch1_lbl"}
    y_in2 = {k: v["y_top"] for k, v in base_node_positions.items() if v["col"] == "ch2_lbl"}
    
    step1_bands = []
    for _, row in df_step1.iterrows():
        c1, c2, c3, cnt = row["ch1_lbl"], row["ch2_lbl"], row["ch3_lbl"], row["count"]
        h1 = (cnt / total_patients) * usable_heights["ch1_lbl"]
        h2 = (cnt / total_patients) * usable_heights["ch2_lbl"]
        
        s_top = y_out1[c1]
        s_bot = s_top - h1
        y_out1[c1] = s_bot
        
        t_top = y_in2[c2]
        t_bot = t_top - h2
        y_in2[c2] = t_bot
        
        step1_bands.append({
            "c1": c1, "c2": c2, "c3": c3,
            "x0": base_node_positions[c1]["x"] + node_width,
            "s_top": s_top, "s_bot": s_bot,
            "x1": base_node_positions[c2]["x"],
            "t_top": t_top, "t_bot": t_bot,
        })

    df_step2 = alluvial_flows.sort_values(by=["rank2", "rank1", "rank3"]).copy()
    y_out2 = {k: v["y_top"] for k, v in base_node_positions.items() if v["col"] == "ch2_lbl"}
    y_in3 = {k: v["y_top"] for k, v in base_node_positions.items() if v["col"] == "ch3_lbl"}
    
    step2_bands = []
    for _, row in df_step2.iterrows():
        c1, c2, c3, cnt = row["ch1_lbl"], row["ch2_lbl"], row["ch3_lbl"], row["count"]
        h2 = (cnt / total_patients) * usable_heights["ch2_lbl"]
        h3 = (cnt / total_patients) * usable_heights["ch3_lbl"]
        
        s_top = y_out2[c2]
        s_bot = s_top - h2
        y_out2[c2] = s_bot
        
        t_top = y_in3[c3]
        t_bot = t_top - h3
        y_in3[c3] = t_bot
        
        step2_bands.append({
            "c1": c1, "c2": c2, "c3": c3,
            "x0": base_node_positions[c2]["x"] + node_width,
            "s_top": s_top, "s_bot": s_bot,
            "x1": base_node_positions[c3]["x"],
            "t_top": t_top, "t_bot": t_bot,
        })

    # Gray background ribbons
    for b in step1_bands:
        if b["c1"] != highlight_c1:
            draw_ribbon(ax, b["x0"], b["s_top"], b["s_bot"], b["x1"], b["t_top"], b["t_bot"],
                        color="#E5E8EB", edge="none", alpha=0.5, zorder=1)
    for b in step2_bands:
        if b["c1"] != highlight_c1:
            draw_ribbon(ax, b["x0"], b["s_top"], b["s_bot"], b["x1"], b["t_top"], b["t_bot"],
                        color="#E5E8EB", edge="none", alpha=0.5, zorder=1)

    # Foreground highlighted ribbons
    for b in step1_bands:
        if b["c1"] == highlight_c1:
            draw_ribbon(ax, b["x0"], b["s_top"], b["s_bot"], b["x1"], b["t_top"], b["t_bot"],
                        color="#5B9BD5", edge="#2C6DB2", alpha=0.85, zorder=3)
    for b in step2_bands:
        if b["c1"] == highlight_c1:
            draw_ribbon(ax, b["x0"], b["s_top"], b["s_bot"], b["x1"], b["t_top"], b["t_bot"],
                        color="#5B9BD5", edge="#2C6DB2", alpha=0.85, zorder=3)

    # Node blocks
    for l_idx, nodes in enumerate(layers):
        color = layer_colors[l_idx]
        for node in nodes:
            info = base_node_positions[node]
            rect = mpatches.Rectangle(
                (info["x"], info["y_bottom"]),
                node_width,
                info["height"],
                facecolor=color,
                edgecolor="black",
                linewidth=0.6,
                zorder=4,
            )
            ax.add_patch(rect)

            if show_labels:
                y_center = info["y_bottom"] + info["height"] / 2
                if l_idx == 0:
                    ax.text(info["x"] - 0.012, y_center, node, va="center", ha="right", fontsize=7.5, fontweight="bold")
                elif l_idx == 2:
                    ax.text(info["x"] + node_width + 0.012, y_center, node, va="center", ha="left", fontsize=7.5, fontweight="bold")
                else:
                    ax.text(info["x"] - 0.010, y_center, node, va="center", ha="right", fontsize=6.8, fontweight="bold")

    ax.set_title(f"Focus: {highlight_c1}", fontsize=10.5, fontweight="bold", pad=5)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.02, total_height + 0.04)
    ax.axis("off")

# -------------------------------------------------------------
# 5. Export Functions for Column-Wise Figures
# -------------------------------------------------------------
out_dir = ROOT_SEQUENCE
out_dir.mkdir(parents=True, exist_ok=True)

c1_clusters = layers[0]

# --- Figure 1: First 2 clusters (1 column, 2 rows) ---
clusters_part1 = c1_clusters[:4]
fig1, axes1 = plt.subplots(
    len(clusters_part1), 
    1, 
    figsize=(9, 3.8 * len(clusters_part1)), 
    gridspec_kw={"hspace": 0.15}
)
if len(clusters_part1) == 1:
    axes1 = [axes1]

for ax, label in zip(axes1, clusters_part1):
    render_alluvial_subplot(ax, highlight_c1=label, show_labels=True)

plt.savefig(
    out_dir / "alluvial_part1_2clusters.png",
    bbox_inches="tight",
    pad_inches=0.02,
    dpi=300,
)
plt.close(fig1)

# --- Figure 2: Remaining 5 clusters (1 column, 5 rows) ---
clusters_part2 = c1_clusters[4:]
fig2, axes2 = plt.subplots(
    len(clusters_part2), 
    1, 
    figsize=(9, 3.8 * len(clusters_part2)), 
    gridspec_kw={"hspace": 0.15}
)
if len(clusters_part2) == 1:
    axes2 = [axes2]

for ax, label in zip(axes2, clusters_part2):
    render_alluvial_subplot(ax, highlight_c1=label, show_labels=True)

plt.savefig(
    out_dir / "alluvial_part2_5clusters.png",
    bbox_inches="tight",
    pad_inches=0.02,
    dpi=300,
)
plt.close(fig2)