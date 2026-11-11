# Code thesis: Multichannel State Sequence Analysis of Care Pathways for Phenotyping and Risk Prediction: A Case Study on Pediatric Respiratory Diseases

The objective of the thesis is to model pediatric respiratory care pathways from routinely collected data using  **State Sequence Analysis (SSA)** and to implement **sequence-based risk prediction** through survival models.

> The raw data are not included in this repository (sensitive health data). This README only describes how the code is organized.

## Repository structure

```
THESIS_BREATH/
├── data/              # raw data (not tracked)
├── data_exploration/  # exploratory analysis of the raw data
└── SSA/
    ├── src/           # SSA pipeline code
    ├── datasets/      # intermediate datasets (not tracked)
    ├── results/       # figures, tables, dissimilarity matrices, clusters (not tracked)
    └── prediction/    # prediction results (not tracked)
```

- **`data_exploration/`**: descriptive analyses of the raw data, done before building the sequences.
- **`SSA/`**: the main part of the thesis: construction of the sequences, their analysis and clustering, and the prediction of future acute respiratory events.

## SSA pipeline

The pipeline is defined in **`SSA/src/main.py`**, which runs three steps in order. All the parameters are set in **`SSA/src/config.py`**.

```
python main.py                  # all steps
python main.py --step dataset   # step 1 only
python main.py --step analysis  # step 2 only
python main.py --step predict   # step 3 only
```

### Configuration (`config.py`)
Single place where the analysis is defined. It contains:
- the **channels** each with its data source, states, state priority, colors and substitution costs;
- the **sequence parameters** (alignment by age or by season, time unit, sequence length);
- the **multichannel setting** (number of channels and weights);
- the **dissimilarity measure** and the **clustering** settings (algorithm, linkage, number of clusters);
- the **prediction** settings (target, features, model);
- the paths of all input and output folders.

### Step 1: Sequence dataset (`dataset_sequences.py`, `utils_sequences.py`)
Selects the clinical events of the chosen channel, maps ICD-9 / ATC / care codes to states, selects the patient cohort, and builds one sequence per patient (one state per time unit, with a priority rule when several events fall in the same unit). It also saves descriptive plots and tables of the preprocessed data.
The cohort is selected from a primary channel (diagnoses); the other channels load the saved cohort.

### Step 2: Sequence analysis and clustering
- **Single channel** (`sequence_analysis.py`, `dissimilarity.py`, `utils_SSA.py`): sequence visualization, sequence characteristics (e.g. entropy, turbulence, complexity), pairwise dissimilarity matrix, hierarchical clustering, and cluster profiling.
- **Multichannel** (`multi_channel_clustering.py`): loads the dissimilarity matrices of each channel, normalizes and combines them with the channel weights, and clusters the combined matrix. Used when `MULTICHANNEL=True`.

### Step 3: Risk prediction (`prediction.py`)
Defines the outcome in a follow-up window after the end of the sequence (e.g. lower respiratory tract infections, obstructive airway drug prescriptions), builds the features (patient characteristics, cluster membership, sequence characteristics) and fits the selected model: Cox proportional hazards (also stratified or with time-varying covariates), or logistic / lasso regression. It reports hazard ratios and the concordance index.

## Files in `SSA/src/`

| File | Role |
|------|------|
| `main.py` | Pipeline entry point (steps 1 → 3) |
| `config.py` | All parameters, channels and paths |
| `dataset_sequences.py` | Step 1: builds the sequence dataset |
| `utils_sequences.py` | Helpers for step 1 (cohort selection, preprocessing, exploration) |
| `sequence_analysis.py` | Step 2: single-channel analysis |
| `dissimilarity.py` | Dissimilarity matrix computation |
| `multi_channel_clustering.py` | Step 2: multichannel clustering |
| `utils_SSA.py` | Helpers for step 2 (visualization, characteristics, clustering, cluster analysis) |
| `prediction.py` | Step 3: targets, features and prediction models |
