################################################################################################################
#################################### STATE SEQUENCE ANALYSIS CONFIGURATION #####################################
################################################################################################################
#This file contains the configuration parameters for the state sequence analysis. 
# It defines the channels, their specific parameters, and other settings for the analysis.
#
# CONTENTS OF THIS FILE
#   1. CHANNELS SET-UP
#        - `channels` dictionary: for each channel it defines the dataset file,
#          the state/date columns, the covariates, the state labels, the state priority,
#          the colors and the substitution costs
#        - active channel(s) (CHANNEL, CHANNEL_1..3) and multichannel setting
#          (MULTICHANNEL, NUM_CHANNELS, channel weights W_0..W_3)
#   2. SEQUENCE PARAMETERS
#        - alignment (SEASON: by season or by age)
#        - time granularity (TIME_UNIT, MONTHS) and sequence length (SEQ_LENGTH, YEARS)
#        - alphabet (STATES_LABELS) and state priority for overlapping events (STATE_PRIORITY)
#   3. SEQUENCE DATA CONTENT
#        - dataset path (DATASET_PATH)
#        - column names (COL_ID, COL_DATE, COL_STATE) and covariates (COVARIATES)
#   4. SEQUENCE VISUALIZATION & RESULTS PATH
#        - color utilities and color palettes (CUSTOM_COLORS..)
#        - sequence names used to save/load results (SEQUENCE_NAME..)
#   5. SEQUENCE DISSIMILARITY AND CLUSTERING
#        - fixed substitution costs (COSTS)
#        - dissimilarity measure (DISTANCE_METRIC..)
#        - clustering algorithm (CLUSTERING), linkage (LINKAGE)
#        - number of clusters (MAX_CLUSTERS, NUM_CLUSTERS)
#   6. SEQUENCE-BASED RISK PREDICTION MODELING
#        - prediction target (PRED_NAME), features (FEATURES, CAT_FEATURES)
#        - model (MODEL), stratification variable (STRATA)
#        - multichannel clusters in prediction (FLAG)
#

##########################################################################################
################################ 1. CHANNELS SET-UP ######################################
##########################################################################################
channels={
    #DIAGNOSES SEQUENCES specific parameters
    'diagnoses': {
        'dataset_path': 'contatti.xlsx',
        'col_state': 'diagnosi1_icd9',
        'col_date': 'datacontatto',
        'covariates': ['datan','sesso','idmedico','EF','adi'],
        'states_labels': {
            'PNE': 'Pneumonia',
            'BRC': 'Bronchiolitis',
            'AST': 'Asthma',
            'BRN': 'Bronchitis',
            'WHZ': 'Wheezing',
            'URI': 'Acute Upper Respiratory Infections',
            'BSP': 'Bronchospasm',
            'FLU': 'Influenza',
            'COV': 'Covid',
            'PHT': 'Pharyngotonsillitis',
            'TON': 'Tonsillitis',
            'OTI': 'Otitis',
            'RNF': 'Rhinopharyngitis',
            'LAR': 'Laryngitis',
            'PHA': 'Pharyngitis',
            'SIN': 'Sinusitis',
            'RHN': 'Rhinitis',
            'FEV': 'Fever',
            'COU': 'Cough',
            'OTH': 'Other',
            'NO': 'No Diagnosis',
        },
        'state_priority':{
            'NO': 0,
            'OTH': 1,
            'COU': 2,
            'FEV': 3,
            'RHN': 4,
            'SIN': 5,
            'PHA': 6,
            'LAR': 7,
            'RNF': 8,
            'OTI': 9,
            'TON': 10,
            'PHT': 11,
            'COV': 12,
            'FLU': 13,
            'BSP': 14,
            'URI': 15,
            'WHZ': 16,
            'BRN': 17,
            'AST': 18,
            'BRC': 19,
            'PNE': 20,
        },
        'costs': {
            'PNE': {'PNE': 0.0, 'BRC': 0.5, 'AST': 1.0, 'BRN': 0.5, 'WHZ': 1.0, 'URI': 1.0, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 1.0, 'TON': 1.0, 'OTI': 1.0, 'RNF': 1.0, 'LAR': 1.0, 'PHA': 1.0, 'SIN': 1.0, 'RHN': 1.0, 'FEV': 1.0, 'COU': 1.0, 'OTH': 0.5, 'NO': 1.0},
            'BRC': {'PNE': 0.5, 'BRC': 0.0, 'AST': 1.0, 'BRN': 0.5, 'WHZ': 1.0, 'URI': 1.0, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 1.0, 'TON': 1.0, 'OTI': 1.0, 'RNF': 1.0, 'LAR': 1.0, 'PHA': 1.0, 'SIN': 1.0, 'RHN': 1.0, 'FEV': 1.0, 'COU': 1.0, 'OTH': 0.5, 'NO': 1.0},
            'AST': {'PNE': 1.0, 'BRC': 1.0, 'AST': 0.0, 'BRN': 1.0, 'WHZ': 0.5, 'URI': 1.0, 'BSP': 0.5, 'FLU': 1.0, 'COV': 1.0, 'PHT': 1.0, 'TON': 1.0, 'OTI': 1.0, 'RNF': 1.0, 'LAR': 1.0, 'PHA': 1.0, 'SIN': 1.0, 'RHN': 1.0, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'BRN': {'PNE': 0.5, 'BRC': 0.5, 'AST': 1.0, 'BRN': 0.0, 'WHZ': 1.0, 'URI': 1.0, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 1.0, 'TON': 1.0, 'OTI': 1.0, 'RNF': 1.0, 'LAR': 1.0, 'PHA': 1.0, 'SIN': 1.0, 'RHN': 1.0, 'FEV': 1.0, 'COU': 1.0, 'OTH': 0.5, 'NO': 1.0},
            'WHZ': {'PNE': 1.0, 'BRC': 1.0, 'AST': 0.5, 'BRN': 1.0, 'WHZ': 0.0, 'URI': 1.0, 'BSP': 0.5, 'FLU': 1.0, 'COV': 1.0, 'PHT': 1.0, 'TON': 1.0, 'OTI': 1.0, 'RNF': 1.0, 'LAR': 1.0, 'PHA': 1.0, 'SIN': 1.0, 'RHN': 1.0, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'URI': {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 0.0, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 0.5, 'TON': 0.5, 'OTI': 0.5, 'RNF': 0.5, 'LAR': 0.5, 'PHA': 0.5, 'SIN': 0.5, 'RHN': 0.5, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'BSP': {'PNE': 1.0, 'BRC': 1.0, 'AST': 0.5, 'BRN': 1.0, 'WHZ': 0.5, 'URI': 1.0, 'BSP': 0.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 1.0, 'TON': 1.0, 'OTI': 1.0, 'RNF': 1.0, 'LAR': 1.0, 'PHA': 1.0, 'SIN': 1.0, 'RHN': 1.0, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'FLU': {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 1.0, 'BSP': 1.0, 'FLU': 0.0, 'COV': 0.5, 'PHT': 1.0, 'TON': 1.0, 'OTI': 1.0, 'RNF': 1.0, 'LAR': 1.0, 'PHA': 1.0, 'SIN': 1.0, 'RHN': 1.0, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'COV': {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 1.0, 'BSP': 1.0, 'FLU': 0.5, 'COV': 0.0, 'PHT': 1.0, 'TON': 1.0, 'OTI': 1.0, 'RNF': 1.0, 'LAR': 1.0, 'PHA': 1.0, 'SIN': 1.0, 'RHN': 1.0, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'PHT': {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 0.5, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 0.0, 'TON': 0.5, 'OTI': 0.5, 'RNF': 0.5, 'LAR': 0.5, 'PHA': 0.5, 'SIN': 0.5, 'RHN': 0.5, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'TON': {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 0.5, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 0.5, 'TON': 0.0, 'OTI': 0.5, 'RNF': 0.5, 'LAR': 0.5, 'PHA': 0.5, 'SIN': 0.5, 'RHN': 0.5, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'OTI': {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 0.5, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 0.5, 'TON': 0.5, 'OTI': 0.0, 'RNF': 0.5, 'LAR': 0.5, 'PHA': 0.5, 'SIN': 0.5, 'RHN': 0.5, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'RNF': {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 0.5, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 0.5, 'TON': 0.5, 'OTI': 0.5, 'RNF': 0.0, 'LAR': 0.5, 'PHA': 0.5, 'SIN': 0.5, 'RHN': 0.5, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'LAR': {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 0.5, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 0.5, 'TON': 0.5, 'OTI': 0.5, 'RNF': 0.5, 'LAR': 0.0, 'PHA': 0.5, 'SIN': 0.5, 'RHN': 0.5, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'PHA': {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 0.5, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 0.5, 'TON': 0.5, 'OTI': 0.5, 'RNF': 0.5, 'LAR': 0.5, 'PHA': 0.0, 'SIN': 0.5, 'RHN': 0.5, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'SIN': {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 0.5, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 0.5, 'TON': 0.5, 'OTI': 0.5, 'RNF': 0.5, 'LAR': 0.5, 'PHA': 0.5, 'SIN': 0.0, 'RHN': 0.5, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'RHN': {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 0.5, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 0.5, 'TON': 0.5, 'OTI': 0.5, 'RNF': 0.5, 'LAR': 0.5, 'PHA': 0.5, 'SIN': 0.5, 'RHN': 0.0, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO': 1.0},
            'FEV': {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 1.0, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 1.0, 'TON': 1.0, 'OTI': 1.0, 'RNF': 1.0, 'LAR': 1.0, 'PHA': 1.0, 'SIN': 1.0, 'RHN': 1.0, 'FEV': 0.0, 'COU': 0.5, 'OTH': 1.0, 'NO': 1.0},
            'COU': {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 1.0, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 1.0, 'TON': 1.0, 'OTI': 1.0, 'RNF': 1.0, 'LAR': 1.0, 'PHA': 1.0, 'SIN': 1.0, 'RHN': 1.0, 'FEV': 0.5, 'COU': 0.0, 'OTH': 1.0, 'NO': 1.0},
            'OTH': {'PNE': 0.5, 'BRC': 0.5, 'AST': 1.0, 'BRN': 0.5, 'WHZ': 1.0, 'URI': 1.0, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 1.0, 'TON': 1.0, 'OTI': 1.0, 'RNF': 1.0, 'LAR': 1.0, 'PHA': 1.0, 'SIN': 1.0, 'RHN': 1.0, 'FEV': 1.0, 'COU': 1.0, 'OTH': 0.0, 'NO': 1.0},
            'NO':  {'PNE': 1.0, 'BRC': 1.0, 'AST': 1.0, 'BRN': 1.0, 'WHZ': 1.0, 'URI': 1.0, 'BSP': 1.0, 'FLU': 1.0, 'COV': 1.0, 'PHT': 1.0, 'TON': 1.0, 'OTI': 1.0, 'RNF': 1.0, 'LAR': 1.0, 'PHA': 1.0, 'SIN': 1.0, 'RHN': 1.0, 'FEV': 1.0, 'COU': 1.0, 'OTH': 1.0, 'NO':  0.0},
        }
    },


    #DIAGNOSES MACROGROUPS SEQUENCES specific parameters
    'macrodiagnoses': {
        'dataset_path': 'contatti.xlsx',
                'col_state': 'diagnosi1_icd9',
                'col_date': 'datacontatto',
                'covariates': ['datan','sesso','idmedico','EF','adi'],
                'states_labels': {
    'URT': 'Upper Respiratory Tract Infections',
    'AURI': 'Acute Upper Respiratory Infections',
    'LRT': 'Lower Respiratory Tract Infections',
    'OBS': 'Obstructive Airway Diseases',
    'VIR': 'Influenza / COVID-19',
    'SYM': 'General Symptoms',
    'OTH': 'Other Respiratory Diseases',
    'NO': 'No Diagnosis',
},
    'state_priority':
    {   'NO':0,
        'SYM':1,
        'OTH':2,
        'URT':3,
        'AURI':4,
        'LRT':5,
        'OBS':6,
        'VIR':7
    },
    'colors': {
            'URT':  "#2EC046",  
            'AURI': "#DF1818",  
            'LRT':  "#6839CE",  
            'OBS':  "#DD7F3B",  
            'VIR':  "#488DD7",  
            'SYM':  "#D740BB",  
            'OTH':  "#DFD625", 
            'NO':   '#CCCCCC',  
        },
     'costs' : {
        'URT': {'URT': 0.00, 'AURI': 0.08, 'LRT': 0.12, 'OBS': 0.15, 'VIR': 0.14, 'SYM': 0.17, 'OTH': 0.21, 'NO': 0.27},
        'AURI': {'URT': 0.08, 'AURI': 0.00, 'LRT': 0.13, 'OBS': 0.17, 'VIR': 0.15, 'SYM': 0.18, 'OTH': 0.21, 'NO': 0.28},
        'LRT': {'URT': 0.12, 'AURI': 0.13, 'LRT': 0.00, 'OBS': 0.16, 'VIR': 0.16, 'SYM': 0.18, 'OTH': 0.20, 'NO': 0.26},
        'OBS': {'URT': 0.15, 'AURI': 0.17, 'LRT': 0.16, 'OBS': 0.00, 'VIR': 0.17, 'SYM': 0.15, 'OTH': 0.19, 'NO': 0.24},
        'VIR': {'URT': 0.14, 'AURI': 0.15, 'LRT': 0.16, 'OBS': 0.17, 'VIR': 0.00, 'SYM': 0.12, 'OTH': 0.15, 'NO': 0.19},
        'SYM': {'URT': 0.17, 'AURI': 0.18, 'LRT': 0.18, 'OBS': 0.15, 'VIR': 0.12, 'SYM': 0.00, 'OTH': 0.12, 'NO': 0.15},
        'OTH': {'URT': 0.21, 'AURI': 0.21, 'LRT': 0.20, 'OBS': 0.19, 'VIR': 0.15, 'SYM': 0.12, 'OTH': 0.00, 'NO': 0.14},
        'NO':  {'URT': 0.27, 'AURI': 0.28, 'LRT': 0.26, 'OBS': 0.24, 'VIR': 0.19, 'SYM': 0.15, 'OTH': 0.14, 'NO': 0.00},
    }
    },

    #DIAGNOSES SUPER MACROGROUPS SEQUENCES specific parameters
    'supermacrodiagnoses': {
        'dataset_path': 'contatti.xlsx',
        'col_state': 'diagnosi1_icd9',
        'col_date': 'datacontatto',
        'covariates': ['datan','sesso','idmedico','EF','adi'],
        'states_labels': {
        'INF_SUP': 'Infezioni delle vie aree superiori',
        'INF_ACUTE_SUP': 'Infezioni acute delle vie aree superiori',
        'INF_INF_OSTR': 'Infezioni delle vie aree inferiori e patologie ostruttive',
        'SIM_ASPEC': 'Sintomi aspecifici',
        'ALT': 'Altre patologie respiratorie',
        'NO':  'Nessuna diagnosi',
    },
    'state_priority':
    {   'NO':0,
        'SIM_ASPEC':1,
        'ALT':2,
        'INF_SUP':3,
        'INF_ACUTE_SUP':4,
        'INF_INF_OSTR':5
    },
    'costs': None
    },

    #DRUGS SEQUENCES specific parameters
     'drugs': {
         'dataset_path': 'farmaci_selected.csv',
         'col_state': 'atc',    
         'col_date': 'dataricetta',
         'covariates': ['datan','sesso','idmedico','EF','adi'],
         'states_labels': {
                'ANT': 'Antibiotics',               # J01
                'NAS': 'Nasal preparations',        # R01
                'THR': 'Throat preparations',       # R02
                'OAD': 'Obstructive airway drugs',  # R03
                'COC': 'Cough and cold preparations', # R05
                'ANH': 'Antihistamines',            # R06
                'NO':  'No prescription',
            },
        'state_priority':
        {   'NO':0,
            'THR':1,
            'NAS':2,
            'COC':3,
            'ANH':4,
            'OAD':5,
            'ANT':6
        },
        'colors': {
            'ANT': "#227F71",  
            'OAD': "#A92CC8",  
            'COC': "#C4AF26",  
            'ANH': "#78DF9E",  
            'NAS': "#E7445A",  
            'THR': "#E991CB", 
            'NO':  '#CCCCCC',  
        },
        'costs': None

    },
    
    #SECONDARY CARE SEQUENCES specific parameters
    'secondary_care': {
        'dataset_path': 'care_setting.csv',
        'col_state': 'care_setting',    
        'col_date': 'date',
        'covariates': ['datan','sesso','idmedico','EF','adi'],
        'states_labels': {

            'HOS': 'Hospitalization',

            'ER': 'Emergency room access',

            'SV': 'Specialist outpatient visit',

            'NO': 'No event',

        },  
        'state_priority':
        {   'NO':0,
            'SV':1,
            'ER':2,
            'HOS':3
        },
        'colors': {
            'HOS': "#DFEE0B",  
            'ER':  "#17138C",  
            'SV':  "#24CCE2",  
            'NO':  '#CCCCCC', 
        },
        'costs': None
    },

    # HEALTH SEQUENCES specific parameters
    'health': {
        'dataset_path': 'contatti.xlsx',
        'col_state': 'diagnosi1_icd9',
        'col_date': 'datacontatto',
        'covariates': ['datan','sesso','idmedico','EF','adi'],
        'states_labels': {
            'SICK': 'Sick',
            'NO':   'No diagnosis',
        },
        'state_priority':
        {   'NO':0,
            'SICK':1
        },
        'costs': None
    }    

}


# ---- SEQUENCE CHANNELS ----
#options: 'diagnoses','macrodiagnoses', 'supermacrodiagnoses','drugs','secondary_care','health'
CHANNEL='macrodiagnoses'
CHANNEL_1='drugs' 
CHANNEL_2='secondary_care' 
CHANNEL_3='health'
   

#---- MULTICHANNEL SETTING ----
#Multichannel flag: if True the analysis will consider NUM_CHANNELS channels
                   #if False, the analysis will consider only the CHANNEL channel
MULTICHANNEL=False
NUM_CHANNELS=3  

#Weights assigned to each channel in the multichannel analysis
W_0=1 #weight for channel 0
W_1=1 #weight for channel 1
W_2=1  #weight for channel 2
W_3=1  #weight for channel 3


##########################################################################################
################################ 2. SEQUENCE PARAMETERS ##################################
##########################################################################################

#---- SEQUENCES ALIGNMENT OPTION/CHOICE ----
#if season is True, sequences are aligned by season, otherwise by age 
SEASON=False

# ---- SEQUENCE TIME GRANULARITY ----
#time unit in days
TIME_UNIT  = 30
#define time unit for the other channels , they can be different from the main channel
TIME_UNIT_1  = 30
TIME_UNIT_2  = 30
TIME_UNIT_3  = 30

#if season is true, specify number of months in each time unit 
MONTHS = 1


#---- SEQUENCE LENGTH ----
#number of time units in the sequence (if time unit is month, then 48 months = 4 years, 36 months=3 years)    
SEQ_LENGTH = 36
#define sequence length for the other channels, they can be different from the main channel
SEQ_LENGTH_1 = 36
SEQ_LENGTH_2 = 36
SEQ_LENGTH_3 = 36

#if season is true, specify number of years for sequence length
YEARS=3

# ---- ALPHABET ----
#state labels and code in the alphabet for the sequence channel, they are defined in the channels dictionary
STATES_LABELS = channels[CHANNEL]['states_labels']
STATES_LABELS_1 = channels[CHANNEL_1]['states_labels']
STATES_LABELS_2 = channels[CHANNEL_2]['states_labels']
STATES_LABELS_3 = channels[CHANNEL_3]['states_labels']

# ---- STATE PRIORITY HIERARCHY ----
#state priority for overlapping events in the same time unit, they are defined in the channels dictionary
STATE_PRIORITY = channels[CHANNEL]['state_priority']


##########################################################################################
################################ 3. SEQUENCE DATA CONTENT ################################
##########################################################################################

# ---- DATA ROOT DIRECTORY ----
#path to the root directory where the dataset is stored containing the dataset file specified in the channels dictionary
DATASET_PATH=channels[CHANNEL]['dataset_path']

#---- COLUMN NAMES IN THE DATASET ----
#column in the dataset that contains the patient identifier
COL_ID= 'pedianet_id' 
#column in the dataset that contains the date of the event that defines the state of the sequence at that time unit
COL_DATE  = channels[CHANNEL]['col_date'] 
#column in the dataset that define the state of the sequence at that time unit
COL_STATE = channels[CHANNEL]['col_state'] 

#----COVARIATES ----
#covariates to retain to describe the patients in the dataset
COVARIATES=channels[CHANNEL]['covariates']


##########################################################################################
############################ 4.SEQUENCE VISUALIZATION & RESULTS PATH #####################
##########################################################################################



# ----- COLOR UTILITIES ----
import colorsys
NO_COLOR = '#CCCCCC'  # fixed color for NO state across all channels

def _hex_from_rgb(rgb):
    return '#{:02X}{:02X}{:02X}'.format(
        int(round(rgb[0] * 255)),
        int(round(rgb[1] * 255)),
        int(round(rgb[2] * 255))
    )

def make_custom_colors(states_labels: dict) -> list:
    """Assigns visually distinct colors to states, reserving NO_COLOR for 'NO'."""
    keys = list(states_labels.keys())
    # Count how many states actually need dynamic colors (excluding 'NO')
    num_colored = sum(1 for k in keys if k != 'NO')
    
    colors = []
    color_idx = 0
    
    for state in keys:
        if state == 'NO':
            colors.append(NO_COLOR)
        else:
            # Distribute hues evenly across the full 360-degree circle
            hue = color_idx / max(num_colored, 1)
            
            # Alternate lightness only when there are many states (> 6)
            # Otherwise, maintain a balanced, punchy lightness (0.50)
            if num_colored > 6:
                lightness = 0.45 if color_idx % 2 == 0 else 0.62
            else:
                lightness = 0.50
                
            saturation = 0.85
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(_hex_from_rgb(rgb))
            
            color_idx += 1
            
    return colors

#if colors are defined in the channels dictionary
def get_channel_colors(channel_name: str) -> list:
    cfg = channels[channel_name]
    labels = cfg['states_labels']

    if 'colors' in cfg and cfg['colors']:
        return [cfg['colors'][state] for state in labels.keys()]
    
    # Fallback on make_custom_colors if colors are not yet defined
    return make_custom_colors(labels)

# Define or generate colors for each channel
CUSTOM_COLORS   = get_channel_colors(CHANNEL)
CUSTOM_COLORS_1 = get_channel_colors(CHANNEL_1)
CUSTOM_COLORS_2 = get_channel_colors(CHANNEL_2)
CUSTOM_COLORS_3 = get_channel_colors(CHANNEL_3)


# ---- SEQUENCE NAME ----
#rename channels if season is true
if SEASON:
    suffix = '_season' 
    CHANNEL   = f'{CHANNEL}{suffix}'
    CHANNEL_1 = f'{CHANNEL_1}{suffix}'
    CHANNEL_2 = f'{CHANNEL_2}{suffix}'
    CHANNEL_3 = f'{CHANNEL_3}{suffix}'

if not MULTICHANNEL:
    SEQUENCE_NAME = f'seq_{CHANNEL}_{TIME_UNIT}_{SEQ_LENGTH}'
else:
    SEQUENCE_NAME=f'seq_multichannel_{NUM_CHANNELS}_{TIME_UNIT}_{SEQ_LENGTH}'
SEQUENCE_NAME_0 = f'seq_{CHANNEL}_{TIME_UNIT}_{SEQ_LENGTH}'
SEQUENCE_NAME_1 = f'seq_{CHANNEL_1}_{TIME_UNIT_1}_{SEQ_LENGTH_1}'
SEQUENCE_NAME_2 = f'seq_{CHANNEL_2}_{TIME_UNIT_2}_{SEQ_LENGTH_2}'
SEQUENCE_NAME_3 = f'seq_{CHANNEL_3}_{TIME_UNIT_3}_{SEQ_LENGTH_3}'

# --- RESULTS PATH ----
#path to the directory where the results will be stored
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR        = ROOT / 'data'
DATASETS_DIR    = ROOT / 'SSA' / 'datasets'
RESULTS_DIR     = ROOT / 'SSA' / 'results'
SEQ_RESULTS_DIR = RESULTS_DIR / 'sequences'
EXPLORATION_DIR = RESULTS_DIR / 'exploration'
CHAR_DIR        = SEQ_RESULTS_DIR / 'sequence_characteristics'
VISUAL_DIR      = SEQ_RESULTS_DIR / 'sequence_visualizations'
CLUSTERING_DIR  = SEQ_RESULTS_DIR / 'sequence_clustering'
DISSIM_DIR      = SEQ_RESULTS_DIR / 'sequence_dissimilarities'
PRED_DIR        = ROOT / 'SSA' / 'prediction'

for d in (DATASETS_DIR, EXPLORATION_DIR, CHAR_DIR, VISUAL_DIR, CLUSTERING_DIR, DISSIM_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)


##########################################################################################
######################### 5. SEQUENCE DISSIMILARITY AND CLUSTERING #######################
##########################################################################################

#----- COSTS BETWEEN STATES ----
#costs between states for following dissimilarity measures 
COSTS= channels[CHANNEL]['costs']

#---- DISSIMILARITY MEASURES ----
# The following dissimilarity measures are available:
#'OM_CONST', 'OM_TRATE', 'OM_EMBED',
#'OM_INDELSLOG', 'OM_INDELSLOG_F', 'OM_INDELSQRT', 'OM_INDELSRT_F' 
#'HAM', 'HAM_WIND', 'HAM_WIND_LOG', 'DHAM' 
#'LCS','LCS_NO', 'LCS_NO_WEIGHTED'

DISTANCE_METRIC = 'LCS_NO_WEIGHTED' 
#dissimilarity measures for the other channels, they can be different from the main channel
DISTANCE_METRIC_1= 'OM_INDELSLOG' 
DISTANCE_METRIC_2= 'LCS'
DISTANCE_METRIC_3= 'LCS'

#---- CLUSTERING ALGORITHM ----
if MULTICHANNEL:
    CLUSTERING='multi_channel_hierarchical' #options: 'multi_channel_hierarchical', 'multi_channel_k_medoids'
else:
    CLUSTERING='hierarchical' #options: 'hierarchical_sequenzo','hierarchical', 'k_medoids'

#---- CLUSTERING LINKAGE ----
#linkage method if hierarchical clustering is used, options: 'ward', 'complete', 'average', 'single','centroid','median'
LINKAGE='ward' 

#---- NUMBER OF CLUSTERS ----
#maximum number of clusters to consider when evaluating cluster quality indicators
MAX_CLUSTERS = 10 
#number of clusters to use for the final clustering solution (can be chosen based on the evaluation of cluster quality indicators)
NUM_CLUSTERS =8

##########################################################################################
######################### 6. SEQUENCE-BASED RISK PREDICTION MODELING #####################
##########################################################################################

#----PREDICTION TARGET----
#name of the prediction task: 'target_lrti' or 'target_obs_drugs' or 'target_obesity' or 'target_bmi'
PRED_NAME='target_lrti' 

#----PREDICTION FEATURES----
#possible features are: 'sex','age','cluster','entropy','turbulence','num_transitions','adi','num_subsequences','complexity','variance_of_spell','count', 'months_last_event','last_year_events'
FEATURES=['age','sex','adi','entropy','num_transitions','variance_of_spell','complexity','turbulence','months_last_event'] #features to use for prediction
#select categorical features among the FEATURES list, they will be one-hot encoded for prediction
CAT_FEATURES=['sex','adi'] #categorical features to use for prediction

#----PREDICTION MODEL----
#model to use for prediction: 
#'logistic_regression' or 'lasso_regression' 
#'cox_regression'
#'cox_time_varying' or 'cox_time_varying_stata'  or 'cox_strata_numerical' or cox_strata_categorical'
MODEL='cox_regression' 

#stratification variable for models: 'cox_strata_numerical' or cox_strata_categorical'
STRATA='last_year_events' 

#----PREDICTION in the MULTICHANNEL setting----
#flag to use multichannel clusters instead of single channel partitions
#a different flag is used since features are extracted from a single channel
FLAG_MULTICHANNEL_CLUSTERS=False  

