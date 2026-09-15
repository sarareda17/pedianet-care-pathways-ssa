import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from utils import get_icd9, get_atc, select_events, dataset_preprocessing,dataset_preprocessing_other, explore_dataset_preprocessed,create_dataset_sequences,dataset_preprocessing_season,dataset_preprocessing_other_season
from config import TIME_UNIT, SEQ_LENGTH, COL_DATE, COL_STATE, SEQUENCE_NAME,COVARIATES, DATASET_PATH, CHANNEL,SEASON,MONTHS, YEARS, STATE_PRIORITY
ROOT = Path(__file__).resolve().parent.parent.parent

####### DATASET to build sequences #######

#import patients dataset
patients=pd.read_excel(ROOT/'data'/'pazienti.xlsx')
#import dataset from which to extract the information to build the sequences
#excel file 
if DATASET_PATH.endswith('.xlsx'):
    dataset=pd.read_excel(ROOT/'data'/DATASET_PATH)
#csv file
elif DATASET_PATH.endswith('.csv'):
    dataset=pd.read_csv(ROOT/'data'/DATASET_PATH)

#define states based on the selected channel and the corresponding codes
if CHANNEL=='diagnoses' or CHANNEL=='diagnoses_season':
    STATES = {
        'PNE': {'label': 'Pneumonia',                          'codes': get_icd9(('480','481','482','483','484','485','486','507','514'), dataset)},
        'BRC': {'label': 'Bronchiolitis',                      'codes': ['466.11', '466.1']},
        'AST': {'label': 'Asthma',                             'codes': get_icd9('493', dataset)},
        'BRN': {'label': 'Bronchitis',                         'codes': [c for c in get_icd9(('466','491'), dataset)
                                                                        if not c.startswith(('466.11', '466.1'))]},
        'WHZ': {'label': 'Wheezing',                           'codes': ['786.07']},
        'URI': {'label': 'Acute Upper Respiratory Infections', 'codes': get_icd9('465', dataset)},
        'BSP': {'label': 'Bronchospasm',                       'codes': get_icd9('519', dataset)},
        'FLU': {'label': 'Influenza',                          'codes': get_icd9('487', dataset)},
        'COV': {'label': 'Covid',                              'codes': ['079.89', '079.82']},
        'PHT': {'label': 'Pharyngotonsillitis',                'codes': ['034.0']},
        'TON': {'label': 'Tonsillitis',                        'codes': get_icd9(('463','474','475'), dataset)},
        'OTI': {'label': 'Otitis',                             'codes': get_icd9(('380','381','382'), dataset)},
        'RNF': {'label': 'Rhinopharyngitis',                   'codes': get_icd9('460', dataset)},
        'LAR': {'label': 'Laryngitis',                         'codes': get_icd9(('464','476'), dataset)},
        'PHA': {'label': 'Pharyngitis',                        'codes': get_icd9('462', dataset) + ['784.1']},
        'SIN': {'label': 'Sinusitis',                          'codes': get_icd9(('461','473'), dataset)},
        'RHN': {'label': 'Rhinitis',                           'codes': get_icd9(('472','477'), dataset)},
        'FEV': {'label': 'Fever',                              'codes': ['780.6']},
        'COU': {'label': 'Cough',                              'codes': ['786.2']},
        'OTH': {'label': 'Other',                              'codes': get_icd9(('478','490','516','518','512'), dataset)},
        'NO':  {'label': 'No Diagnosis',                       'codes': []}
    }

elif CHANNEL=='macrodiagnoses' or CHANNEL=='macrodiagnoses_season':
    STATES = {
    'URT': {
        'label': 'Upper Respiratory Tract Infections',
        'codes': (
            get_icd9('460', dataset) +                          # Rhinopharyngitis
            get_icd9('462', dataset) + ['784.1'] +              # Pharyngitis
            get_icd9(('463', '474', '475'), dataset) +          # Tonsillitis
            ['034.0'] +                                         # Pharyngotonsillitis
            get_icd9(('380', '381', '382'), dataset)            # Otitis
        )
    },
    'AURI': {
        'label': 'Acute Upper Respiratory Infections',
        'codes': (
            get_icd9(('464', '476'), dataset) +                  # Laryngitis
            get_icd9(('461', '473'), dataset) +                  # Sinusitis
            get_icd9(('472', '477'), dataset) +                  # Rhinitis
            get_icd9('465', dataset)                            # Acute Upper Respiratory Infections
        )
    },
    'LRT': {
        'label': 'Lower Respiratory Tract Infections',
        'codes': (
            [c for c in get_icd9(('466', '491'), dataset)
             if not c.startswith(('466.11', '466.1'))] +        # Bronchitis
            ['466.11', '466.1'] +                               # Bronchiolitis
            get_icd9(('480', '481', '482', '483', '484',
                      '485', '486', '507', '514'), dataset)     # Pneumonia
        )
    },
    'OBS': {
        'label': 'Obstructive Airway Diseases',
        'codes': (
            ['786.07'] +                                        # Wheezing
            get_icd9('519', dataset) +                          # Bronchospasm
            get_icd9('493', dataset)                            # Asthma
        )
    },
    'VIR': {
        'label': 'Influenza / COVID-19',
        'codes': (
            ['079.89', '079.82'] +                              # Covid
            get_icd9('487', dataset)                            # Influenza
        )
    },
    'SYM': {
        'label': 'General Symptoms',
        'codes': (
            ['786.2'] +                                         # Cough
            ['780.6']                                           # Fever
        )
    },
    'OTH': {
        'label': 'Other Respiratory Diseases',
        'codes': get_icd9(('478', '490', '516', '518', '512'), dataset)
    },
    'NO': {
        'label': 'No Diagnosis',
        'codes': []
    }
}
elif CHANNEL == 'supermacrodiagnoses' or CHANNEL == 'supermacrodiagnoses_season':
    STATES = {
    'INF_SUP': {
        'label': 'Infezioni delle vie aeree superiori',
        'codes': (
            get_icd9('460', dataset) +                          # Rinofaringite
            get_icd9('462', dataset) + ['784.1'] +              # Faringite
            get_icd9(('463','474','475'), dataset) +            # Tonsillite
            ['034.0']                                           # Faringotonsillite
        )
    },
    'INF_ACUTE_SUP': {
        'label': 'Infezioni acute delle vie aeree superiori',
        'codes': (
            get_icd9(('464','476'), dataset) +                  # Laringite
            get_icd9(('461','473'), dataset) +                  # Sinusite
            get_icd9(('472','477'), dataset) +                  # Rinite
            get_icd9('465', dataset)                            # Infezioni acute vie superiori
        )
    },
    'INF_INF_OSTR': {
        'label': 'Infezioni delle vie aeree inferiori e patologie ostruttive',
        'codes': (
            # Infezioni inferiori
            [c for c in get_icd9(('466','491'), dataset)
             if not c.startswith(('466.11', '466.1'))] +        # Bronchite
            ['466.11', '466.1'] +                               # Bronchiolite
            get_icd9(('480','481','482','483','484',
                      '485','486','507','514'), dataset) +       # Polmonite
            # Patologie ostruttive
            ['786.07'] +                                        # Wheezing
            get_icd9('519', dataset) +                          # Broncospasmo
            get_icd9('493', dataset)                            # Asma
        )
    },
    'SIM_ASPEC': {
        'label': 'Sintomi aspecifici',
        'codes': (
            # Sintomi generali
            ['786.2'] +                                         # Tosse
            ['780.6'] +                                         # Febbre
            # Influenza e Covid
            ['079.89', '079.82'] +                              # Covid
            get_icd9('487', dataset)                            # Influenza
        )
    },
    'ALT': {
        'label': 'Altre malattie respiratorie',
        'codes': get_icd9(('478','490','516','518','512'), dataset)
    },
    'NO': {
        'label': 'Nessuna diagnosi',
        'codes': []
    }
}
elif CHANNEL=='drugs' or CHANNEL=='drugs_season':
    STATES={
            'ANT': {'label': 'Antibiotics',               'codes': get_atc('J01', dataset)},  # J01
            'NAS': {'label': 'Nasal preparations',        'codes': get_atc('R01', dataset)},  # R01
            'THR': {'label': 'Throat preparations',       'codes': get_atc('R02', dataset)},  # R02
            'OAD': {'label': 'Drugs for obstructive airway diseases',   'codes': get_atc('R03', dataset)},  # R03
            'COC': {'label': 'Cough and cold preparations', 'codes': get_atc('R05', dataset)},  # R05
            'ANH': {'label': 'Antihistamines',            'codes': get_atc('R06', dataset)},  # R06
            'NO':  {'label': 'No prescription',      'codes': []}
    }
elif CHANNEL=='care_setting' or CHANNEL=='care_setting_season':
    STATES={
            'RIC': {'label': 'Ricovero', 'codes': ['RIC']},
            'PS':  {'label': 'Pronto Soccorso', 'codes': ['PS']},
            'VS':  {'label': 'Visita Specialistica', 'codes': [
                # ALLERGOLOGIA (branca 6)
                '89.7A.7_6', '89.7A.7_2',   # Prima visita allergologica pediatrica
                '91.90.4_2', '91.90.4_3',   # Screening allergologico prick test
                '89.01.7_7', '89.01.7_2',   # Visita allergologica pediatrica di controllo
                '91.90.6_0',                 # Test percutanei e intracutanei prick test
                # ORL (branca 21)
                '89.7B.8_5', '89.7B.8_2',   # Prima visita ORL pediatrica
                '89.01.H_8', '89.01.H_2',   # Visita ORL pediatrica di controllo
                '31.42_4',                   # Rinofaringoscopia / rinoscopia
                '31.42_2',                   # Laringoscopia a fibre ottiche
                '89.17_0',                   # Polisonnografia (ORL)
                # RADIOLOGIA DIAGNOSTICA (branca 8)
                '87.44.1_2',                 # RX torace mediastino
                '87.44.1_3',                 # RX torace controllo
                '88.73.3_2',                 # Eco polmonare
                # PNEUMOLOGIA (branca 22)
                '89.7B.9_2', '89.7B.9_3',   # Prima visita pneumologica pediatrica
                '89.01.L_3',                 # Visita pneumologica pediatrica di controllo
                '89.37.4_0',                 # Test di broncodilatazione farmacologica
                '89.65.5_0',                 # Monitoraggio saturazione / pulsossimetria
                '89.37.1_2', '89.37.1_3',   # Spirometria semplice e globale
                '89.38.6_0',                 # Valutazione ventilazione e gas espirati
                '89.44.1_0',                 # Prova da sforzo cardiorespiratorio
                '89.61.2_0',                 # Pulsossimetria notturna
                # ALTRE PRESTAZIONI SPECIALISTICHE (branca 26)
                '89.91.2_0',                 # Seduta inalatoria
            ]},  
            'NO':  {'label': 'Nessuno', 'codes': ['NO']}   
    }
elif CHANNEL == 'health' or CHANNEL == 'health_season':
    STATES = {
        'SICK': {
            'label': 'Sick',
            'codes': (
                get_icd9('460', dataset) +
                get_icd9('462', dataset) + ['784.1'] +
                get_icd9(('463','474','475'), dataset) +
                ['034.0'] +                                         # INF_SUP
                get_icd9(('464','476'), dataset) +
                get_icd9(('461','473'), dataset) +
                get_icd9(('472','477'), dataset) +
                get_icd9('465', dataset) +                          # INF_ACUTE_SUP
                [c for c in get_icd9(('466','491'), dataset)
                 if not c.startswith(('466.11', '466.1'))] +
                ['466.11', '466.1'] +
                get_icd9(('480','481','482','483','484',
                          '485','486','507','514'), dataset) +      # INF_INF
                ['786.07'] +
                get_icd9('519', dataset) +
                get_icd9('493', dataset) +                          # PAT_OSTR
                ['079.89', '079.82'] +
                get_icd9('487', dataset) +                          # INF_COV
                ['786.2', '780.6'] +                                # SIM_GEN
                get_icd9(('478','490','516','518','512'), dataset)  # ALT
            )
        },
        'NO': {
            'label': 'No diagnosis',
            'codes': []
        }
    }

#new dataset with only selected events
print(type(STATES))
dataset_events=select_events(dataset, STATES, COL_STATE)
dataset_events.to_csv(ROOT/'SSA'/'datasets'/f'{CHANNEL}_breath.csv', index=False)

#preprocess the dataset to keep only useful info and select patients cohort
if SEASON:
    if CHANNEL=='diagnoses_season'  or CHANNEL=='macrodiagnoses_season':
        patients_sequences, dataset_preprocessed = dataset_preprocessing_season(patients, dataset, COL_DATE, COL_STATE, STATES, TIME_UNIT, SEQ_LENGTH,CHANNEL,MONTHS,YEARS)
    else:
        selected_patients=pd.read_csv(ROOT/'SSA'/'datasets'/f'selected_patients_diagnoses_season_{YEARS}.csv')
        patients_sequences,dataset_preprocessed = dataset_preprocessing_other_season(selected_patients, dataset_events, COL_DATE, COL_STATE, STATES, TIME_UNIT, SEQ_LENGTH,CHANNEL,MONTHS,YEARS)
else:
    if CHANNEL=='diagnoses'  or CHANNEL=='macrodiagnoses':
        patients_sequences, dataset_preprocessed = dataset_preprocessing(patients, dataset, COL_DATE, COL_STATE, STATES, TIME_UNIT, SEQ_LENGTH,CHANNEL)
    else:
        selected_patients=pd.read_csv(ROOT/'SSA'/'datasets'/f'selected_patients_diagnoses.csv')
        patients_sequences,dataset_preprocessed = dataset_preprocessing_other(selected_patients, dataset_events, COL_DATE, COL_STATE, STATES, TIME_UNIT, SEQ_LENGTH,CHANNEL)
dataset_preprocessed.to_csv(ROOT/'SSA'/'datasets'/f'{CHANNEL}_preprocessed.csv', index=False)

#explore the preprocessed dataset
summary_df=explore_dataset_preprocessed(patients, patients_sequences, dataset, dataset_preprocessed,SEQUENCE_NAME,SEASON)

#create the dataset to build the sequences and save it
dataset_sequences=create_dataset_sequences(patients_sequences,dataset_preprocessed, SEQ_LENGTH,COVARIATES, STATE_PRIORITY)
dataset_sequences.to_csv(ROOT/'SSA'/'datasets'/ f'{SEQUENCE_NAME}.csv', index=False)

