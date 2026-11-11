import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import ROOT, EXPLORATION_DIR, DATASETS_DIR

#function to get all icd9 codes that start with the given prefixes in the dataset
def get_icd9(prefixes, dataset):
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    return list(
        dataset[dataset['diagnosi1_icd9'].str.startswith(prefixes, na=False)]['diagnosi1_icd9'].unique()
    )

#function to get all atc codes that start with the given prefixes in the dataset
def get_atc(prefixes, dataset):
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    return list(
        dataset[dataset['atc'].str.startswith(prefixes)]['atc'].unique()
    )

#function to select events in the dataset based on the states dictionary
def select_events(dataset,states,col_state):
    all_codes = [code for v in states.values() for code in v['codes']]
    dataset_events=dataset[dataset[col_state].isin(all_codes)].copy() 
    #map icd9 to state
    code_to_state = {code: state for state, v in states.items() for code in v['codes']}
    dataset_events['state'] = dataset_events[col_state].map(code_to_state)  

    #map state to label
    state_to_label = {state: v['label'] for state, v in states.items()}
    dataset_events['state_label'] = dataset_events['state'].map(state_to_label) 
    return dataset_events

#main channel dataset preprocessing 
#function to preprocess the dataset to keep only useful info and create the state column
def dataset_preprocessing(patients,dataset,col_date,col_state, states, time_unit, seq_length,channel):
    
    #select patient with end of follow up at least SEQ_LENGTH time units after the start of follow up
    patients_sequences=patients[(pd.to_datetime(patients['EF']) - pd.to_datetime(patients['datan'])).dt.days // time_unit >= seq_length]
        
    #select rows where pedianet_id is in patients_sequences
    dataset=dataset[dataset['pedianet_id'].isin(patients_sequences['pedianet_id'])]
    
    #add age
    datan_map = patients_sequences.drop_duplicates('pedianet_id').set_index('pedianet_id')['datan']
    dataset['age'] = (
    pd.to_datetime(dataset[col_date]) - 
    pd.to_datetime(dataset['pedianet_id'].map(datan_map))
).dt.days

    #add time unit column
    dataset['time_unit'] = dataset['age'] // time_unit

    #select rows where time_unit is less than seq_length
    dataset = dataset[dataset['time_unit'] < seq_length]


    #select rows where col_state is in states dictionary
    dataset=select_events(dataset, states, col_state)

    #select patients with at least one event in the dataset
    print(f"Number of patients before selecting those with at least one event: {len(patients_sequences)}")
    patients_sequences=patients_sequences[patients_sequences['pedianet_id'].isin(dataset['pedianet_id'])]
    print(f"Number of patients after selecting those with at least one event: {len(patients_sequences)}")
    
    #save selected patients
    patients_sequences.to_csv(DATASETS_DIR/f'selected_patients_diagnoses.csv', index=False)

    return patients_sequences,dataset

#other channel dataset preprocessing
#function to preprocess the dataset to keep only useful info and create the state column
def dataset_preprocessing_other(selected_patients,dataset,col_date,col_state, states, time_unit, seq_length,channel):
            
    #select rows where pedianet_id is in patients_sequences
    dataset=dataset[dataset['pedianet_id'].isin(selected_patients['pedianet_id'])]
    
    #add age
    dataset['age'] = (
        pd.to_datetime(dataset[col_date]) - 
        pd.to_datetime(dataset['pedianet_id'].map(selected_patients.set_index('pedianet_id')['datan']))
    ).dt.days

    #add time unit column
    dataset['time_unit'] = dataset['age'] // time_unit

    #select rows where time_unit is less than seq_length
    dataset = dataset[dataset['time_unit'] < seq_length]


    #select rows where col_state is in states dictionary
    dataset=select_events(dataset, states, col_state)


    return selected_patients,dataset


#main channel dataset preprocessing when SEASON is true 
#sequnces are aligned by seasonality 
#function to preprocess the dataset to keep only useful info and create the state column
def dataset_preprocessing_season(patients,dataset,col_date,col_state, states, time_unit, seq_length,channel,MONTHS,YEARS):

    #patients start year is the year of the start follow up (=birth) if it is in January, 
    #otherwise it is the year of the start of follow up + 1
    patients['start_year'] = np.where(patients['datan'].dt.month == 1, patients['datan'].dt.year, patients['datan'].dt.year + 1)
    #patients start date is the first day of the start year
    patients['start_date'] = np.where(patients['datan'].dt.month == 1, pd.to_datetime(patients['datan']), pd.to_datetime(patients['start_year'].astype(str) + '-01-01'))
    #patient end date is the last day of the start year +YEARS
    patients['end_date'] = pd.to_datetime((patients['start_year'] + YEARS-1).astype(str) + '-12-31')
    
    #select patient with end of follow up after end date of the patient
    patients_sequences=patients[(pd.to_datetime(patients['EF']) >= pd.to_datetime(patients['end_date']))]
        
    #select rows where pedianet_id is in patients_sequences
    dataset=dataset[dataset['pedianet_id'].isin(patients_sequences['pedianet_id'])]

    #remove rows with events before the the start date of the patient
    dataset = dataset[pd.to_datetime(dataset[col_date]) >= pd.to_datetime(patients_sequences.set_index('pedianet_id').loc[dataset['pedianet_id']]['start_date'].values)]

    #add time unit column based on the start date of the patient
    #the time unit is the month of the event + 12 * (year of the event - start year of the patient)
    dataset['time_unit'] = ((
    (pd.to_datetime(dataset[col_date]).dt.year - 
     pd.to_datetime(patients_sequences.set_index('pedianet_id')
     .loc[dataset['pedianet_id']]['start_date']).dt.year.values) * 12 + 
     pd.to_datetime(dataset[col_date]).dt.month) - 1 ) // MONTHS

    #select rows where time_unit is less than seq_length
    dataset = dataset[dataset['time_unit'] < YEARS * 12 // MONTHS]

    #select rows where col_state is in states dictionary
    dataset=select_events(dataset, states, col_state)

    #select patients with at least one event in the dataset
    print(f"Number of patients before selecting those with at least one event: {len(patients_sequences)}")
    patients_sequences=patients_sequences[patients_sequences['pedianet_id'].isin(dataset['pedianet_id'])]
    print(f"Number of patients after selecting those with at least one event: {len(patients_sequences)}")

    #save selected patients
    patients_sequences.to_csv(DATASETS_DIR/f'selected_patients_diagnoses_season_{YEARS}.csv', index=False)

    return patients_sequences,dataset

#other channel dataset preprocessing
#function to preprocess the dataset to keep only useful info and create the state column
def dataset_preprocessing_other_season(selected_patients,dataset,col_date,col_state, states, time_unit, seq_length,channel,MONTHS,YEARS):
            
    #select rows where pedianet_id is in patients_sequences
    dataset=dataset[dataset['pedianet_id'].isin(selected_patients['pedianet_id'])]

    #remove rows with events before the the start date of the patient
    dataset = dataset[pd.to_datetime(dataset[col_date]) >= pd.to_datetime(selected_patients.set_index('pedianet_id').loc[dataset['pedianet_id']]['start_date'].values)]
    
    #add time unit column based on the start date of the patient
    #the time unit is the month of the event + 12 * (year of the event - start year of the patient)
    dataset['time_unit'] = ((
    (pd.to_datetime(dataset[col_date]).dt.year - 
    pd.to_datetime(selected_patients.set_index('pedianet_id')
    .loc[dataset['pedianet_id']]['start_date']).dt.year.values) * 12 + 
    pd.to_datetime(dataset[col_date]).dt.month) - 1) // MONTHS

    #select rows where time_unit is less than seq_length
    dataset = dataset[dataset['time_unit'] < YEARS * 12 // MONTHS]

    #select rows where col_state is in states dictionary
    dataset=select_events(dataset, states, col_state)

    return selected_patients,dataset



#function to create the dataset to build the sequences
def create_dataset_sequences(patients_sequences,dataset_preprocessed, seq_length, COVARIATES,STATE_PRIORITY):
    #create new dataset for all selected patients with covariates
    dataset_sequences = patients_sequences[['pedianet_id'] + COVARIATES].copy()
    
    #add time units columns
    for i in range(seq_length):
        dataset_sequences[f'{i}'] = 'NO'
    
    #fill time units columns
    for _, row in dataset_preprocessed.iterrows():
        patient_id = row['pedianet_id']
        time_unit = row['time_unit']
        state = row['state']
        if time_unit < seq_length :
            mask= (dataset_sequences['pedianet_id'] == patient_id)
            current_state= dataset_sequences.loc[mask, f'{time_unit}'].values[0]
            if STATE_PRIORITY[state] > STATE_PRIORITY[current_state]:
                dataset_sequences.loc[mask, f'{time_unit}'] = state


    return dataset_sequences




#function to explore the preprocessed dataset and save the results in the exploration folder
def explore_dataset_preprocessed(patients,patients_sequences, dataset, dataset_preprocessed, SEQUENCE_NAME,SEASON):
    #create a folder in EXPLORATION_DIR with the name of the sequence if it doesn't exist
    (EXPLORATION_DIR / SEQUENCE_NAME).mkdir(parents=True, exist_ok=True)
    
    n_patients_total = patients['pedianet_id'].nunique()
    n_patients_selected = patients_sequences['pedianet_id'].nunique()
    n_patients_at_least_one_event = dataset_preprocessed['pedianet_id'].nunique()
    n_patients_with__one_event = dataset_preprocessed.groupby('pedianet_id').size().eq(1).sum()
    n_patients_with_two_events = dataset_preprocessed.groupby('pedianet_id').size().eq(2).sum()
    n_patients_with_three_events = dataset_preprocessed.groupby('pedianet_id').size().eq(3).sum()
    n_patients_with_four_events = dataset_preprocessed.groupby('pedianet_id').size().eq(4).sum()
    n_patients_with_five_events = dataset_preprocessed.groupby('pedianet_id').size().eq(5).sum()
    n_patients_with_more_than_five_events = dataset_preprocessed.groupby('pedianet_id').size().gt(5).sum()
    n_events_preprocessed = len(dataset_preprocessed)
    n_events_total = len(dataset)

    q1 = dataset_preprocessed.groupby('pedianet_id').size().quantile(0.25)
    q3 = dataset_preprocessed.groupby('pedianet_id').size().quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    n_outliers = dataset_preprocessed.groupby('pedianet_id').size()[
        (dataset_preprocessed.groupby('pedianet_id').size() < lower_bound) |
        (dataset_preprocessed.groupby('pedianet_id').size() > upper_bound)
    ].count()

    summary_data_preprocessing_df = pd.DataFrame([
        {
            'Info':       'Patients selected',
            'Count':      n_patients_selected,
            'Count (original)': n_patients_total,
            'Percentage': f"{n_patients_selected / n_patients_total * 100:.2f}%"
        },
        {
            'Info':       'Patients with at least one event',
            'Count':      n_patients_at_least_one_event,
            'Count (original)': n_patients_selected,
            'Percentage': f"{n_patients_at_least_one_event / n_patients_selected * 100:.2f}%"
        },
        {
            'Info':       'Patients with  one event',
            'Count':      n_patients_with__one_event,
            'Count (original)': n_patients_selected,
            'Percentage': f"{n_patients_with__one_event / n_patients_selected * 100:.2f}%"
        },
        {
            'Info':       'Patients with two events',
            'Count':      n_patients_with_two_events,
            'Count (original)': n_patients_selected,
            'Percentage': f"{n_patients_with_two_events / n_patients_selected * 100:.2f}%"
        },
        {
            'Info':       'Patients with three events',
            'Count':      n_patients_with_three_events,
            'Count (original)': n_patients_selected,
            'Percentage': f"{n_patients_with_three_events / n_patients_selected * 100:.2f}%"
        },
        {
            'Info':       'Patients with four events',
            'Count':      n_patients_with_four_events,
            'Count (original)': n_patients_selected,
            'Percentage': f"{n_patients_with_four_events / n_patients_selected * 100:.2f}%"
        },
        {
            'Info':       'Patients with five events',
            'Count':      n_patients_with_five_events,
            'Count (original)': n_patients_selected,
            'Percentage': f"{n_patients_with_five_events / n_patients_selected * 100:.2f}%"
        },
        {
            'Info':       'Patients with more than five events',
            'Count':      n_patients_with_more_than_five_events,
            'Count (original)': n_patients_selected,
            'Percentage': f"{n_patients_with_more_than_five_events / n_patients_selected * 100:.2f}%"
        },
        {
            'Info':       'Events in preprocessed dataset',
            'Count':      n_events_preprocessed,
            'Count (original)': n_events_total,
            'Percentage': f"{n_events_preprocessed / n_events_total * 100:.2f}%"
        },
        {
            'Info':       'Mean events per patient',
            'Count':      round(dataset_preprocessed.groupby('pedianet_id').size().mean(), 2),
            'Count (original)': round(dataset.groupby('pedianet_id').size().mean(), 2),
            'Percentage': '-'
        },
        {
            'Info':       'Outlier patients by number of events (IQR method)',
            'Count':      n_outliers,
            'Count (original)': '-',
            'Percentage': f"{n_outliers / n_patients_selected * 100:.2f}%"
        },
    ])
    summary_data_preprocessing_df.to_csv(EXPLORATION_DIR / SEQUENCE_NAME / 'summary_preprocessing.csv', index=False)
    summary_data_preprocessing_df.to_latex(EXPLORATION_DIR / SEQUENCE_NAME / 'summary_preprocessing.tex', index=False)

    #plot number of events per patient in the preprocessed dataset
    plt.figure(figsize=(10,6))
    dataset_preprocessed.groupby('pedianet_id').size().value_counts().sort_index().plot(kind='bar')
    plt.xlabel('Number of Events')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Events per Patient in the Preprocessed Dataset')
    plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'events_per_patient_preprocessed.png')

    #plot number of events per time unit in the preprocessed dataset
    plt.figure(figsize=(10,6))
    dataset_preprocessed.groupby('time_unit').size().plot()
    plt.xlabel('Time Unit')
    plt.ylabel('Number of Events')
    plt.title('Number of Events per Time Unit in the Preprocessed Dataset')
    plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'events_per_time_unit.png') 

    #boxplot of number of events per patient in the preprocessed dataset
    plt.figure(figsize=(10,6))
    dataset_preprocessed.groupby('pedianet_id').size().plot(kind='box')
    plt.xlabel('Patients')
    plt.ylabel('Number of Events')
    plt.title('Boxplot of Number of Events per Patient in the Preprocessed Dataset')
    plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'boxplot_events_per_patient_preprocessed.png') 

    
    #plot mean number of events per patient per time unit in the preprocessed dataset
    plt.figure(figsize=(10,6))
    dataset_preprocessed.groupby(['pedianet_id', 'time_unit']).size().groupby('time_unit').mean().plot()
    plt.xlabel('Time Unit')
    plt.ylabel('Mean Number of Events')
    plt.title('Mean Number of Events per Patient per Time Unit')
    plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'mean_events_per_patient_per_time_unit.png')
    
    #plot max number of events per patient per time unit in the preprocessed dataset
    plt.figure(figsize=(10,6))
    dataset_preprocessed.groupby(['pedianet_id', 'time_unit']).size().unstack(fill_value=0).max().plot()
    plt.xlabel('Time Unit')
    plt.ylabel('Max Number of Events')
    plt.title('Max Number of Events per Patient per Time Unit')
    plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'max_events_per_patient_per_time_unit.png')

    #plot distribution of states in the preprocessed dataset
    plt.figure(figsize=(10,6))
    ax = dataset_preprocessed['state_label'].value_counts().plot(kind='bar')
    ax.bar_label(ax.containers[0])
    plt.xlabel('State')
    plt.ylabel('Frequency')
    plt.title('Distribution of States in the Preprocessed Dataset')
    plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'distribution_states_preprocessed.png')
    #save in latex format the distribution of states in the preprocessed dataset
    state_distribution_df = dataset_preprocessed['state_label'].value_counts().reset_index()
    state_distribution_df.columns = ['State', 'Frequency']
    state_distribution_df.to_latex(EXPLORATION_DIR / SEQUENCE_NAME / 'distribution_states_preprocessed.tex', index=False)

    #plot number of patients with at least one event per time-unit
    plt.figure(figsize=(10,6))
    ax = dataset_preprocessed.groupby(['time_unit'])['pedianet_id'].nunique().plot(kind='bar', color='#9EB7C2', linewidth=2)
    ax.bar_label(ax.containers[0])
    plt.xlabel('Time Unit')
    plt.ylabel('Number of Patients')
    plt.title('Number of Patients with at least one diagnosis for each time unit')
    plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'patients_per_time_unit.png')



    #plot number of patients with more than one event for each time unit in the preprocessed dataset
    plt.figure(figsize=(10,6))
    dataset_preprocessed.groupby(['pedianet_id', 'time_unit']).size().groupby(['time_unit']).apply(lambda x: (x > 1).sum()).plot()
    plt.xlabel('Time Unit')
    plt.ylabel('Number of Patients with More than One Event')
    plt.title('Number of Patients with two or more Events for Each Time Unit')
    plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'patients_with_more_than_one_event_per_time_unit.png')
    
    

    # Plot the percentage of patients with two or more different events per time unit
    total_patients = dataset_preprocessed['pedianet_id'].nunique()
    plt.figure(figsize=(10, 6))
    (
        dataset_preprocessed.groupby(['pedianet_id', 'time_unit'])['state']
        .nunique()
        .groupby('time_unit')
        .apply(lambda x: (x > 1).sum() / total_patients * 100)
        .plot(color='#9EB7C2', linewidth=2)
    )
    plt.xlabel('Time Unit')
    plt.ylabel('Percentage of Patients (%)')
    plt.title('Percentage of Patients with two or more different events for each time units')
    plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'percentage_patients_with_more_than_one_different_event_per_time_unit.png',bbox_inches='tight')

    # Plot the percentage of multi-event patients among active patients (with >= 1 event) per time unit
    plt.figure(figsize=(10, 6))
    (
        dataset_preprocessed.groupby(['pedianet_id', 'time_unit'])['state']
        .nunique()
        .groupby('time_unit')
        .apply(lambda x: (x > 1).sum() / (x >= 1).sum() * 100)
        .plot(color='#9EB7C2', linewidth=2)
    )
    plt.xlabel('Time Unit')
    plt.ylabel('Percentage of Active Patients (%)')
    plt.title('Percentage of Patients with two or more different events among active patients for each time unit')
    plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'percentage_active_patients_with_more_than_one_different_event_per_time_unit.png',bbox_inches='tight')
    
    #plot number of patients with two or more different events for each time unit in the preprocessed dataset
    plt.figure(figsize=(10,6))
    dataset_preprocessed.groupby(['pedianet_id', 'time_unit'])['state'].nunique().groupby(['time_unit']).apply(lambda x: (x > 1).sum()).plot(color='#9EB7C2', linewidth=2)
    plt.xlabel('Time Unit')
    plt.ylabel('Number of Patients')
    plt.title('Number of Patients with two or more different events for each time unit')
    plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'patients_with_more_than_one_different_event_per_time_unit.png')

    #plot number of patients with more than two events for each time unit in the preprocessed dataset
    plt.figure(figsize=(10,6))
    dataset_preprocessed.groupby(['pedianet_id', 'time_unit']).size().groupby(['time_unit']).apply(lambda x: (x > 2).sum()).plot()
    plt.xlabel('Time Unit')
    plt.ylabel('Number of Patients with More than Two Events')
    plt.title('Number of Patients with More than Two Events for Each Time Unit')
    plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'patients_with_more_than_two_events_per_time_unit.png')

    #plot number of patients with three or more different events for each time unit in the preprocessed dataset
    plt.figure(figsize=(10,6))
    dataset_preprocessed.groupby(['pedianet_id', 'time_unit'])['state'].nunique().groupby(['time_unit']).apply(lambda x: (x > 2).sum()).plot()
    plt.xlabel('Time Unit')
    plt.ylabel('Number of Patients with More than Two Different Events')
    plt.title('Number of Patients with More than Two Different Events for Each Time Unit')
    plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'patients_with_more_than_two_different_events_per_time_unit.png')

    if SEASON:
        #plot age distribution at the start of the sequence for the preprocessed dataset
        plt.figure(figsize=(10,6))
        patients_sequences['start_age'] = (pd.to_datetime(patients_sequences['start_date']) - pd.to_datetime(patients_sequences['datan'])).dt.days // 30
        patients_sequences['start_age'].plot(kind='hist', bins=20)
        plt.xlabel('Age at Start of Sequence (months)')
        plt.ylabel('Frequency')
        plt.title('Age Distribution at Start of Sequence for the Preprocessed Dataset')
        plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'age_distribution_start_sequence.png')
        

        #plot age distribution at the end of the sequence for the preprocessed dataset
        plt.figure(figsize=(10,6))
        patients_sequences['end_age'] = (pd.to_datetime(patients_sequences['end_date']) - pd.to_datetime(patients_sequences['datan'])).dt.days // 30
        patients_sequences['end_age'].plot(kind='hist', bins=20)
        plt.xlabel('Age at End of Sequence (months)')
        plt.ylabel('Frequency')
        plt.title('Age Distribution at End of Sequence for the Preprocessed Dataset')
        plt.savefig(EXPLORATION_DIR / SEQUENCE_NAME / 'age_distribution_end_sequence.png')
    



    return summary_data_preprocessing_df