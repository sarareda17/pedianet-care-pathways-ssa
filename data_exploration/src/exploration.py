import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi
from pathlib import Path
ROOT= Path(__file__).resolve().parent.parent.parent
ROOT_RESULTS=ROOT/'data_exploration'/'results'


def general_exploration(data):
    print(f"Data shape: {data.shape}")
    
    print("Data Columns:")
    for col in data.columns:
        print(f"{col}: {data[col].dtype},Missing values: {data[col].isna().sum()/len(data)*100:.2f}%")
        if data[col].dtype == 'object':
            print(f"  Unique values: {data[col].nunique()}")

    
def patients_exploration(data):
    print("Patients Exploration:")
    #build dataframe with column name, type, number of missing values and percentage of missing values
    dataset_info = pd.DataFrame({
        'Column Name': data.columns,
        'Data Type': [data[col].dtype for col in data.columns],
        'Missing Values': [data[col].isna().sum() for col in data.columns],
        'Percentage Missing': [(data[col].isna().sum()/len(data)*100) for col in data.columns]
    })
    dataset_info.to_csv(ROOT_RESULTS/'tables'/'patients'/'dataset_info.csv', index=False)
    dataset_info.to_latex(ROOT_RESULTS/'tables'/'patients'/'dataset_info.tex', index=False)


    #number of patients in the dataset
    print(f"Unique Patients: {data['pedianet_id'].nunique()}")

    #gender of patients
    male_patients = data[data['sesso'] == 'M']['pedianet_id'].nunique()
    female_patients= data[data['sesso'] == 'F']['pedianet_id'].nunique()
    #cake plot (pink/blue)
    plt.figure(figsize=(6,6))
    plt.pie([male_patients, female_patients], labels=['Male', 'Female'], colors=['lightblue', 'lightpink'], autopct='%1.1f%%', startangle=90)
    plt.title("Gender Distribution of Patients")
    plt.savefig(ROOT_RESULTS/'figures'/'patients'/'gender_distribution.png', dpi=300, bbox_inches="tight")


    #is start follow up (SF) == pedianet ingresso?
    print(f"SF==Pedianet Ingresso: {(data['SF'] == data['pedianet_ingresso']).sum()/len(data)*100:.2f}%")
    print(f"SF before Pedianet Ingresso: {(data['SF'] < data['pedianet_ingresso']).sum()/len(data)*100:.2f}%")
    print(f"SF after Pedianet Ingresso: {(data['SF'] > data['pedianet_ingresso']).sum()/len(data)*100:.2f}%")
    print(data['pedianet_ingresso'].describe())

    #adi (deprivationindex) distribution
    plt.figure(figsize=(8,6))
    plt.pie(data['adi'].value_counts(), labels=data['adi'].value_counts().index, autopct='%1.1f%%', startangle=90)
    plt.title("Distribution of Adi (Deprivation Index)")
    plt.savefig(ROOT_RESULTS/'figures'/'patients'/'adi_distribution.png', dpi=300, bbox_inches="tight")

    #check flag FSE
    print(f"Patients with FSE flag: {data['flag_FSE'].sum()/len(data)*100:.2f}%")
    #cake plot for FSE flag
    plt.figure(figsize=(6,6))
    plt.pie([data['flag_FSE'].sum(), len(data) - data['flag_FSE'].sum()], labels=[' FSE', ' No FSE'], colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
    plt.title("Distribution of FSE Flag")
    plt.savefig(ROOT_RESULTS/'figures'/'patients'/'fse_distribution.png', dpi=300, bbox_inches="tight")


    #Mean observation time
    print(f"Mean Observation Time per patient (years):{((data['EF']-data['SF']).dt.days/365.25).mean():.2f}")
    
    #Distribution of observation time per patient
    obs_time = (data['EF']-data['SF']).dt.days/365.25
    plt.figure(figsize=(8,6))
    plt.hist(obs_time, bins=20)
    plt.title("Observation Time per Patient (years)")
    plt.xlabel("Observation Time (years)")
    plt.ylabel("Frequency")
    plt.savefig(ROOT_RESULTS/'figures'/'patients'/'observation_time.png', dpi=300, bbox_inches="tight")

    #Cumulative distribution of observation time (reverse)
    plt.figure(figsize=(8,6))
    plt.hist(obs_time, bins=20, cumulative=-1)
    plt.title("Patients with Observation Time ≥ t")
    plt.xlabel("Observation Time (years)")
    plt.ylabel("Number of patients")

    plt.savefig(ROOT_RESULTS/'figures'/'patients'/'observation_time_cumulative_reverse.png',
            dpi=300, bbox_inches="tight")

    #cumulative distribution of observation time (reverse) in months from birth (SF is birth)
    plt.figure(figsize=(8, 5))

    # Follow-up in months from birth (SF is birth)
    obs_time_month = (data['EF'] - data['SF']).dt.days / 30.4375

    # Integer bins for each individual month up to the maximum observed
    max_months = int(np.ceil(obs_time_month.max()))
    bins = np.arange(0, max_months + 1, 1)

    # Reverse cumulative histogram (patients with follow-up >= t)
    plt.hist(obs_time_month, bins=bins, cumulative=-1, 
            color="#5FA4BB", edgecolor="#4C8193", alpha=0.85)

    plt.title(r"Patients with Follow-up $\geq$ t$ Months from Birth", fontsize=12, fontweight='bold')
    plt.xlabel("Age from Birth (Months)", fontsize=11)
    plt.ylabel("Number of Patients", fontsize=11)
    plt.xlim(0, max_months)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(ROOT_RESULTS / 'figures' / 'patients' / 'observation_time_cumulative_reverse_month.png',
                dpi=300, bbox_inches="tight")

    #milestone plot 
     # Follow-up in months from birth
    obs_months = (data['EF'] - data['datan']).dt.days / 30.4375

    milestones = [6, 12, 24, 36, 48, 60, 72, 84]
    labels = [f'{m}m\n({m/12}y)' if m >= 12 else f'{m}m' for m in milestones]
    counts = [(obs_months >= m).sum() for m in milestones]
    percentages = [(c / len(data)) * 100 for c in counts]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, counts, color="#80B1D3", edgecolor='#9EB7C2', width=0.55)

    # Add exact count and percentage above each bar
    for bar, pct, n in zip(bars, percentages, counts):
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y + len(data) * 0.015,
                f'{pct:.1f}%\n({n:,})', ha='center', va='bottom', fontsize=9)

    plt.ylim(0, len(data) * 1.18)
    plt.title("Cohort Retention across observation windows", fontsize=12)
    plt.xlabel("Minimum Follow-up from Birth", fontsize=11)
    plt.ylabel("Number of Active Patients", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(ROOT_RESULTS / 'figures' / 'patients' / 'milestone_plot.png', dpi=300, bbox_inches="tight")
    
    #tables that for each year of age say how many patients are in pedianet using cumulative observation time
    age_years = []
    for ot in obs_time:
        for year in range(0, int(ot) + 1):
            age_years.append(year)

    number_patient_at_age = pd.Series(age_years).value_counts().sort_index()
    number_patient_at_age.index.name = "Age (years)"
    number_patient_at_age.name = "Number of Patients"

    # crea dataframe con Age come colonna
    number_patient_at_age = pd.DataFrame(number_patient_at_age).reset_index()

    number_patient_at_age.to_csv(ROOT_RESULTS/'tables'/'patients'/'patients_at_age.csv', index=False)
    number_patient_at_age.to_latex(ROOT_RESULTS/'tables'/'patients'/'patients_at_age.tex', index=False)  


   #Distribution of start of follow-up (SF)  over time
    plt.figure(figsize=(8,6))
    plt.hist(data['SF'], bins=80, alpha=0.5, label='Start of Follow-up (SF)')  
    plt.title("Distribution of Start of Follow-up (SF) Over Time")  
    plt.xlabel("Time")  
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(ROOT_RESULTS/'figures'/'patients'/'SF_over_time.png', dpi=300, bbox_inches="tight")  

    #tables where for each year from 2018 to 2025 are reported the number of patients with start of follow-up (SF) in that year and the percentage over total patients
    data['year_SF'] = data['SF'].dt.year
    patients_per_year_SF = data.groupby('year_SF')['pedianet_id'].nunique().reset_index()
    patients_per_year_SF.columns = ['Year', 'Number of Patients with SF in that Year']
    patients_per_year_SF['Percentage'] = patients_per_year_SF['Number of Patients with SF in that Year']/data['pedianet_id'].nunique()*100
    patients_per_year_SF.to_csv(ROOT_RESULTS/'tables'/'patients'/'patients_per_year_SF.csv', index=False)
    patients_per_year_SF.to_latex(ROOT_RESULTS/'tables'/'patients'/'patients_per_year_SF.tex', index=False)

    #Distribution of end of follow-up (EF)  over time
    plt.figure(figsize=(8,6))
    plt.hist(data['EF'], bins=80, alpha=0.5, label='End of Follow-up (EF)')
    plt.title("Distribution of End of Follow-up (EF) Over Time")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(ROOT_RESULTS/'figures'/'patients'/'EF_over_time.png', dpi=300, bbox_inches="tight")  

    #cumulative number of total patients in Pedianet per year: for each year consider patients 
    #with start of follow-up (SF) in that year and sum them up to get the cumulative number of patients 
    #in Pedianet per year
    patients_per_year_SF['Cumulative Patients'] = patients_per_year_SF['Number of Patients with SF in that Year'].cumsum()
    plt.figure(figsize=(8,6))
    plt.plot(patients_per_year_SF['Year'], patients_per_year_SF['Cumulative Patients'], marker='o')
    plt.title("Cumulative Number of Patients in Pedianet Over Time")
    plt.xlabel("Year")
    plt.ylabel("Cumulative Number of Patients")
    plt.savefig(ROOT_RESULTS/'figures'/'patients'/'cumulative_patients_over_time.png', dpi=300, bbox_inches="tight")

    
    #Distribution of age at first follow-up
    age_SF = (data['SF'] - data['datan']).dt.days/365.25
    plt.figure(figsize=(8,6))
    plt.hist(age_SF, bins=20)
    plt.title("Age at First Follow-up")
    plt.xlabel("Age (years)")
    plt.ylabel("Frequency")
    plt.savefig(ROOT_RESULTS/'figures'/'patients'/'patient_age_SF.png', dpi=300, bbox_inches="tight")

    #Distribution of age in months at first follow-up
    age_SF_months = (data['SF'] - data['datan']).dt.days//30
    plt.figure(figsize=(8,6))
    plt.hist(age_SF_months, bins=20)
    plt.title("Age at First Follow-up (Months)")
    plt.xlabel("Age (months)")
    plt.ylabel("Frequency")
    plt.savefig(ROOT_RESULTS/'figures'/'patients'/'patient_age_SF_months.png', dpi=300, bbox_inches="tight")

    #Distribution of age at end of follow-up
    age_EF = (data['EF'] - data['datan']).dt.days/365.25
    plt.figure(figsize=(8,6))
    plt.hist(age_EF, bins=20)   
    plt.title("Age at End of Follow-up")
    plt.xlabel("Age (years)")
    plt.ylabel("Frequency")
    plt.savefig(ROOT_RESULTS/'figures'/'patients'/'patient_age_EF.png', dpi=300, bbox_inches="tight")

    #distribution of patients per doctor
    patients_per_doctor=data.groupby('idmedico')['pedianet_id'].nunique()
    plt.figure(figsize=(8,6))
    plt.plot(patients_per_doctor.index, patients_per_doctor.values, marker='o')
    plt.title("Number of Patients per Doctor")
    plt.xlabel("Doctor ID")
    plt.ylabel("Number of Patients")
    plt.savefig(ROOT_RESULTS/'figures'/'patients'/'patients_per_doctor.png', dpi=300, bbox_inches="tight")



    #create table with main info about patients
    patients_summary = pd.DataFrame(
        {
            "Total Patients": [data['pedianet_id'].nunique()],
            "Mean Observation Time (years)": [((data['EF']-data['SF']).dt.days/365.25).mean()],
            "Min Observation Time (years)": [((data['EF']-data['SF']).dt.days/365.25).min()],
            "Max Observation Time (years)": [((data['EF']-data['SF']).dt.days/365.25).max()],
            "Mean Age at First Follow-up (years)": [((data['SF'] - data['datan']).dt.days/365.25).mean()],
            "Mean Age at End of Follow-up (years)": [((data['EF'] - data['datan']).dt.days/365.25).mean()],
            "Min Age at First Follow-up (years)": [((data['SF'] - data['datan']).dt.days/365.25).min()],
            "Max Age at First Follow-up (years)": [((data['SF'] - data['datan']).dt.days/365.25).max()],
            "Number of doctors": [data['idmedico'].nunique()],
            "Mean patients per doctor": [data.groupby('idmedico')['pedianet_id'].nunique().mean()],
            "Min patients per doctor": [data.groupby('idmedico')['pedianet_id'].nunique().min()],   
            "Max patients per doctor": [data.groupby('idmedico')['pedianet_id'].nunique().max()],
            "Patients who died during follow-up": [data['deceduto'].sum()]
        }
    )
    #save
    patients_summary.to_csv(ROOT_RESULTS/'tables'/'patients'/'patients_summary.csv', index=False)
    patients_summary.to_latex(ROOT_RESULTS/'tables'/'patients'/'patients_summary.tex', index=False)



def main_diagnoses_exploration(diagnoses_data,patients_data):
    print("Diagnoses Exploration:")
    #table with column name, type, number of missing values and percentage of missing values
    dataset_info = pd.DataFrame({
        'Column Name': diagnoses_data.columns,
        'Data Type': [diagnoses_data[col].dtype for col in diagnoses_data.columns],
        'Missing Values': [diagnoses_data[col].isna().sum() for col in diagnoses_data.columns],
        'Percentage Missing': [(diagnoses_data[col].isna().sum()/len(diagnoses_data)*100) for col in diagnoses_data.columns]
    })
    dataset_info.to_csv(ROOT_RESULTS/'tables'/'diagnoses'/'dataset_info.csv', index=False)
    dataset_info.to_latex(ROOT_RESULTS/'tables'/'diagnoses'/'dataset_info.tex', index=False)

    print(f"Unique Diagnoses: {diagnoses_data['tipo'].nunique()}")
    print(f"Diagnoses: {diagnoses_data['tipo'].unique()}")

    #add age at diagnoses dataset
    diagnoses_data = diagnoses_data.merge(
    patients_data[['pedianet_id', 'datan']], 
    on='pedianet_id', 
    how='left'
    ) 
    print(diagnoses_data['datan'].head())
    diagnoses_data['age_at_diagnosis'] = (diagnoses_data['data'] - diagnoses_data['datan']).dt.days/365.25
    #add month_year column
    diagnoses_data['month_year'] = diagnoses_data['data'].dt.to_period('M')
    patients_data['month_year_SF'] = patients_data['SF'].dt.to_period('M')
    
    #Diagnoses distribution
    plt.figure(figsize=(8,6))
    plt.hist(diagnoses_data['tipo'], bins=diagnoses_data['tipo'].nunique())
    plt.title("Distribution of Main Diagnoses")
    plt.xlabel("Diagnosis")
    plt.ylabel("Frequency")
    plt.savefig(ROOT_RESULTS/'figures'/'diagnoses'/'diagnoses_distribution.png', dpi=300, bbox_inches="tight")

    #Diagnoses frequency per patient
    diagnoses_per_patient = diagnoses_data.groupby('pedianet_id').size()
    print(f"Patients with more than 3 diagnoses:{(diagnoses_per_patient > 3).sum()}")
    print(f"Patients with more than 5 diagnoses:{(diagnoses_per_patient > 5).sum()}")
    print(f"Patients with more than 7 diagnoses:{(diagnoses_per_patient > 7).sum()}")

    #patients with 0 diagnoses
    patients_with_diagnoses = diagnoses_data['pedianet_id'].unique()
    patients_without_diagnoses = patients_data[~patients_data['pedianet_id'].isin(patients_with_diagnoses)]
    print(f"Patients with 0 diagnoses: {patients_without_diagnoses.shape[0]}")


    #Plot number of patients for different number of diagnoses
    diagnoses_count = diagnoses_per_patient.value_counts().sort_index()
    plt.figure(figsize=(8,6))
    plt.bar(diagnoses_count.index, diagnoses_count.values)
    plt.title("Number of Patients for Different Number of Diagnoses")
    plt.xlabel("Number of Diagnoses")
    plt.ylabel("Number of Patients")
    plt.savefig(ROOT_RESULTS/'figures'/'diagnoses'/'patients_per_diagnoses_count.png', dpi=300, bbox_inches="tight")

    #Unique diagnoses per patient
    unique_diagnoses_per_patient = diagnoses_data.groupby('pedianet_id')['tipo'].nunique()
    print(f"Mean Unique Diagnoses per Patient: {unique_diagnoses_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    plt.hist(unique_diagnoses_per_patient, bins=unique_diagnoses_per_patient.nunique())
    plt.title("Number of Unique Diagnoses per Patient")
    plt.xlabel("Number of Unique Diagnoses")
    plt.ylabel("Frequency")
    plt.savefig(ROOT_RESULTS/'figures'/'diagnoses'/'unique_diagnoses_per_patient.png', dpi=300, bbox_inches="tight") 

    #Distribution of diagnoses over time
    diagnoses_over_time = diagnoses_data.groupby('month_year').size()
    plt.figure(figsize=(10,6))
    plt.hist(diagnoses_over_time, bins=diagnoses_data['month_year'].nunique())
    plt.title("Number of Diagnoses Over Time")
    plt.xlabel("Time (Month-Year)")
    plt.ylabel("Number of Diagnoses")
    plt.savefig(ROOT_RESULTS/'figures'/'diagnoses'/'diagnoses_over_time.png', dpi=300, bbox_inches="tight")

    #Distribution of diagnoses over age at diagnosis
    plt.figure(figsize=(8,6))
    plt.hist(diagnoses_data['age_at_diagnosis'], bins=20)
    plt.title("Age at Diagnosis Distribution")
    plt.xlabel("Age at Diagnosis (years)")
    plt.ylabel("Frequency")
    plt.savefig(ROOT_RESULTS/'figures'/'diagnoses'/'age_at_diagnosis_distribution.png', dpi=300, bbox_inches="tight")

    #Distribution of diagnoses over time for every diagnoses
    bronchite_diagnoses= diagnoses_data[diagnoses_data['tipo'] == 'bronchite']
    print(f"Bronchite Diagnoses: {bronchite_diagnoses.shape[0]}")
    laringite_diagnoses= diagnoses_data[diagnoses_data['tipo'] == 'laringite']
    print(f"Laringite Diagnoses: {laringite_diagnoses.shape[0]}")
    bronchiolite_diagnoses= diagnoses_data[diagnoses_data['tipo'] == 'bronchiolite']
    print(f"Bronchiolite Diagnoses: {bronchiolite_diagnoses.shape[0]}")
    wheezing_diagnoses= diagnoses_data[diagnoses_data['tipo'] == 'wheezing']
    print(f"Wheezing Diagnoses: {wheezing_diagnoses.shape[0]}")
    asma_diagnoses= diagnoses_data[diagnoses_data['tipo'] == 'asma']
    print(f"Asma Diagnoses: {asma_diagnoses.shape[0]}")
    
    fig, ax = plt.subplots(2, 3, figsize=(10, 8))

    ax[0, 0].hist(bronchite_diagnoses['data'], bins=80)
    ax[0,0].set_title("Bronchite Diagnoses Over Time")
    ax[0, 1].hist(laringite_diagnoses['data'], bins=80)
    ax[0, 1].set_title("Laringite Diagnoses Over Time")
    ax[0, 2].hist(bronchiolite_diagnoses['data'], bins=80)
    ax[0, 2].set_title("Bronchiolite Diagnoses Over Time")
    ax[1, 0].hist(wheezing_diagnoses['data'], bins=80)
    ax[1, 0].set_title("Wheezing Diagnoses Over Time")
    ax[1, 1].hist(asma_diagnoses['data'], bins=80)   
    ax[1, 1].set_title("Asma Diagnoses Over Time")  
    plt.tight_layout()
    plt.savefig(ROOT_RESULTS/'figures'/'diagnoses'/'diagnoses_over_time.png', dpi=300, bbox_inches="tight")

    #Distribution of diagnoses over time for every diagnoses but normalized for the number of patients in that time (to see if the increase in diagnoses is due to an increase in patients or an increase in diagnoses per patient)
    #get cumulative number of patients in Pedianet per month_year
    patients_per_month_year = patients_data.groupby('month_year_SF')['pedianet_id'].nunique().cumsum()
    fig, ax = plt.subplots(2, 3, figsize=(10, 8))
    ax[0, 0].hist(bronchite_diagnoses['data'], bins=80, weights=np.ones(len(bronchite_diagnoses)) / patients_per_month_year.loc[bronchite_diagnoses['month_year']].values)
    ax[0,0].set_title("Bronchite Diagnoses Over Time (normalized)")
    ax[0, 1].hist(laringite_diagnoses['data'], bins=80, weights=np.ones(len(laringite_diagnoses)) / patients_per_month_year.loc[laringite_diagnoses['month_year']].values)
    ax[0, 1].set_title("Laringite Diagnoses Over Time (normalized)")
    ax[0, 2].hist(bronchiolite_diagnoses['data'], bins=80, weights=np.ones(len(bronchiolite_diagnoses)) / patients_per_month_year.loc[bronchiolite_diagnoses['month_year']].values)
    ax[0, 2].set_title("Bronchiolite Diagnoses Over Time (normalized)")
    ax[1, 0].hist(wheezing_diagnoses['data'], bins=80, weights=np.ones(len(wheezing_diagnoses)) / patients_per_month_year.loc[wheezing_diagnoses['month_year']].values)
    ax[1, 0].set_title("Wheezing Diagnoses Over Time (normalized)")
    ax[1, 1].hist(asma_diagnoses['data'], bins=80, weights=np.ones(len(asma_diagnoses)) / patients_per_month_year.loc[asma_diagnoses['month_year']].values)
    ax[1, 1].set_title("Asma Diagnoses Over Time (normalized)")
    plt.tight_layout()
    plt.savefig(ROOT_RESULTS/'figures'/'diagnoses'/'diagnoses_over_time_normalized.png', dpi=300, bbox_inches="tight")


    #Distribution over age for every diagnosis
    fig, ax = plt.subplots(2, 3, figsize=(10, 8))

    ax[0, 0].hist(bronchite_diagnoses['age_at_diagnosis'], bins=8)
    ax[0,0].set_title("Bronchite Diagnoses Age Distribution")
    ax[0, 1].hist(laringite_diagnoses['age_at_diagnosis'], bins=8)
    ax[0, 1].set_title("Laringite Diagnoses Age Distribution")
    ax[0, 2].hist(bronchiolite_diagnoses['age_at_diagnosis'], bins=8)
    ax[0, 2].set_title("Bronchiolite Diagnoses Age Distribution")
    ax[1, 0].hist(wheezing_diagnoses['age_at_diagnosis'], bins=8)
    ax[1, 0].set_title("Wheezing Diagnoses Age Distribution")
    ax[1, 1].hist(asma_diagnoses['age_at_diagnosis'], bins=8)   
    ax[1, 1].set_title("Asma Diagnoses Age Distribution")  
    plt.tight_layout()
    plt.savefig(ROOT_RESULTS/'figures'/'diagnoses'/'diagnoses_age_distribution.png', dpi=300, bbox_inches="tight")

    #Choose 10 patients with more diagnoses and plot their diagnoses over time with different colors for every diagnosis
    top_patients = diagnoses_per_patient.sort_values(ascending=False).head(10).index
    top_patients_diagnoses = diagnoses_data[diagnoses_data['pedianet_id'].isin(top_patients)]

    fig, axes = plt.subplots(5, 2, figsize=(16, 20))

    for i, patient in enumerate(top_patients):
        row, col = i // 2, i % 2
        patient_diagnoses = top_patients_diagnoses[top_patients_diagnoses['pedianet_id'] == patient]
        axes[row, col].scatter(patient_diagnoses['data'], patient_diagnoses['tipo'], s=100)
        axes[row, col].set_title(f"Patient {patient} Diagnoses Over Time")
        axes[row, col].set_xlabel("Time")
        axes[row, col].set_ylabel("Diagnosis")
        axes[row, col].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(ROOT_RESULTS/'figures'/'diagnoses'/'top_patients_diagnoses_over_time.png', dpi=300, bbox_inches="tight")



    #diagnoses summary table
    diagnoses_summary = pd.DataFrame(
        {
            "Total Diagnoses": [diagnoses_data.shape[0]],
            "Unique Diagnoses": [diagnoses_data['tipo'].nunique()],
            "Type of diagnoses": [diagnoses_data['tipo'].unique()],
            "Diagnoses per Type": [diagnoses_data['tipo'].value_counts().to_dict()],
            "Percentage of diagnoses per Type": [(diagnoses_data['tipo'].value_counts()/diagnoses_data.shape[0]*100).to_dict()],
            "Mean Diagnoses per Patient": [diagnoses_per_patient.mean()],
            "Mean Unique Diagnoses per Patient": [unique_diagnoses_per_patient.mean()],
            "Patients with 0 Diagnoses": [patients_without_diagnoses.shape[0]],
            "Patients with 1 Diagnosis": [(diagnoses_per_patient == 1).sum()],
            "Patients with 2 Diagnoses": [(diagnoses_per_patient == 2).sum()],  
            "patients with 3 Diagnoses": [(diagnoses_per_patient == 3).sum()],
            "Patients with >3 Diagnoses": [(diagnoses_per_patient > 3).sum()],
            "Patients with >5 Diagnoses": [(diagnoses_per_patient > 5).sum()],
            "Patients with >7 Diagnoses": [(diagnoses_per_patient > 7).sum()],
            "Date first diagnoses": [diagnoses_data['data'].min()],
            "Date last diagnoses": [diagnoses_data['data'].max()]
        }
    )
    diagnoses_summary.to_csv(ROOT_RESULTS/'tables'/'diagnoses'/'diagnoses_summary.csv', index=False)
    diagnoses_summary.to_latex(ROOT_RESULTS/'tables'/'diagnoses'/'diagnoses_summary.tex', index=False)





def vaccines_exploration(vaccines_data,patients_data,vaccination_counts):
    #table with column name, type, number of missing values and percentage of missing values
    dataset_info = pd.DataFrame({
        'Column Name': vaccines_data.columns,
        'Data Type': [vaccines_data[col].dtype for col in vaccines_data.columns],
        'Missing Values': [vaccines_data[col].isna().sum() for col in vaccines_data.columns],
        'Percentage Missing': [(vaccines_data[col].isna().sum()/len(vaccines_data)*100) for col in vaccines_data.columns]
    })
    dataset_info.to_csv(ROOT_RESULTS/'tables'/'vaccines'/'dataset_info.csv', index=False)
    dataset_info.to_latex(ROOT_RESULTS/'tables'/'vaccines'/'dataset_info.tex', index=False)

    #add age at vaccination
    vaccines_data = vaccines_data.merge(
    patients_data[['pedianet_id', 'datan']], 
    on='pedianet_id',   
    how='left'
    )
    vaccines_data['age_at_vaccination'] = (vaccines_data['datarecord'] - vaccines_data['datan']).dt.days/365.25

    print("Vaccines Exploration:")
    print(f"Unique Vaccines Drugs: {vaccines_data['farmaco'].nunique()}, Unique Types: {vaccines_data['farmaco'].unique()}")
    #save different drugs used in a table
    vaccines_drugs = pd.DataFrame(vaccines_data['farmaco'].value_counts())
    vaccines_drugs.reset_index(inplace=True)
    vaccines_drugs.columns = ['Vaccine', 'Count']
    vaccines_drugs['Percentage'] = vaccines_drugs['Count'] / vaccines_drugs['Count'].sum() * 100
    vaccines_drugs.to_csv(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_drugs.csv', index=False)
    vaccines_drugs.to_latex(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_drugs.tex', index=False)

    #table top 20 most used drugs
    vaccines_top_drugs = vaccines_drugs.head(20)
    vaccines_top_drugs.to_csv(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_top_drugs.csv', index=False)
    vaccines_top_drugs.to_latex(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_top_drugs.tex', index=False)

    print(f"Unique Vaccines ATC: {vaccines_data['atc'].nunique()}, Unique Types: {vaccines_data['atc'].unique()}")
    total = vaccines_data['farmaco'].nunique()
    unique_atc = sum(
    1 for _, group in vaccines_data.groupby('farmaco')
    if group['atc'].nunique() == 1      
    )
    print(f"Drugs with unique ATC code: {unique_atc/total*100:.2f}%")
    #save different ATC codes used in a table
    vaccines_atc = pd.DataFrame(vaccines_data['atc'].value_counts())
    vaccines_atc.reset_index(inplace=True)
    vaccines_atc.columns = ['ATC Code', 'Count']
    vaccines_atc['Percentage'] = vaccines_atc['Count'] / vaccines_atc['Count'].sum() * 100
    vaccines_atc.to_csv(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_atc.csv', index=False)
    vaccines_atc.to_latex(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_atc.tex', index=False)

    
    print(f"Description of virus types: {vaccines_data['descrizione_virus'].nunique()}, Unique descriptions: {vaccines_data['descrizione_virus'].unique()}")
    #save different virus types in a table
    vaccines_virus = pd.DataFrame(vaccines_data['descrizione_virus'].value_counts())
    vaccines_virus.reset_index(inplace=True)
    vaccines_virus.columns = ['Virus Description', 'Count']
    vaccines_virus['Percentage'] = vaccines_virus['Count'] / vaccines_virus['Count'].sum() * 100
    vaccines_virus.to_csv(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_virus.csv', index=False)
    vaccines_virus.to_latex(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_virus.tex', index=False)

    total_virus = vaccines_data['descrizione_virus'].nunique()
    unique_atc = sum(
    1 for _, group in vaccines_data.groupby('descrizione_virus')
    if group['atc'].nunique() == 1      
    )
    print(f"Virus descriptions with unique ATC code: {unique_atc/total_virus*100:.2f}%")
    
    #save different virus types in a table
    vaccines_virus_atc = vaccines_data.groupby('descrizione_virus')['atc'].nunique().reset_index()
    vaccines_virus_atc.columns = ['Virus Description', 'Unique ATC Count']
    vaccines_virus_atc.to_csv(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_virus_atc.csv', index=False)
    vaccines_virus_atc.to_latex(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_virus_atc.tex', index=False)   

    #save codices in a table
    vaccines_codice = pd.DataFrame(vaccines_data['codice'].value_counts())
    vaccines_codice.reset_index(inplace=True)
    vaccines_codice.columns = ['Code', 'Count']
    vaccines_codice['Percentage'] = vaccines_codice['Count'] / vaccines_codice['Count'].sum() * 100
    vaccines_codice.to_csv(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_codice.csv', index=False)
    vaccines_codice.to_latex(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_codice.tex', index=False)


    
    
    print(vaccines_data['dose'].nunique(), vaccines_data['dose'].unique())
    print(vaccines_data['sede_somministrazione'].nunique(), vaccines_data['sede_somministrazione'].unique())
    print(vaccines_data['via_somministrazione'].nunique(), vaccines_data['via_somministrazione'].unique())
    print(f"Unique codes: {vaccines_data['codice'].nunique()}, Unique ATC: {vaccines_data['codice'].unique()}")
    print(f"Unique ATC codes: {vaccines_data['atc'].nunique()}, Unique ATC codes: {vaccines_data['atc'].unique()}")

    
    #filter only relevant virus vaccination by virus description and vaccination_counts excel
    relevant_virus_descriptions = vaccination_counts[vaccination_counts['relevant'] == 'y']
    relevant_virus_descriptions_by_group= relevant_virus_descriptions.groupby('group')['count'].sum()
    
    #make table with vaccination group, their count and percentage over total relevant virus descriptions
    vaccination_group_table = pd.DataFrame({
    'Vaccination Group': relevant_virus_descriptions_by_group.index,
    'Count': relevant_virus_descriptions_by_group.values,
    'Percentage': relevant_virus_descriptions_by_group.values / vaccines_data.shape[0] * 100
    })
    vaccination_group_table.to_csv(ROOT_RESULTS/'tables'/'vaccines'/'vaccination_group_table.csv', index=False)
    vaccination_group_table.to_latex(ROOT_RESULTS/'tables'/'vaccines'/'vaccination_group_table.tex', index=False)
    
    #total number of relevant vaccines
    total_relevant_vaccines = vaccines_data[vaccines_data['descrizione_virus'].isin(relevant_virus_descriptions['descrizione'])].shape[0]
    print(f"Total Relevant Vaccines: {total_relevant_vaccines}")
    
    #mean number of relevant vaccines per patient
    mean_relevant_vaccines_per_patient = vaccines_data[vaccines_data['descrizione_virus'].isin(relevant_virus_descriptions['descrizione'])].groupby('pedianet_id').size().mean()
    print(f"Mean Relevant Vaccines per Patient: {mean_relevant_vaccines_per_patient:.2f}")

    #number of patients with at least one relevant vaccine
    patients_with_relevant_vaccine = vaccines_data[vaccines_data['descrizione_virus'].isin(relevant_virus_descriptions['descrizione'])]['pedianet_id'].nunique()
    print(f"Patients with at least one relevant vaccine: {patients_with_relevant_vaccine}")
    print(f"Percentage of patients with at least one relevant vaccine: {patients_with_relevant_vaccine/patients_data['pedianet_id'].nunique()*100:.2f}%")
    
    
    #distribution of vaccines over time
    plt.figure(figsize=(8,6))
    plt.hist(vaccines_data['datarecord'], bins=80)
    plt.title("Number of Vaccinations Over Time")
    plt.xlabel("Time")
    plt.ylabel("Number of Vaccinations")
    plt.savefig(ROOT_RESULTS/'figures'/'vaccines'/'vaccinations_over_time.png', dpi=300, bbox_inches="tight")



    #distribution of vaccines over age at vaccination
    plt.figure(figsize=(8,6))
    plt.hist(vaccines_data['age_at_vaccination'], bins=20)
    plt.title("Age at Vaccination Distribution")
    plt.xlabel("Age at Vaccination (years)")
    plt.ylabel("Frequency")
    plt.savefig(ROOT_RESULTS/'figures'/'vaccines'/'age_at_vaccination_distribution.png', dpi=300, bbox_inches="tight")




    #vaccines per patient
    #distribution of number of vaccines per patient
    vaccines_per_patient = vaccines_data.groupby('pedianet_id').size()
    print(f"Mean Vaccines per Patient: {vaccines_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))

    plt.hist(vaccines_per_patient, bins=20)
    plt.xlabel("Number of Vaccines per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Vaccines per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'vaccines'/'vaccines_per_patient.png')

    
    #distribution of unique vaccines (atc) per patient
    unique_vaccines_per_patient = vaccines_data.groupby('pedianet_id')['atc'].nunique()
    print(f"Mean Unique Vaccines per Patient: {unique_vaccines_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    plt.hist(unique_vaccines_per_patient, bins=unique_vaccines_per_patient.nunique())
    plt.xlabel("Number of Unique Vaccines per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Unique Vaccines (atc) per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'vaccines'/'unique_vaccines_per_patient.png', dpi=300, bbox_inches="tight")

    #distribution of unique vaccines (virus) per patient
    unique_virus_per_patient = vaccines_data.groupby('pedianet_id')['descrizione_virus'].nunique()
    print(f"Mean Unique Virus Descriptions per Patient: {unique_virus_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    plt.hist(unique_virus_per_patient, bins=unique_virus_per_patient.nunique())
    plt.xlabel("Number of Unique Virus Descriptions per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Unique Virus Descriptions per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'vaccines'/'unique_virus_per_patient.png', dpi=300, bbox_inches="tight")

    #vaccines summary table
    vaccines_summary = pd.DataFrame(
        {
            "Total Vaccinations": [vaccines_data.shape[0]],
            "Unique Vaccine Drugs": [vaccines_data['farmaco'].value_counts(normalize=True).to_dict()],
            "Unique Virus Descriptions": [vaccines_data['descrizione_virus'].value_counts(normalize=True).to_dict()],
            "Unique Codes": [vaccines_data['codice'].value_counts(normalize=True).to_dict()],
            "Unique ATC": [vaccines_data['atc'].value_counts(normalize=True).to_dict()],
            "Unique Sides of Administration": [vaccines_data['sede_somministrazione'].value_counts(normalize=True).to_dict()],
            "Unique Routes of Administration": [vaccines_data['via_somministrazione'].value_counts(normalize=True).to_dict()],
            "Unique Doses": [vaccines_data['dose'].value_counts(normalize=True).to_dict()],
            "Mean Vaccines per Patient": [vaccines_per_patient.mean()],
            "Mean Unique Vaccines (atc) per Patient": [unique_vaccines_per_patient.mean()],
            "Mean Unique Vaccines (drug) per Patient": [vaccines_data.groupby('pedianet_id')['farmaco'].nunique().mean()],
            "Mean Unique Virus Descriptions per Patient": [vaccines_data.groupby('pedianet_id')['descrizione_virus'].nunique().mean()],
            "Drugs with unique ATC code": [sum(
                1 for _, group in vaccines_data.groupby('farmaco')
                if group['atc'].nunique() == 1      
            )/total*100],
            "Virus descriptions with unique ATC code": [sum(
                1 for _, group in vaccines_data.groupby('descrizione_virus')
                if group['atc'].nunique() == 1      
             )/total_virus*100]    

        }
    )
    vaccines_summary.to_csv(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_summary.csv', index=False)
    vaccines_summary.to_latex(ROOT_RESULTS/'tables'/'vaccines'/'vaccines_summary.tex', index=False)   



def drugs_exploration(drugs_data,patients_data):
    print("Drugs Exploration:")
    #table with column name, type, number of missing values and percentage of missing values
    dataset_info = pd.DataFrame({
        'Column Name': drugs_data.columns,
        'Data Type': [drugs_data[col].dtype for col in drugs_data.columns],
        'Missing Values': [drugs_data[col].isna().sum() for col in drugs_data.columns]
    })
    dataset_info['Percentage Missing'] = (dataset_info['Missing Values'] / len(drugs_data)) * 100
    dataset_info.to_csv(ROOT_RESULTS/'tables'/'drugs'/'dataset_info.csv', index=False)
    dataset_info.to_latex(ROOT_RESULTS/'tables'/'drugs'/'dataset_info.tex', index=False)

    

    #add age at drug administration
    drugs_data = drugs_data.merge(
    patients_data[['pedianet_id', 'datan']], 
    on='pedianet_id',   
    how='left'
    )
    drugs_data['age_at_drug'] = (drugs_data['dataricetta'] - drugs_data['datan']).dt.days/365.25

    print("Drugs Exploration:")
    print(f"Unique AIC code: {drugs_data['codice'].nunique()}, Unique Types: {drugs_data['codice'].unique()}")
    print(f"Unique principio: {drugs_data['principio'].nunique()}, Unique Types: {drugs_data['principio'].unique()}")
    print(f"Unique codice principio: {drugs_data['codiceprincipio'].nunique()}, Unique Types: {drugs_data['codiceprincipio'].unique()}")
    print(f"Unique ATC code: {drugs_data['atc'].nunique()}, Unique Types: {drugs_data['atc'].unique()}")
    print(f"Unique ATC descrizione: {drugs_data['atc_descrizione'].nunique()}, Unique Types: {drugs_data['atc_descrizione'].unique()}")

    print(f"Unique codice gruppo equivalenza: {drugs_data['codicegruppoequivalenza'].nunique()}, Unique Types: {drugs_data['codicegruppoequivalenza'].unique()}")
    print(f"Unique descrizione gruppo equivalenza: {drugs_data['descrizionegruppoequivalenza'].nunique()}, Unique Types: {drugs_data['descrizionegruppoequivalenza'].unique()}")
    
    #distribution of drug prescriptions per patient 
    drugs_per_patient = drugs_data.groupby('pedianet_id').size()
    print(f"Mean Drugs per Patient: {drugs_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    drugs_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Drugs per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Drugs per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'drugs'/'drugs_per_patient.png', dpi=300, bbox_inches="tight")

    # distribution of unique drugs (atc) per patient
    unique_drugs_per_patient = drugs_data.groupby('pedianet_id')['atc'].nunique()
    print(f"Mean Unique Drugs per Patient: {unique_drugs_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    unique_drugs_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Unique Drugs (ATC) per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Unique Drugs (ATC) per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'drugs'/'unique_drugs_per_patient.png', dpi=300, bbox_inches="tight")

    # save different atc in a table with count and percentage
    atc_table = pd.DataFrame({
        'ATC': drugs_data['atc'].value_counts().index,
        'Count': drugs_data['atc'].value_counts().values
    })
    atc_table['Percentage'] = atc_table['Count'] / atc_table['Count'].sum() * 100
    atc_table = atc_table.reset_index(drop=True)
    atc_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'atc.csv', index=False)
    atc_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'atc.tex', index=False)

    # retain top 15 atc
    top15_atc = atc_table.head(15)
    top15_atc.to_csv(ROOT_RESULTS/'tables'/'drugs'/'top15_atc.csv', index=False)
    top15_atc.to_latex(ROOT_RESULTS/'tables'/'drugs'/'top15_atc.tex', index=False)

    #save different atc_descrizione in a table with count and percentage
    atc_descrizione_table = pd.DataFrame({
        'ATC_Descrizione': drugs_data['atc_descrizione'].value_counts().index,
        'Count': drugs_data['atc_descrizione'].value_counts().values
    })
    atc_descrizione_table['Percentage'] = atc_descrizione_table['Count'] / atc_descrizione_table['Count'].sum() * 100
    atc_descrizione_table = atc_descrizione_table.reset_index(drop=True)
    atc_descrizione_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'atc_descrizione.csv', index=False)
    atc_descrizione_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'atc_descrizione.tex', index=False)

    # retain top 15 atc_descrizione
    top15_atc_descrizione = atc_descrizione_table.head(15)
    top15_atc_descrizione.to_csv(ROOT_RESULTS/'tables'/'drugs'/'top15_atc_descrizione.csv', index=False)
    top15_atc_descrizione.to_latex(ROOT_RESULTS/'tables'/'drugs'/'top15_atc_descrizione.tex', index=False)


    #distribution of unique drugs (codiceprincipio) per patient
    unique_principio_per_patient = drugs_data.groupby('pedianet_id')['codiceprincipio'].nunique()
    print(f"Mean Unique Principio per Patient: {unique_principio_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    unique_principio_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Unique Principio per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Unique Principio per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'drugs'/'unique_principio_per_patient.png', dpi=300, bbox_inches="tight")

    #save different codiceprincipio used in a table with count and percentage
    unique_codiceprincipio = drugs_data['codiceprincipio'].unique()

    codiceprincipio_table = pd.DataFrame({
        'CodicePrincipio': unique_codiceprincipio,
        'Count': [(drugs_data['codiceprincipio'] == cod).sum() for cod in unique_codiceprincipio]
    })

    codiceprincipio_table['Percentage'] = codiceprincipio_table['Count'] / codiceprincipio_table['Count'].sum() * 100

    codiceprincipio_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'unique_codiceprincipio.csv', index=False)
    codiceprincipio_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'unique_codiceprincipio.tex', index=False)

    #save in a table top 15 codiceprincipio
    top15_codiceprincipio = codiceprincipio_table.nlargest(15, 'Count')
    top15_codiceprincipio.to_csv(ROOT_RESULTS/'tables'/'drugs'/'top15_codiceprincipio.csv', index=False)
    top15_codiceprincipio.to_latex(ROOT_RESULTS/'tables'/'drugs'/'top15_codiceprincipio.tex', index=False)





    #distribution of unique drugs (codicegruppoequivalenza) per patient
    unique_gruppoequivalenza_per_patient = drugs_data.groupby('pedianet_id')['codicegruppoequivalenza'].nunique()
    print(f"Mean Unique Codice Gruppo Equivalenza per Patient: {unique_gruppoequivalenza_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    unique_gruppoequivalenza_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Unique Codice Gruppo Equivalenza per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Unique Codice Gruppo Equivalenza per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'drugs'/'unique_gruppoequivalenza_per_patient.png', dpi=300, bbox_inches="tight")

    # save gruppoequivalenza in a table with count and percentages
    gruppoequivalenza_table = pd.DataFrame({
        'CodiceGruppoEquivalenza': drugs_data['codicegruppoequivalenza'].unique(),
        'Count': [(drugs_data['codicegruppoequivalenza'] == cod).sum() 
                for cod in drugs_data['codicegruppoequivalenza'].unique()]
    })
    gruppoequivalenza_table['Percentage'] = (gruppoequivalenza_table['Count'] / 
                                            gruppoequivalenza_table['Count'].sum() * 100)
    gruppoequivalenza_table = gruppoequivalenza_table.sort_values('Count', ascending=False).reset_index(drop=True)

    gruppoequivalenza_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'gruppoequivalenza.csv', index=False)
    gruppoequivalenza_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'gruppoequivalenza.tex', index=False)


    #get info on fascia 
    drug_fascia=drugs_data['fascia']
    unique_fascia=drugs_data['fascia'].nunique()
    print(f"Number of different fascia are {unique_fascia}")
    plt.figure(figsize=(8,6))
    drug_fascia.value_counts().plot(kind='bar')
    plt.xlabel("Fascia")
    plt.ylabel("Number of Drugs")
    plt.title("Distribution of Drugs per Fascia")
    plt.savefig(ROOT_RESULTS/'figures'/'drugs'/'drug_per_fascia.png', dpi=300, bbox_inches="tight")

    #save a table where for each fascia you get count and percentage of drugs
    fascia_table = pd.DataFrame({
        'Fascia': drug_fascia.value_counts().index,
        'Count': drug_fascia.value_counts().values
    })
    fascia_table['Percentage'] = fascia_table['Count'] / fascia_table['Count'].sum() * 100
    fascia_table = fascia_table.reset_index(drop=True)

    fascia_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'fascia.csv', index=False)
    fascia_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'fascia.tex', index=False)

    # considers only fascia A drugs
    fascia_a_drugs = drugs_data[drugs_data['fascia'] == 'A']

    # distribution per patient
    fascia_a_per_patient = fascia_a_drugs.groupby('pedianet_id')['atc'].nunique()
    plt.figure(figsize=(8,6))
    fascia_a_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Unique Fascia A Drugs per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Fascia A Drugs per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'drugs'/'fascia_a_per_patient.png', dpi=300, bbox_inches="tight")

    # save in table with count and percentage - ATC
    fascia_a_atc_table = pd.DataFrame({
        'ATC': fascia_a_drugs['atc'].value_counts().index,
        'Count': fascia_a_drugs['atc'].value_counts().values
    })
    fascia_a_atc_table['Percentage'] = fascia_a_atc_table['Count'] / fascia_a_atc_table['Count'].sum() * 100
    fascia_a_atc_table = fascia_a_atc_table.reset_index(drop=True)
    fascia_a_atc_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_atc.csv', index=False)
    fascia_a_atc_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_atc.tex', index=False)

    #table with top 20 ATC for fascia A
    fascia_a_top20_atc_table = fascia_a_atc_table.head(20)
    fascia_a_top20_atc_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_top20_atc.csv', index=False)
    fascia_a_top20_atc_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_top20_atc.tex', index=False)   

    # save in table with count and percentage - ATC descrizione
    fascia_a_atc_descrizione_table = pd.DataFrame({
        'ATC_Descrizione': fascia_a_drugs['atc_descrizione'].value_counts().index,
        'Count': fascia_a_drugs['atc_descrizione'].value_counts().values
    })
    fascia_a_atc_descrizione_table['Percentage'] = fascia_a_atc_descrizione_table['Count'] / fascia_a_atc_descrizione_table['Count'].sum() * 100        
    fascia_a_atc_descrizione_table = fascia_a_atc_descrizione_table.reset_index(drop=True)
    fascia_a_atc_descrizione_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_atc_descrizione.csv', index=False)
    fascia_a_atc_descrizione_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_atc_descrizione.tex', index=False)   

    #table with top 20 ATC descrizione for fascia A
    fascia_a_top20_atc_descrizione_table = fascia_a_atc_descrizione_table.head(20)
    fascia_a_top20_atc_descrizione_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_top20_atc_descrizione.csv', index=False)     
    fascia_a_top20_atc_descrizione_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_top20_atc_descrizione.tex', index=False)

    # save in table with count and percentage - codice principio
    fascia_a_codiceprincipio_table = pd.DataFrame({
        'CodicePrincipio': fascia_a_drugs['codiceprincipio'].value_counts().index,
        'Count': fascia_a_drugs['codiceprincipio'].value_counts().  values
    })
    fascia_a_codiceprincipio_table['Percentage'] = fascia_a_codiceprincipio_table['Count'] / fascia_a_codiceprincipio_table['Count'].sum() * 100 
    fascia_a_codiceprincipio_table = fascia_a_codiceprincipio_table.reset_index(drop=True)
    fascia_a_codiceprincipio_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_codiceprincipio.csv', index=False)
    fascia_a_codiceprincipio_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_codiceprincipio.tex', index=False)

    #table with top 20 codice principio for fascia A
    fascia_a_top20_codiceprincipio_table = fascia_a_codiceprincipio_table.head(20)
    fascia_a_top20_codiceprincipio_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_top20_codiceprincipio.csv', index=False)
    fascia_a_top20_codiceprincipio_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_top20_codiceprincipio.tex', index=False)

    #save in table with count and percentage - principio
    fascia_a_principio_table = pd.DataFrame({
        'Principio': fascia_a_drugs['principio'].value_counts().index,  
        'Count': fascia_a_drugs['principio'].value_counts().values
    })
    fascia_a_principio_table['Percentage'] = fascia_a_principio_table['Count'] / fascia_a_principio_table['Count'].sum() * 100
    fascia_a_atc_table = fascia_a_principio_table.reset_index(drop=True)
    fascia_a_principio_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_principio.csv', index=False)
    fascia_a_principio_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_principio.tex', index=False)

    #table with top 20 principio for fascia A
    fascia_a_top20_principio_table = fascia_a_principio_table.head(20)
    fascia_a_top20_principio_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_top20_principio.csv', index=False)
    fascia_a_top20_principio_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_top20_principio.tex', index=False)
       

    # save in table with count and percentage - GruppoEquivalenza
    fascia_a_gruppo_table = pd.DataFrame({
        'CodiceGruppoEquivalenza': fascia_a_drugs['codicegruppoequivalenza'].value_counts().index,
        'Count': fascia_a_drugs['codicegruppoequivalenza'].value_counts().values
    })
    fascia_a_gruppo_table['Percentage'] = fascia_a_gruppo_table['Count'] / fascia_a_gruppo_table['Count'].sum() * 100
    fascia_a_gruppo_table = fascia_a_gruppo_table.reset_index(drop=True)
    fascia_a_gruppo_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_gruppoequivalenza.csv', index=False)
    fascia_a_gruppo_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'fascia_a_gruppoequivalenza.tex', index=False)

    #antibiotics, consider drugs with ATC code starting with J01
    antibiotics = drugs_data[drugs_data['atc'].str.startswith('J01')]
    print(f"Number of Antibiotics Prescriptions: {antibiotics.shape[0]}, Percentage of Antibiotics Prescriptions: {antibiotics.shape[0]/drugs_data.shape[0]*100:.2f}%")
    antibiotics_per_patient = antibiotics.groupby('pedianet_id').size()
    print(f"Mean Antibiotics per Patient: {antibiotics_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    antibiotics_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Antibiotics per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Antibiotics per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'drugs'/'antibiotics_per_patient.png')   

    #save in a table all antibiotics presciptions rows
    antibiotics.to_csv(ROOT_RESULTS/'tables'/'drugs'/'antibiotics.csv', index=False)

    #table with antibiotics classified by the first 4 digits of the ATC code with count and percentage
    antibiotics_atc4_table = pd.DataFrame({
        'ATC4': antibiotics['atc'].str[:4].value_counts().index,
        'Count': antibiotics['atc'].str[:4].value_counts().values
    })
    antibiotics_atc4_table['Percentage'] = antibiotics_atc4_table['Count'] / antibiotics_atc4_table['Count'].sum() * 100
    antibiotics_atc4_table = antibiotics_atc4_table.reset_index(drop=True)
    antibiotics_atc4_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'antibiotics_atc4.csv', index=False)
    antibiotics_atc4_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'antibiotics_atc4.tex', index=False)

    #respiratory drugs, consider drugs with ATC code starting with R
    respiratory_drugs = drugs_data[drugs_data['atc'].str.startswith('R')]
    print(f"Number of Respiratory Drugs Prescriptions: {respiratory_drugs.shape[0]}, Percentage of Respiratory Drugs Prescriptions: {respiratory_drugs.shape[0]/drugs_data.shape[0]*100:.2f}%")
    respiratory_drugs_per_patient = respiratory_drugs.groupby('pedianet_id').size()
    print(f"Mean Respiratory Drugs per Patient: {respiratory_drugs_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    respiratory_drugs_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Respiratory Drugs per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Respiratory Drugs per Patient")    
    plt.savefig(ROOT_RESULTS/'figures'/'drugs'/'respiratory_drugs_per_patient.png')
    #save in a table all respiratory drugs presciptions rows
    respiratory_drugs.to_csv(ROOT_RESULTS/'tables'/'drugs'/'respiratory_drugs.csv', index=False)
    

    #table with respiratory drugs classified by the first 3 digits of the ATC code with count and percentage
    respiratory_drugs_atc3_table = pd.DataFrame({
        'ATC3': respiratory_drugs['atc'].str[:3].value_counts().index,
        'Count': respiratory_drugs['atc'].str[:3].value_counts().values
    })
    respiratory_drugs_atc3_table['Percentage'] = respiratory_drugs_atc3_table['Count'] / respiratory_drugs_atc3_table['Count'].sum() * 100
    respiratory_drugs_atc3_table = respiratory_drugs_atc3_table.reset_index(drop=True)
    respiratory_drugs_atc3_table.to_csv(ROOT_RESULTS/'tables'/'drugs'/'respiratory_drugs_atc3.csv', index=False)
    respiratory_drugs_atc3_table.to_latex(ROOT_RESULTS/'tables'/'drugs'/'respiratory_drugs_atc3.tex', index=False)
    

    # drugs summary table
    drugs_fascia_a = drugs_data[drugs_data['fascia'] == 'A']

    drugs_summary = pd.DataFrame(
        {
            "Total Drug Prescriptions": [drugs_data.shape[0]],
            "Unique AIC code": [drugs_data['codice'].nunique()],
            "Unique Principio": [drugs_data['principio'].nunique()],
            "Unique Codice Principio": [drugs_data['codiceprincipio'].nunique()],
            "Unique ATC code": [drugs_data['atc'].nunique()],
            "Unique ATC descrizione": [drugs_data['atc_descrizione'].nunique()],
            "Unique Codice Gruppo Equivalenza": [drugs_data['codicegruppoequivalenza'].nunique()],
            "Mean Drugs per Patient": [drugs_per_patient.mean()],
            "Mean Unique AIC code per Patient": [drugs_data.groupby('pedianet_id')['codice'].nunique().mean()],
            "Mean Unique Principio per Patient": [drugs_data.groupby('pedianet_id')['principio'].nunique().mean()],
            "Mean Unique Codice Principio per Patient": [drugs_data.groupby('pedianet_id')['codiceprincipio'].nunique().mean()],
            "Mean Unique ATC code per Patient": [drugs_data.groupby('pedianet_id')['atc'].nunique().mean()],
            "Mean Unique ATC descrizione per Patient": [drugs_data.groupby('pedianet_id')['atc_descrizione'].nunique().mean()],
            "Mean Unique Codice Gruppo Equivalenza per Patient": [drugs_data.groupby('pedianet_id')['codicegruppoequivalenza'].nunique().mean()],
            # Fascia A
            "Total Fascia A Prescriptions": [drugs_fascia_a.shape[0]],
            "Percentage Fascia A Prescriptions": [drugs_fascia_a.shape[0] / drugs_data.shape[0] * 100],
            "Unique ATC code (Fascia A)": [drugs_fascia_a['atc'].nunique()],
            "Unique Codice Gruppo Equivalenza (Fascia A)": [drugs_fascia_a['codicegruppoequivalenza'].nunique()],
            "Mean Fascia A Drugs per Patient": [drugs_fascia_a.groupby('pedianet_id')['atc'].count().mean()],
            "Mean Unique ATC code per Patient (Fascia A)": [drugs_fascia_a.groupby('pedianet_id')['atc'].nunique().mean()],
            "Mean Unique Codice Gruppo Equivalenza per Patient (Fascia A)": [drugs_fascia_a.groupby('pedianet_id')['codicegruppoequivalenza'].nunique().mean()],
        }
    )
    drugs_summary.to_csv(ROOT_RESULTS/'tables'/'drugs'/'drugs_summary.csv', index=False)
    drugs_summary.to_latex(ROOT_RESULTS/'tables'/'drugs'/'drugs_summary.tex', index=False)


def outpatients_exploration(outpatients_data,branche_data,patients_data):
    
    #table with column name, type, number of missing values and percentage of missing values
    dataset_info = pd.DataFrame({
        'Column Name': outpatients_data.columns,
        'Data Type': [outpatients_data[col].dtype for col in outpatients_data.columns],
        'Missing Values': [outpatients_data[col].isna().sum() for col in outpatients_data.columns]
    })
    dataset_info['Percentage Missing'] = (dataset_info['Missing Values'] / len(outpatients_data)) * 100
    dataset_info.to_csv(ROOT_RESULTS/'tables'/'outpatients'/'dataset_info.csv', index=False)
    dataset_info.to_latex(ROOT_RESULTS/'tables'/'outpatients'/'dataset_info.tex', index=False)

    #add age at outpatient visit
    outpatients_data = outpatients_data.merge(
    patients_data[['pedianet_id', 'datan']], 
    on='pedianet_id',   
    how='left'
    )
    outpatients_data['age_at_outpatient'] = (outpatients_data['dataricetta'] - outpatients_data['datan']).dt.days/365.25

    print("Outpatients Exploration:")
    print(f"Unique Outpatient Types (codice): {outpatients_data['codice'].nunique()}, Unique Types: {outpatients_data['codice'].unique()}")
    print(f"unique Outpatient dmr: {outpatients_data['dmr'].nunique()}, Unique Types: {outpatients_data['dmr'].unique()}")
    print(f"Unique Category of specialistic visit: {outpatients_data['branca'].nunique()}, Unique Types: {outpatients_data['branca'].unique()}")
    print(f"Unique referto diagnoses: {outpatients_data['referto_diagnosi'].nunique()}, Unique Types: {outpatients_data['referto_diagnosi'].unique()}")

    #distribution of outpatient visits per patient
    outpatients_per_patient = outpatients_data.groupby('pedianet_id').size()
    print(f"Mean Outpatient Visits per Patient: {outpatients_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    outpatients_data.groupby('pedianet_id').size().value_counts().plot(kind='bar')
    plt.xlabel("Number of Outpatient Visits per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Outpatient Visits per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'outpatients'/'outpatients_per_patient.png')

    #distribution of unique outpatient types codice  per patient
    unique_outpatient_types_per_patient = outpatients_data.groupby('pedianet_id')['codice'].nunique()
    print(f"Mean Unique Outpatient Types per Patient: {unique_outpatient_types_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    unique_outpatient_types_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Unique Outpatient Types per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Unique Outpatient Types (codice) per Patient")  
    plt.savefig(ROOT_RESULTS/'figures'/'outpatients'/'unique_outpatient_types_per_patient.png')

    #table with descrizione with count and percentage
    descrizione_table = pd.DataFrame({
        'Descrizione': outpatients_data['descrizione'].value_counts().index,
        'Count': outpatients_data['descrizione'].value_counts().values
    })
    descrizione_table['Percentage'] = descrizione_table['Count'] / descrizione_table['Count'].sum() * 100
    descrizione_table = descrizione_table.reset_index(drop=True)
    descrizione_table.to_csv(ROOT_RESULTS/'tables'/'outpatients'/'descrizione.csv', index=False)
    descrizione_table.to_latex(ROOT_RESULTS/'tables'/'outpatients'/'descrizione.tex', index=False)

    #top 20 descrizione
    top20_descrizione_table = descrizione_table.head(20)
    top20_descrizione_table.to_csv(ROOT_RESULTS/'tables'/'outpatients'/'top20_descrizione.csv', index=False)
    top20_descrizione_table.to_latex(ROOT_RESULTS/'tables'/'outpatients'/'top20_descrizione.tex', index=False)




    #table with codice with count and percentage
    codice_table = pd.DataFrame({
        'Codice': outpatients_data['codice'].value_counts().index,
        'Count': outpatients_data['codice'].value_counts().values
    })
    codice_table['Percentage'] = codice_table['Count'] / codice_table['Count'].sum() * 100
    codice_table = codice_table.reset_index(drop=True)
    codice_table.to_csv(ROOT_RESULTS/'tables'/'outpatients'/'codice.csv', index=False)
    codice_table.to_latex(ROOT_RESULTS/'tables'/'outpatients'/'codice.tex', index=False)

    #table with top 15 codice with count and percentage
    top15_codice_table = codice_table.head(30)
    top15_codice_table.to_csv(ROOT_RESULTS/'tables'/'outpatients'/'top30_codice.csv', index=False)
    top15_codice_table.to_latex(ROOT_RESULTS/'tables'/'outpatients'/'top30_codice.tex', index=False) 



    #distribution of unique outpatient categories (branca) per patient
    unique_outpatient_branca_per_patient = outpatients_data.groupby('pedianet_id')['branca'].nunique()
    print(f"Mean Unique Outpatient Categories (Branca) per Patient: {unique_outpatient_branca_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    unique_outpatient_branca_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Unique Outpatient Categories (Branca) per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Unique Outpatient Categories (Branca) per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'outpatients'/'unique_outpatient_branca_per_patient.png') 

    
    

    #count nan
    nan_branca_count = outpatients_data['branca'].isna().sum()
    print(f"Number of NaN values in 'branca' column: {nan_branca_count}")

    #convert branca in int64 from float64
    outpatients_data['branca'] = outpatients_data['branca'].fillna(-1).astype(int)


    #check type branca before merging
    print(f"Data type of 'branca' in outpatients_data: {outpatients_data['branca'].dtype}")
    print(f"Data type of 'branca' in branche_data: {branche_data['branca'].dtype}")
    
    
    #add branca description to outpatients data
    
    outpatients_data = outpatients_data.merge(
        branche_data[['branca', 'denominazione_branca']],
        on='branca',
        how='left'
    )

    print(f"Unique Branca Descriptions: {outpatients_data['denominazione_branca'].nunique()}, Unique Types: {outpatients_data['denominazione_branca'].unique()}")

    #table with branca with count and percentage
    branca_table = pd.DataFrame({
        'branca': outpatients_data['branca'].value_counts().index,
        'Count': outpatients_data['branca'].value_counts().values
    })
    #add description to branca table
    branca_table = branca_table.merge(
        branche_data[['branca', 'denominazione_branca']],
        on='branca',
        how='left'
    )
    branca_table['Percentage'] = branca_table['Count'] / branca_table['Count'].sum() * 100
    branca_table = branca_table.reset_index(drop=True)  
    branca_table.to_csv(ROOT_RESULTS/'tables'/'outpatients'/'branca.csv', index=False)
    branca_table.to_latex(ROOT_RESULTS/'tables'/'outpatients'/'branca.tex', index=False)

    #table with descrizione list on interesting branca (21, 6,22)
    interesting_branca = [21, 6, 22,8,26]
    for branca in interesting_branca:
        branca_data = outpatients_data[outpatients_data['branca'] == branca]
        
        # Conta le descrizioni
        descrizione_counts = branca_data['description_from_catalogo'].value_counts()
        
        # Per ogni descrizione, prendi il primo codice associato
        descrizione_codice = (
            branca_data.groupby('description_from_catalogo')['codice']
            .first()
        )
        
        branca_descrizione_table = pd.DataFrame({
            'Codice': descrizione_codice[descrizione_counts.index].values,
            'Descrizione': descrizione_counts.index,
            'Count': descrizione_counts.values
        })
        
        branca_descrizione_table['Percentage'] = branca_descrizione_table['Count'] / outpatients_data.shape[0] * 100
        branca_descrizione_table = branca_descrizione_table.reset_index(drop=True)
        branca_descrizione_table.to_csv(ROOT_RESULTS/'tables'/'outpatients'/'branca_{branca}_descrizione_catalogo.csv', index=False)
        branca_descrizione_table.to_latex(ROOT_RESULTS/'tables'/'outpatients'/'branca_{branca}_descrizione_catalogo.tex', index=False)

    #distribution of unique dmr per patient
    unique_dmr_per_patient = outpatients_data.groupby('pedianet_id')['dmr'].nunique()
    print(f"Mean Unique DMR per Patient: {unique_dmr_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))       
    unique_dmr_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Unique DMR per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Unique DMR per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'outpatients'/'unique_dmr_per_patient.png')
    
    #table with dmr with count and percentage
    # table with dmr with count and percentage
    dmr_table = pd.DataFrame({
        'DMR': outpatients_data['dmr'].value_counts().index,
        'Count': outpatients_data['dmr'].value_counts().values
    })
    dmr_table['Percentage'] = dmr_table['Count'] / outpatients_data.shape[0] * 100
    dmr_table = dmr_table.reset_index(drop=True)
    dmr_table.to_csv(ROOT_RESULTS/'tables'/'outpatients'/'dmr.csv', index=False)
    dmr_table.to_latex(ROOT_RESULTS/'tables'/'outpatients'/'dmr.tex', index=False)

    #top 15 dmr
    top15_dmr_table = dmr_table.head(15)
    top15_dmr_table.to_csv(ROOT_RESULTS/'tables'/'outpatients'/'top15_dmr.csv', index=False)     
    top15_dmr_table.to_latex(ROOT_RESULTS/'tables'/'outpatients'/'top15_dmr.tex', index=False)


    #table with descrizione and description from catalogo
    compare_descrizioni=pd.DataFrame({
        'Descrizione': outpatients_data['descrizione'],
        'Description from Catalogo': outpatients_data['description_from_catalogo']      
    })
    compare_descrizioni.to_csv(ROOT_RESULTS/'tables'/'outpatients'/'compare_descrizioni.csv', index=False)
    compare_descrizioni.to_latex(ROOT_RESULTS/'tables'/'outpatients'/'compare_descrizioni.tex', index=False)

    #outpatients summary table
    outpatients_summary = pd.DataFrame(
        {
            "Total Outpatient Visits": [outpatients_data.shape[0]],
            "Number of patients with at least one outpatient visit": [outpatients_data['pedianet_id'].nunique()],
            "Number of patients without outpatient visits": [patients_data.shape[0] - outpatients_data['pedianet_id'].nunique()],
            "Unique Outpatient Codes": [outpatients_data['codice'].nunique()],
            "Unique DMR": [outpatients_data['dmr'].nunique()],
            "Unique Outpatient Categories": [outpatients_data['branca'].nunique() ],
            "Mean Outpatient Visits per Patient": [outpatients_per_patient.mean()],
            "Mean Unique Outpatient Codes per Patient": [outpatients_data.groupby('pedianet_id')['codice'].nunique().mean()],
            "Mean Unique Outpatient Categories per Patient": [outpatients_data.groupby('pedianet_id')['branca'].nunique().mean()],
            "Mean Unique DMR per Patient": [outpatients_data.groupby('pedianet_id')['dmr'].nunique().mean()]
        }
    )
    outpatients_summary.to_csv(ROOT_RESULTS/'tables'/'outpatients'/'outpatients_summary.csv', index=False)
    outpatients_summary.to_latex(ROOT_RESULTS/'tables'/'outpatients'/'outpatients_summary.tex', index=False)

    

def contacts_exploration(contacts_data,patients_data):

    print("Contacts Exploration:")
    #table with column name, type, missing values and percentage of missing values
    contacts_info = pd.DataFrame({
        'Column Name': contacts_data.columns,
        'Data Type': contacts_data.dtypes,
        'Missing Values': contacts_data.isnull().sum(),
        'Percentage Missing': contacts_data.isnull().mean() * 100
    })
    contacts_info.to_csv(ROOT_RESULTS/'tables'/'contacts'/'contacts_info.csv', index=False)
    contacts_info.to_latex(ROOT_RESULTS/'tables'/'contacts'/'contacts_info.tex', index=False)

    #explore column guid 
    print (f"Unique GUID: {contacts_data['guid'].nunique()}, Unique Types: {contacts_data['guid'].unique()}")

    #add age at contact
    contacts_data = contacts_data.merge(
    patients_data[['pedianet_id', 'datan']], 
    on='pedianet_id',   
    how='left'
    )
    contacts_data['age_at_contact'] = (contacts_data['datacontatto'] - contacts_data['datan']).dt.days/365.25

    #tables with different motivocontatto and their counts and percentages
    print("Contacts Exploration:")
    print(f"Unique Contact Reasons (motivo contatto): {contacts_data['motivocontatto'].nunique()}, Unique Types: {contacts_data['motivocontatto'].unique()}")
    contact_reasons = pd.DataFrame(contacts_data['motivocontatto'].value_counts())
    contact_reasons.reset_index(inplace=True)
    contact_reasons.columns = ['Contact Reason', 'Count']
    contact_reasons['Percentage'] = contact_reasons['Count']/contact_reasons['Count'].sum()*100
    contact_reasons.to_csv(ROOT_RESULTS/'tables'/'contacts'/'contact_reasons.csv', index=False)
    contact_reasons.to_latex(ROOT_RESULTS/'tables'/'contacts'/'contact_reasons.tex', index=False)


    #distribution of contacts per patient
    contacts_per_patient = contacts_data.groupby('pedianet_id').size()
    print(f"Mean Contacts per Patient: {contacts_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    plt.hist(contacts_per_patient, bins=50)
    plt.xlabel("Number of Contacts per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Contacts per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'contacts'/'contacts_per_patient.png', dpi=300, bbox_inches="tight")

    #distribution of unique contact reasons per patient
    unique_contact_reasons_per_patient = contacts_data.groupby('pedianet_id')['motivocontatto'].nunique()
    print(f"Mean Unique Contact Reasons per Patient: {unique_contact_reasons_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    unique_contact_reasons_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Unique Contact Reasons per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Unique Contact Reasons per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'contacts'/'unique_contact_reasons_per_patient.png', dpi=300, bbox_inches="tight")

    #info about diagnosi columns 
    print(f"Unique Diagnoses1 in Contacts: {contacts_data['diagnosi1'].nunique()}, Unique Types: {contacts_data['diagnosi1'].unique()}")
    print(f"Unique Diagnoses1 ICD9 in Contacts: {contacts_data['diagnosi1_icd9'].nunique()}, Unique Types: {contacts_data['diagnosi1_icd9'].unique()}")

    #tables with diagnosis 1 and their counts and percentages
    diagnosis1_table = pd.DataFrame(contacts_data['diagnosi1'].value_counts())
    diagnosis1_table.reset_index(inplace=True)
    diagnosis1_table.columns = ['Diagnosis 1', 'Count']
    diagnosis1_table['Percentage'] = diagnosis1_table['Count']/diagnosis1_table['Count'].sum()*100
    diagnosis1_table.to_csv(ROOT_RESULTS/'tables'/'contacts'/'diagnosis1.csv', index=False)
    diagnosis1_table.to_latex(ROOT_RESULTS/'tables'/'contacts'/'diagnosis1.tex', index=False)
    

    #top 30 diagnosis1
    top30_diagnosis1 = diagnosis1_table.head(30)
    top30_diagnosis1.to_csv(ROOT_RESULTS/'tables'/'contacts'/'top30_diagnosis1.csv', index=False)
    top30_diagnosis1.to_latex(ROOT_RESULTS/'tables'/'contacts'/'top30_diagnosis1.tex', index=False)

    #table with diagnosis1_icd9 and their counts and percentages
    diagnosis1_icd9_table = pd.DataFrame(contacts_data['diagnosi1_icd9'].value_counts())
    diagnosis1_icd9_table.reset_index(inplace=True)
    diagnosis1_icd9_table.columns = ['Diagnosis 1 ICD9', 'Count']
    diagnosis1_icd9_table['Percentage'] = diagnosis1_icd9_table['Count']/diagnosis1_icd9_table['Count'].sum()*100
    diagnosis1_icd9_table.to_csv(ROOT_RESULTS/'tables'/'contacts'/'diagnosis1_icd9.csv', index=False)
    diagnosis1_icd9_table.to_latex(ROOT_RESULTS/'tables'/'contacts'/'diagnosis1_icd9.tex', index=False)  

    #top 30 diagnosis1_icd9
    top50_diagnosis1_icd9 = diagnosis1_icd9_table.head(50)
    top50_diagnosis1_icd9.to_csv(ROOT_RESULTS/'tables'/'contacts'/'top50_diagnosis1_icd9.csv', index=False)
    top50_diagnosis1_icd9.to_latex(ROOT_RESULTS/'tables'/'contacts'/'top50_diagnosis1_icd9.tex', index=False)


    #add most frequent diagnosis1 to the icd9 table
    top50_diagnosis1_icd9['Most Frequent Diagnosis 1'] = top50_diagnosis1_icd9['Diagnosis 1 ICD9'].map(
        contacts_data.groupby('diagnosi1_icd9')['diagnosi1'].agg(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan)
    )
    top50_diagnosis1_icd9.to_csv(ROOT_RESULTS/'tables'/'contacts'/'top50_diagnosis1_icd9_with_most_frequent_diagnosis1.csv', index=False)
    top50_diagnosis1_icd9.to_latex(ROOT_RESULTS/'tables'/'contacts'/'top50_diagnosis1_icd9_with_most_frequent_diagnosis1.tex', index=False)

    #check precision ICD9 codes for diagnosis1_icd9, how many have 3 digits, how many have 4 digits, how many have 5 digits (without considering the dot)
    contacts_data['diagnosi1_icd9_digits'] = contacts_data['diagnosi1_icd9'].str.replace('.', '', regex=False).str.len()
    #store in a table the count and percentage of each number of digits
    icd9_digits_table = pd.DataFrame(contacts_data['diagnosi1_icd9_digits'].value_counts())
    icd9_digits_table.reset_index(inplace=True)
    icd9_digits_table.columns = ['Number of Digits', 'Count']
    icd9_digits_table['Percentage'] = icd9_digits_table['Count'] / icd9_digits_table['Count'].sum() * 100
    icd9_digits_table.to_csv(ROOT_RESULTS/'tables'/'contacts'/'diagnosis1_icd9_digits.csv', index=False)
    icd9_digits_table.to_latex(ROOT_RESULTS/'tables'/'contacts'/'diagnosis1_icd9_digits.tex', index=False)

    
    #number of doctors in the contacts data
    print(f"Number of unique doctors in contacts data: {contacts_data['idmedico'].nunique()}, Unique Types: {contacts_data['idmedico'].unique()}")




    
    #table with icd9 codes from 2 categories relevant for respiratory diseases(diseases of the respiratory system 460-519 and 001-139 infectious and parasitic diseases) and their counts and percentages
    #respiratory system ICD9 codes are from 460 to 519
    prefixes = tuple(str(i) for i in range(460, 520))

    respiratory_system_icd9 = contacts_data[
        contacts_data['diagnosi1_icd9'].str.startswith(prefixes)
    ]

    print(f"Number of contacts with respiratory system ICD9 codes: {respiratory_system_icd9.shape[0]}")
    print(f"Percentage of contacts with respiratory system ICD9 codes: {respiratory_system_icd9.shape[0] / contacts_data.shape[0] * 100:.2f}% ")

    respiratory_system_icd9_table = pd.DataFrame({
        'Diagnosis 1 ICD9': respiratory_system_icd9['diagnosi1_icd9'].value_counts().index,
        'Count': respiratory_system_icd9['diagnosi1_icd9'].value_counts().values
    })
    #percentage over total diagnosis1_icd9
    respiratory_system_icd9_table['Percentage']= respiratory_system_icd9_table['Count'] / contacts_data['diagnosi1_icd9'].value_counts().sum() * 100
    #add most frequent diagnosis1 to the respiratory system icd9 table
    respiratory_system_icd9_table['Most Frequent Diagnosis 1'] = respiratory_system_icd9_table['Diagnosis 1 ICD9'].map(
        contacts_data.groupby('diagnosi1_icd9')['diagnosi1'].agg(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan)
    )

    respiratory_system_icd9_table.to_csv(ROOT_RESULTS/'tables'/'contacts'/'respiratory_system_icd9.csv', index=False)
    respiratory_system_icd9_table.to_latex(ROOT_RESULTS/'tables'/'contacts'/'respiratory_system_icd9.tex', index=False)

    #looking at respiratory system diseases , aggregate codes looking at the first 3 digits
    respiratory_system_icd9_grouped = respiratory_system_icd9.groupby(respiratory_system_icd9['diagnosi1_icd9'].str.slice(0, 3)).size()
    respiratory_system_icd9_grouped_table = pd.DataFrame({
        'Diagnosis 1 ICD9 (first 3 digits)': respiratory_system_icd9_grouped.index,
        'Count': respiratory_system_icd9_grouped.values
    })
    #add percentage over total diagnosis1_icd9
    respiratory_system_icd9_grouped_table['Percentage'] = respiratory_system_icd9_grouped_table['Count'] / contacts_data['diagnosi1_icd9'].value_counts().sum() * 100
    #add most frequent diagnosis1 to the respiratory system icd9 grouped table
    respiratory_system_icd9_grouped_table['Most Frequent Diagnosis 1'] = respiratory_system_icd9_grouped_table['Diagnosis 1 ICD9 (first 3 digits)'].map(
        contacts_data.groupby(contacts_data['diagnosi1_icd9'].str.slice(0, 3))['diagnosi1'].agg(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan)
    )
    respiratory_system_icd9_grouped_table.to_csv(ROOT_RESULTS/'tables'/'contacts'/'respiratory_system_icd9_grouped.csv', index=False)
    respiratory_system_icd9_grouped_table.to_latex(ROOT_RESULTS/'tables'/'contacts'/'respiratory_system_icd9_grouped.tex', index=False)  

    #infectious and parasitic diseases ICD9 codes are from 001 to 139
    prefixes = tuple(str(i).zfill(3) for i in range(1, 140))
    infectious_parasitic_icd9 = contacts_data[
        contacts_data['diagnosi1_icd9'].str.startswith(prefixes)
    ]

    print(f"Number of contacts with infectious and parasitic diseases ICD9 codes: {infectious_parasitic_icd9.shape[0]}")
    print(f"Percentage of contacts with infectious and parasitic diseases ICD9 codes: {infectious_parasitic_icd9.shape[0] / contacts_data.shape[0] * 100:.2f}% ")
    infectious_parasitic_icd9_table = pd.DataFrame({
        'Diagnosis 1 ICD9': infectious_parasitic_icd9['diagnosi1_icd9'].value_counts().index,
        'Count': infectious_parasitic_icd9['diagnosi1_icd9'].value_counts().values
    })
    infectious_parasitic_icd9_table['Percentage'] = infectious_parasitic_icd9_table['Count'] / contacts_data['diagnosi1_icd9'].value_counts().sum() * 100
    #add most frequent diagnosis1 to the infectious parasitic icd9 table
    infectious_parasitic_icd9_table['Most Frequent Diagnosis 1'] = infectious_parasitic_icd9_table['Diagnosis 1 ICD9'].map(
        contacts_data.groupby('diagnosi1_icd9')['diagnosi1'].agg(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan)
    )
    infectious_parasitic_icd9_table.to_csv(ROOT_RESULTS/'tables'/'contacts'/'infectious_parasitic_icd9.csv', index=False)
    infectious_parasitic_icd9_table.to_latex(ROOT_RESULTS/'tables'/'contacts'/'infectious_parasitic_icd9.tex', index=False)
    
   #table with  Disease of the nervous system and sense organ ICD9 codes (320-389) and their counts and percentages
    prefixes = tuple(str(i) for i in range(320, 390))
    nervous_system_icd9 = contacts_data[
        contacts_data['diagnosi1_icd9'].str.startswith(prefixes)
    ]   
    print(f"Number of contacts with nervous system and sense organ ICD9 codes: {nervous_system_icd9.shape[0]}")
    print(f"Percentage of contacts with nervous system and sense organ ICD9 codes: {nervous_system_icd9.shape[0] / contacts_data.shape[0] * 100:.2f}% ")
    nervous_system_icd9_table = pd.DataFrame({
        'Diagnosis 1 ICD9': nervous_system_icd9['diagnosi1_icd9'].value_counts().index,
        'Count': nervous_system_icd9['diagnosi1_icd9'].value_counts().values
    })
    nervous_system_icd9_table['Percentage'] = nervous_system_icd9_table['Count'] / contacts_data['diagnosi1_icd9'].value_counts().sum() * 100
    #add most frequent diagnosis1 to the nervous system icd9 table
    nervous_system_icd9_table['Most Frequent Diagnosis 1'] = nervous_system_icd9_table['Diagnosis 1 ICD9'].map(
        contacts_data.groupby('diagnosi1_icd9')['diagnosi1'].agg(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan)
    )
    nervous_system_icd9_table.to_csv(ROOT_RESULTS/'tables'/'contacts'/'nervous_system_icd9.csv', index=False)
    nervous_system_icd9_table.to_latex(ROOT_RESULTS/'tables'/'contacts'/'nervous_system_icd9.tex', index=False)
    
    #table with General symptoms ICD9 codes (780-789) and their counts and percentages
    prefixes = tuple(str(i) for i in range(780, 790))
    general_symptoms_icd9 = contacts_data[
        contacts_data['diagnosi1_icd9'].str.startswith(prefixes)
    ]
    print(f"Number of contacts with general symptoms ICD9 codes: {general_symptoms_icd9.shape[0]}")
    print(f"Percentage of contacts with general symptoms ICD9 codes: {general_symptoms_icd9.shape[0] / contacts_data.shape[0] * 100:.2f}% ")
    general_symptoms_icd9_table = pd.DataFrame({
        'Diagnosis 1 ICD9': general_symptoms_icd9['diagnosi1_icd9'].value_counts().index,
        'Count': general_symptoms_icd9['diagnosi1_icd9'].value_counts().values
    })
    general_symptoms_icd9_table['Percentage'] = general_symptoms_icd9_table['Count'] / contacts_data['diagnosi1_icd9'].value_counts().sum() *   100
    #add most frequent diagnosis1 to the general symptoms icd9 table
    general_symptoms_icd9_table['Most Frequent Diagnosis 1'] = general_symptoms_icd9_table['Diagnosis 1 ICD9'].map(
        contacts_data.groupby('diagnosi1_icd9')['diagnosi1'].agg(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else np.nan)
    )
    general_symptoms_icd9_table.to_csv(ROOT_RESULTS/'tables'/'contacts'/'general_symptoms_icd9.csv', index=False)
    general_symptoms_icd9_table.to_latex(ROOT_RESULTS/'tables'/'contacts'/'general_symptoms_icd9.tex', index=False)  
    
    #missing values in diagnosis1, check doctors
    missing_diagnosis1 = contacts_data[contacts_data['diagnosi1'].isnull()]
    print(f"Number of contacts with missing diagnosis1: {missing_diagnosis1.shape[0]}")
    print(f"Unique doctors for contacts with missing diagnosis1: {missing_diagnosis1['idmedico'].nunique()}, Unique Types: {missing_diagnosis1['idmedico'].unique()}")  

    #table with the most frequent doctors for contacts with missing diagnosis1
    top_doctors_missing_diagnosis1 = missing_diagnosis1['idmedico'].value_counts().head(30)
    top_doctors_missing_diagnosis1_table = pd.DataFrame({
        'Doctor ID': top_doctors_missing_diagnosis1.index,
        'Count': top_doctors_missing_diagnosis1.values,
        'Percentage': top_doctors_missing_diagnosis1.values / top_doctors_missing_diagnosis1.values.sum() * 100
    })
    top_doctors_missing_diagnosis1_table.to_csv(ROOT_RESULTS/'tables'/'contacts'/'top_doctors_missing_diagnosis1.csv', index=False)
    top_doctors_missing_diagnosis1_table.to_latex(ROOT_RESULTS/'tables'/'contacts'/'top_doctors_missing_diagnosis1.tex', index=False)


    #missing values in ICD9 diagnosis1, check doctors
    missing_diagnosis1_icd9 = contacts_data[contacts_data['diagnosi1_icd9'].isnull()]
    print(f"Number of contacts with missing diagnosis1_icd9: {missing_diagnosis1_icd9.shape[0]}")
    print(f"Unique doctors for contacts with missing diagnosis1_icd9: {missing_diagnosis1_icd9['idmedico'].nunique()}, Unique Types: {missing_diagnosis1_icd9['idmedico'].unique()}")

    #table with the most frequent doctors for contacts with missing diagnosis1_icd9
    top_doctors_missing_diagnosis1_icd9 = missing_diagnosis1_icd9['idmedico'].value_counts().head(30)
    top_doctors_missing_diagnosis1_icd9_table = pd.DataFrame({
        'Doctor ID': top_doctors_missing_diagnosis1_icd9.index,
        'Count': top_doctors_missing_diagnosis1_icd9.values,
        'Percentage over total missing': top_doctors_missing_diagnosis1_icd9.values / top_doctors_missing_diagnosis1_icd9.values.sum() * 100,
        'Percentage over total contacts': top_doctors_missing_diagnosis1_icd9.values / contacts_data.shape[0] * 100,
        'Percentage over contacts of the doctor': top_doctors_missing_diagnosis1_icd9.values / contacts_data[contacts_data['idmedico'].isin(top_doctors_missing_diagnosis1_icd9.index)].shape[0] * 100  
    })
    top_doctors_missing_diagnosis1_icd9_table.to_csv(ROOT_RESULTS/'tables'/'contacts'/'top_doctors_missing_diagnosis1_icd9.csv', index=False)
    top_doctors_missing_diagnosis1_icd9_table.to_latex(ROOT_RESULTS/'tables'/'contacts'/'top_doctors_missing_diagnosis1_icd9.tex', index=False)  

    #number of infezioni acute delle vie respiratorie superiori (diagnosi1) per patient 
    infezione_acuta_per_patient = contacts_data[contacts_data['diagnosi1'] == "Infezione acuta delle vie respiratorie superiori"].groupby('pedianet_id').size()
    print(f"Mean Infezione Acuta delle Vie Respiratorie Superiori per Patient: {infezione_acuta_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    infezione_acuta_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Infezione Acuta delle Vie Respiratorie Superiori per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Infezione Acuta delle Vie Respiratorie Superiori per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'contacts'/'infezione_acuta_per_patient.png')

    #distribution of contacts over time
    plt.figure(figsize=(8,6))
    plt.hist(contacts_data['datacontatto'], bins=80)
    plt.title("Number of Contacts Over Time")
    plt.xlabel("Time")
    plt.ylabel("Number of Contacts")
    plt.savefig(ROOT_RESULTS/'figures'/'contacts'/'contacts_over_time.png', dpi=300, bbox_inches="tight")

    #ditribution of contacts over age at contact
    plt.figure(figsize=(8,6))
    plt.hist(contacts_data['age_at_contact'], bins=7)
    plt.title("Age at Contact Distribution")
    plt.xlabel("Age at Contact (years)")
    plt.ylabel("Frequency")
    plt.savefig(ROOT_RESULTS/'figures'/'contacts'/'age_at_contact_distribution.png', dpi=300, bbox_inches="tight")

    #summary table for contacts
    contacts_summary = pd.DataFrame(
        {
            "Total Contacts": [contacts_data.shape[0]],
            "Patients with at least one contact": [contacts_data['pedianet_id'].nunique()],
            "Patients without contacts": [patients_data.shape[0] - contacts_data['pedianet_id'].nunique()],
            "Unique Contact Reasons": [contacts_data['motivocontatto'].nunique()],
            "Unique Diagnoses in Contacts": [contacts_data['diagnosi1'].nunique()],
            "Unique Diagnoses ICD9 in Contacts": [contacts_data['diagnosi1_icd9'].nunique()],
            "Mean Contacts per Patient": [contacts_per_patient.mean()],
            "Mean Unique Contact Reasons per Patient": [contacts_data.groupby('pedianet_id')['motivocontatto'].nunique().mean()],
            "Date first contact": [contacts_data['datacontatto'].min()],
            "Date last contact": [contacts_data['datacontatto'].max()],
            "Patients with Infezione acuta delle vie respiratorie superiori": [contacts_data[contacts_data['diagnosi1'] == "Infezione acuta delle vie respiratorie superiori"]['pedianet_id'].nunique()],
            "Number of doctors with missing diagnosis1": [missing_diagnosis1['idmedico'].nunique()],
            "Number of doctors with missing diagnosis1_icd9": [missing_diagnosis1_icd9['idmedico'].nunique()],
            "Number of unique doctors in contacts": [contacts_data['idmedico'].nunique()]
        }
    )
    contacts_summary.to_csv(ROOT_RESULTS/'tables'/'contacts'/'contacts_summary.csv', index=False)
    contacts_summary.to_latex(ROOT_RESULTS/'tables'/'contacts'/'contacts_summary.tex', index=False)


def exemptions_exploration(exemptions_data,comorbidities_exemptions_data,patients_data):
    #table with column name, type, missing values and percentage of missing values
    exemptions_info = pd.DataFrame({
        'Column Name': exemptions_data.columns, 
        'Data Type': exemptions_data.dtypes,
        'Missing Values': exemptions_data.isnull().sum(),
        'Percentage Missing': exemptions_data.isnull().mean() * 100
    })
    exemptions_info.to_csv(ROOT_RESULTS/'tables'/'exemptions'/'exemptions_info.csv', index=False)
    exemptions_info.to_latex(ROOT_RESULTS/'tables'/'exemptions'/'exemptions_info.tex', index=False)

    #add age at exemption
    exemptions_data = exemptions_data.merge(
    patients_data[['pedianet_id', 'datan']], 
    on='pedianet_id',   
    how='left'
    )
    exemptions_data['age_at_exemption'] = (exemptions_data['decorrenza'] - exemptions_data['datan']).dt.days/365.25

    print("Exemptions Exploration:")
    #types of exemptions and their counts and percentages
    print(f"Unique Exemption Types: {exemptions_data['tipo'].nunique()}, Unique Types: {exemptions_data['tipo'].unique()}")
    exemption_types = pd.DataFrame(exemptions_data['tipo'].value_counts())
    exemption_types.reset_index(inplace=True)
    exemption_types.columns = ['Exemption Type', 'Count']
    exemption_types['Percentage'] = exemption_types['Count']/exemption_types['Count'].sum()*100
    exemption_types.to_csv(ROOT_RESULTS/'tables'/'exemptions'/'exemption_types.csv', index=False)
    exemption_types.to_latex(ROOT_RESULTS/'tables'/'exemptions'/'exemption_types.tex', index=False) 

    #distribution of exemptions types
    plt.figure(figsize=(8,6))
    plt.hist(exemptions_data['tipo'], bins=exemptions_data['tipo'].nunique())
    plt.title("Distribution of Exemption Types")
    plt.xlabel("Exemption Type")
    plt.ylabel("Frequency")
    plt.savefig(ROOT_RESULTS/'figures'/'exemptions'/'exemption_types_distribution.png', dpi=300, bbox_inches="tight")

    #distribution of exemptions per patient
    exemptions_per_patient = exemptions_data.groupby('pedianet_id').size()
    print(f"Mean Exemptions per Patient: {exemptions_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    exemptions_per_patient.value_counts().sort_index().plot(kind='bar')
    plt.xlabel("Number of Exemptions per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Exemptions per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'exemptions'/'exemptions_per_patient.png', dpi=300, bbox_inches="tight")

    #distribution of unique exemption types per patient
    unique_exemption_types_per_patient = exemptions_data.groupby('pedianet_id')['tipo'].nunique()
    print(f"Mean Unique Exemption Types per Patient: {unique_exemption_types_per_patient.mean():.2f}")
    plt.figure(figsize=(8,6))
    unique_exemption_types_per_patient.value_counts().sort_index().plot(kind='bar')
    plt.xlabel("Number of Unique Exemption Types per Patient")
    plt.ylabel("Number of Patients")
    plt.title("Distribution of Number of Unique Exemption Types per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'exemptions'/'unique_exemption_types_per_patient.png', dpi=300, bbox_inches="tight")

    #tables codes exemptions and their counts, percentages, descriptions
    print(f"Unique Exemption Codes: {exemptions_data['codice'].nunique()}, Unique Types: {exemptions_data['codice'].unique()}")
    exemption_codes = pd.DataFrame(exemptions_data['codice'].value_counts())
    exemption_codes.reset_index(inplace=True)
    exemption_codes.columns = ['Exemption Code', 'Count']
    exemption_codes['Percentage'] = exemption_codes['Count']/exemption_codes['Count'].sum()*100
    exemption_codes.to_csv(ROOT_RESULTS/'tables'/'exemptions'/'exemption_codes.csv', index=False)       
    exemption_codes.to_latex(ROOT_RESULTS/'tables'/'exemptions'/'exemption_codes.tex', index=False)

    #save table comorbidities exemptions codes and their descriptions
    comorbidities_exemptions_data.to_csv(ROOT_RESULTS/'tables'/'exemptions'/'comorbidities_exemptions.csv', index=False)

    #check type code in comorbidities exemptions data and in exemptions data, if they are the same
    
    #convert in string the exemption code in comorbidities exemptions data (add 0 in front if the code is less than 3 digits)
    comorbidities_exemptions_data['exemption_code'] = comorbidities_exemptions_data['exemption_code'].astype(str).str.zfill(3)
    print(f"Unique Exemption Codes in Comorbidities Exemptions Data: {comorbidities_exemptions_data['exemption_code'].nunique()}, Unique Types: {comorbidities_exemptions_data['exemption_code'].unique()}")
    print(f"Unique Exemption Codes in Exemptions Data: {exemptions_data['codice'].nunique()}, Unique Types: {exemptions_data['codice'].unique()}")

    #table with exemption that have code in comorbidities exemptions and their counts, percentages, descriptions
    comorbidities_exemptions = exemptions_data[exemptions_data['codice'].isin(comorbidities_exemptions_data['exemption_code'])]

    comorbidities_exemptions_table = pd.DataFrame(comorbidities_exemptions['codice'].value_counts())
    comorbidities_exemptions_table.reset_index(inplace=True)
    comorbidities_exemptions_table.columns = ['Exemption Code', 'Count']    
    #add description to comorbidities exemptions table
    comorbidities_exemptions_table = comorbidities_exemptions_table.merge(
        comorbidities_exemptions_data[['exemption_code', 'disease']],
        left_on='Exemption Code',
        right_on='exemption_code',
        how='left'
    )
    
    comorbidities_exemptions_table['Percentage'] = comorbidities_exemptions_table['Count']/comorbidities_exemptions_table['Count'].sum()*100
    comorbidities_exemptions_table.to_csv(ROOT_RESULTS/'tables'/'exemptions'/'comorbidities_in_exemptions.csv', index=False)
    comorbidities_exemptions_table.to_latex(ROOT_RESULTS/'tables'/'exemptions'/'comorbidities_in_exemptions.tex', index=False)    

    #get info on descrizione codice
    print(f"Unique Exemption Code Descriptions: {exemptions_data['descrizione'].nunique()}, Unique Types: {exemptions_data['descrizione'].unique()}")
    exemption_code_descriptions = pd.DataFrame(exemptions_data['descrizione'].value_counts())
    exemption_code_descriptions.reset_index(inplace=True)
    exemption_code_descriptions.columns = ['Exemption Code Description', 'Count']
    exemption_code_descriptions['Percentage'] = exemption_code_descriptions['Count']/exemption_code_descriptions['Count'].sum()*100
    exemption_code_descriptions.to_csv(ROOT_RESULTS/'tables'/'exemptions'/'exemption_code_descriptions.csv', index=False)       
    exemption_code_descriptions.to_latex(ROOT_RESULTS/'tables'/'exemptions'/'exemption_code_descriptions.tex', index=False)


   #duratation of exemption "patologia" in years
    patologia_exemptions = exemptions_data[exemptions_data['tipo'] == 'Patologia']
    print("Patologia Exemptions: ", patologia_exemptions.shape[0])
    patologia_exemptions['duration_years'] = (patologia_exemptions['scadenza'] - patologia_exemptions['decorrenza']).dt.days/365.25
    print(f"Mean Duration of Patologia Exemptions: {patologia_exemptions['duration_years'].mean():.2f} years")
    plt.figure(figsize=(8,6))
    plt.hist(patologia_exemptions['duration_years'], bins=20)
    plt.xlabel("Duration of Patologia Exemption (years)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Duration of Patologia Exemptions")
    plt.savefig(ROOT_RESULTS/'figures'/'exemptions'/'patologia_exemption_duration.png', dpi=300, bbox_inches="tight")

    #Code description for "patologia" exemption
    patologia_code_descriptions = exemptions_data[exemptions_data['tipo'] == 'Patologia']['descrizione'].unique()
    print(f"Unique Patologia Exemption Code Descriptions: {len(patologia_code_descriptions)}")
    print("Patologia Exemption Code Descriptions:")
    for desc in patologia_code_descriptions:
        print(f"  - {desc}")
    #tables with count and percentages
    patologia_code_table = pd.DataFrame({'Exemption Code Description': patologia_code_descriptions, 'Count': [exemptions_data[(exemptions_data['tipo'] == 'Patologia') & (exemptions_data['descrizione'] == desc)].shape[0] for desc in patologia_code_descriptions]})
    patologia_code_table['Percentage'] = patologia_code_table['Count']/patologia_code_table['Count'].sum()*100
    patologia_code_table.to_csv(ROOT_RESULTS/'tables'/'exemptions'/'patologia_exemption_codes.csv', index=False)
    patologia_code_table.to_latex(ROOT_RESULTS/'tables'/'exemptions'/'patologia_exemption_codes.tex', index=False)

    
    #duration of patologia "invalidità" in years
    invalidita_exemptions = exemptions_data[exemptions_data['tipo'] == 'Invalidità']
    print("Invalidità Exemptions: ", invalidita_exemptions.shape[0])
    invalidita_exemptions['duration_years'] = (invalidita_exemptions['scadenza'] - invalidita_exemptions['decorrenza']).dt.days/365.25
    print(f"Mean Duration of Invalidità Exemptions: {invalidita_exemptions['duration_years'].mean():.2f} years")
    plt.figure(figsize=(8,6))
    plt.hist(invalidita_exemptions['duration_years'], bins=20)
    plt.xlabel("Duration of Invalidità Exemption (years)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Duration of Invalidità Exemptions")
    plt.savefig(ROOT_RESULTS/'figures'/'exemptions'/'invalidita_exemption_duration.png', dpi=300, bbox_inches="tight")

    #code description for "invalidità" exemption
    invalidita_code_descriptions = exemptions_data[exemptions_data['tipo'] == 'Invalidità']['descrizione'].unique()
    print(f"Unique Invalidità Exemption Code Descriptions: {len(invalidita_code_descriptions)}")
    print("Invalidità Exemption Code Descriptions:")
    for desc in invalidita_code_descriptions:
        print(f"  - {desc}")
    #tables with count and percentages
    invalidita_code_table = pd.DataFrame({'Exemption Code Description': invalidita_code_descriptions, 'Count': [exemptions_data[(exemptions_data['tipo'] == 'Invalidità') & (exemptions_data['descrizione'] == desc)].shape[0] for desc in invalidita_code_descriptions]})
    invalidita_code_table['Percentage'] = invalidita_code_table['Count']/invalidita_code_table['Count'].sum()*100
    invalidita_code_table.to_csv(ROOT_RESULTS/'tables'/'exemptions'/'invalidita_exemption_codes.csv', index=False)
    invalidita_code_table.to_latex(ROOT_RESULTS/'tables'/'exemptions'/'invalidita_exemption_codes.tex', index=False)


    #summary table for exemptions
    exemptions_summary = pd.DataFrame(
        {
            "Total Exemptions": [exemptions_data.shape[0]],
            "Unique Exemption Types": [exemptions_data['tipo'].nunique()],
            "Mean Exemptions per Patient": [exemptions_per_patient.mean()],
            "Mean Unique Exemption Types per Patient": [exemptions_data.groupby('pedianet_id')['tipo'].nunique().mean()],
            "Number of patients with Patologia exemptions": [patologia_exemptions['pedianet_id'].nunique()],
            "Number of patients with Invalidità exemptions": [invalidita_exemptions['pedianet_id'].nunique()]
        }
    )
    exemptions_summary.to_csv(ROOT_RESULTS/'tables'/'exemptions'/'exemptions_summary.csv', index=False)
    exemptions_summary.to_latex(ROOT_RESULTS/'tables'/'exemptions'/'exemptions_summary.tex', index=False)


def family_diseases_exploration(family_diseases_data,patients_data):
    #table with column name, type, missing values and percentage of missing values
    family_diseases_info = pd.DataFrame({
        'Column Name': family_diseases_data.columns,
        'Data Type': family_diseases_data.dtypes,
        'Missing Values': family_diseases_data.isnull().sum(),
        'Percentage Missing': family_diseases_data.isnull().mean() * 100
    })  
    family_diseases_info.to_csv(ROOT_RESULTS/'tables'/'family_diseases'/'family_diseases_info.csv', index=False)
    family_diseases_info.to_latex(ROOT_RESULTS/'tables'/'family_diseases'/'family_diseases_info.tex', index=False)
    
    
    print("Family Diseases Exploration:")
    #tables with different family relationships and their counts and percentages
    print(f"Unique Family Relationships: {family_diseases_data['familiare'].nunique()}, Unique Types: {family_diseases_data['familiare'].unique()}")
    family_relationships = pd.DataFrame(family_diseases_data['familiare'].value_counts())
    family_relationships.reset_index(inplace=True)
    family_relationships.columns = ['Family Relationship', 'Count']
    family_relationships['Percentage'] = family_relationships['Count']/family_relationships['Count'].sum()*100
    family_relationships.to_csv(ROOT_RESULTS/'tables'/'family_diseases'/'family_relationships.csv', index=False)
    family_relationships.to_latex(ROOT_RESULTS/'tables'/'family_diseases'/'family_relationships.tex', index=False) 

    #plot the distribution 
    plt.figure(figsize=(8,6))
    plt.hist(family_diseases_data['familiare'], bins=family_diseases_data['familiare'].nunique())
    plt.title("Distribution of Family Relationships")
    plt.xlabel("Family Relationship")
    plt.ylabel("Frequency") 
    plt.xticks(rotation=90, ha='right')  
    plt.tight_layout()  
    plt.savefig(ROOT_RESULTS/'figures'/'family_diseases'/'family_relationships_distribution.png', dpi=300, bbox_inches="tight")

    


    #tables with different diseases and their counts and percentages
    print(f"Unique Diseases: {family_diseases_data['descrizione'].nunique()}, Unique Types: {family_diseases_data['descrizione'].unique()}")
    family_disease_types = pd.DataFrame(family_diseases_data['descrizione'].value_counts())
    family_disease_types.reset_index(inplace=True)
    family_disease_types.columns = ['Family Disease Type', 'Count']
    family_disease_types['Percentage'] = family_disease_types['Count']/family_disease_types['Count'].sum()*100
    family_disease_types.to_csv(ROOT_RESULTS/'tables'/'family_diseases'/'family_disease_types.csv', index=False)
    family_disease_types.to_latex(ROOT_RESULTS/'tables'/'family_diseases'/'family_disease_types.tex', index=False) 

    #tables with the 20 most frequent family diseases
    print(f"Top 20 Most Frequent Family Diseases:")
    top_20_diseases = family_disease_types.head(20)
    top_20_diseases.to_csv(ROOT_RESULTS/'tables'/'family_diseases'/'top_20_diseases.csv', index=False)
    top_20_diseases.to_latex(ROOT_RESULTS/'tables'/'family_diseases'/'top_20_diseases.tex', index=False)

    #distribution of family diseases per patient
    family_diseases_per_patient = family_diseases_data.groupby('pedianet_id').size()
    plt.figure(figsize=(8,6))
    family_diseases_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Family Diseases per Patient")
    plt.ylabel("Frequency")
    plt.title("Distribution of Family Diseases per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'family_diseases'/'family_diseases_per_patient.png', dpi=300, bbox_inches="tight")

    #number of patient with more than 1 family disease
    num_patients_with_multiple_diseases = family_diseases_per_patient[family_diseases_per_patient > 1].shape[0]
    print(f"Number of Patients with Multiple Family Diseases: {num_patients_with_multiple_diseases}")
    #summary table for family diseases

    #distribution of unique family diseases per patient
    plt.figure(figsize=(8,6))
    unique_family_diseases_per_patient = family_diseases_data.groupby('pedianet_id')['descrizione'].nunique()
    unique_family_diseases_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Unique Family Diseases per Patient")
    plt.ylabel("Frequency")
    plt.title("Distribution of Unique Family Diseases per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'family_diseases'/'unique_family_diseases_per_patient.png', dpi=300, bbox_inches="tight")

    #patients with multiple same family disease
    multiple_same_diseases = family_diseases_data.groupby(['pedianet_id', 'descrizione']).size()
    multiple_same_diseases = multiple_same_diseases[multiple_same_diseases > 1]
    print(f"Patients with Multiple Same Family Diseases: {multiple_same_diseases.shape[0]}")


    family_diseases_summary = pd.DataFrame(
        {
            "Total Family Diseases": [family_diseases_data.shape[0]],
            "Patients with at least one family disease": [family_diseases_data['pedianet_id'].nunique()],
            "Patients with more than 1 family disease": [num_patients_with_multiple_diseases],
            "Patients without family diseases": [patients_data['pedianet_id'].nunique() - family_diseases_data['pedianet_id'].nunique()],
            "Unique Family Relationships": [family_diseases_data['familiare'].nunique()],
            "Unique Disease Types": [family_diseases_data['descrizione'].nunique()],
            "Mean family disease per patient": [family_diseases_data.groupby('pedianet_id').size().mean()],
            "Number of Patients with Multiple Family Diseases": [num_patients_with_multiple_diseases],
            "Number of Patients with Multiple Same Family Diseases": [multiple_same_diseases.shape[0]]
        }
    )
    family_diseases_summary.to_csv(ROOT_RESULTS/'tables'/'family_diseases'/'family_diseases_summary.csv', index=False)
    family_diseases_summary.to_latex(ROOT_RESULTS/'tables'/'family_diseases'/'family_diseases_summary.tex', index=False)


def fse_exploration (fse_data,patients_data):
    #transform datarecord to datetime
    fse_data['datarecord'] = pd.to_datetime(fse_data['datarecord'], errors='coerce')
    #add age at record
    fse_data = fse_data.merge(
    patients_data[['pedianet_id', 'datan']], 
    on='pedianet_id',   
    how='left'
    )
    fse_data['age_at_record'] = (fse_data['datarecord'] - fse_data['datan']).dt.days/365.25


    print("fse exploration")
    #table with column name, type, missing values and percentage of missing values
    fse_info = pd.DataFrame({
        'Column Name': fse_data.columns,
        'Data Type': fse_data.dtypes,
        'Missing Values': fse_data.isnull().sum(),
        'Percentage Missing': fse_data.isnull().mean() * 100
    })
    fse_info.to_csv(ROOT_RESULTS/'tables'/'fse'/'fse_info.csv', index=False)
    fse_info.to_latex(ROOT_RESULTS/'tables'/'fse'/'fse_info.tex', index=False)
       

    # Number of records
    print(f"Total Records in FSE Data: {fse_data.shape[0]}")
    #Number of patients 
    print(f"Total Patients in FSE Data: {fse_data['pedianet_id'].nunique()}")
    #Patients without any record
    print(f"Patients without any FSE Record: {patients_data['pedianet_id'].nunique() - fse_data['pedianet_id'].nunique()}")

    #distribution of records per patient
    records_per_patient = fse_data.groupby('pedianet_id').size()
    plt.figure(figsize=(8,6))
    records_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of Records per patient")
    plt.ylabel("Frequency")
    plt.title("Distribution of Records per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'fse'/'records_per_patient.png', dpi=300, bbox_inches="tight")

    #distribution of record over time
    plt.figure(figsize=(8,6))
    plt.hist(fse_data['datarecord'], bins=50)
    plt.xlabel("Month")
    plt.ylabel("Number of Records")
    plt.title("Distribution of Records over Time")
    plt.savefig(ROOT_RESULTS/'figures'/'fse'/'records_over_time.png', dpi=300, bbox_inches="tight")

    #distribution of record over age
    plt.figure(figsize=(8,6))
    plt.hist(fse_data['age_at_record'], bins=50)
    plt.xlabel("Age at Record")
    plt.ylabel("Number of Records")
    plt.title("Distribution of Records over Age")
    plt.savefig(ROOT_RESULTS/'figures'/'fse'/'records_over_age.png', dpi=300, bbox_inches="tight")

    #info about "tipo" document in fse
    tipo_document_info = fse_data['tipo'].value_counts()
    print(f"Tipo Document Counts:")
    for tipo, count in tipo_document_info.items():
        print(f"  - {tipo}: {count}")
    #distribution of tipo document 
    plt.figure(figsize=(8,6))
    tipo_document_info.plot(kind='bar')
    plt.xlabel("Tipo Document")
    plt.ylabel("Frequency")
    plt.title("Distribution of Tipo Documents")
    plt.savefig(ROOT_RESULTS/'figures'/'fse'/'tipo_documents.png', dpi=300, bbox_inches="tight")

    #table with tipo documento count and percentages
    tipo_document_table = pd.DataFrame(tipo_document_info)
    tipo_document_table.reset_index(inplace=True)
    tipo_document_table.columns = ['Tipo Document', 'Count']
    tipo_document_table['Percentage'] = tipo_document_table['Count']/tipo_document_table['Count'].sum()*100
    tipo_document_table.to_csv(ROOT_RESULTS/'tables'/'fse'/'tipo_documents.csv', index=False)
    tipo_document_table.to_latex(ROOT_RESULTS/'tables'/'fse'/'tipo_documents.tex', index=False)  

    #distribution of unique tipo documents per patient
    unique_tipo_documents_per_patient = fse_data.groupby('pedianet_id')['tipo'].nunique()
    plt.figure(figsize=(8,6))
    unique_tipo_documents_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of unique tipo")
    plt.ylabel("Frequency")
    plt.title("Distribution of Unique Tipo Documents per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'fse'/'unique_tipo_documents_per_patient.png', dpi=300, bbox_inches="tight")

    #number of patient in emergency room 
    fse_PS=fse_data[fse_data['tipo'] == 'PS']
    emergency_room=fse_data[fse_data['tipo'] == 'PS'].shape[0]
    emergency_room_patients = fse_data[fse_data['tipo'] == 'PS']['pedianet_id'].nunique()

    #distribution of ER per patient
    er_per_patient = fse_PS.groupby('pedianet_id').size()
    plt.figure(figsize=(8,6))
    er_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of PS Records per patient")
    plt.ylabel("Frequency")
    plt.title("Distribution of PS Records per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'fse'/'er_per_patient.png', dpi=300, bbox_inches="tight")

    #distribution of PS over age
    plt.figure(figsize=(8,6))
    plt.hist(fse_PS['age_at_record'], bins=50)
    plt.xlabel("Age at Record")
    plt.ylabel("Number of Records")
    plt.title("Distribution of PS Records over Age")
    plt.savefig(ROOT_RESULTS/'figures'/'fse'/'er_over_age.png', dpi=300, bbox_inches="tight")

    #number of patients in hospital
    fse_RIC=fse_data[fse_data['tipo'] == 'RIC']
    hospital=fse_data[fse_data['tipo'] == 'RIC'].shape[0]
    hospital_patients = fse_data[fse_data['tipo'] == 'RIC']['pedianet_id'].nunique()


    #distribution of RIC per patient
    ric_per_patient = fse_RIC.groupby('pedianet_id').size()
    plt.figure(figsize=(8,6))
    ric_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of RIC Records")
    plt.ylabel("Frequency")
    plt.title("Distribution of RIC Records per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'fse'/'ric_per_patient.png', dpi=300, bbox_inches="tight")

    #distribution of RIC over age
    plt.figure(figsize=(8,6))
    plt.hist(fse_RIC['age_at_record'], bins=50)
    plt.xlabel("Age at Record")
    plt.ylabel("Number of Records")
    plt.title("Distribution of RIC Records over Age")
    plt.savefig(ROOT_RESULTS/'figures'/'fse'/'ric_over_age.png', dpi=300, bbox_inches="tight")


    #number of lab test
    fse_LAB=fse_data[fse_data['tipo'] == 'LAB']
    lab_tests=fse_data[fse_data['tipo'] == 'LAB'].shape[0]
    lab_tests_patients = fse_data[fse_data['tipo'] == 'LAB']['pedianet_id'].nunique()

    #distribution of LAB per patient
    lab_per_patient = fse_LAB.groupby('pedianet_id').size()
    plt.figure(figsize=(8,6))
    lab_per_patient.value_counts().plot(kind='bar')
    plt.xlabel("Number of LAB Records")
    plt.ylabel("Frequency")
    plt.title("Distribution of LAB Records per Patient")
    plt.savefig(ROOT_RESULTS/'figures'/'fse'/'lab_per_patient.png', dpi=300, bbox_inches="tight")

    #distribution of LAB over age
    plt.figure(figsize=(8,6))
    plt.hist(fse_LAB['age_at_record'], bins=50)
    plt.xlabel("Age at Record")
    plt.ylabel("Number of Records")
    plt.title("Distribution of LAB Records over Age")
    plt.savefig(ROOT_RESULTS/'figures'/'fse'/'lab_over_age.png', dpi=300 , bbox_inches="tight")



    #number of different "testo" values
    print(f"Number of Different 'testo' Values: {fse_data['testo'].nunique()}")

    #to give an idea of testo, print  and save first
    print(f"First testo values: {fse_data['testo'].head(10).tolist()}") 
    fse_data['testo'].head(1).to_csv(ROOT_RESULTS/'tables'/'fse'/'testo_samples.csv', index=False)
    fse_data['testo'].head(1).to_latex(ROOT_RESULTS/'tables'/'fse'/'testo_samples.tex', index=False)
    


    #summary table
    summary_table = pd.DataFrame({
        "Number of Records": [fse_data.shape[0]],
        "Number of Patients": [fse_data['pedianet_id'].nunique()],
        "Number of Unique Tipo Documents": [fse_data['tipo'].nunique()],
        "Number of Unique 'testo' Values": [fse_data['testo'].nunique()],
        "NUmber of patient in ER (PS)": [emergency_room_patients],
        "Number of patient in hospital (RIC)": [hospital_patients],
        "Number of patient with lab tests (LAB)": [lab_tests_patients],
        "Mean ER Records per Patient": [er_per_patient.mean()],
        "Mean RIC Records per Patient": [ric_per_patient.mean()],
        "Mean LAB Records per Patient": [lab_per_patient.mean()],
        "Mean Records per Patient": [fse_data.groupby('pedianet_id').size().mean()],
        "Mean unique Tipo Documents per Patient": [fse_data.groupby('pedianet_id')['tipo'].nunique().mean()]

    })
    summary_table.to_csv(ROOT_RESULTS/'tables'/'fse'/'fse_summary.csv', index=False)
    summary_table.to_latex(ROOT_RESULTS/'tables'/'fse'/'fse_summary.tex', index=False)

