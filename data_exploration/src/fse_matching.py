import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
ROOT= Path(__file__).resolve().parent.parent.parent

#import dataset fse
fse=pd.read_csv(ROOT/'data'/'fse.csv')
#import dataset diagnosis preprocessed
diagnoses=pd.read_csv(ROOT/'SSA'/'datasets'/'diagnoses_breath.csv')
patients=pd.read_excel(ROOT/'data'/'pazienti.xlsx')

#select patients in the fse dataset
fse_matching=fse[fse['pedianet_id'].isin(patients['pedianet_id'])]

#add age (in days) at datarecord 
fse_matching=pd.merge(fse_matching, patients[['pedianet_id', 'datan']], on='pedianet_id', how='left')
fse_matching['age_at_datarecord']=(pd.to_datetime(fse_matching['datarecord']) - pd.to_datetime(fse_matching['datan'])).dt.days

#select only tipo RIC (ricovero) o PS (pronto soccorso):
fse_matching=fse_matching[fse_matching['tipo'].isin(['RIC','PS'])]

print(f"Total number of fse in Pedianet: {fse.shape[0]}")
print(f"Number of fse records before filtering: {fse_matching.shape[0]}")

#function to match fse records with diagnoses within a certain time distance (e.g. 7 days)
def match_fse_diagnoses(fse_matching,diagnoses,dist=7):
    fse_breath=fse_matching.copy()
    indexes=[]
    for index, record in fse_matching.iterrows():
        pedianet_id = record['pedianet_id']
        date_record= record['datarecord']
        #check if the patient has a diagnosis which is closer than 7 days to the date of the fse record
        diagnoses_patient = diagnoses[diagnoses['pedianet_id'] == pedianet_id]
        diagnoses_patient['date_diff'] = (pd.to_datetime(diagnoses_patient['datacontatto']) - pd.to_datetime(date_record)).dt.days.abs()
        closest_diagnosis = diagnoses_patient[diagnoses_patient['date_diff'] <= dist]
        #if closest_diagnosis is empty, drop the record from fse_matching
        if closest_diagnosis.empty:
            indexes.append(index)
    fse_breath=fse_breath.drop(indexes)
    return fse_breath

#match fse records with diagnoses within 7 days
fse_breath_7=match_fse_diagnoses(fse_matching,diagnoses,dist=7)
print(f"Number of fse records after filtering: {fse_breath_7.shape[0]} using 7-day window and percentage: {fse_breath_7.shape[0]/fse_matching.shape[0]*100:.2f}%")

fse_breath_14=match_fse_diagnoses(fse_matching,diagnoses,dist=14)
print(f"Number of fse records after filtering: {fse_breath_14.shape[0]} using 14-day window and percentage: {fse_breath_14.shape[0]/fse_matching.shape[0]*100:.2f}%")

#save the matched fse records to csv
fse_breath_7.to_csv(ROOT/'data'/'fse_matched_7.csv', index=False)
fse_breath_14.to_csv(ROOT/'data'/'fse_matched_14.csv', index=False)
