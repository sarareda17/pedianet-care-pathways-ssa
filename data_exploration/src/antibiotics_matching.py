import pandas as pd
import numpy as np
from pathlib import Path
ROOT= Path(__file__).resolve().parent.parent.parent

#import dataset farmaci
farmaci=pd.read_excel(ROOT/'data'/'farmaci.xlsx')
#import dataset diagnosis preprocessed
diagnoses=pd.read_csv(ROOT/'SSA'/'datasets'/'diagnoses_breath.csv')
patients=pd.read_excel(ROOT/'data'/'pazienti.xlsx')

#select respiratory drugs with atc code starting with R
respiratory_drugs=farmaci[farmaci['atc'].str.startswith('R')]

#select antibiotics with atc code starting with J01
antibiotics=farmaci[farmaci['atc'].str.startswith('J01')]
#select only antibiotics with a diagnosis of respiratory infection within 7 days
def match_antibiotics_diagnoses(antibiotics,diagnoses,dist=7):
    antibiotics_matching=antibiotics.copy()
    indexes=[]
    for index, record in antibiotics.iterrows():
        pedianet_id = record['pedianet_id']
        date_record= record['dataricetta']
        #check if the patient has a diagnosis which is closer than 7 days to the date of the antibiotic record
        diagnoses_patient = diagnoses[diagnoses['pedianet_id'] == pedianet_id]
        diagnoses_patient['date_diff'] = (pd.to_datetime(diagnoses_patient['datacontatto']) - pd.to_datetime(date_record)).dt.days.abs()
        closest_diagnosis = diagnoses_patient[diagnoses_patient['date_diff'] <= dist]
        #if closest_diagnosis is empty, drop the record from antibiotics_matching
        if closest_diagnosis.empty:
            indexes.append(index)
    antibiotics_matching=antibiotics_matching.drop(indexes)
    return antibiotics_matching

antibiotics_matching_7=match_antibiotics_diagnoses(antibiotics,diagnoses,dist=7)
print(f"Number of antibiotics records after filtering: {antibiotics_matching_7.shape[0]} using 7-day window and percentage: {antibiotics_matching_7.shape[0]/antibiotics.shape[0]*100:.2f}%")

#build new dataset
farmaci_selected=pd.concat([respiratory_drugs, antibiotics_matching_7])
#save dataset
farmaci_selected.to_csv(ROOT/'data'/'farmaci_selected.csv', index=False)
