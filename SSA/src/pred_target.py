#explore data to find the target to predict 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
ROOT= Path(__file__).parent.parent.parent
PRED_FOLDER=ROOT/'SSA'/'prediction'


patients_sequences=pd.read_csv(ROOT/'SSA'/'datasets'/'selected_patients_diagnoses_season_3.csv')
print(f"Total number of patients:{len(patients_sequences)}")

### count patient that have at least 1 more year of data after the end of the sequence
patients_pred=patients_sequences[(pd.to_datetime(patients_sequences['EF'])-pd.to_datetime(patients_sequences['end_date'])).dt.days>365]
print(f"Number of patients that have 1 more year of data after the end of the sequence: {len(patients_pred)} which is {len(patients_pred)/len(patients_sequences)*100:.2f}% of the total patients")
patients_pred.to_csv(PRED_FOLDER/'selected_patients.csv', index=False)
### count patient that have at least 2 more years of data after the end of the sequence
patients_pred_2=patients_sequences[(pd.to_datetime(patients_sequences['EF'])-pd.to_datetime(patients_sequences['end_date'])).dt.days>730]
print(f"Number of patients that have 2 more years of data after the end of the sequence: {len(patients_pred_2)} which is {len(patients_pred_2)/len(patients_sequences)*100:.2f}% of the total patients")


############### look for possible target to predict #######################
diag_breath=pd.read_csv(ROOT/'SSA'/'datasets'/'diagnoses_season_breath.csv')
macrodiag_breath=pd.read_csv(ROOT/'SSA'/'datasets'/'macrodiagnoses_season_breath.csv')
drugs_breath=pd.read_csv(ROOT/'SSA'/'datasets'/'drugs_season_breath.csv')
care_setting_breath=pd.read_csv(ROOT/'SSA'/'datasets'/'care_setting_season_breath.csv')


#count how many of the selected patients have: 
### asma in the follow-up period 
asma_icd_codes=('493')
#select rows correponding to asma
diag_asma=diag_breath[diag_breath['diagnosi1_icd9'].astype(str).str.startswith(asma_icd_codes)]
#for each selected patient, chack if the patient has a asma diagnosis in the year after the end date of the sequence
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_diag_asma=diag_asma[diag_asma['pedianet_id']==patient['pedianet_id']]
    diag_asma_after_end_date=patient_diag_asma[(pd.to_datetime(patient_diag_asma['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_asma['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    if len(diag_asma_after_end_date)>0:
        count_patient+=1
print(f"Number of patients that have asma diagnosis in the year after the end date of the sequence: {count_patient} which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")


### wheezing and asma in the follow-up period
asma_wheezing_icd_codes=('493', '786.07')
#select rows correponding to asma
diag_asma_wheezing=diag_breath[diag_breath['diagnosi1_icd9'].astype(str).str.startswith(asma_wheezing_icd_codes)]
#for each selected patient, chack if the patient has a asma diagnosis after the end date of the sequence
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_diag_asma_wheezing=diag_asma_wheezing[diag_asma_wheezing['pedianet_id']==patient['pedianet_id']]
    diag_asma_wheezing_after_end_date=patient_diag_asma_wheezing[(pd.to_datetime(patient_diag_asma_wheezing['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_asma_wheezing['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    if len(diag_asma_wheezing_after_end_date)>0:
        count_patient+=1
print(f"Number of patients that have asma or wheezing diagnosis in the year after the end date of the sequence: {count_patient} which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")

### wheezing and asma in  the year after the end of the sequence sepated by at least 90 days
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_diag_asma=diag_asma_wheezing[diag_asma_wheezing['pedianet_id']==patient['pedianet_id']]
    diag_asma_after_end_date=patient_diag_asma[(pd.to_datetime(patient_diag_asma['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_asma['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    diag_asma_after_end_date=diag_asma_after_end_date.sort_values(by='datacontatto')
    if len(diag_asma_after_end_date)>1:
        #check if there are two diagnosis separated by at least 90 days
        for i in range(len(diag_asma_after_end_date)-1):
            date1=pd.to_datetime(diag_asma_after_end_date.iloc[i]['datacontatto'])
            date2=pd.to_datetime(diag_asma_after_end_date.iloc[i+1]['datacontatto'])
            if (date2-date1).days>=90:
                count_patient+=1
                break
print(f"Number of patients that have two asma or wheezing diagnosis separated by at least 90 days in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")

#patients that have 1 macrodiagnosis in the follow up period of 1 year after the end of the sequence
macro=['PAT_OSTR']
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_diag_breath=macrodiag_breath[macrodiag_breath['pedianet_id']==patient['pedianet_id']]
    diag_breath_after_end_date=patient_diag_breath[(pd.to_datetime(patient_diag_breath['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_breath['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    if len(diag_breath_after_end_date[diag_breath_after_end_date['state'].isin(macro)])>0:
        count_patient+=1
print(f"Number of patients that have at least 1 diagnosis of the states {macro} in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")

#patients that have 2 macrodiagnosis in the follow up period of 1 year after the end of the sequence
macro=['PAT_OSTR']
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_diag_breath=macrodiag_breath[macrodiag_breath['pedianet_id']==patient['pedianet_id']]
    diag_breath_after_end_date=patient_diag_breath[(pd.to_datetime(patient_diag_breath['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_breath['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    if len(diag_breath_after_end_date[diag_breath_after_end_date['state'].isin(macro)])>1:
        count_patient+=1
print(f"Number of patients that have at least 2 diagnosis of the states {macro} in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")


#at least 2 macrodiagnoses in the follow up period of 1 year after the end of the 
#sequence and separated by at least 90 days
macro=['PAT_OSTR']
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_diag_breath=macrodiag_breath[macrodiag_breath['pedianet_id']==patient['pedianet_id']]
    diag_breath_after_end_date=patient_diag_breath[(pd.to_datetime(patient_diag_breath['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_breath['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    diag_breath_after_end_date=diag_breath_after_end_date.sort_values(by='datacontatto')
    #check if there are at least 2 diagnosis of the states in the follow up period separated by at least 90 days
    diag_breath_after_end_date_states=diag_breath_after_end_date[diag_breath_after_end_date['state'].isin(macro)]
    if len(diag_breath_after_end_date_states)>1:
        for i in range(len(diag_breath_after_end_date_states)-1):
            date1=pd.to_datetime(diag_breath_after_end_date_states.iloc[i]['datacontatto'])
            date2=pd.to_datetime(diag_breath_after_end_date_states.iloc[i+1]['datacontatto'])
            if (date2-date1).days>=90:
                count_patient+=1
                break
print(f"Number of patients that have at least 2 diagnosis of the states {macro} in the year after the end date of the sequence and separated by at least 90 days: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")



#at least 3 of macrodiagnoses in the follow up period of 1 year after the end of the sequence
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_diag_breath=macrodiag_breath[macrodiag_breath['pedianet_id']==patient['pedianet_id']]
    diag_breath_after_end_date=patient_diag_breath[(pd.to_datetime(patient_diag_breath['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_breath['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    diag_breath_after_end_date=diag_breath_after_end_date.sort_values(by='datacontatto')
    #check if there are at least 3 diagnosis of the states in the follow up period
    if len(diag_breath_after_end_date[diag_breath_after_end_date['state'].isin(macro)])>=3:
        count_patient+=1
print(f"Number of patients that have at least 3 diagnosis of the states {macro} in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data") 

#patients that have at least 2 prescription of R03 atc drugs in the follow up period of 1 year after the end of the sequence
target_patients_obs_drugs=pd.DataFrame(columns=[ 'target'],index=patients_pred['pedianet_id'])
target_patients_obs_drugs['target']=0
atc_r03_codes=('R03')
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_drugs_breath=drugs_breath[drugs_breath['pedianet_id']==patient['pedianet_id']]
    drugs_breath_after_end_date=patient_drugs_breath[(pd.to_datetime(patient_drugs_breath['dataricetta'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_drugs_breath['dataricetta'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    #check if there are at least 2 prescription of R03 atc drugs in the follow up period
    if len(drugs_breath_after_end_date[drugs_breath_after_end_date['atc'].astype(str).str.startswith(atc_r03_codes)])>=2:
        count_patient+=1
        target_patients_obs_drugs.loc[patient['pedianet_id'],'target']=1

print(f"Number of patients that have at least 2 prescription of R03 atc drugs in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")
target_patients_obs_drugs.to_csv(PRED_FOLDER/'target'/'target_obs_drugs.csv', index=True)


#patients with at least 1 macrodiagnoses in the follow up period of 1 year after the end of the sequence
target_patients=pd.DataFrame(columns=[ 'target','months'],index=patients_pred['pedianet_id'])
target_patients['target']=0
macro=['PAT_OSTR','INF_INF']
count_patient=0
for idx, patient in patients_pred.iterrows():   
    patient_diag_breath=macrodiag_breath[macrodiag_breath['pedianet_id']==patient['pedianet_id']]
    diag_breath_after_end_date=patient_diag_breath[(pd.to_datetime(patient_diag_breath['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_breath['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    diag_breath_after_end_date=diag_breath_after_end_date.sort_values(by='datacontatto')
    #check if there are at least 2 diagnosis of the states in the follow up period
    diag_macro=diag_breath_after_end_date[diag_breath_after_end_date['state'].isin(macro)]
    first_diag_date=diag_macro['datacontatto'].min()
    if len(diag_macro)>=1:
        count_patient+=1
        target_patients.loc[patient['pedianet_id'],'target']=1  
        target_patients.loc[patient['pedianet_id'],'months']= round((pd.to_datetime(first_diag_date)-pd.to_datetime(patient['end_date'])).days/30,1)
print(f"Number of patients that have at least 1 diagnosis of the states {macro} in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")
target_patients.to_csv(PRED_FOLDER/'target'/'target_lrti_obs.csv', index=True)




#at least 2 macrodiagnoses in the follow up period of 1 year after the end of the 
#sequence and separated by at least 90 days

macro=['PAT_OSTR','INF_INF']
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_diag_breath=macrodiag_breath[macrodiag_breath['pedianet_id']==patient['pedianet_id']]
    diag_breath_after_end_date=patient_diag_breath[(pd.to_datetime(patient_diag_breath['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_breath['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    diag_breath_after_end_date=diag_breath_after_end_date.sort_values(by='datacontatto')
    #check if there are at least 2 diagnosis of the states in the follow up period separated by at least 90 days
    diag_breath_after_end_date_states=diag_breath_after_end_date[diag_breath_after_end_date['state'].isin(macro)]
    if len(diag_breath_after_end_date_states)>1:
        for i in range(len(diag_breath_after_end_date_states)-1):
            date1=pd.to_datetime(diag_breath_after_end_date_states.iloc[i]['datacontatto'])
            date2=pd.to_datetime(diag_breath_after_end_date_states.iloc[i+1]['datacontatto'])
            if (date2-date1).days>=90:
                count_patient+=1          
                break
print(f"Number of patients that have at least 2 diagnosis of the states {macro} in the year after the end date of the sequence and separated by at least 90 days: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")


#at least 2 macrodiagnoses in the follow up period of 1 year after the end of the 
#sequence 
count_patient=0
for idx, patient in patients_pred.iterrows():   
    patient_diag_breath=macrodiag_breath[macrodiag_breath['pedianet_id']==patient['pedianet_id']]
    diag_breath_after_end_date=patient_diag_breath[(pd.to_datetime(patient_diag_breath['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_breath['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    diag_breath_after_end_date=diag_breath_after_end_date.sort_values(by='datacontatto')
    #check if there are at least 2 diagnosis of the states in the follow up period
    if len(diag_breath_after_end_date[diag_breath_after_end_date['state'].isin(macro)])>=2:
        count_patient+=1
print(f"Number of patients that have at least 2 diagnosis of the states {macro} in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")

#patients that have at least 1 INF_INF macrodiagnoses in the follow up period of 1 year after the end of the sequence
macro=['INF_INF']
count_patient=0
for idx, patient in patients_pred.iterrows():   
    patient_diag_breath=macrodiag_breath[macrodiag_breath['pedianet_id']==patient['pedianet_id']]
    diag_breath_after_end_date=patient_diag_breath[(pd.to_datetime(patient_diag_breath['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_breath['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    diag_breath_after_end_date=diag_breath_after_end_date.sort_values(by='datacontatto')
    #check if there are at least 1 diagnosis of the states in the follow up period
    if len(diag_breath_after_end_date[diag_breath_after_end_date['state'].isin(macro)])>=1:
        count_patient+=1
print(f"Number of patients that have at least 1 diagnosis of the states {macro} in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")



###### using the care setting information to define the target:

#patients that have at least 1 hospitalization in the year after the end of the sequence
care_setting_hosp=['RIC']
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_care_setting_breath=care_setting_breath[care_setting_breath['pedianet_id']==patient['pedianet_id']]
    care_setting_breath_after_end_date=patient_care_setting_breath[(pd.to_datetime(patient_care_setting_breath['date'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_care_setting_breath['date'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    care_setting_breath_after_end_date=care_setting_breath_after_end_date.sort_values(by='date')
    #check if there are at least 1 hospitalization in the follow up period
    if len(care_setting_breath_after_end_date[care_setting_breath_after_end_date['care_setting'].isin(care_setting_hosp)])>=1:
        count_patient+=1
print(f"Number of patients that have at least 1 hospitalization in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")

#patients that have at least 1 ER access in the year after the end of the sequence
care_setting_er=['PS']
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_care_setting_breath=care_setting_breath[care_setting_breath['pedianet_id']==patient['pedianet_id']]
    care_setting_breath_after_end_date=patient_care_setting_breath[(pd.to_datetime(patient_care_setting_breath['date'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_care_setting_breath['date'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    care_setting_breath_after_end_date=care_setting_breath_after_end_date.sort_values(by='date')
    #check if there are at least 1 ER access in the follow up period
    if len(care_setting_breath_after_end_date[care_setting_breath_after_end_date['care_setting'].isin(care_setting_er)])>=1:
        count_patient+=1
print(f"Number of patients that have at least 1 ER access in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")

#patients that have at least 1 VS access in the year after the end of the sequence
care_setting_vs=['VS']
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_care_setting_breath=care_setting_breath[care_setting_breath['pedianet_id']==patient['pedianet_id']]
    care_setting_breath_after_end_date=patient_care_setting_breath[(pd.to_datetime(patient_care_setting_breath['date'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_care_setting_breath['date'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    care_setting_breath_after_end_date=care_setting_breath_after_end_date.sort_values(by='date')
    #check if there are at least 1 VS access in the follow up period
    if len(care_setting_breath_after_end_date[care_setting_breath_after_end_date['care_setting'].isin(care_setting_vs)])>=1:
        count_patient+=1
print(f"Number of patients that have at least 1 VS access in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")

"""

######### using all the breath diagnoses 
#patients that have at least 1 diagnosis in the year after the end of the sequence
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_diag_breath=diag_breath[diag_breath['pedianet_id']==patient['pedianet_id']]
    diag_breath_after_end_date=patient_diag_breath[(pd.to_datetime(patient_diag_breath['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_breath['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    diag_breath_after_end_date=diag_breath_after_end_date.sort_values(by='datacontatto')
    #check if there are at least 1 diagnosis in the follow up period
    if len(diag_breath_after_end_date)>=1:
        count_patient+=1
print(f"Number of patients that have at least 1 diagnosis in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")


#patients that have at least 2 diagnosis in the year after the end of the sequence
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_diag_breath=diag_breath[diag_breath['pedianet_id']==patient['pedianet_id']]
    diag_breath_after_end_date=patient_diag_breath[(pd.to_datetime(patient_diag_breath['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_breath['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    diag_breath_after_end_date=diag_breath_after_end_date.sort_values(by='datacontatto')
    #check if there are at least 2 diagnosis in the follow up period
    if len(diag_breath_after_end_date)>=2:
        count_patient+=1
print(f"Number of patients that have at least 2 diagnosis in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")    

#patients that have at least 3 diagnosis in the year after the end of the sequence
count_patient=0
for idx, patient in patients_pred.iterrows():
    patient_diag_breath=diag_breath[diag_breath['pedianet_id']==patient['pedianet_id']]
    diag_breath_after_end_date=patient_diag_breath[(pd.to_datetime(patient_diag_breath['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_breath['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
    diag_breath_after_end_date=diag_breath_after_end_date.sort_values(by='datacontatto')
    #check if there are at least 3 diagnosis in the follow up period
    if len(diag_breath_after_end_date)>=3:
        count_patient+=1
print(f"Number of patients that have at least 3 diagnosis in the year after the end date of the sequence: {count_patient}, which is {count_patient/len(patients_pred)*100:.2f}% of the patients with at least 1 year of follow up data")



"""
