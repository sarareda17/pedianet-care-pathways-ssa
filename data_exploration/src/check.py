#compare diagnosi_principali and contatti 
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

#upload data
df_main_diagnoses=pd.read_excel(ROOT/'data'/'main_diagnoses.xlsx')
df_selected_contacts=pd.read_csv(ROOT/'SSA'/'datasets'/'diagnoses_breath.csv')

df_selected_contacts['datacontatto'] = pd.to_datetime(df_selected_contacts['datacontatto'])

#types of main diagnoses 
types_diag=df_main_diagnoses['tipo'].unique()

#check if there is a corrispondence between main_diagnoses and selected contacts
for t in types_diag:
    #for each type of diagnosis, print diagnosi1 value_counts of corrispondent contacts (having same pediane_id and date)
    print(f"Type of diagnosis: {t}")
    diag_subset=df_main_diagnoses[df_main_diagnoses['tipo']==t]
    count=0
    diag=[]
    icd=[]
    for _,row in diag_subset.iterrows():
        ped_id=row['pedianet_id']
        date=row['data']
        #check if there is a contact with same ped_id and date
        contact_subset=df_selected_contacts[(df_selected_contacts['pedianet_id']==ped_id) & (df_selected_contacts['datacontatto']==date)]
        if contact_subset.empty:
            count+=1
        else:
            diag.append(contact_subset['diagnosi1'].values[0])
            icd.append(contact_subset['diagnosi1_icd9'].values[0])
    
    print(f"Number of main diagnoses without a corresponding contact: {count/len(diag_subset):.2%}")
    print(f"Diagnoses in corresponding contacts: {pd.Series(diag).value_counts()}")
    print(f"ICD codes in corresponding contacts: {pd.Series(icd).value_counts()}")




