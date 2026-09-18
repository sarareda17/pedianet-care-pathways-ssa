import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
from config import YEARS

data=pd.read_excel(ROOT/'data'/'Misure_BMIzscore.xlsx')
patients_with_BMI=data['pedianet_id'].unique()
print(f'The number of patients with BMI measurements is: {len(patients_with_BMI)}')
print(data.head())
print(data.shape)


patients_sequences=pd.read_csv(ROOT/'SSA'/'datasets'/f'selected_patients_diagnoses.csv')
patients_sequences['end_date']=pd.to_datetime(patients_sequences['datan'])+ pd.DateOffset(years=YEARS)
patients_pred=patients_sequences[(pd.to_datetime(patients_sequences['EF'])-pd.to_datetime(patients_sequences['end_date'])).dt.days>365]

#count how many patients with BMI measurements are in prediction dataset
patients_pred_with_BMI=patients_pred[patients_pred['pedianet_id'].isin(patients_with_BMI)].shape[0]
print(f'The number of patients with BMI measurements in the prediction dataset is: {patients_pred_with_BMI}/{len(patients_pred)}')


#select measurements of patients in the prediction dataset
data_selected=data[data['pedianet_id'].isin(patients_pred['pedianet_id'])]
print(f'The number of measurements in the selected dataset is: {data_selected.shape[0]}')

################## BMI as baseline covariate (look only at measurements in the follow up window of the sequence)
data_cov=data_selected.copy()
#remove measurements that are not in the follow up window of the sequence
data_cov=data_cov[data_cov['agemos']<=12*YEARS]

print(f'The number of measurements in the covariate dataset is: {data_cov.shape[0]}')
print(f'The number of patients in the covariate dataset is: {data_cov["pedianet_id"].nunique()}/{len(patients_pred)}')

################## BMI as target
data_target=data_selected.copy()
#remove measurements that are not in the prediction window
data_target=data_target[data_target['agemos']>12*YEARS]
print(f'The number of measurements in the target dataset is: {data_target.shape[0]}')
print(f'The number of patients in the target dataset is: {data_target["pedianet_id"].nunique()}/{len(patients_pred)}')

#add target column: 1 for overweight and 2 for obesity, 0 otherwise following these rules:
#for childeren over 5 years (agemos > 60):BMIwho > 1 overweight,BMI who > 2 obesity
#for children under 5 years (agemos <= 60):BMIwho > 2 overweight, BMI who > 3 obesity

data_target['target']=0
data_target.loc[(data_target['agemos']>60) & (data_target['BMIwho']>1),'target']=1
data_target.loc[(data_target['agemos']>60) & (data_target['BMIwho']>2),'target']=2
data_target.loc[(data_target['agemos']<=60) & (data_target['BMIwho']>2),'target']=1
data_target.loc[(data_target['agemos']<=60) & (data_target['BMIwho']>3),'target']=2

#count children with at least one overweight or obesity measurement in the prediction window
overweight_children=data_target[data_target['target']==1]['pedianet_id'].nunique()
obese_children=data_target[data_target['target']==2]['pedianet_id'].nunique()
overweight_obesity_children=data_target[data_target['target']>0]['pedianet_id'].nunique()
print(f'The number of children with at least one overweight measurement in the prediction window is: {overweight_children}/{len(patients_pred)}')
print(f'The number of children with at least one obesity measurement in the prediction window is: {obese_children}/{len(patients_pred)}')
print(f'The number of children with at least one overweight or obesity measurement in the prediction window is: {overweight_obesity_children}/{len(patients_pred)}')    
