import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
ROOT= Path(__file__).resolve().parent.parent.parent

#in this file I will create a join dataset for care setting taking info from specialistics visits and fse records 

#load datasets 
visite_specialistiche=pd.read_excel(ROOT/'data'/'specialistica.xlsx')
fse=pd.read_csv(ROOT/'data'/'fse_matched_7.csv')

#create new dataset
#keep the colomn pedianet_id, datarecord, tipo from fse and rename them as pedianet_id, date, care_setting
fse_care_setting=fse[['pedianet_id', 'datarecord', 'tipo']].copy()
fse_care_setting.rename(columns={'datarecord': 'date', 'tipo': 'care_setting'}, inplace=True)
#keep the column pedianet_id, dataricetta, codice from visite_specialistiche and rename them as pedianet_id, date, care_setting
visite_care_setting=visite_specialistiche[['pedianet_id', 'dataricetta', 'codice']].copy()
visite_care_setting.rename(columns={'dataricetta': 'date', 'codice': 'care_setting'}, inplace=True) 
#remove time from date 
visite_care_setting['date'] = visite_care_setting['date'].dt.date

#concatanate the two datasets
care_setting=pd.concat([fse_care_setting, visite_care_setting], ignore_index=True)
#save the dataset to csv
care_setting.to_csv(ROOT/'data'/'care_setting.csv', index=False)