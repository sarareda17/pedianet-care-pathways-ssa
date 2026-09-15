from datetime import datetime
import pandas as pd
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent

# load dataset outpatients 
outpatients = pd.read_excel(ROOT / 'data' / 'specialistica.xlsx')

# get the list of datetime from the file names in cataloghi_ambulatoriale_veneto
path = ROOT / 'data_exploration'/'cataloghi_ambulatoriale_veneto'
files = os.listdir(path)

# parse dates from filenames and sort chronologically
datetimes = []
for file in files:
    if not file.endswith('.xlsx'):
        continue
    parts = file.replace('.xlsx', '').split('-')
    try:
        date = datetime(int(parts[3]), int(parts[4]), int(parts[5]))
        datetimes.append((file, date))
    except (IndexError, ValueError):
        print(f"Skipping {file}: cannot parse date")

datetimes.sort(key=lambda x: x[1])

# load ALL catalogs into memory once to avoid reading from disk 80k times
print("Loading catalogs...")
catalogs = {}
for file, date in datetimes:
    catalog_path = os.path.join(path, file)
    try:
        df = pd.read_excel(catalog_path, dtype=str)
        # keep only codice and descrizione columns
        df = df[['050000', 'descrizione']].copy()
        df.columns = ['codice', 'descrizione']
        df = df.drop_duplicates(subset='codice')
        # store as dictionary {codice: descrizione} for fast lookup
        catalogs[date] = df.set_index('codice')['descrizione'].to_dict()
    except Exception as e:
        print(f"Skipping {file}: {e}")

catalog_dates = sorted(catalogs.keys())
print(f"Loaded {len(catalogs)} catalogs")

# for each record date, find the most recent catalog date that is <= record date
def get_catalog_date(record_date):
    result = None
    for d in catalog_dates:
        if d <= record_date:
            result = d
        else:
            break
    return result

# ensure dataricetta is datetime for comparison
outpatients['dataricetta'] = pd.to_datetime(outpatients['dataricetta'])

# for each record, look up the description from the appropriate catalog in memory
print("Matching descriptions...")
def get_description(row):
    cat_date = get_catalog_date(row['dataricetta'])
    if cat_date is None:
        return None
    # get description from the dictionary, None if code not found
    return catalogs[cat_date].get(str(row['codice']), None)

outpatients['description_from_catalogo'] = outpatients.apply(get_description, axis=1)

# save the resulting dataframe to a new excel file
outpatients.to_excel(ROOT / 'data' / 'outpatients_with_descriptions.xlsx', index=False)

#print how many None descriptions we have
num_none = outpatients['description_from_catalogo'].isna().sum()
print(f"Number of records without description: {num_none} out of {len(outpatients)}")

#print codice in outpatients con None description
none_descriptions = outpatients[outpatients['description_from_catalogo'].isna()]['codice'].unique()
print(f"Unique codici with None description: {none_descriptions}")   

# check pipeline
print(outpatients.head())