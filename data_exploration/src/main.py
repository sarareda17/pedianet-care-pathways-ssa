from exploration import general_exploration
from exploration import patients_exploration
from exploration import main_diagnoses_exploration
from exploration import vaccines_exploration
from exploration import drugs_exploration
from exploration import outpatients_exploration
from exploration import contacts_exploration
from exploration import exemptions_exploration
from exploration import family_diseases_exploration
from exploration import fse_exploration

#libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
ROOT= Path(__file__).resolve().parent.parent.parent


def main():
    # Load the dataset
    path1=ROOT/'data'/'pazienti.xlsx'
    patients = pd.read_excel(path1)
    path2=ROOT/'data'/'main_diagnoses.xlsx'
    #diagnoses = pd.read_excel(path2)
    path3=ROOT/'data'/'farmaci.xlsx'
    #drugs = pd.read_excel(path3)
    path4=ROOT/'data'/'vaccinazioni.xlsx'
    vaccines = pd.read_excel(path4)
    path5=ROOT/'data'/'specialistica.xlsx'
    #outpatients = pd.read_excel(path5)
    path6=ROOT/'data'/'contatti.xlsx'
    #contacts = pd.read_excel(path6)
    path7=ROOT/'data'/'esenzioni.xlsx'
    #exemptions = pd.read_excel(path7)
    path8=ROOT/'data'/'patologie_fam.xlsx'
    #family_diseases = pd.read_excel(path8)
    path9=ROOT/'data'/'fse.csv'
    #fse=pd.read_csv(path9)
    path10=ROOT/'data'/'branche_specialistica.csv'
    #branche=pd.read_csv(path10)
    path11=ROOT/'data'/'outpatients_with_descriptions.xlsx'
    #outpatients_with_descriptions = pd.read_excel(path11)
    path12=ROOT/'data'/'comorbidities_exemptions.csv'
    #comorbidities_exemptions = pd.read_csv(path12)
    path13=ROOT/'data'/'vaccination_counts.xlsx'
    vaccination_counts = pd.read_excel(path13)

    # Perform general exploration
    #general_exploration(patients)
    #general_exploration(diagnoses)
    #general_exploration(drugs)
    #general_exploration(vaccines)
    #general_exploration(outpatients)
    #general_exploration(contacts)
    #general_exploration(exemptions)
    #general_exploration(fse)
    #patients_exploration(patients)
    #main_diagnoses_exploration(diagnoses,patients)
    vaccines_exploration(vaccines,patients,vaccination_counts)
    #drugs_exploration(drugs,patients)
    #outpatients_exploration(outpatients_with_descriptions,branche,patients)
    #contacts_exploration(contacts,patients)
    #exemptions_exploration(exemptions,comorbidities_exemptions,patients)
    #family_diseases_exploration(family_diseases,patients)
    #fse_exploration(fse,patients)




if __name__ == "__main__":
    main()