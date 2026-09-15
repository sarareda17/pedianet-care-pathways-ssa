#set up and run prediction models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
CLUSTERING_FOLDER=ROOT/'SSA'/'results'/'sequences'/'sequence_clustering'
CHAR_FOLDER=ROOT/'SSA'/'results'/'sequences'/'sequence_characteristics'
PRED_FOLDER=ROOT/'SSA'/'prediction'
from config import SEQUENCE_NAME, DISTANCE_METRIC, CLUSTERING,LINKAGE,NUM_CLUSTERS,YEARS,SEASON,SEQ_LENGTH,STATES_LABELS

def run_prediction():
    """
    Sets up target variables (follow-up events), extracts patient and trajectory
    features, performs correlation checks, and fits survival/classification models.
    """

    ########### SET UP ###########
    PRED_NAME='target_obs_drugs' #name of the prediction task: 'target_lrti' or 'target_obs_drugs'
    #possible features are: 'sex','age','cluster','entropy','turbulence','num_transitions','adi','num_subsequences','complexity','variance_of_spell','count', 'months_last_event','last_year_events'
    FEATURES=['sex','age','cluster','entropy','num_transitions','complexity','turbulence','num_subsequences','variance_of_spell','count', 'months_last_event','last_year_events'] #features to use for prediction
    CAT_FEATURES=['sex','cluster'] #categorical features to use for prediction
    MODEL='cox_strata_cluster' #model to use for prediction: 'logistic_regression' or 'cox_regression' or 'lasso_regression' or 'cox_time_varying' or cox_strata_entropy' or 'cox_time_varying_strata' or 'cox_strata_cluster'

    (PRED_FOLDER/PRED_NAME).mkdir(parents=True, exist_ok=True)

    if SEASON==True:
        #load patients included in the sequence analysis and select only those included in the prediction dataset
        patients_sequences=pd.read_csv(ROOT/'SSA'/'datasets'/f'selected_patients_diagnoses_season_{YEARS}.csv')
        patients_pred=patients_sequences[(pd.to_datetime(patients_sequences['EF'])-pd.to_datetime(patients_sequences['end_date'])).dt.days>365]
        #load datasets useful to set up the prediction target: patients with the event and time to event
        macrodiag_breath=pd.read_csv(ROOT/'SSA'/'datasets'/'macrodiagnoses_season_breath.csv')
        drugs_breath=pd.read_csv(ROOT/'SSA'/'datasets'/'drugs_season_breath.csv')   
    else:
        #load patients included in the sequence analysis and select only those included in the prediction dataset
        patients_sequences=pd.read_csv(ROOT/'SSA'/'datasets'/f'selected_patients_diagnoses.csv')
        patients_sequences['end_date']=pd.to_datetime(patients_sequences['datan'])+ pd.DateOffset(years=YEARS)
        patients_pred=patients_sequences[(pd.to_datetime(patients_sequences['EF'])-pd.to_datetime(patients_sequences['end_date'])).dt.days>365]
        #load datasets useful to set up the prediction target: patients with the event and time to event
        macrodiag_breath=pd.read_csv(ROOT/'SSA'/'datasets'/'macrodiagnoses_breath.csv')
        drugs_breath=pd.read_csv(ROOT/'SSA'/'datasets'/'drugs_breath.csv')
    
    print(f'The number of patients included in the prediction dataset is: {len(patients_pred)}/{len(patients_sequences)}')



    if PRED_NAME=='target_lrti':
        #target: patients that have at least 1 INF_INF macrodiagnoses in the follow up period of 1 year after the end of the sequence
        macro=['INF_INF']
        target_patients = pd.DataFrame(
            {'target': 0, 'months': 12.0}, 
            index=patients_pred['pedianet_id']
        )
        target_patients['months'] = target_patients['months'].astype(float)
        count_patient=0
        for idx, patient in patients_pred.iterrows():   
            patient_diag_breath=macrodiag_breath[macrodiag_breath['pedianet_id']==patient['pedianet_id']]
            diag_breath_after_end_date=patient_diag_breath[(pd.to_datetime(patient_diag_breath['datacontatto'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_diag_breath['datacontatto'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
            selected_diag=diag_breath_after_end_date[diag_breath_after_end_date['state'].isin(macro)]
            #check if there are at least 1 diagnosis of the states in the follow up period
            if len(selected_diag)>=1:
                count_patient+=1
                target_patients.loc[patient['pedianet_id'],'target']=1  
                selected_diag=selected_diag.sort_values(by='datacontatto')
                first_diag_date=selected_diag['datacontatto'].min()
                delta_days = (pd.to_datetime(first_diag_date) - pd.to_datetime(patient['end_date'])).days
                months_val = round(delta_days / 30.4375, 2)
                # max(0.1, ...) preventing crashes
                target_patients.loc[patient['pedianet_id'], 'months'] = max(0.1, months_val)
        print(f'The number of patients with at least 1 diagnosis of {macro} in the follow up period is: {count_patient}/{len(patients_pred)}')

    elif PRED_NAME=='target_obs_drugs':
        #target: patients that have at least 2 prescription of R03 atc drugs in the follow up period of 1 year after the end of the sequence
        target_patients = pd.DataFrame(
            {'target': 0, 'months': 12.0}, 
            index=patients_pred['pedianet_id']
        )
        target_patients['months'] = target_patients['months'].astype(float)
        atc_r03_codes=('R03')
        count_patient=0
        for idx, patient in patients_pred.iterrows():
            patient_drugs_breath=drugs_breath[drugs_breath['pedianet_id']==patient['pedianet_id']]
            drugs_breath_after_end_date=patient_drugs_breath[(pd.to_datetime(patient_drugs_breath['dataricetta'])>pd.to_datetime(patient['end_date'])) & (pd.to_datetime(patient_drugs_breath['dataricetta'])<pd.to_datetime(patient['end_date'])+pd.DateOffset(years=1))]
            drugs_breath_after_end_date=drugs_breath_after_end_date.sort_values(by='dataricetta')
            selected_drug=drugs_breath_after_end_date[drugs_breath_after_end_date['atc'].astype(str).str.startswith(atc_r03_codes)]
            #check if there are at least 2 prescription of R03 atc drugs in the follow up period
            if len(selected_drug)>=2:
                count_patient+=1
                target_patients.loc[patient['pedianet_id'],'target']=1
                first_drug_date=selected_drug['dataricetta'].min()
                selected_drug=selected_drug.sort_values(by='dataricetta')
                delta_days = (pd.to_datetime(first_drug_date) - pd.to_datetime(patient['end_date'])).days
                months_val = round(delta_days / 30.4375, 2)
                # max(0.1, ...) preventing crashes
                target_patients.loc[patient['pedianet_id'], 'months'] = max(0.1, months_val)
        print(f'The number of patients with at least 2 prescription of R03 atc drugs in the follow up period is: {count_patient}/{len(patients_pred)}')


    #load features we want to use to predict the target
    features=pd.DataFrame(index=patients_pred['pedianet_id'])
    patients_pred=patients_pred.set_index('pedianet_id')

    #set up min max scaler 
    from sklearn.preprocessing import MinMaxScaler

    ## - patient level features from patient dataset
    #sex as 0 for female and 1 for male
    features['sex'] = patients_pred['sesso'].map({'F': 0, 'M': 1})

    if SEASON==True:
        #age in months
        features['age'] = ((pd.to_datetime(patients_pred['end_date']) - pd.to_datetime(patients_pred['datan'])).dt.days / 30.4375).astype(int)
        print(f'Age : {features['age'].head(10)}')
        features['age'] = MinMaxScaler().fit_transform(features[['age']])
    else:
        if 'age' in FEATURES:
            FEATURES.remove('age')


    features['adi']=patients_pred['adi']

    ## - cluster labels
    cluster_labels=pd.read_csv(CLUSTERING_FOLDER/SEQUENCE_NAME/CLUSTERING/DISTANCE_METRIC/f'cluster_labels_{DISTANCE_METRIC}.csv', index_col='Entity ID')
    features['cluster'] = cluster_labels['Cluster'].reindex(features.index)

    print('Cluster head:\n', features['cluster'].value_counts())

    ###sequence characteristics
    seq_char=pd.read_csv(CHAR_FOLDER/SEQUENCE_NAME/f'characteristics_by_sequence.csv', index_col=0)

    ## - entropy values
    features['entropy'] = seq_char['entropy'].reindex(features.index)
    features['entropy'] = MinMaxScaler().fit_transform(features[['entropy']])
    ##-entropy squared values
    features['entropy_sq'] = features['entropy']**2
    ## -number of transitions
    features['num_transitions'] = seq_char['number of transitions'].reindex(features.index)
    features['num_transitions'] = MinMaxScaler().fit_transform(features[['num_transitions']])
    ## - turbulence values
    features['turbulence'] = seq_char['turbulence'].reindex(features.index)
    features['turbulence'] = MinMaxScaler().fit_transform(features[['turbulence']])
    ## - number of subsequences
    features['num_subsequences'] = seq_char['number of subsequences'].reindex(features.index)
    features['num_subsequences'] = MinMaxScaler().fit_transform(features[['num_subsequences']])
    ## - complexity values
    features['complexity'] = seq_char['complexity index'].reindex(features.index)
    features['complexity'] = MinMaxScaler().fit_transform(features[['complexity']])
    ## - variance of spell values
    features['variance_of_spell'] = seq_char['variance of spell durations'].reindex(features.index)
    features['variance_of_spell'] = MinMaxScaler().fit_transform(features[['variance_of_spell']])


    #other features 
    sequences=pd.read_csv(ROOT/'SSA'/'datasets'/f'{SEQUENCE_NAME}.csv', index_col='pedianet_id')  
    sequences=sequences[[str(i) for i in range(SEQ_LENGTH) if str(i) in sequences.columns]] #keep only the columns with the sequence states

    ## - months since the last event in the sequence 
    last_event_month=np.where((sequences!= 'NO'), sequences.columns.astype(int), -1).max(axis=1)
    months_last_event=SEQ_LENGTH-last_event_month-1
    features['months_last_event'] = pd.Series(months_last_event, index=sequences.index).reindex(features.index)
    features['months_last_event'] = MinMaxScaler().fit_transform(features[['months_last_event']])

    ## - number of events in the last year of the sequence (last 12 months)
    last_year_events = (sequences.iloc[:, -12:] != 'NO').sum(axis=1)
    features['last_year_events'] = pd.Series(last_year_events, index=sequences.index).reindex(features.index)
    features['last_year_events'] = MinMaxScaler().fit_transform(features[['last_year_events']])

    ## - counts of events in each state in the sequence
    if 'count' in FEATURES:
        FEATURES.remove('count')
        for state, label in STATES_LABELS.items():
            if state != 'NO':  # Exclude the 'NO' state from counting
                features[f'count_{state}'] = (sequences == state).sum(axis=1)
                features[f'count_{state}'] = MinMaxScaler().fit_transform(features[[f'count_{state}']])
                FEATURES.append(f'count_{state}')


    #create the dataset for prediction: features + target
    #keep only the feauters selected above
    features=features[FEATURES]
    target = target_patients['target'].reindex(features.index)
    duration=target_patients['months'].reindex(features.index)
    pred_dataset_cox=pd.DataFrame({'target':target,'duration':duration}, index=features.index).join(features)
    pred_dataset_log=pd.DataFrame({'target':target}, index=features.index).join(features)

    #choose the dataset based on the model selected
    if MODEL=='logistic_regression' or MODEL=='lasso_regression':
        pred_dataset=pred_dataset_log
    elif MODEL=='cox_regression' or MODEL=='cox_time_varying_strata' or MODEL=='cox_time_varying' or MODEL=='cox_strata_entropy' or MODEL=='cox_strata_cluster':
        pred_dataset=pred_dataset_cox

    ########### DATASET PREPROCESSING ###########
    #check for missing values
    missing_values=pred_dataset.isnull().sum()
    print(f"Missing values in the prediction dataset:\n{missing_values}")
    #check for correlation on numeric features
    numeric_features = [col for col in FEATURES if col not in CAT_FEATURES]
    correlation_matrix = pred_dataset[numeric_features].corr()
    print(f"Correlation matrix:\n{correlation_matrix}") 
    #plot correlation matrix
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation Matrix of Numeric Features')
    plt.savefig(PRED_FOLDER/PRED_NAME/'correlation_matrix.png')
    plt.show()
    #remove highly correlated features (correlation > 0.8)
    highly_correlated_features = set()
    for i in range(len(numeric_features)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                highly_correlated_features.add(correlation_matrix.columns[i])
    #if non linear terms are inlcuded, they are not removed
    if 'entropy_sq' in highly_correlated_features:
        highly_correlated_features.remove('entropy_sq')
    pred_dataset.drop(columns=highly_correlated_features, inplace=True)
    print(f"Highly correlated features removed: {highly_correlated_features}")


    # summary statistics of entropy across clusters
    if 'cluster' and 'entropy' in pred_dataset.columns:
        entropy_summary = pred_dataset.groupby('cluster')['entropy'].agg(
            n_patients='count',
            mean='mean',
            std='std',
            median='median',
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75)
        ).reset_index()

        print("--- Entropy Distribution Across Trajectory Clusters ---")
        print(entropy_summary.to_string(index=False))

    #create dummy variables for categorical features
    pred_dataset[CAT_FEATURES]=pred_dataset[CAT_FEATURES].astype('category')
    if 'cluster' in CAT_FEATURES:
        most_frequent_cluster = pred_dataset['cluster'].value_counts().idxmax()
        #set the cluster with the highest number of patients as the reference category for dummy variables
        categories = [most_frequent_cluster] + [c for c in pred_dataset['cluster'].unique() if c != most_frequent_cluster]
        pred_dataset['cluster'] = pd.Categorical(pred_dataset['cluster'], categories=categories)


    pred_dataset = pd.get_dummies(
        pred_dataset,
        columns=CAT_FEATURES,
        drop_first=True,
        dtype=int  # set 0 and 1 as integer type for dummy variables
    )

    #dummies variables have different name now, save the new names of the features to use in the prediction model
    features_name=[col for col in pred_dataset.columns if col != 'target' and col != 'duration']

    #dataset distribution and target distribution
    print(pred_dataset.head())
    print(f'Dataset shape: {pred_dataset.shape}')
    print(f'Target distribution:\n{pred_dataset["target"].value_counts(normalize=True)}')


    # Point-biserial correlations between cluster dummies and continuous variables
    cluster_cols = sorted([col for col in pred_dataset.columns if col.startswith('cluster_')])
    numeric_cols = [col for col in pred_dataset.columns if col not in ['target', 'duration'] + cluster_cols + CAT_FEATURES]

    if cluster_cols and numeric_cols:
        from scipy.stats import pointbiserialr
        corr_matrix = pd.DataFrame(index=cluster_cols, columns=numeric_cols, dtype=float)

        for cluster_col in cluster_cols:
            for num_col in numeric_cols:
                corr, _ = pointbiserialr(pred_dataset[cluster_col], pred_dataset[num_col])
                corr_matrix.loc[cluster_col, num_col] = corr

        # heatmap of point-biserial correlations
        fig_width = max(10, len(numeric_cols) * 0.9)
        fig_height = max(5, len(cluster_cols) * 0.8)
        
        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Point-Biserial Correlation ($r_{pb}$)'},
            linewidths=0.5,
            linecolor='white'
        )

        plt.title('Point-Biserial Correlations: Trajectory Clusters vs Continuous Features', fontsize=12, pad=12)
        plt.xlabel('Continuous Variables', fontsize=10)
        plt.ylabel('Cluster Dummies', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(PRED_FOLDER / PRED_NAME / 'point_biserial_correlations_heatmap.png', dpi=300)
        

    ############ PREDICTION MODEL ############
    if MODEL=='logistic_regression':
        #split train and test set stratified by target
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(pred_dataset[features_name], pred_dataset['target'], test_size=0.2, random_state=42, stratify=pred_dataset['target'])
        print(f'Train set shape: {X_train.shape}, Test set shape: {X_test.shape}')
        #run logistic regression model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        #make predictions on the test set
        #y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = np.where(y_pred_proba >= 0.5, 1, 0)

        #analyze results 
        from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=['No', 'Yes']).plot()
        
        plt.savefig(PRED_FOLDER/PRED_NAME/'confusion_matrix.png')
        print(f'ROC AUC Score: {roc_auc_score(y_test, y_pred):.3f}')
        print(f"AUPRC: {average_precision_score(y_test, y_pred_proba):.3f}")

        #print logistic regression coefficients
        coefficients = pd.DataFrame({'Feature': features_name, 'Coefficient': model.coef_[0]})
        coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
        print(coefficients)

        
    elif MODEL=='lasso_regression':
        #split train and test set stratified by target
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(pred_dataset[features_name], pred_dataset['target'], test_size=0.2, random_state=42, stratify=pred_dataset['target'])
        print(f'Train set shape: {X_train.shape}, Test set shape: {X_test.shape}')
        #run lasso regression model
        from sklearn.linear_model import LogisticRegressionCV
        model = LogisticRegressionCV(cv=5, penalty='l1', solver='saga', max_iter=1000, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        #make predictions on the test set
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = np.where(y_pred_proba >= 0.5, 1, 0)

        #analyze results 
        from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=['No', 'Yes']).plot()
        
        plt.savefig(PRED_FOLDER/PRED_NAME/'confusion_matrix_lasso.png')
        print(f'ROC AUC Score: {roc_auc_score(y_test, y_pred):.3f}')
        print(f"AUPRC: {average_precision_score(y_test, y_pred_proba):.3f}")

        #print lasso regression coefficients
        coefficients = pd.DataFrame({'Feature': features_name, 'Coefficient': model.coef_[0]})
        coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
        print(coefficients)

    elif MODEL=='cox_regression':
        ############ COX PROPORTIONAL HAZARDS MODEL (full dataset, no train/test split) ############
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index

        # Fit the Cox model on the entire dataset
        cph = CoxPHFitter()
        cph.fit(pred_dataset[features_name + ['target', 'duration']],
                duration_col='duration', event_col='target')

        # Full model summary: coefficients, hazard ratios, p-values, 95% CI
        cph.print_summary()

        ############ 1. HAZARD RATIOS TABLE ############
        hr_table = cph.summary[['coef', 'exp(coef)', 'p', 
                                'exp(coef) lower 95%', 'exp(coef) upper 95%']]
        hr_table = hr_table.sort_values(by='exp(coef)', ascending=False)
        print("\n--- Hazard Ratios ---")
        print(hr_table)

        ############ 2. CONCORDANCE INDEX (in-sample, on full dataset) ############
        c_index = concordance_index(
            pred_dataset['duration'],
            -cph.predict_partial_hazard(pred_dataset[features_name]),
            pred_dataset['target']
        )
        print(f"\nConcordance Index (in-sample): {c_index:.3f}")

        ############ 3. CHECK PROPORTIONAL HAZARDS ASSUMPTION ############
        print("\n--- Proportional Hazards Assumption Check ---")
        cph.check_assumptions(pred_dataset[features_name + ['target', 'duration']], 
                            p_value_threshold=0.05)
        

        ############ 4. SURVIVAL CURVES FOR KEY COVARIATES (e.g., cluster membership) ############
        fig, ax = plt.subplots(figsize=(8, 6))
        cph.plot_partial_effects_on_outcome(covariates='cluster_3', values=[0, 1], ax=ax)
        plt.title('Estimated Survival Function by Cluster 3 Membership')
        plt.xlabel('Months since end of sequence')
        plt.ylabel('Survival probability (no event)')
        plt.savefig(PRED_FOLDER/PRED_NAME/'cox_survival_curves.png')

        ############ 5. BASELINE HAZARD / CUMULATIVE HAZARD (optional, descriptive) ############
        cph.baseline_hazard_.plot()
        plt.title('Baseline Cumulative Hazard')
        plt.xlabel('Months')
        plt.ylabel('Cumulative Hazard')
        plt.savefig(PRED_FOLDER/PRED_NAME/'cox_baseline_hazard.png')
        plt.close()

    elif MODEL=='cox_time_varying':
        ############ EXTENDED COX MODEL WITH TIME-VARYING COVARIATES (Andersen-Gill) ############
        from lifelines import CoxTimeVaryingFitter
        from lifelines.utils import to_episodic_format, concordance_index

        # prepare the dataset for time-varying covariates
        df_base = pred_dataset.copy()
        df_base['duration'] = df_base['duration'].clip(lower=0.1).astype(float)
        df_base['target'] = df_base['target'].astype(int)

        # expand the dataset into episodic format for time-varying covariates
        # time_gaps=1.0 generates unit intervals: (0, 1], (1, 2], ..., (11, 12]
        long_df = to_episodic_format(
            df_base,
            duration_col='duration',
            event_col='target',  
            time_gaps=1.0
        )

        long_df['pedianet_id'] = long_df['id']
        #check
        print(f"Pazienti verificati in long_df: {long_df['pedianet_id'].nunique()}/{len(df_base)}")

        # define the features for the time-varying model
        tv_features = [f for f in features_name]  # all features are considered time-varying

        fit_cols = ['pedianet_id', 'start', 'stop', 'target'] + tv_features

        # Fit the Cox model with time-varying covariates
        ctv = CoxTimeVaryingFitter(penalizer=0.1)
        ctv.fit(
            long_df[fit_cols],
            id_col='pedianet_id',
            event_col='target',
            start_col='start',
            stop_col='stop',
            show_progress=True
        )

        # Summary of the time-varying Cox model
        ctv.print_summary() 

        ############ 1. HAZARD RATIOS TABLE ############
        hr_table_tv = ctv.summary[['coef', 'exp(coef)', 'p', 
                                'exp(coef) lower 95%', 'exp(coef) upper 95%']]
        hr_table_tv = hr_table_tv.sort_values(by='exp(coef)', ascending=False)
        print("\n--- Hazard Ratios (Time-Varying Model) ---")
        print(hr_table_tv)

        ############ 2. CONCORDANCE INDEX ############
        # Compute predicted partial hazard for each interval: exp(beta * X)
        predicted_partial_hazard = ctv.predict_partial_hazard(long_df[tv_features])
        long_df['pred_hazard'] = predicted_partial_hazard

        # Compute the concordance index
        c_index = concordance_index(long_df['target'], long_df['pred_hazard'])
        print(f"\n--- Concordance Index (Time-Varying Model) ---")
        print(f"Concordance Index: {c_index:.3f}")

    elif MODEL=='cox_time_varying_strata':
        ############ EXTENDED COX MODEL WITH TIME-VARYING COVARIATES (Andersen-Gill) and strata on is_peak############
        from lifelines import CoxTimeVaryingFitter
        from lifelines.utils import to_episodic_format, concordance_index

        # prepare the dataset for time-varying covariates
        df_base = pred_dataset.copy()
        df_base['duration'] = df_base['duration'].clip(lower=0.1).astype(float)
        df_base['target'] = df_base['target'].astype(int)

        # expand the dataset into episodic format for time-varying covariates
        # time_gaps=1.0 generates unit intervals: (0, 1], (1, 2], ..., (11, 12]
        long_df = to_episodic_format(
        df_base,
        duration_col='duration',
        event_col='target',  
        time_gaps=1.0
        )

        long_df['pedianet_id'] = long_df['id']
        #check
        print(f"Pazienti verificati in long_df: {long_df['pedianet_id'].nunique()}/{len(df_base)}")

        # create a binary indicator for peak months (1, 2, 3, 10, 11, 12) to capture seasonal effects
        long_df['calendar_month'] = np.ceil(long_df['stop']).clip(lower=1, upper=12).astype(int)
        peak_months = [1, 2, 3, 10, 11, 12]
        long_df['is_peak'] = long_df['calendar_month'].isin(peak_months).astype(int)

        # define the features for the time-varying model
        tv_features = [f for f in features_name if f not in ['is_peak']]  # exclude 'is_peak' if it's already in features_name

        fit_cols = ['pedianet_id', 'start', 'stop', 'target','is_peak'] + tv_features

        # Fit the Cox model with time-varying covariates
        ctv = CoxTimeVaryingFitter(penalizer=0.1)
        ctv.fit(
            long_df[fit_cols],
            id_col='pedianet_id',
            event_col='target',
            start_col='start',
            stop_col='stop',
            strata=['is_peak'],
            show_progress=True
        )

        # Summary of the time-varying Cox model
        ctv.print_summary()

        ############ 1. HAZARD RATIOS TABLE ############
        hr_table_tv = ctv.summary[['coef', 'exp(coef)', 'p', 
                                'exp(coef) lower 95%', 'exp(coef) upper 95%']]
        hr_table_tv = hr_table_tv.sort_values(by='exp(coef)', ascending=False)
        print("\n--- Hazard Ratios (Time-Varying Model) ---")
        print(hr_table_tv)

        ############ 2. CONCORDANCE INDEX ############
        # Compute predicted partial hazard for each interval: exp(beta * X)
        predicted_partial_hazard = ctv.predict_partial_hazard(long_df[['is_peak'] + tv_features])
        long_df['pred_hazard'] = predicted_partial_hazard

        # --- METHOD 1: Global Concordance Index (Standard Andersen-Gill) ---
        c_index_global = concordance_index(
            long_df['stop'],
            -long_df['pred_hazard'],
            long_df['target']
        )
        print(f"\n[1] Global Concordance Index (Andersen-Gill format): {c_index_global:.3f}")

        # --- METHOD 2: Stratified Concordance Index (Intra-Season Evaluation) ---
        c_indices = {}
        event_counts = {}

        print("\n[2] Stratified Concordance Index (Intra-Season):")
        for stratum_val, group in long_df.groupby('is_peak'):
            stratum_label = "Peak Months (is_peak=1)" if stratum_val == 1 else "Calm Months (is_peak=0)"
            n_events = int(group['target'].sum())
            
            if n_events > 0:
                c_val = concordance_index(
                    group['stop'],
                    -group['pred_hazard'],
                    group['target']
                )
                c_indices[stratum_val] = c_val
                event_counts[stratum_val] = n_events
                print(f"    - {stratum_label}: C-index = {c_val:.3f} ({n_events} events)")

        total_events = sum(event_counts.values())
        stratified_c_index = (
            sum(c_indices[s] * event_counts[s] for s in c_indices) / total_events
            if total_events > 0 else np.nan
        )
        print(f"    --> Overall Stratified C-index (weighted by events): {stratified_c_index:.3f}")

    elif MODEL == 'cox_strata_entropy':
        ############ COX PROPORTIONAL HAZARDS MODEL (Stratified by Entropy) ############
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index

        df_strata = pred_dataset.copy()

        # 1. Discretize continuous entropy into balanced quantiles 
        df_strata['entropy_strata'] = pd.qcut(
            df_strata['entropy'],
            q=4,
            labels=['entropy_1', 'entropy_2', 'entropy_3', 'entropy_4']
        )

        # 2. Exclude 'entropy' from covariates to estimate
        features_without_entropy = [f for f in features_name if f != 'entropy']

        # 3. Fit the stratified Cox model
        cph = CoxPHFitter()
        cph.fit(
            df_strata[features_without_entropy + ['entropy_strata', 'target', 'duration']],
            duration_col='duration',
            event_col='target',
            strata='entropy_strata'
        )

        # Full model summary: coefficients, hazard ratios, p-values, 95% CI
        cph.print_summary()

        ############ 1. HAZARD RATIOS TABLE ############
        hr_table = cph.summary[['coef', 'exp(coef)', 'p', 
                                'exp(coef) lower 95%', 'exp(coef) upper 95%']]
        hr_table = hr_table.sort_values(by='exp(coef)', ascending=False)
        print("\n--- Hazard Ratios ---")
        print(hr_table)

        ############ 2. STRATIFIED CONCORDANCE INDEX ############
        # Compute intra-stratum C-index to evaluate discrimination within matched risk baselines
        pred_risk = cph.predict_partial_hazard(df_strata[features_without_entropy])
        df_strata['pred_hazard'] = pred_risk

        c_indices = {}
        event_counts = {}
        print("\n--- Concordance Index by Entropy Stratum ---")
        for s_name, group in df_strata.groupby('entropy_strata'):
            n_ev = int(group['target'].sum())
            if n_ev > 0:
                c_val = concordance_index(group['duration'], -group['pred_hazard'], group['target'])
                c_indices[s_name] = c_val
                event_counts[s_name] = n_ev
                print(f"  * Stratum {s_name}: C-index = {c_val:.3f} ({n_ev} events)")

        stratified_c_index = sum(c_indices[s] * event_counts[s] for s in c_indices) / sum(event_counts.values())
        print(f"--> Overall Stratified C-index: {stratified_c_index:.3f}")

        ############ 3. CHECK PROPORTIONAL HAZARDS ASSUMPTION ############
        print("\n--- Proportional Hazards Assumption Check ---")
        # Entropy will no longer trigger a violation flag because it has been absorbed by the strata
        cph.check_assumptions(
            df_strata[features_without_entropy + ['entropy_strata', 'target', 'duration']],
            p_value_threshold=0.05
        )

        ############ 4. STRATIFIED BASELINE CUMULATIVE HAZARD ############
        # Plot the distinct non-proportional baseline curves across entropy tertiles
        fig, ax = plt.subplots(figsize=(8, 6))
        cph.baseline_cumulative_hazard_.plot(ax=ax)
        plt.title('Baseline Cumulative Hazard Stratified by Entropy Tertile')
        plt.xlabel('Months')
        plt.ylabel('Cumulative Hazard')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(PRED_FOLDER / PRED_NAME / 'cox_entropy_stratified_baseline.png')
        plt.close()  

    elif MODEL == 'cox_strata_cluster':
        ############ COX PROPORTIONAL HAZARDS MODEL (Stratified by Cluster) ############
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index

        df_strata = pred_dataset.copy()

        # 1. Exclude 'cluster' from covariates to estimate
        #cluster is not in the features_name list because it is a categorical variable that has been one-hot encoded into multiple dummy variables
        df_strata['cluster'] = df_strata[[col for col in df_strata.columns if col.startswith('cluster_')]].idxmax(axis=1).str.replace('cluster_', '').astype(int)
        features_without_cluster = [f for f in features_name if not f.startswith('cluster_')]

        # 2. Fit the stratified Cox model
        cph = CoxPHFitter()
        cph.fit(
            df_strata[features_without_cluster + ['cluster', 'target', 'duration']],
            duration_col='duration',
            event_col='target',
            strata='cluster'
        )

        # Full model summary: coefficients, hazard ratios, p-values, 95% CI
        cph.print_summary()

        ############ 1. HAZARD RATIOS TABLE ############
        hr_table = cph.summary[['coef', 'exp(coef)', 'p', 
                                'exp(coef) lower 95%', 'exp(coef) upper 95%']]
        hr_table = hr_table.sort_values(by='exp(coef)', ascending=False)
        print("\n--- Hazard Ratios ---")
        print(hr_table)

        ############ 2. STRATIFIED CONCORDANCE INDEX ############
        # Compute intra-stratum C-index to evaluate discrimination within matched risk baselines
        pred_risk = cph.predict_partial_hazard(df_strata[features_without_cluster])
        df_strata['pred_hazard'] = pred_risk

        c_indices = {}
        event_counts = {}
        print("\n--- Concordance Index by Cluster Stratum ---")
        for s_name, group in df_strata.groupby('cluster'):
            n_ev = int(group['target'].sum())
            if n_ev > 0:
                c_val = concordance_index(group['duration'], -group['pred_hazard'], group['target'])
                c_indices[s_name] = c_val
                event_counts[s_name] = n_ev
                print(f"  * Stratum {s_name}: C-index = {c_val:.3f} ({n_ev} events)")

        stratified_c_index = sum(c_indices[s] * event_counts[s] for s in c_indices) / sum(event_counts.values())
        print(f"--> Overall Stratified C-index: {stratified_c_index:.3f}")


if __name__ == "__main__":
    run_prediction()
