import csv
import os
from itertools import islice
from typing import Union

import gower
import pandas as pd
from imblearn.under_sampling import NearMiss, TomekLinks
from sklearn.compose import make_column_transformer
from sklearn.neighbors import LocalOutlierFactor


def prep_campus_dataframe(validation_set: pd.DataFrame, training_set: pd.DataFrame, testing_set: pd.DataFrame, data_vars: list, campus: list) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	campus_logit_df = training_set[training_set['adj_acad_prog_primary_campus'].isin(campus)][data_vars].dropna().drop(columns=['emplid'])

	campus_validation_set = validation_set[(validation_set['adj_acad_prog_primary_campus'].isin(campus))][data_vars].dropna()

	campus_training_set = training_set[(training_set['adj_acad_prog_primary_campus'].isin(campus))][data_vars].dropna()

	campus_testing_set = testing_set[(testing_set['adj_acad_prog_primary_campus'].isin(campus))][data_vars].dropna().drop(columns=['enrl_ind'])

	campus_testing_set = campus_testing_set.reset_index()

	campus_shap_outcome = campus_testing_set['emplid'].copy(deep=True).values.tolist()

	campus_pred_outcome = campus_testing_set[[ 
							'emplid',
							# 'enrl_ind'
							]].copy(deep=True)

	campus_aggregate_outcome = campus_testing_set[[ 
							'emplid',
							'male',
							'underrep_minority',
							'first_gen_flag',
							'resident'
							# 'enrl_ind'
							]].copy(deep=True)

	campus_current_outcome = campus_testing_set[[ 
							'emplid',
							# 'enrl_ind'
							]].copy(deep=True)
		
	return campus_logit_df, campus_validation_set, campus_training_set, campus_testing_set, campus_shap_outcome, campus_pred_outcome, campus_aggregate_outcome, campus_current_outcome
	

def prep_system_dataframe(validation_set: pd.DataFrame, training_set: pd.DataFrame, testing_set: pd.DataFrame, data_vars: list, campus: list) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	system_logit_df = training_set[data_vars].dropna().drop(columns=['emplid'])

	system_validation_set = validation_set[data_vars].dropna()

	system_training_set = training_set[data_vars].dropna()

	system_testing_set = testing_set[(testing_set['adj_acad_prog_primary_campus'].isin(campus))][data_vars].dropna().drop(columns=['enrl_ind'])

	system_testing_set = system_testing_set.reset_index()

	system_shap_outcome = system_testing_set['emplid'].copy(deep=True).values.tolist()

	system_pred_outcome = system_testing_set[[ 
							'emplid',
							# 'enrl_ind'
							]].copy(deep=True)

	system_aggregate_outcome = system_testing_set[[ 
							'emplid',
							'male',
							'underrep_minority',
							'first_gen_flag',
							'resident'
							# 'enrl_ind'
							]].copy(deep=True)

	system_current_outcome = system_testing_set[[ 
							'emplid',
							# 'enrl_ind'
							]].copy(deep=True)
		   
	return system_logit_df, system_validation_set, system_training_set, system_testing_set, system_shap_outcome, system_pred_outcome, system_aggregate_outcome, system_current_outcome


def remove_outliers(campus_validation_set: pd.DataFrame, campus_training_set: pd.DataFrame, campus_x_training_outlier, campus_x_validation_outlier, campus: str, model_descr: str) -> Union[pd.DataFrame, pd.DataFrame]:
	campus_x_training_gower = gower.gower_matrix(campus_x_training_outlier)
	campus_x_validation_gower = gower.gower_matrix(campus_x_validation_outlier)

	campus_training_set['mask'] = LocalOutlierFactor(metric='precomputed', n_jobs=-1).fit_predict(campus_x_training_gower)
	campus_validation_set['mask'] = LocalOutlierFactor(metric='precomputed', n_jobs=-1).fit_predict(campus_x_validation_gower)

	campus_training_outlier_set = campus_training_set.drop(campus_training_set[campus_training_set['mask'] == 1].index)
	campus_training_outlier_set.to_csv(f'Z:\\Nathan\\Models\\student_risk\\outliers\\{campus}_{model_descr}_training_outlier_set.csv', encoding='utf-8', index=False)
	campus_validation_outlier_set = campus_validation_set.drop(campus_validation_set[campus_validation_set['mask'] == 1].index)
	campus_validation_outlier_set.to_csv(f'Z:\\Nathan\\Models\\student_risk\\outliers\\{campus}_{model_descr}_validation_outlier_set.csv', encoding='utf-8', index=False)

	campus_training_set = campus_training_set.drop(campus_training_set[campus_training_set['mask'] == -1].index)
	campus_training_set = campus_training_set.drop(columns='mask')
	campus_validation_set = campus_validation_set.drop(campus_validation_set[campus_validation_set['mask'] == -1].index)
	campus_validation_set = campus_validation_set.drop(columns='mask')
	
	return campus_validation_set, campus_training_set


def tomek_undersample(campus_validation_set: pd.DataFrame, campus_training_set: pd.DataFrame, campus_x_train: pd.DataFrame, campus_x_cv: pd.DataFrame, campus_y_train: pd.Series, campus_y_cv: pd.Series, campus_tomek_prep: object, campus_feat_names: list, campus: str, model_descr: str) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	
	for name, transformer, features, _ in campus_tomek_prep._iter(fitted=True):
		if transformer != 'passthrough':
			try:
				campus_feat_names.extend(campus_tomek_prep.named_transformers_[name].get_feature_names())
			except AttributeError:
				campus_feat_names.extend(features)
				
		if transformer == 'passthrough':
			campus_feat_names.extend(campus_tomek_prep._feature_names_in[features])

	campus_under_train = TomekLinks(sampling_strategy='all', n_jobs=-1)
	campus_under_valid = TomekLinks(sampling_strategy='all', n_jobs=-1)

	campus_x_train, campus_y_train = campus_under_train.fit_resample(campus_x_train, campus_y_train)
	campus_x_cv, campus_y_cv = campus_under_valid.fit_resample(campus_x_cv, campus_y_cv)

	campus_tomek_train_index = campus_under_train.sample_indices_
	campus_tomek_valid_index = campus_under_valid.sample_indices_
	campus_training_set = campus_training_set.reset_index(drop=True)
	campus_validation_set = campus_validation_set.reset_index(drop=True)

	campus_tomek_train_set = campus_training_set.drop(campus_tomek_train_index)
	campus_tomek_train_set.to_csv(f'Z:\\Nathan\\Models\\student_risk\\outliers\\{campus}_{model_descr}_tomek_training_set.csv', encoding='utf-8', index=False)
	campus_tomek_valid_set = campus_validation_set.drop(campus_tomek_valid_index)
	campus_tomek_valid_set.to_csv(f'Z:\\Nathan\\Models\\student_risk\\outliers\\{campus}_{model_descr}_tomek_validation_set.csv', encoding='utf-8', index=False)
	
	return campus_x_train, campus_x_cv, campus_y_train, campus_y_cv


def aggregate_outcome(campus_aggregate_outcome: pd.DataFrame, campus_xgbrf_pred_probs, campus: str, model_descr: str) -> None:
	campus_aggregate_outcome['emplid'] = campus_aggregate_outcome['emplid'].astype(str).str.zfill(9)
	campus_aggregate_outcome['risk_prob'] = 1 - pd.DataFrame(campus_xgbrf_pred_probs).round(4)

	campus_aggregate_outcome = campus_aggregate_outcome.rename(columns={"male": "sex_ind"})
	campus_aggregate_outcome.loc[campus_aggregate_outcome['sex_ind'] == 1, 'sex_descr'] = 'Male'
	campus_aggregate_outcome.loc[campus_aggregate_outcome['sex_ind'] == 0, 'sex_descr'] = 'Female'

	campus_aggregate_outcome = campus_aggregate_outcome.rename(columns={"underrep_minority": "underrep_minority_ind"})
	campus_aggregate_outcome.loc[campus_aggregate_outcome['underrep_minority_ind'] == 1, 'underrep_minority_descr'] = 'Minority'
	campus_aggregate_outcome.loc[campus_aggregate_outcome['underrep_minority_ind'] == 0, 'underrep_minority_descr'] = 'Non-minority'

	campus_aggregate_outcome = campus_aggregate_outcome.rename(columns={"resident": "resident_ind"})
	campus_aggregate_outcome.loc[campus_aggregate_outcome['resident_ind'] == 1, 'resident_descr'] = 'Resident'
	campus_aggregate_outcome.loc[campus_aggregate_outcome['resident_ind'] == 0, 'resident_descr'] = 'non-Resident'

	campus_aggregate_outcome.loc[campus_aggregate_outcome['first_gen_flag'] == 'Y', 'first_gen_flag'] = 1
	campus_aggregate_outcome.loc[campus_aggregate_outcome['first_gen_flag'] == 'N', 'first_gen_flag'] = 0

	campus_aggregate_outcome = campus_aggregate_outcome.rename(columns={"first_gen_flag": "first_gen_ind"})
	campus_aggregate_outcome.loc[campus_aggregate_outcome['first_gen_ind'] == 1, 'first_gen_descr'] = 'non-First Gen'
	campus_aggregate_outcome.loc[campus_aggregate_outcome['first_gen_ind'] == 0, 'first_gen_descr'] = 'First Gen'

	campus_aggregate_outcome.to_csv(f'Z:\\Nathan\\Models\\student_risk\\predictions\\{campus}\\{campus}_{model_descr}_aggregate_outcome.csv', encoding='utf-8', index=False)


def fairness_output(auto_engine, model_id: int, model_type: str, model_descr: str, metric_frame: object, run_date: object, campus_var: list) -> None:
	fairness_df = pd.DataFrame(metric_frame.by_group)
	fairness_df = fairness_df.reset_index()
	fairness_df = fairness_df.astype({"male":"int", "underrep_minority":"int"})
	
	fairness_df['date'] = run_date
	fairness_df['model_id'] = model_id
	fairness_df['model_type'] = model_type
	fairness_df['model_descr'] = model_descr
	fairness_df['matrix'] = fairness_df.matrix.astype(str)
	fairness_df['adj_acad_prog_primary_campus'] = campus_var[0]

	fairness_df.astype(str).to_sql('student_fairness', con=auto_engine, if_exists='append', index=False, schema='oracle_int.dbo')


def results_output(auto_engine, model_id: int, run_date: object, campus_current_outcome: pd.DataFrame, campus_xgbrf_pred_probs, campus: str, model_descr: str) -> None:
	campus_current_outcome['emplid'] = campus_current_outcome['emplid'].astype(str).str.zfill(9)
	campus_current_outcome['risk_prob'] = 1 - pd.DataFrame(campus_xgbrf_pred_probs).round(4)

	campus_current_outcome['date'] = run_date
	campus_current_outcome['model_id'] = model_id

	if not os.path.isfile(f'Z:\\Nathan\\Models\\student_risk\\predictions\\{campus}\\{campus}_{model_descr}_student_outcome.csv'):
		campus_current_outcome.to_csv(f'Z:\\Nathan\\Models\\student_risk\\predictions\\{campus}\\{campus}_{model_descr}_student_outcome.csv', encoding='utf-8', index=False)
		campus_current_outcome.to_sql('student_outcome', con=auto_engine, if_exists='append', index=False, schema='oracle_int.dbo')
	else:
		campus_prior_outcome = pd.read_csv(f'Z:\\Nathan\\Models\\student_risk\\predictions\\{campus}\\{campus}_{model_descr}_student_outcome.csv', encoding='utf-8', low_memory=False)
		campus_prior_outcome.to_csv(f'Z:\\Nathan\\Models\\student_risk\\predictions\\{campus}\\{campus}_{model_descr}_student_backup.csv', encoding='utf-8', index=False)
		campus_student_outcome = pd.concat([campus_prior_outcome, campus_current_outcome])
		campus_student_outcome.to_csv(f'Z:\\Nathan\\Models\\student_risk\\predictions\\{campus}\\{campus}_{model_descr}_student_outcome.csv', encoding='utf-8', index=False)
		campus_current_outcome.to_sql('student_outcome', con=auto_engine, if_exists='append', index=False, schema='oracle_int.dbo')


def shap_output(engine, student_shap, top_N: int, model_id: int, run_date: object, campus_shap_zip: dict, campus: str, model_descr: str) -> None:
	campus_shap_file = open(f'Z:\\Nathan\\Models\\student_risk\\shap\\{campus}\\{campus}_{model_descr}_shap.csv', 'w', newline='')
	campus_shap_writer = csv.writer(campus_shap_file)
	campus_shap_insert = []

	campus_shap_writer.writerow(['emplid','shap_values'])

	for emplid in campus_shap_zip:
		campus_shap_writer.writerow([emplid, list(islice(campus_shap_zip[emplid].items(), top_N))])
		campus_shap_sql = [emplid, list(islice(campus_shap_zip[emplid].items(), top_N))]
	
		campus_shap_insert.append(str(campus_shap_sql[0]).zfill(9))

		for index in range(top_N):
			shap_str, shap_float = campus_shap_sql[1][index]
			campus_shap_insert.append(shap_str) 
			campus_shap_insert.append(round(shap_float, 4))

	campus_shap_file.close()

	while campus_shap_insert:
		ins = student_shap.insert().values(emplid=campus_shap_insert.pop(0), 
											shap_descr_1=campus_shap_insert.pop(0), shap_value_1=campus_shap_insert.pop(0), 
											shap_descr_2=campus_shap_insert.pop(0), shap_value_2=campus_shap_insert.pop(0), 
											shap_descr_3=campus_shap_insert.pop(0), shap_value_3=campus_shap_insert.pop(0), 
											shap_descr_4=campus_shap_insert.pop(0), shap_value_4=campus_shap_insert.pop(0), 
											shap_descr_5=campus_shap_insert.pop(0), shap_value_5=campus_shap_insert.pop(0), 
											date=run_date, model_id=model_id)
		engine.execute(ins)

