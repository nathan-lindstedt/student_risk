#%%
import datetime
import pathlib
import time
import urllib
from datetime import date

import joblib
import numpy as np
import pandas as pd
import pyodbc
import saspy
import sklearn
import sqlalchemy
from fairlearn.metrics import MetricFrame, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate, selection_rate, count
from patsy.highlevel import dmatrices
from sklearn.compose import make_column_transformer
from sklearn.ensemble import VotingClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import MetaData, Table
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier, XGBRFClassifier

import shap
from student_risk import build_ft_tr_2yr_prod, config, helper_funcs

#%%
# Database connection
cred = pathlib.Path('Z:\\Nathan\\Models\\student_risk\\login.bin').read_text().split('|')
params = urllib.parse.quote_plus(f'TRUSTED_CONNECTION=YES; DRIVER={{SQL Server Native Client 11.0}}; SERVER={cred[0]}; DATABASE={cred[1]}')
engine = sqlalchemy.create_engine(f'mssql+pyodbc:///?odbc_connect={params}')
auto_engine = engine.execution_options(autocommit=True, isolation_level='AUTOCOMMIT')
metadata_engine = MetaData(engine.execution_options(autocommit=True, isolation_level='AUTOCOMMIT'))
student_shap = Table('student_shap', metadata_engine, autoload=True)

#%%
# Global variable initializaiton
strm: str = None
outcome: str = 'year'
top_N: int = 5
model_id: int = 7
model_descr: str = 'ft_tr_2yr'
run_date: date = date.today()
unwanted_vars: list = ['emplid','enrl_ind']

#%%
# Global XGBoost hyperparameter initialization
min_child_weight: int = 8
max_bin: int = 48
num_parallel_tree: int = 96
subsample: float = 0.8
colsample_bytree: float = 0.8
colsample_bynode: float = 0.8
verbose: bool = False

#%%
# SAS dataset builder
build_ft_tr_2yr_prod.DatasetBuilderProd.build_census_prod(outcome)

#%%
# Import pre-split data
validation_set = pd.read_sas(f'Z:\\Nathan\\Models\\student_risk\\datasets\\{model_descr}_validation_set.sas7bdat', encoding='latin1')
training_set = pd.read_sas(f'Z:\\Nathan\\Models\\student_risk\\datasets\\{model_descr}_training_set.sas7bdat', encoding='latin1')
testing_set = pd.read_sas(f'Z:\\Nathan\\Models\\student_risk\\datasets\\{model_descr}_testing_set.sas7bdat', encoding='latin1')

#%%
# Prepare dataframes
print('\nPrepare dataframes and preprocess data...')

# Pullman variables
pullm_data_vars = [
'emplid',
'enrl_ind', 
'male',
'underrep_minority',
# 'acad_year',
# 'age_group',
# 'age',
# 'race_hispanic',
# 'race_american_indian',
# 'race_alaska',
# 'race_asian',
# 'race_black',
# 'race_native_hawaiian',
# 'race_white',
# 'min_week_from_term_begin_dt',
# 'max_week_from_term_begin_dt',
# 'count_week_from_term_begin_dt',
'acad_level_bot_descr',
# 'acs_mi',
# 'distance',
# 'pop_dens',
# 'ipeds_ethnic_group_descrshort',
'pell_eligibility_ind', 
# 'pell_recipient_ind',
'first_gen_flag', 
'first_gen_flag_mi',
# 'LSAMP_STEM_Flag',
# 'anywhere_STEM_Flag',
'honors_program_ind',
# 'afl_greek_indicator',
# 'high_school_gpa',
'fall_term_gpa',
'fall_term_gpa_mi',
# 'fall_term_D_grade_count',
# 'fall_term_F_grade_count',
# 'fall_term_S_grade_count',
# 'fall_term_W_grade_count',
'spring_term_gpa',
'spring_term_gpa_mi',
'spring_term_D_grade_count',
'spring_term_F_grade_count',
# 'spring_term_S_grade_count',
# 'spring_term_W_grade_count',
# 'spring_midterm_gpa_change',
# 'awe_instrument',
# 'cdi_instrument',
# 'fall_avg_difficulty',
# 'fall_avg_pct_withdrawn',
# 'fall_avg_pct_CDFW',
# 'fall_avg_pct_CDF',
# 'fall_avg_pct_DFW',
# 'fall_avg_pct_DF',
# 'spring_avg_difficulty',
# 'spring_avg_pct_withdrawn',
# 'spring_avg_pct_CDFW',
# 'spring_avg_pct_CDF',
# 'spring_avg_pct_DFW',
# 'spring_avg_pct_DF',
# 'fall_lec_count',
# 'fall_lab_count',
# 'fall_lec_contact_hrs',
# 'fall_lab_contact_hrs',
'spring_lec_count',
'spring_lab_count',
'spring_stu_count',
'spring_oth_count',
# 'spring_lec_contact_hrs',
# 'spring_lab_contact_hrs',
# 'total_fall_contact_hrs',
# 'total_spring_contact_hrs',
# 'fall_midterm_gpa_avg',
# 'fall_midterm_gpa_avg_ind',
# 'spring_midterm_gpa_avg',
# 'spring_midterm_gpa_avg_mi',
'cum_adj_transfer_hours',
'resident',
# 'father_wsu_flag',
# 'mother_wsu_flag',
'parent1_highest_educ_lvl',
'parent2_highest_educ_lvl',
# 'citizenship_country',
# 'gini_indx',
# 'pvrt_rate',
# 'median_inc',
# 'median_value',
# 'educ_rate',
# 'pct_blk',
# 'pct_ai',
# 'pct_asn',
# 'pct_hawi',
# 'pct_oth',
# 'pct_two',
# 'pct_non',
# 'pct_hisp',
# 'city_large',
# 'city_mid',
# 'city_small',
# 'suburb_large',
# 'suburb_mid',
# 'suburb_small',
# 'town_fringe',
# 'town_distant',
# 'town_remote',
# 'rural_fringe',
# 'rural_distant',
# 'rural_remote',
# 'AD_DTA',
# 'AD_AST',
'AP',
'RS',
'CHS',
# 'IB',
# 'AICE',
'IB_AICE', 
'spring_credit_hours',
# 'total_spring_units',
'spring_withdrawn_hours',
# 'athlete',
'remedial',
# 'ACAD_PLAN',
# 'plan_owner_org',
'business',
'cahnrs_anml',
# 'cahnrs_envr',
'cahnrs_econ',
'cahnrext',
'cas_chem',
'cas_crim',
'cas_math',
'cas_psyc',
'cas_biol',
'cas_engl',
'cas_phys',
'cas',
'comm',
'education',
'medicine',
'nursing',
# 'pharmacy',
# 'provost',
'vcea_bioe',
'vcea_cive',
'vcea_desn',
'vcea_eecs',
'vcea_mech',
'vcea',
'vet_med',
# 'last_sch_proprietorship',
# 'sat_erws',
# 'sat_mss',
# 'sat_comp',
# 'attendee_alive',
# 'attendee_campus_visit',
# 'attendee_cashe',
# 'attendee_destination',
# 'attendee_experience',
# 'attendee_fcd_pullman',
# 'attendee_fced',
# 'attendee_fcoc',
# 'attendee_fcod',
# 'attendee_group_visit',
# 'attendee_honors_visit',
# 'attendee_imagine_tomorrow',
# 'attendee_imagine_u',
# 'attendee_la_bienvenida',
# 'attendee_lvp_camp',
# 'attendee_oos_destination',
# 'attendee_oos_experience',
# 'attendee_preview',
# 'attendee_preview_jrs',
# 'attendee_shaping',
# 'attendee_top_scholars',
# 'attendee_transfer_day',
# 'attendee_vibes',
# 'attendee_welcome_center',
# 'attendee_any_visitation_ind',
# 'attendee_total_visits',
# 'qvalue',
# 'fed_efc',
# 'fed_need',
'unmet_need_ofr',
'unmet_need_ofr_mi'
]

pullm_campus_var = ['PULLM']

pullm_logit_df: pd.DataFrame
pullm_validation_set: pd.DataFrame
pullm_training_set: pd.DataFrame
pullm_testing_set: pd.DataFrame
pullm_shap_outcome: list
pullm_pred_outcome: pd.DataFrame
pullm_aggregate_outcome: pd.DataFrame
pullm_current_outcome: pd.DataFrame

pullm_x_vars = [x for x in pullm_data_vars if x not in unwanted_vars]

# Pullman dataframes
pullm_logit_df, pullm_validation_set, pullm_training_set, pullm_testing_set, pullm_shap_outcome, pullm_pred_outcome, pullm_aggregate_outcome, pullm_current_outcome = helper_funcs.prep_campus_dataframe(validation_set, training_set, testing_set, pullm_data_vars, pullm_campus_var)

#%%
# University variables
univr_data_vars = [
'emplid',
'enrl_ind', 
'male',
'underrep_minority',
# 'acad_year',
# 'age_group',
# 'age',
# 'race_hispanic',
# 'race_american_indian',
# 'race_alaska',
# 'race_asian',
# 'race_black',
# 'race_native_hawaiian',
# 'race_white',
# 'min_week_from_term_begin_dt',
# 'max_week_from_term_begin_dt',
# 'count_week_from_term_begin_dt',
'acad_level_bot_descr',
# 'acs_mi',
# 'distance',
# 'pop_dens',
# 'ipeds_ethnic_group_descrshort',
'pell_eligibility_ind', 
# 'pell_recipient_ind',
'first_gen_flag', 
'first_gen_flag_mi',
# 'LSAMP_STEM_Flag',
# 'anywhere_STEM_Flag',
# 'honors_program_ind',
# 'afl_greek_indicator',
# 'high_school_gpa',
'fall_term_gpa',
'fall_term_gpa_mi',
# 'fall_term_D_grade_count',
# 'fall_term_F_grade_count',
# 'fall_term_S_grade_count',
# 'fall_term_W_grade_count',
'spring_term_gpa',
'spring_term_gpa_mi',
'spring_term_D_grade_count',
'spring_term_F_grade_count',
# 'spring_term_S_grade_count',
# 'spring_term_W_grade_count',
# 'spring_midterm_gpa_change',
# 'awe_instrument',
# 'cdi_instrument',
# 'fall_avg_difficulty',
# 'fall_avg_pct_withdrawn',
# 'fall_avg_pct_CDFW',
# 'fall_avg_pct_CDF',
# 'fall_avg_pct_DFW',
# 'fall_avg_pct_DF',
# 'spring_avg_difficulty',
# 'spring_avg_pct_withdrawn',
# 'spring_avg_pct_CDFW',
# 'spring_avg_pct_CDF',
# 'spring_avg_pct_DFW',
# 'spring_avg_pct_DF',
# 'fall_lec_count',
# 'fall_lab_count',
# 'fall_lec_contact_hrs',
# 'fall_lab_contact_hrs',
'spring_lec_count',
'spring_lab_count',
# 'spring_lec_contact_hrs',
# 'spring_lab_contact_hrs',
# 'total_fall_contact_hrs',
# 'total_spring_contact_hrs',
# 'fall_midterm_gpa_avg',
# 'fall_midterm_gpa_avg_ind',
# 'spring_midterm_gpa_avg',
# 'spring_midterm_gpa_avg_mi',
'cum_adj_transfer_hours',
'resident',
# 'father_wsu_flag',
# 'mother_wsu_flag',
'parent1_highest_educ_lvl',
'parent2_highest_educ_lvl',
# 'citizenship_country',
# 'gini_indx',
# 'pvrt_rate',
# 'median_inc',
# 'median_value',
# 'educ_rate',
# 'pct_blk',
# 'pct_ai',
# 'pct_asn',
# 'pct_hawi',
# 'pct_oth',
# 'pct_two',
# 'pct_non',
# 'pct_hisp',
# 'city_large',
# 'city_mid',
# 'city_small',
# 'suburb_large',
# 'suburb_mid',
# 'suburb_small',
# 'town_fringe',
# 'town_distant',
# 'town_remote',
# 'rural_fringe',
# 'rural_distant',
# 'rural_remote',
# 'AD_DTA',
# 'AD_AST',
# 'AP',
# 'RS',
# 'CHS',
# 'IB',
# 'AICE',
# 'IB_AICE', 
'spring_credit_hours',
# 'total_spring_units',
'spring_withdrawn_hours',
# 'athlete',
'remedial',
# 'ACAD_PLAN',
# 'plan_owner_org',
# 'business',
# 'cahnrs_anml',
# 'cahnrs_envr',
# 'cahnrs_econ',
# 'cahnrext',
# 'cas_chem',
# 'cas_crim',
# 'cas_math',
# 'cas_psyc',
# 'cas_biol',
# 'cas_engl',
# 'cas_phys',
# 'cas',
# 'comm',
# 'education',
# 'medicine',
# 'nursing',
# 'pharmacy',
# 'provost',
# 'vcea_bioe',
# 'vcea_cive',
# 'vcea_desn',
# 'vcea_eecs',
# 'vcea_mech',
# 'vcea',
# 'vet_med',
# 'last_sch_proprietorship',
# 'sat_erws',
# 'sat_mss',
# 'sat_comp',
# 'attendee_alive',
# 'attendee_campus_visit',
# 'attendee_cashe',
# 'attendee_destination',
# 'attendee_experience',
# 'attendee_fcd_pullman',
# 'attendee_fced',
# 'attendee_fcoc',
# 'attendee_fcod',
# 'attendee_group_visit',
# 'attendee_honors_visit',
# 'attendee_imagine_tomorrow',
# 'attendee_imagine_u',
# 'attendee_la_bienvenida',
# 'attendee_lvp_camp',
# 'attendee_oos_destination',
# 'attendee_oos_experience',
# 'attendee_preview',
# 'attendee_preview_jrs',
# 'attendee_shaping',
# 'attendee_top_scholars',
# 'attendee_transfer_day',
# 'attendee_vibes',
# 'attendee_welcome_center',
# 'attendee_any_visitation_ind',
# 'attendee_total_visits',
# 'qvalue',
# 'fed_efc',
# 'fed_need',
'unmet_need_ofr',
'unmet_need_ofr_mi'
]

univr_system_var = ['EVERE','ONLIN','SPOKA','TRICI','VANCO']

univr_logit_df: pd.DataFrame
univr_validation_set: pd.DataFrame
univr_training_set: pd.DataFrame
univr_testing_set: pd.DataFrame
univr_shap_outcome: list
univr_pred_outcome: pd.DataFrame
univr_aggregate_outcome: pd.DataFrame
univr_current_outcome: pd.DataFrame

univr_x_vars = [x for x in univr_data_vars if x not in unwanted_vars]

# University dataframes
univr_logit_df, univr_validation_set, univr_training_set, univr_testing_set, univr_shap_outcome, univr_pred_outcome, univr_aggregate_outcome, univr_current_outcome = helper_funcs.prep_system_dataframe(validation_set, training_set, testing_set, univr_data_vars, univr_system_var)

#%%
# Detect and remove outliers
# https://scikit-learn.org/dev/auto_examples/neighbors/plot_lof_outlier_detection.html
# https://en.wikipedia.org/wiki/Gower%27s_distance
print('\nDetect and remove outliers...')

# Pullman outliers
pullm_x_training_outlier = pullm_training_set.drop(columns=['enrl_ind','emplid'])
pullm_x_validation_outlier = pullm_validation_set.drop(columns=['enrl_ind','emplid'])

pullm_onehot_vars = pullm_x_training_outlier.select_dtypes(include='object').columns.tolist()

pullm_outlier_prep = make_column_transformer(
	(OneHotEncoder(drop='first'), pullm_onehot_vars),
	remainder='passthrough'
)

pullm_x_training_outlier = pullm_outlier_prep.fit_transform(pullm_x_training_outlier)
pullm_x_validation_outlier = pullm_outlier_prep.transform(pullm_x_validation_outlier)

pullm_validation_set, pullm_training_set = helper_funcs.remove_outliers(pullm_validation_set, pullm_training_set, pullm_x_training_outlier, pullm_x_validation_outlier, 'pullm', model_descr)

#%%
# University outliers
univr_x_training_outlier = univr_training_set.drop(columns=['enrl_ind','emplid'])
univr_x_validation_outlier = univr_validation_set.drop(columns=['enrl_ind','emplid'])

univr_onehot_vars = univr_x_training_outlier.select_dtypes(include='object').columns.tolist()

univr_outlier_prep = make_column_transformer(
	(OneHotEncoder(drop='first'), univr_onehot_vars),
	remainder='passthrough'
)

univr_x_training_outlier = univr_outlier_prep.fit_transform(univr_x_training_outlier)
univr_x_validation_outlier = univr_outlier_prep.transform(univr_x_validation_outlier)

univr_validation_set, univr_training_set = helper_funcs.remove_outliers(univr_validation_set, univr_training_set, univr_x_training_outlier, univr_x_validation_outlier, 'univr', model_descr)

#%%
# Create Tomek Link undersampled validation and training sets
# https://imbalanced-learn.org/stable/under_sampling.html#tomek-s-links

# Pullman undersample
pullm_x_train = pullm_training_set.drop(columns=['enrl_ind','emplid'])
pullm_x_cv = pullm_validation_set.drop(columns=['enrl_ind','emplid'])

pullm_x_test = pullm_testing_set[pullm_x_vars]

pullm_y_train = pullm_training_set['enrl_ind']
pullm_y_cv = pullm_validation_set['enrl_ind']
# pullm_y_test = pullm_testing_set['enrl_ind']

pullm_binary_vars = pullm_x_train.columns[pullm_x_train.isin([0,1]).all()].tolist()
pullm_filter_vars = pullm_binary_vars + pullm_onehot_vars
pullm_centered_vars = [b for b in pullm_x_vars if all(a not in b for a in pullm_filter_vars)]

pullm_tomek_prep = make_column_transformer(
	# (StandardScaler(), pullm_centered_vars),
	(OneHotEncoder(drop='first'), pullm_onehot_vars),
	remainder='passthrough'
)

pullm_x_train = pullm_tomek_prep.fit_transform(pullm_x_train)
pullm_x_cv = pullm_tomek_prep.transform(pullm_x_cv)
pullm_x_test = pullm_tomek_prep.transform(pullm_x_test)

pullm_feat_names: list = []

pullm_x_train, pullm_x_cv, pullm_y_train, pullm_y_cv = helper_funcs.tomek_undersample(pullm_validation_set, pullm_training_set, pullm_x_train, pullm_x_cv, pullm_y_train, pullm_y_cv, pullm_tomek_prep, pullm_feat_names, 'pullm', model_descr)

#%%
# University undersample
univr_x_train = univr_training_set.drop(columns=['enrl_ind','emplid'])
univr_x_cv = univr_validation_set.drop(columns=['enrl_ind','emplid'])

univr_x_test = univr_testing_set[univr_x_vars]

univr_y_train = univr_training_set['enrl_ind']
univr_y_cv = univr_validation_set['enrl_ind']
# univr_y_test = univr_testing_set['enrl_ind']

univr_binary_vars = univr_x_train.columns[univr_x_train.isin([0,1]).all()].to_list()
univr_filter_vars = univr_binary_vars + univr_onehot_vars
univr_centered_vars = [b for b in univr_x_vars if all(a not in b for a in univr_filter_vars)]

univr_tomek_prep = make_column_transformer(
	# (StandardScaler(), univr_centered_vars),
	(OneHotEncoder(drop='first'), univr_onehot_vars),
	remainder='passthrough'
)

univr_x_train = univr_tomek_prep.fit_transform(univr_x_train)
univr_x_cv = univr_tomek_prep.transform(univr_x_cv)
univr_x_test = univr_tomek_prep.transform(univr_x_test)

univr_feat_names: list = []

univr_x_train, univr_x_cv, univr_y_train, univr_y_cv = helper_funcs.tomek_undersample(univr_validation_set, univr_training_set, univr_x_train, univr_x_cv, univr_y_train, univr_y_cv, univr_tomek_prep, univr_feat_names, 'univr', model_descr)

#%%
# Standard logistic model

# Pullman standard model
print('\nStandard logistic model for Pullman second-year transfers...\n')

try:
	pullm_y, pullm_x = dmatrices('enrl_ind ~ ' + ' + '.join(pullm_x_vars), data=pullm_logit_df, return_type='dataframe')

	pullm_logit_mod = Logit(pullm_y, pullm_x)
	pullm_logit_res = pullm_logit_mod.fit(maxiter=500, method='bfgs')
	print(pullm_logit_res.summary())

	# Pullman VIF
	print('\nVIF for Pullman...\n')
	pullm_vif = pd.DataFrame()
	pullm_vif['vif factor'] = [variance_inflation_factor(pullm_x.values, i) for i in range(pullm_x.shape[1])]
	pullm_vif['features'] = pullm_x.columns
	pullm_vif.sort_values(by=['vif factor'], ascending=False, inplace=True, ignore_index=True)
	print(pullm_vif.round(1).to_string())
	print('\n')
	
except:
	print('Failed to converge or model misspecification: Linear combination, singular matrix, divide by zero, or separation\n')

print('\n')

#%%
# University standard model
print('\nStandard logistic model for University second-year transfers...\n')

try:
	univr_y, univr_x = dmatrices('enrl_ind ~ ' + ' + '.join(univr_x_vars), data=univr_logit_df, return_type='dataframe')

	univr_logit_mod = Logit(univr_y, univr_x)
	univr_logit_res = univr_logit_mod.fit(maxiter=500, method='bfgs')
	print(univr_logit_res.summary())

	# University VIF
	print('\nVIF for University...\n')
	univr_vif = pd.DataFrame()
	univr_vif['vif factor'] = [variance_inflation_factor(univr_x.values, i) for i in range(univr_x.shape[1])]
	univr_vif['features'] = univr_x.columns
	univr_vif.sort_values(by=['vif factor'], ascending=False, inplace=True, ignore_index=True)
	print(univr_vif.round(1).to_string())
	print('\n')

except:
	print('Failed to converge or model misspecification: Linear combination, singular matrix, divide by zero, or separation\n')

print('\n')

#%%
print('Run machine learning models for second-year transfers...\n')

# Logistic model

# Pullman logistic
# pullm_lreg = LogisticRegression(penalty='elasticnet', class_weight='balanced', solver='saga', max_iter=5000, l1_ratio=0.0, C=1.0, n_jobs=-1, verbose=False).fit(pullm_x_train, pullm_y_train)

# pullm_lreg_probs = pullm_lreg.predict_proba(pullm_x_train)
# pullm_lreg_probs = pullm_lreg_probs[:, 1]
# pullm_lreg_auc = roc_auc_score(pullm_y_train, pullm_lreg_probs)

# print(f'Overall accuracy for Pullman logistic model (training): {pullm_lreg.score(pullm_x_train, pullm_y_train):.4f}')
# print(f'ROC AUC for Pullman logistic model (training): {pullm_lreg_auc:.4f}')
# print(f'Overall accuracy for Pullman logistic model (validation): {pullm_lreg.score(pullm_x_cv, pullm_y_cv):.4f}\n')

#%%
# University logistic
# univr_lreg = LogisticRegression(penalty='elasticnet', class_weight='balanced', solver='saga', max_iter=5000, l1_ratio=0.0, C=1.0, n_jobs=-1, verbose=False).fit(univr_x_train, univr_y_train)

# univr_lreg_probs = univr_lreg.predict_proba(univr_x_train)
# univr_lreg_probs = univr_lreg_probs[:, 1]
# univr_lreg_auc = roc_auc_score(univr_y_train, univr_lreg_probs)

# print(f'Overall accuracy for University logistic model (training): {univr_lreg.score(univr_x_train, univr_y_train):.4f}')
# print(f'ROC AUC for University logistic model (training): {univr_lreg_auc:.4f}')
# print(f'Overall accuracy for University logistic model (validation): {univr_lreg.score(univr_x_cv, univr_y_cv):.4f}\n')

#%%
# Stochastic gradient descent model

# Pullman SGD
# pullm_sgd = SGDClassifier(loss='modified_huber', penalty='elasticnet', class_weight='balanced', early_stopping=False, max_iter=5000, l1_ratio=0.0, learning_rate='adaptive', eta0=0.0001, tol=0.0001, n_iter_no_change=100, n_jobs=-1, verbose=False).fit(pullm_x_train, pullm_y_train)

# pullm_sgd_probs = pullm_sgd.predict_proba(pullm_x_train)
# pullm_sgd_probs = pullm_sgd_probs[:, 1]
# pullm_sgd_auc = roc_auc_score(pullm_y_train, pullm_sgd_probs)

# print(f'Overall accuracy for Pullman SGD model (training): {pullm_sgd.score(pullm_x_train, pullm_y_train):.4f}')
# print(f'ROC AUC for Pullman SGD model (training): {pullm_sgd_auc:.4f}')
# print(f'Overall accuracy for Pullman SGD model (validation): {pullm_sgd.score(pullm_x_cv, pullm_y_cv):.4f}\n')

#%%
# University SGD
# univr_sgd = SGDClassifier(loss='modified_huber', penalty='elasticnet', class_weight='balanced', early_stopping=False, max_iter=5000, l1_ratio=0.0, learning_rate='adaptive', eta0=0.0001, tol=0.0001, n_iter_no_change=100, n_jobs=-1, verbose=False).fit(univr_x_train, univr_y_train)

# univr_sgd_probs = univr_sgd.predict_proba(univr_x_train)
# univr_sgd_probs = univr_sgd_probs[:, 1]
# univr_sgd_auc = roc_auc_score(univr_y_train, univr_sgd_probs)

# print(f'Overall accuracy for University SGD model (training): {univr_sgd.score(univr_x_train, univr_y_train):.4f}')
# print(f'ROC AUC for University SGD model (training): {univr_sgd_auc:.4f}')
# print(f'Overall accuracy for University SGD model (validation): {univr_sgd.score(univr_x_cv, univr_y_cv):.4f}\n')

#%%
# Multi-layer perceptron model

# Pullman MLP
# pullm_mlp = MLPClassifier(hidden_layer_sizes=(75,50,25), activation='relu', solver='sgd', alpha=2.5, learning_rate_init=0.001, n_iter_no_change=25, max_iter=5000, verbose=False).fit(pullm_x_train, pullm_y_train)

# pullm_mlp_probs = pullm_mlp.predict_proba(pullm_x_train)
# pullm_mlp_probs = pullm_mlp_probs[:, 1]
# pullm_mlp_auc = roc_auc_score(pullm_y_train, pullm_mlp_probs)

# print(f'\nOverall accuracy for multi-layer perceptron model (training): {pullm_mlp.score(pullm_x_train, pullm_y_train):.4f}')
# print(f'ROC AUC for multi-layer perceptron model (training): {pullm_mlp_auc:.4f}\n')

#%%
# University MLP
# univr_mlp = MLPClassifier(hidden_layer_sizes=(75,50,25), activation='relu', solver='sgd', alpha=2.5, learning_rate_init=0.001, n_iter_no_change=25, max_iter=5000, verbose=False).fit(univr_x_train, univr_y_train)

# univr_mlp_probs = univr_mlp.predict_proba(univr_x_train)
# univr_mlp_probs = univr_mlp_probs[:, 1]
# univr_mlp_auc = roc_auc_score(univr_y_train, univr_mlp_probs)

# print(f'\nOverall accuracy for University multi-layer perceptron model (training): {univr_mlp.score(univr_x_train, univr_y_train):.4f}')
# print(f'ROC AUC for University multi-layer perceptron model (training): {univr_mlp_auc:.4f}\n')

#%%
# XGBoost model

# Pullman XGBoost tuning
# pullm_class_weight = pullm_y_train[pullm_y_train == 0].count() / pullm_y_train[pullm_y_train == 1].count()
# pullm_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'learning_rate': [0.01, 0.5, 1.0]}]

# pullm_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', scale_pos_weight=pullm_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), pullm_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=False, n_jobs=-1)
# pullm_best_model = pullm_gridsearch.fit(pullm_x_train, pullm_y_train)

# print(f'Best Pullman XGB parameters: {pullm_gridsearch.best_params_}')

#%%
# Pullman XGBoost
# pullm_class_weight = pullm_y_train[pullm_y_train == 0].count() / pullm_y_train[pullm_y_train == 1].count()
# pullm_xgb = XGBClassifier(tree_method='hist', grow_policy='depthwise', scale_pos_weight=pullm_class_weight, 
# 								eval_metric='logloss', **pullm_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(pullm_x_train, pullm_y_train, eval_set=[(pullm_x_cv, pullm_y_cv)], early_stopping_rounds=20, verbose=False)

# pullm_xgb_probs = pullm_xgb.predict_proba(pullm_x_train)
# pullm_xgb_probs = pullm_xgb_probs[:, 1]
# pullm_xgb_auc = roc_auc_score(pullm_y_train, pullm_xgb_probs)

# print(f'Overall accuracy for Pullman XGB model (training): {pullm_xgb.score(pullm_x_train, pullm_y_train):.4f}')
# print(f'ROC AUC for Pullman XGB model (training): {pullm_xgb_auc:.4f}')
# print(f'Overall accuracy for Pullman XGB model (validation): {pullm_xgb.score(pullm_x_cv, pullm_y_cv):.4f}\n')

#%%
# University XGBoost tuning
# univr_class_weight = univr_y_train[univr_y_train == 0].count() / univr_y_train[univr_y_train == 1].count()
# univr_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'learning_rate': [0.01, 0.5, 1.0]}]

# univr_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', scale_pos_weight=univr_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), univr_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=False, n_jobs=-1)
# univr_best_model = univr_gridsearch.fit(univr_x_train, univr_y_train)

# print(f'Best University XGB parameters: {univr_gridsearch.best_params_}')

#%%
# University XGBboost
# univr_class_weight = univr_y_train[univr_y_train == 0].count() / univr_y_train[univr_y_train == 1].count()
# univr_xgb = XGBClassifier(tree_method='hist', grow_policy='depthwise', scale_pos_weight=univr_class_weight, 
# 								eval_metric='logloss', **univr_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(univr_x_train, univr_y_train, eval_set=[(univr_x_cv, univr_y_cv)], early_stopping_rounds=20, verbose=False)

# univr_xgb_probs = univr_xgb.predict_proba(univr_x_train)
# univr_xgb_probs = univr_xgb_probs[:, 1]
# univr_xgb_auc = roc_auc_score(univr_y_train, univr_xgb_probs)

# print(f'Overall accuracy for University XGB model (training): {univr_xgb.score(univr_x_train, univr_y_train):.4f}')
# print(f'ROC AUC for University XGB model (training): {univr_xgb_auc:.4f}')
# print(f'Overall accuracy for University XGB model (validation): {univr_xgb.score(univr_x_cv, univr_y_cv):.4f}\n')

#%%
# Pullman Random Forest tuning
# pullm_class_weight = pullm_y_train[pullm_y_train == 0].count() / pullm_y_train[pullm_y_train == 1].count()
# pullm_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True)}]

# pullm_gridsearch = HalvingGridSearchCV(XGBRFClassifier(tree_method='hist', grow_policy='depthwise', subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pullm_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), pullm_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=False, n_jobs=-1)
# pullm_best_model = pullm_gridsearch.fit(pullm_x_train, pullm_y_train)

# print(f'Best Pullman Random Forest parameters: {pullm_gridsearch.best_params_}')

#%%
# Pullman Random Forest
# pullm_class_weight = pullm_y_train[pullm_y_train == 0].count() / pullm_y_train[pullm_y_train == 1].count()
# pullm_rf = XGBRFClassifier(tree_method='hist', grow_policy='depthwise', subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pullm_class_weight, 
# 								eval_metric='logloss', **pullm_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(pullm_x_train, pullm_y_train, verbose=False)

# pullm_rf_probs = pullm_rf.predict_proba(pullm_x_train)
# pullm_rf_probs = pullm_rf_probs[:, 1]
# pullm_rf_auc = roc_auc_score(pullm_y_train, pullm_rf_probs)

# print(f'Overall accuracy for Pullman Random Forest model (training): {pullm_rf.score(pullm_x_train, pullm_y_train):.4f}')
# print(f'ROC AUC for Pullman Random Forest model (training): {pullm_rf_auc:.4f}')
# print(f'Overall accuracy for Pullman Random Forest model (validation): {pullm_rf.score(pullm_x_cv, pullm_y_cv):.4f}\n')

#%%
# University Random Forest tuning
# univr_class_weight = univr_y_cv[univr_y_cv == 0].count() / univr_y_cv[univr_y_cv == 1].count()
# univr_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True)}]

# univr_gridsearch = HalvingGridSearchCV(XGBRFClassifier(tree_method='hist', grow_policy='depthwise', subsample=0.8, colsample_bytree=0.8, scale_pos_weight=univr_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), univr_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=False, n_jobs=-1)
# univr_best_model = univr_gridsearch.fit(univr_x_cv, univr_y_cv)

# print(f'Best University Random Forest parameters: {univr_gridsearch.best_params_}')

#%%
# University Random Forest
# univr_class_weight = univr_y_train[univr_y_train == 0].count() / univr_y_train[univr_y_train == 1].count()
# univr_rf = XGBRFClassifier(tree_method='hist', grow_policy='depthwise', subsample=0.8, colsample_bytree=0.8, scale_pos_weight=univr_class_weight, 
# 								eval_metric='logloss', **univr_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(univr_x_train, univr_y_train, verbose=False)

# univr_rf_probs = univr_rf.predict_proba(univr_x_train)
# univr_rf_probs = univr_rf_probs[:, 1]
# univr_rf_auc = roc_auc_score(univr_y_train, univr_rf_probs)

# print(f'Overall accuracy for University Random Forest model (training): {univr_rf.score(univr_x_train, univr_y_train):.4f}')
# print(f'ROC AUC for University Random Forest model (training): {univr_rf_auc:.4f}')
# print(f'Overall accuracy for University Random Forest model (validation): {univr_rf.score(univr_x_cv, univr_y_cv):.4f}\n')

#%%
# Pullman XGBoost Random Forest model selection
if build_ft_tr_2yr_prod.DatasetBuilderProd.valid_pass == 0 and build_ft_tr_2yr_prod.DatasetBuilderProd.training_pass == 0:
	pullm_start = time.perf_counter()

	pullm_class_weight = pullm_y_train[pullm_y_train == 0].count() / pullm_y_train[pullm_y_train == 1].count()
	pullm_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
							'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
							'learning_rate': [0.01, 0.5, 1.0]}]

	pullm_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, colsample_bynode=colsample_bynode, scale_pos_weight=pullm_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), pullm_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=verbose, n_jobs=-1)
	pullm_best_model = pullm_gridsearch.fit(pullm_x_train, pullm_y_train, eval_set=[(pullm_x_cv, pullm_y_cv)], early_stopping_rounds=20, verbose=False)

	pullm_stop = time.perf_counter()

	print(f'Pullman XGB Random Forest model trained in {(pullm_stop - pullm_start)/60:.1f} minutes')
	print(f'Best Pullman XGB Random Forest parameters: {pullm_gridsearch.best_params_}')

	pullm_xgbrf = XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, colsample_bynode=colsample_bynode, scale_pos_weight=pullm_class_weight, 
									eval_metric='logloss', **pullm_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(pullm_x_train, pullm_y_train, eval_set=[(pullm_x_cv, pullm_y_cv)], early_stopping_rounds=20, verbose=False)
	
	joblib.dump(pullm_xgbrf, f'Z:\\Nathan\\Models\\student_risk\\models\\pullm_{model_descr}_model_v{sklearn.__version__}.pkl')

else:
	pullm_xgbrf = joblib.load(f'Z:\\Nathan\\Models\\student_risk\\models\\pullm_{model_descr}_model_v{sklearn.__version__}.pkl')

#%%
# Pullman metrics
pullm_xgbrf_train_probs = pullm_xgbrf.predict_proba(pullm_x_train)
pullm_xgbrf_train_probs = pullm_xgbrf_train_probs[:, 1]
pullm_xgbrf_train_auc = roc_auc_score(pullm_y_train, pullm_xgbrf_train_probs)

pullm_xgbrf_cv_probs = pullm_xgbrf.predict_proba(pullm_x_cv)
pullm_xgbrf_cv_probs = pullm_xgbrf_cv_probs[:, 1]
pullm_xgbrf_cv_auc = roc_auc_score(pullm_y_cv, pullm_xgbrf_cv_probs)

print(f'Overall accuracy for Pullman XGB Random Forest model (training): {pullm_xgbrf.score(pullm_x_train, pullm_y_train):.4f}')
print(f'ROC AUC for Pullman XGB Random Forest model (training): {pullm_xgbrf_train_auc:.4f}')
print(f'Overall accuracy for Pullman XGB Random Forest model (validation): {pullm_xgbrf.score(pullm_x_cv, pullm_y_cv):.4f}')
print(f'ROC AUC for Pullman XGB Random Forest model (validation): {pullm_xgbrf_cv_auc:.4f}\n')

# Pullman metrics by sensitive features
pullm_metrics = {
	'accuracy': accuracy_score,
    'tpr': true_positive_rate,
    'tnr': true_negative_rate,
    'balanced': balanced_accuracy_score,
    'matrix': confusion_matrix,
    'headcount': count
}

pullm_group_train = pd.DataFrame()
pullm_group_valid = pd.DataFrame()

pullm_group_train['male'] = pullm_x_train[:, pullm_feat_names.index('male')]
pullm_group_train['underrep_minority'] = pullm_x_train[:, pullm_feat_names.index('underrep_minority')]
pullm_group_valid['male'] = pullm_x_cv[:, pullm_feat_names.index('male')]
pullm_group_valid['underrep_minority'] = pullm_x_cv[:, pullm_feat_names.index('underrep_minority')]

pullm_metric_train_frame = MetricFrame(
    metrics=pullm_metrics, y_true=pullm_y_train, y_pred=pullm_xgbrf.predict(pullm_x_train), sensitive_features=pullm_group_train
)

pullm_metric_valid_frame = MetricFrame(
    metrics=pullm_metrics, y_true=pullm_y_cv, y_pred=pullm_xgbrf.predict(pullm_x_cv), sensitive_features=pullm_group_valid
)

print('Pullman metrics by sensitive features (training)\n')
print(pullm_metric_train_frame.by_group)
print('\n')

print('Pullman metrics by sensitive features (validation)\n')
print(pullm_metric_valid_frame.by_group)
print('\n')

helper_funcs.fairness_output(auto_engine, model_id, 'train', model_descr, pullm_metric_train_frame, run_date, pullm_campus_var)
helper_funcs.fairness_output(auto_engine, model_id, 'valid', model_descr, pullm_metric_valid_frame, run_date, pullm_campus_var)

#%%
# University XGBoost Random Forest model selection
if build_ft_tr_2yr_prod.DatasetBuilderProd.valid_pass == 0 and build_ft_tr_2yr_prod.DatasetBuilderProd.training_pass == 0:
	univr_start = time.perf_counter()

	univr_class_weight = univr_y_train[univr_y_train == 0].count() / univr_y_train[univr_y_train == 1].count()
	univr_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
							'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
							'learning_rate': [0.01, 0.5, 1.0]}]

	univr_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, colsample_bynode=colsample_bynode, scale_pos_weight=univr_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), univr_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=verbose, n_jobs=-1)
	univr_best_model = univr_gridsearch.fit(univr_x_train, univr_y_train, eval_set=[(univr_x_cv, univr_y_cv)], early_stopping_rounds=20, verbose=False)

	univr_stop = time.perf_counter()

	print(f'University XGB Random Forest model trained in {(univr_stop - univr_start)/60:.1f} minutes')
	print(f'Best University XGB Random Forest parameters: {univr_gridsearch.best_params_}')

	univr_xgbrf = XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, colsample_bynode=colsample_bynode, scale_pos_weight=univr_class_weight, 
									eval_metric='logloss', **univr_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(univr_x_train, univr_y_train, eval_set=[(univr_x_cv, univr_y_cv)], early_stopping_rounds=20, verbose=False)

	joblib.dump(univr_xgbrf, f'Z:\\Nathan\\Models\\student_risk\\models\\univr_{model_descr}_model_v{sklearn.__version__}.pkl')

else:
	univr_xgbrf = joblib.load(f'Z:\\Nathan\\Models\\student_risk\\models\\univr_{model_descr}_model_v{sklearn.__version__}.pkl')

#%%
# University metrics
univr_xgbrf_train_probs = univr_xgbrf.predict_proba(univr_x_train)
univr_xgbrf_train_probs = univr_xgbrf_train_probs[:, 1]
univr_xgbrf_train_auc = roc_auc_score(univr_y_train, univr_xgbrf_train_probs)

univr_xgbrf_cv_probs = univr_xgbrf.predict_proba(univr_x_cv)
univr_xgbrf_cv_probs = univr_xgbrf_cv_probs[:, 1]
univr_xgbrf_cv_auc = roc_auc_score(univr_y_cv, univr_xgbrf_cv_probs)

print(f'Overall accuracy for University XGB Random Forest model (training): {univr_xgbrf.score(univr_x_train, univr_y_train):.4f}')
print(f'ROC AUC for University XGB Random Forest model (training): {univr_xgbrf_train_auc:.4f}')
print(f'Overall accuracy for University XGB Random Forest model (validation): {univr_xgbrf.score(univr_x_cv, univr_y_cv):.4f}')
print(f'ROC AUC for University XGB Random Forest model (validation): {univr_xgbrf_cv_auc:.4f}\n')

# University metrics by sensitive features 
univr_metrics = {
	'accuracy': accuracy_score,
    'tpr': true_positive_rate,
    'tnr': true_negative_rate,
    'balanced': balanced_accuracy_score,
    'matrix': confusion_matrix,
    'headcount': count
}

univr_group_train = pd.DataFrame()
univr_group_valid = pd.DataFrame()

univr_group_train['male'] = univr_x_train[:, univr_feat_names.index('male')]
univr_group_train['underrep_minority'] = univr_x_train[:, univr_feat_names.index('underrep_minority')]
univr_group_valid['male'] = univr_x_cv[:, univr_feat_names.index('male')]
univr_group_valid['underrep_minority'] = univr_x_cv[:, univr_feat_names.index('underrep_minority')]

univr_metric_train_frame = MetricFrame(
    metrics=univr_metrics, y_true=univr_y_train, y_pred=univr_xgbrf.predict(univr_x_train), sensitive_features=univr_group_train
)

univr_metric_valid_frame = MetricFrame(
    metrics=univr_metrics, y_true=univr_y_cv, y_pred=univr_xgbrf.predict(univr_x_cv), sensitive_features=univr_group_valid
)

print('University metrics by sensitive features (training)\n')
print(univr_metric_train_frame.by_group)
print('\n')

print('University metrics by sensitive features (validation)\n')
print(univr_metric_valid_frame.by_group)
print('\n')

helper_funcs.fairness_output(auto_engine, model_id, 'train', model_descr, univr_metric_train_frame, run_date, ['UNIVR'])
helper_funcs.fairness_output(auto_engine, model_id, 'valid', model_descr, univr_metric_valid_frame, run_date, ['UNIVR'])

#%%
# Ensemble model

# Pullman VCF
# pullm_vcf = VotingClassifier(estimators=[('lreg', pullm_lreg), ('sgd', pullm_sgd)], voting='soft', weights=[1, 1]).fit(pullm_x_train, pullm_y_train)

# pullm_vcf_probs = pullm_vcf.predict_proba(pullm_x_train)
# pullm_vcf_probs = pullm_vcf_probs[:, 1]
# pullm_vcf_auc = roc_auc_score(pullm_y_train, pullm_vcf_probs)

# print(f'\nOverall accuracy for Pullman ensemble model (training): {pullm_vcf.score(pullm_x_train, pullm_y_train):.4f}')
# print(f'ROC AUC for Pullman ensemble model (training): {pullm_vcf_auc:.4f}\n')

#%%
# University VCF
# univr_vcf = VotingClassifier(estimators=[('lreg', univr_lreg), ('sgd', univr_sgd)], voting='soft', weights=[1, 1]).fit(univr_x_train, univr_y_train)

# univr_vcf_probs = univr_vcf.predict_proba(univr_x_train)
# univr_vcf_probs = univr_vcf_probs[:, 1]
# univr_vcf_auc = roc_auc_score(univr_y_train, univr_vcf_probs)

# print(f'\nOverall accuracy for University ensemble model (training): {univr_vcf.score(univr_x_train, univr_y_train):.4f}')
# print(f'ROC AUC for University ensemble model (training): {univr_vcf_auc:.4f}\n')

#%%
# Calculate SHAP values
# https://github.com/slundberg/shap
print('Calculate SHAP values...')

# Pullman SHAP training
pullm_explainer = shap.TreeExplainer(model=pullm_xgbrf, data=pullm_x_train, model_output='predict_proba')

#%%
# Pullman SHAP prediction
pullm_shap_values = pullm_explainer.shap_values(X=pullm_x_test)

#%%
# Pullman SHAP plots
# 	for index in range(len(pullm_shap_values[0])):
# 		shap.plots._waterfall.waterfall_legacy(pullm_explainer.expected_value[0], pullm_shap_values[0][index], pullm_x_test[index], feature_names=pullm_feat_names, max_display=4)

#%%
pullm_shap_results = []

for index in range(len(pullm_shap_values[0])):
	pullm_shap_results.extend(pd.DataFrame(data=pullm_shap_values[0][index].reshape(1, len(pullm_feat_names)), columns=pullm_feat_names).sort_values(by=0, axis=1, key=abs, ascending=False).to_dict(orient='records'))

pullm_shap_zip = dict(zip(pullm_shap_outcome, pullm_shap_results))

#%%
# University SHAP training
univr_explainer = shap.TreeExplainer(model=univr_xgbrf, data=univr_x_train, model_output='predict_proba')

#%%
# University SHAP prediction
univr_shap_values = univr_explainer.shap_values(X=univr_x_test)

#%%
# University SHAP plots
# 	for index in range(len(univr_shap_values[0])):
# 		shap.plots._waterfall.waterfall_legacy(univr_explainer.expected_value[0], univr_shap_values[0][index], univr_x_test[index], feature_names=univr_feat_names, max_display=4)

#%%
univr_shap_results = []

for index in range(len(univr_shap_values[0])):
	univr_shap_results.extend(pd.DataFrame(data=univr_shap_values[0][index].reshape(1, len(univr_feat_names)), columns=univr_feat_names).sort_values(by=0, axis=1, key=abs, ascending=False).to_dict(orient='records'))

univr_shap_zip = dict(zip(univr_shap_outcome, univr_shap_results))

print('Done\n')

#%%
# Prepare model predictions
print('Prepare model predictions...')

# Pullman probabilites
# pullm_lreg_pred_probs = pullm_lreg.predict_proba(pullm_x_test)
# pullm_lreg_pred_probs = pullm_lreg_pred_probs[:, 1]
# pullm_sgd_pred_probs = pullm_sgd.predict_proba(pullm_x_test)
# pullm_sgd_pred_probs = pullm_sgd_pred_probs[:, 1]
# pullm_xgb_pred_probs = pullm_xgb.predict_proba(pullm_x_test)
# pullm_xgb_pred_probs = pullm_xgb_pred_probs[:, 1]
# pullm_rf_pred_probs = pullm_rf.predict_proba(pullm_x_test)
# pullm_rf_pred_probs = pullm_rf_pred_probs[:, 1]
pullm_xgbrf_pred_probs = pullm_xgbrf.predict_proba(pullm_x_test)
pullm_xgbrf_pred_probs = pullm_xgbrf_pred_probs[:, 1]
# pullm_mlp_pred_probs = pullm_mlp.predict_proba(pullm_x_test)
# pullm_mlp_pred_probs = pullm_mlp_pred_probs[:, 1]
# pullm_vcf_pred_probs = pullm_vcf.predict_proba(pullm_x_test)
# pullm_vcf_pred_probs = pullm_vcf_pred_probs[:, 1]

#%%
# University probabilities
# univr_lreg_pred_probs = univr_lreg.predict_proba(univr_x_test)
# univr_lreg_pred_probs = univr_lreg_pred_probs[:, 1]
# univr_sgd_pred_probs = univr_sgd.predict_proba(univr_x_test)
# univr_sgd_pred_probs = univr_sgd_pred_probs[:, 1]
# univr_xgb_pred_probs = univr_xgb.predict_proba(univr_x_test)
# univr_xgb_pred_probs = univr_xgb_pred_probs[:, 1]
# univr_rf_pred_probs = univr_rf.predict_proba(univr_x_test)
# univr_rf_pred_probs = univr_rf_pred_probs[:, 1]
univr_xgbrf_pred_probs = univr_xgbrf.predict_proba(univr_x_test)
univr_xgbrf_pred_probs = univr_xgbrf_pred_probs[:, 1]
# univr_mlp_pred_probs = univr_mlp.predict_proba(univr_x_test)
# univr_mlp_pred_probs = univr_mlp_pred_probs[:, 1]
# univr_vcf_pred_probs = univr_vcf.predict_proba(univr_x_test)
# univr_vcf_pred_probs = univr_vcf_pred_probs[:, 1]

print('Done\n')

#%%
# Output model predictions to file
print('Output model predictions and model...')

# Pullman predicted outcome
# pullm_pred_outcome['lr_prob'] = pd.DataFrame(pullm_lreg_pred_probs)
# pullm_pred_outcome['lr_pred'] = pullm_lreg.predict(pullm_x_test)
# pullm_pred_outcome['sgd_prob'] = pd.DataFrame(pullm_sgd_pred_probs)
# pullm_pred_outcome['sgd_pred'] = pullm_sgd.predict(pullm_x_test)
# pullm_pred_outcome['xgb_prob'] = pd.DataFrame(pullm_xgb_pred_probs)
# pullm_pred_outcome['xgb_pred'] = pullm_xgb.predict(pullm_x_test)
# pullm_pred_outcome['rf_prob'] = pd.DataFrame(pullm_rf_pred_probs)
# pullm_pred_outcome['rf_pred'] = pullm_rf.predict(pullm_x_test)
pullm_pred_outcome['xgbrf_prob'] = pd.DataFrame(pullm_xgbrf_pred_probs)
pullm_pred_outcome['xgbrf_pred'] = pullm_xgbrf.predict(pullm_x_test)
# pullm_pred_outcome['mlp_prob'] = pd.DataFrame(pullm_mlp_pred_probs)
# pullm_pred_outcome['mlp_pred'] = pullm_mlp.predict(pullm_x_test)
# pullm_pred_outcome['vcf_prob'] = pd.DataFrame(pullm_vcf_pred_probs)
# pullm_pred_outcome['vcf_pred'] = pullm_vcf.predict(pullm_x_test)
pullm_pred_outcome.to_csv(f'Z:\\Nathan\\Models\\student_risk\\predictions\\pullm\\pullm_{model_descr}_pred_outcome.csv', encoding='utf-8', index=False)

#%%
# University predicted outcome
# univr_pred_outcome['lr_prob'] = pd.DataFrame(univr_lreg_pred_probs)
# univr_pred_outcome['lr_pred'] = univr_lreg.predict(univr_x_test)
# univr_pred_outcome['sgd_prob'] = pd.DataFrame(univr_sgd_pred_probs)
# univr_pred_outcome['sgd_pred'] = univr_sgd.predict(univr_x_test)
# univr_pred_outcome['xgb_prob'] = pd.DataFrame(univr_xgb_pred_probs)
# univr_pred_outcome['xgb_pred'] = univr_xgb.predict(univr_x_test)
# univr_pred_outcome['rf_prob'] = pd.DataFrame(univr_rf_pred_probs)
# univr_pred_outcome['rf_pred'] = univr_rf.predict(univr_x_test)
univr_pred_outcome['xgbrf_prob'] = pd.DataFrame(univr_xgbrf_pred_probs)
univr_pred_outcome['xgbrf_pred'] = univr_xgbrf.predict(univr_x_test)
# univr_pred_outcome['mlp_prob'] = pd.DataFrame(univr_mlp_pred_probs)
# univr_pred_outcome['mlp_pred'] = univr_mlp.predict(univr_x_test)
# univr_pred_outcome['vcf_prob'] = pd.DataFrame(univr_vcf_pred_probs)
# univr_pred_outcome['vcf_pred'] = univr_vcf.predict(univr_x_test)
univr_pred_outcome.to_csv(f'Z:\\Nathan\\Models\\student_risk\\predictions\\univr\\univr_{model_descr}_pred_outcome.csv', encoding='utf-8', index=False)

#%%
# Pullman output
helper_funcs.aggregate_outcome(pullm_aggregate_outcome, pullm_xgbrf_pred_probs, 'pullm', model_descr)
helper_funcs.results_output(auto_engine, model_id, run_date, pullm_current_outcome, pullm_xgbrf_pred_probs, 'pullm', model_descr)
helper_funcs.shap_output(engine, student_shap, top_N, model_id, run_date, pullm_shap_zip, 'pullm', model_descr)
joblib.dump(pullm_xgbrf, f'Z:\\Nathan\\Models\\student_risk\\models\\pullm_{model_descr}_model_v{sklearn.__version__}.pkl')

#%%
# University output
helper_funcs.aggregate_outcome(univr_aggregate_outcome, univr_xgbrf_pred_probs, 'univr', model_descr)
helper_funcs.results_output(auto_engine, model_id, run_date, univr_current_outcome, univr_xgbrf_pred_probs, 'univr', model_descr)
helper_funcs.shap_output(engine, student_shap, top_N, model_id, run_date, univr_shap_zip, 'univr', model_descr)
joblib.dump(univr_xgbrf, f'Z:\\Nathan\\Models\\student_risk\\models\\univr_{model_descr}_model_v{sklearn.__version__}.pkl')

print('Done\n')