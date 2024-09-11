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
from student_risk import build_ft_ft_2yr_prod, config, helper_funcs

#%%
# Database connection
cred = pathlib.Path('Z:\\Nathan\\Models\\student_risk\\login.bin').read_text().split('|')
params = urllib.parse.quote_plus(f'TRUSTED_CONNECTION=YES; DRIVER={{SQL Server Native Client 11.0}}; SERVER={cred[0]}; DATABASE={cred[1]}')
engine = sqlalchemy.create_engine(f'mssql+pyodbc:///?odbc_connect={params}')
auto_engine = engine.execution_options(autocommit=True, isolation_level='AUTOCOMMIT')
metadata_engine = MetaData(engine.execution_options(autocommit=True, isolation_level='AUTOCOMMIT'))
student_shap = Table('student_shap', metadata_engine, autoload=True)

#%%
# Global variable initialization
strm: str = None
outcome: str = 'year'
top_N: int = 5
model_id: int = 2
model_descr: str = 'ft_ft_2yr'
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
build_ft_ft_2yr_prod.DatasetBuilderProd.build_census_prod(outcome)

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
# 'marital_status',
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
# 'high_school_gpa_mi',
# 'fall_cum_gpa',
'cum_gpa',
'cum_gpa_hours',
# 'spring_midterm_gpa_change',
# 'awe_instrument',
# 'cdi_instrument',
'fall_avg_difficulty',
'fall_avg_pct_withdrawn',
# 'fall_avg_pct_CDFW',
'fall_avg_pct_CDF',
# 'fall_avg_pct_DFW',
# 'fall_avg_pct_DF',
# 'fall_crse_mi',
'fall_lec_count',
'fall_lab_count',
# 'fall_int_count',
'fall_stu_count',
# 'fall_sem_count',
'fall_oth_count',
# 'fall_lec_contact_hrs',
# 'fall_lab_contact_hrs',
# 'fall_int_contact_hrs',
# 'fall_stu_contact_hrs',
# 'fall_sem_contact_hrs',
# 'fall_oth_contact_hrs',
# 'total_fall_contact_hrs',
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
# 'term_credit_hours',
'total_fall_units',
'fall_withdrawn_hours',
'fall_withdrawn_ind',
# 'athlete',
# 'remedial',
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
# Vancouver variables
vanco_data_vars = [
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
# 'marital_status',
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
# 'high_school_gpa_mi',
# 'fall_cum_gpa',
'cum_gpa',
'cum_gpa_hours',
# 'spring_midterm_gpa_change',
# 'awe_instrument',
# 'cdi_instrument',
'fall_avg_difficulty',
'fall_avg_pct_withdrawn',
# 'fall_avg_pct_CDFW',
'fall_avg_pct_CDF',
# 'fall_avg_pct_DFW',
# 'fall_avg_pct_DF',
# 'fall_crse_mi',
'fall_lec_count',
'fall_lab_count',
# 'fall_int_count',
# 'fall_stu_count',
# 'fall_sem_count',
# 'fall_oth_count',
# 'fall_lec_contact_hrs',
# 'fall_lab_contact_hrs',
# 'fall_int_contact_hrs',
# 'fall_stu_contact_hrs',
# 'fall_sem_contact_hrs',
# 'fall_oth_contact_hrs',
# 'total_fall_contact_hrs',
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
# 'term_credit_hours',
'total_fall_units',
'fall_withdrawn_hours',
'fall_withdrawn_ind',
# 'athlete',
# 'remedial',
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

vanco_campus_var = ['VANCO']

vanco_logit_df: pd.DataFrame
vanco_validation_set: pd.DataFrame
vanco_training_set: pd.DataFrame
vanco_testing_set: pd.DataFrame
vanco_shap_outcome: list
vanco_pred_outcome: pd.DataFrame
vanco_aggregate_outcome: pd.DataFrame
vanco_current_outcome: pd.DataFrame

vanco_x_vars = [x for x in vanco_data_vars if x not in unwanted_vars]

# Vancouver dataframes
vanco_logit_df, vanco_validation_set, vanco_training_set, vanco_testing_set, vanco_shap_outcome, vanco_pred_outcome, vanco_aggregate_outcome, vanco_current_outcome = helper_funcs.prep_campus_dataframe(validation_set, training_set, testing_set, vanco_data_vars, vanco_campus_var)

#%%
# Tri-Cities variables
trici_data_vars = [
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
# 'marital_status',
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
# 'high_school_gpa_mi',
# 'fall_cum_gpa',
'cum_gpa',
'cum_gpa_hours',
# 'spring_midterm_gpa_change',
# 'awe_instrument',
# 'cdi_instrument',
'fall_avg_difficulty',
'fall_avg_pct_withdrawn',
# 'fall_avg_pct_CDFW',
'fall_avg_pct_CDF',
# 'fall_avg_pct_DFW',
# 'fall_avg_pct_DF',
# 'fall_crse_mi',
'fall_lec_count',
'fall_lab_count',
# 'fall_int_count',
# 'fall_stu_count',
# 'fall_sem_count',
# 'fall_oth_count',
# 'fall_lec_contact_hrs',
# 'fall_lab_contact_hrs',
# 'fall_int_contact_hrs',
# 'fall_stu_contact_hrs',
# 'fall_sem_contact_hrs',
# 'fall_oth_contact_hrs',
# 'total_fall_contact_hrs',
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
# 'term_credit_hours',
'total_fall_units',
'fall_withdrawn_hours',
'fall_withdrawn_ind',
# 'athlete',
# 'remedial',
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

trici_campus_var = ['TRICI']

trici_logit_df: pd.DataFrame
trici_validation_set: pd.DataFrame
trici_training_set: pd.DataFrame
trici_testing_set: pd.DataFrame
trici_shap_outcome: list
trici_pred_outcome: pd.DataFrame
trici_aggregate_outcome: pd.DataFrame
trici_current_outcome: pd.DataFrame

trici_x_vars = [x for x in trici_data_vars if x not in unwanted_vars]

# Tri-Cities dataframes
trici_logit_df, trici_validation_set, trici_training_set, trici_testing_set, trici_shap_outcome, trici_pred_outcome, trici_aggregate_outcome, trici_current_outcome = helper_funcs.prep_campus_dataframe(validation_set, training_set, testing_set, trici_data_vars, trici_campus_var)

#%%
# University dataframes
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
# 'marital_status',
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
# 'high_school_gpa_mi',
# 'fall_cum_gpa',
'cum_gpa',
'cum_gpa_hours',
# 'spring_midterm_gpa_change',
# 'awe_instrument',
# 'cdi_instrument',
'fall_avg_difficulty',
'fall_avg_pct_withdrawn',
# 'fall_avg_pct_CDFW',
'fall_avg_pct_CDF',
# 'fall_avg_pct_DFW',
# 'fall_avg_pct_DF',
# 'fall_crse_mi',
'fall_lec_count',
'fall_lab_count',
# 'fall_int_count',
# 'fall_stu_count',
# 'fall_sem_count',
# 'fall_oth_count',
# 'fall_lec_contact_hrs',
# 'fall_lab_contact_hrs',
# 'fall_int_contact_hrs',
# 'fall_stu_contact_hrs',
# 'fall_sem_contact_hrs',
# 'fall_oth_contact_hrs',
# 'total_fall_contact_hrs',
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
# 'term_credit_hours',
'total_fall_units',
'fall_withdrawn_hours',
'fall_withdrawn_ind',
# 'athlete',
# 'remedial',
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

univr_system_var = ['EVERE','TRICI','SPOKA','ONLIN']

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
# Vancouver outliers
vanco_x_training_outlier = vanco_training_set.drop(columns=['enrl_ind','emplid'])
vanco_x_validation_outlier = vanco_validation_set.drop(columns=['enrl_ind','emplid'])

vanco_onehot_vars = vanco_x_training_outlier.select_dtypes(include='object').columns.tolist()

vanco_outlier_prep = make_column_transformer(
	(OneHotEncoder(drop='first'), vanco_onehot_vars),
	remainder='passthrough'
)

vanco_x_training_outlier = vanco_outlier_prep.fit_transform(vanco_x_training_outlier)
vanco_x_validation_outlier = vanco_outlier_prep.transform(vanco_x_validation_outlier)

vanco_validation_set, vanco_training_set = helper_funcs.remove_outliers(vanco_validation_set, vanco_training_set, vanco_x_training_outlier, vanco_x_validation_outlier, 'vanco', model_descr)

#%%
# Tri-Cities outliers
trici_x_training_outlier = trici_training_set.drop(columns=['enrl_ind','emplid'])
trici_x_validation_outlier = trici_validation_set.drop(columns=['enrl_ind','emplid'])

trici_onehot_vars = trici_x_training_outlier.select_dtypes(include='object').columns.tolist()

trici_outlier_prep = make_column_transformer(
	(OneHotEncoder(drop='first'), trici_onehot_vars),
	remainder='passthrough'
)

trici_x_training_outlier = trici_outlier_prep.fit_transform(trici_x_training_outlier)
trici_x_validation_outlier = trici_outlier_prep.transform(trici_x_validation_outlier)

trici_validation_set, trici_training_set = helper_funcs.remove_outliers(trici_validation_set, trici_training_set, trici_x_training_outlier, trici_x_validation_outlier, 'trici', model_descr)

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
# Vancouver undersample
vanco_x_train = vanco_training_set.drop(columns=['enrl_ind','emplid'])
vanco_x_cv = vanco_validation_set.drop(columns=['enrl_ind','emplid'])

vanco_x_test = vanco_testing_set[vanco_x_vars]

vanco_y_train = vanco_training_set['enrl_ind']
vanco_y_cv = vanco_validation_set['enrl_ind']
# vanco_y_test = vanco_testing_set['enrl_ind']

vanco_binary_vars = vanco_x_train.columns[vanco_x_train.isin([0,1]).all()].to_list()
vanco_filter_vars = vanco_binary_vars + vanco_onehot_vars
vanco_centered_vars = [b for b in vanco_x_vars if all(a not in b for a in vanco_filter_vars)]

vanco_tomek_prep = make_column_transformer(
	# (StandardScaler(), vanco_centered_vars),
	(OneHotEncoder(drop='first'), vanco_onehot_vars),
	remainder='passthrough'
)

vanco_x_train = vanco_tomek_prep.fit_transform(vanco_x_train)
vanco_x_cv = vanco_tomek_prep.transform(vanco_x_cv)
vanco_x_test = vanco_tomek_prep.transform(vanco_x_test)

vanco_feat_names: list = []

vanco_x_train, vanco_x_cv, vanco_y_train, vanco_y_cv = helper_funcs.tomek_undersample(vanco_validation_set, vanco_training_set, vanco_x_train, vanco_x_cv, vanco_y_train, vanco_y_cv, vanco_tomek_prep, vanco_feat_names, 'vanco', model_descr)

#%%
# Tri-Cities undersample
trici_x_train = trici_training_set.drop(columns=['enrl_ind','emplid'])
trici_x_cv = trici_validation_set.drop(columns=['enrl_ind','emplid'])

trici_x_test = trici_testing_set[trici_x_vars]

trici_y_train = trici_training_set['enrl_ind']
trici_y_cv = trici_validation_set['enrl_ind']
# trici_y_test = trici_testing_set['enrl_ind']

trici_binary_vars = trici_x_train.columns[trici_x_train.isin([0,1]).all()].to_list()
trici_filter_vars = trici_binary_vars + trici_onehot_vars
trici_centered_vars = [b for b in trici_x_vars if all(a not in b for a in trici_filter_vars)]

trici_tomek_prep = make_column_transformer(
	# (StandardScaler(), trici_centered_vars),
	(OneHotEncoder(drop='first'), trici_onehot_vars),
	remainder='passthrough'
)

trici_x_train = trici_tomek_prep.fit_transform(trici_x_train)
trici_x_cv = trici_tomek_prep.transform(trici_x_cv)
trici_x_test = trici_tomek_prep.transform(trici_x_test)

trici_feat_names: list = []

trici_x_train, trici_x_cv, trici_y_train, trici_y_cv = helper_funcs.tomek_undersample(trici_validation_set, trici_training_set, trici_x_train, trici_x_cv, trici_y_train, trici_y_cv, trici_tomek_prep, trici_feat_names, 'trici', model_descr)

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
print('\nStandard logistic model for Pullman sophomores...\n')

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
# Vancouver standard model
print('\nStandard logistic model for Vancouver sophomores...\n')

try:
	vanco_y, vanco_x = dmatrices('enrl_ind ~ ' + ' + '.join(vanco_x_vars), data=vanco_logit_df, return_type='dataframe')

	vanco_logit_mod = Logit(vanco_y, vanco_x)
	vanco_logit_res = vanco_logit_mod.fit(maxiter=500, method='bfgs')
	print(vanco_logit_res.summary())

	# Vancouver VIF
	print('\nVIF for Vancouver...\n')
	vanco_vif = pd.DataFrame()
	vanco_vif['vif factor'] = [variance_inflation_factor(vanco_x.values, i) for i in range(vanco_x.shape[1])]
	vanco_vif['features'] = vanco_x.columns
	vanco_vif.sort_values(by=['vif factor'], ascending=False, inplace=True, ignore_index=True)
	print(vanco_vif.round(1).to_string())
	print('\n')

except:
	print('\nFailed to converge or model misspecification: Linear combination, singular matrix, divide by zero, or separation')

print('\n')

#%%
# Tri-Cities standard model
print('\nStandard logistic model for Tri-Cities sophomores...\n')

try:
	trici_y, trici_x = dmatrices('enrl_ind ~ ' + ' + '.join(trici_x_vars), data=trici_logit_df, return_type='dataframe')

	trici_logit_mod = Logit(trici_y, trici_x)
	trici_logit_res = trici_logit_mod.fit(maxiter=500, method='bfgs')
	print(trici_logit_res.summary())

	# Tri-Cities VIF
	print('\nVIF for Tri-Cities...\n')
	trici_vif = pd.DataFrame()
	trici_vif['vif factor'] = [variance_inflation_factor(trici_x.values, i) for i in range(trici_x.shape[1])]
	trici_vif['features'] = trici_x.columns
	trici_vif.sort_values(by=['vif factor'], ascending=False, inplace=True, ignore_index=True)
	print(trici_vif.round(1).to_string())
	print('\n')
	
except:
	print('Failed to converge or model misspecification: Linear combination, singular matrix, divide by zero, or separation\n')

print('\n')

#%%
# University standard model
print('\nStandard logistic model for University sophomores...\n')

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
print('Run machine learning models for sophomores...\n')

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
# Vancouver logistic
# vanco_lreg = LogisticRegression(penalty='elasticnet', class_weight='balanced', solver='saga', max_iter=5000, l1_ratio=0.0, C=1.0, n_jobs=-1, verbose=False).fit(vanco_x_train, vanco_y_train)

# vanco_lreg_probs = vanco_lreg.predict_proba(vanco_x_train)
# vanco_lreg_probs = vanco_lreg_probs[:, 1]
# vanco_lreg_auc = roc_auc_score(vanco_y_train, vanco_lreg_probs)

# print(f'Overall accuracy for Vancouver logistic model (training): {vanco_lreg.score(vanco_x_train, vanco_y_train):.4f}')
# print(f'ROC AUC for Vancouver logistic model (training): {vanco_lreg_auc:.4f}')
# print(f'Overall accuracy for Vancouver logistic model (validation): {vanco_lreg.score(vanco_x_cv, vanco_y_cv):.4f}\n')

#%%
# Tri-Cities logistic
# trici_lreg = LogisticRegression(penalty='elasticnet', class_weight='balanced', solver='saga', max_iter=5000, l1_ratio=0.0, C=1.0, n_jobs=-1, verbose=False).fit(trici_x_train, trici_y_train)

# trici_lreg_probs = trici_lreg.predict_proba(trici_x_train)
# trici_lreg_probs = trici_lreg_probs[:, 1]
# trici_lreg_auc = roc_auc_score(trici_y_train, trici_lreg_probs)

# print(f'Overall accuracy for Tri-Cities logistic model (training): {trici_lreg.score(trici_x_train, trici_y_train):.4f}')
# print(f'ROC AUC for Tri-Cities logistic model (training): {trici_lreg_auc:.4f}')
# print(f'Overall accuracy for Tri-Cities logistic model (validation): {trici_lreg.score(trici_x_cv, trici_y_cv):.4f}\n')

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
# Vancouver SGD
# vanco_sgd = SGDClassifier(loss='modified_huber', penalty='elasticnet', class_weight='balanced', early_stopping=False, max_iter=5000, l1_ratio=0.0, learning_rate='adaptive', eta0=0.0001, tol=0.0001, n_iter_no_change=100, n_jobs=-1, verbose=False).fit(vanco_x_train, vanco_y_train)

# vanco_sgd_probs = vanco_sgd.predict_proba(vanco_x_train)
# vanco_sgd_probs = vanco_sgd_probs[:, 1]
# vanco_sgd_auc = roc_auc_score(vanco_y_train, vanco_sgd_probs)

# print(f'Overall accuracy for Vancouver SGD model (training): {vanco_sgd.score(vanco_x_train, vanco_y_train):.4f}')
# print(f'ROC AUC for Vancouver SGD model (training): {vanco_sgd_auc:.4f}')
# print(f'Overall accuracy for Vancouver SGD model (validation): {vanco_sgd.score(vanco_x_cv, vanco_y_cv):.4f}\n')

#%%
# Tri-Cities SGD
# trici_sgd = SGDClassifier(loss='modified_huber', penalty='elasticnet', class_weight='balanced', early_stopping=False, max_iter=5000, l1_ratio=0.0, learning_rate='adaptive', eta0=0.0001, tol=0.0001, n_iter_no_change=100, n_jobs=-1, verbose=False).fit(trici_x_train, trici_y_train)

# trici_sgd_probs = trici_sgd.predict_proba(trici_x_train)
# trici_sgd_probs = trici_sgd_probs[:, 1]
# trici_sgd_auc = roc_auc_score(trici_y_train, trici_sgd_probs)

# print(f'Overall accuracy for Tri-Cities SGD model (training): {trici_sgd.score(trici_x_train, trici_y_train):.4f}')
# print(f'ROC AUC for Tri-Cities SGD model (training): {trici_sgd_auc:.4f}')
# print(f'Overall accuracy for Tri-Cities SGD model (validation): {trici_sgd.score(trici_x_cv, trici_y_cv):.4f}\n')

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
# Vancouver MLP
# vanco_mlp = MLPClassifier(hidden_layer_sizes=(75,50,25), activation='relu', solver='sgd', alpha=2.5, learning_rate_init=0.001, n_iter_no_change=25, max_iter=5000, verbose=False).fit(vanco_x_train, vanco_y_train)

# vanco_mlp_probs = vanco_mlp.predict_proba(vanco_x_train)
# vanco_mlp_probs = vanco_mlp_probs[:, 1]
# vanco_mlp_auc = roc_auc_score(vanco_y_train, vanco_mlp_probs)

# print(f'\nOverall accuracy for multi-layer perceptron model (training): {vanco_mlp.score(vanco_x_train, vanco_y_train):.4f}')
# print(f'ROC AUC for multi-layer perceptron model (training): {vanco_mlp_auc:.4f}\n')

#%%
# Tri-Cities MLP
# trici_mlp = MLPClassifier(hidden_layer_sizes=(75,50,25), activation='relu', solver='sgd', alpha=2.5, learning_rate_init=0.001, n_iter_no_change=25, max_iter=5000, verbose=False).fit(trici_x_train, trici_y_train)

# trici_mlp_probs = trici_mlp.predict_proba(trici_x_train)
# trici_mlp_probs = trici_mlp_probs[:, 1]
# trici_mlp_auc = roc_auc_score(trici_y_train, trici_mlp_probs)

# print(f'\nOverall accuracy for multi-layer perceptron model (training): {trici_mlp.score(trici_x_train, trici_y_train):.4f}')
# print(f'ROC AUC for multi-layer perceptron model (training): {trici_mlp_auc:.4f}\n')

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

# pullm_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', scale_pos_weight=pullm_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), pullm_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=verbose, n_jobs=-1)
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
# Vancouver XGBoost tuning
# vanco_class_weight = vanco_y_train[vanco_y_train == 0].count() / vanco_y_train[vanco_y_train == 1].count()
# vanco_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'learning_rate': [0.01, 0.5, 1.0]}]

# vanco_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', scale_pos_weight=vanco_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), vanco_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=verbose, n_jobs=-1)
# vanco_best_model = vanco_gridsearch.fit(vanco_x_train, vanco_y_train)

# print(f'Best Vancouver XGB parameters: {vanco_gridsearch.best_params_}')

#%%
# Vancouver XGBoost
# vanco_class_weight = vanco_y_train[vanco_y_train == 0].count() / vanco_y_train[vanco_y_train == 1].count()
# vanco_xgb = XGBClassifier(tree_method='hist', grow_policy='depthwise', scale_pos_weight=vanco_class_weight, 
# 								eval_metric='logloss', **vanco_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(vanco_x_train, vanco_y_train, eval_set=[(vanco_x_cv, vanco_y_cv)], early_stopping_rounds=20, verbose=False)

# vanco_xgb_probs = vanco_xgb.predict_proba(vanco_x_train)
# vanco_xgb_probs = vanco_xgb_probs[:, 1]
# vanco_xgb_auc = roc_auc_score(vanco_y_train, vanco_xgb_probs)

# print(f'Overall accuracy for Vancouver XGB model (training): {vanco_xgb.score(vanco_x_train, vanco_y_train):.4f}')
# print(f'ROC AUC for Vancouver XGB model (training): {vanco_xgb_auc:.4f}')
# print(f'Overall accuracy for Vancouver XGB model (validation): {vanco_xgb.score(vanco_x_cv, vanco_y_cv):.4f}\n')

#%%
# Tri-Cities XGBoost tuning
# trici_class_weight = trici_y_train[trici_y_train == 0].count() / trici_y_train[trici_y_train == 1].count()
# trici_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'learning_rate': [0.01, 0.5, 1.0]}]

# trici_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', scale_pos_weight=trici_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), trici_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=verbose, n_jobs=-1)
# trici_best_model = trici_gridsearch.fit(trici_x_train, trici_y_train)

# print(f'Best Tri-Cities XGB parameters: {trici_gridsearch.best_params_}')

#%%
# Tri-Cities XGBoost
# trici_class_weight = trici_y_train[trici_y_train == 0].count() / trici_y_train[trici_y_train == 1].count()
# trici_xgb = XGBClassifier(tree_method='hist', grow_policy='depthwise', scale_pos_weight=trici_class_weight, 
# 								eval_metric='logloss', **trici_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(trici_x_train, trici_y_train, eval_set=[(trici_x_cv, trici_y_cv)], early_stopping_rounds=20, verbose=False)

# trici_xgb_probs = trici_xgb.predict_proba(trici_x_train)
# trici_xgb_probs = trici_xgb_probs[:, 1]
# trici_xgb_auc = roc_auc_score(trici_y_train, trici_xgb_probs)

# print(f'Overall accuracy for Tri-Cities XGB model (training): {trici_xgb.score(trici_x_train, trici_y_train):.4f}')
# print(f'ROC AUC for Tri-Cities XGB model (training): {trici_xgb_auc:.4f}')
# print(f'Overall accuracy for Tri-Cities XGB model (validation): {trici_xgb.score(trici_x_cv, trici_y_cv):.4f}\n')

#%%
# University XGBoost tuning
# univr_class_weight = univr_y_train[univr_y_train == 0].count() / univr_y_train[univr_y_train == 1].count()
# univr_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'learning_rate': [0.01, 0.5, 1.0]}]

# univr_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', scale_pos_weight=univr_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), univr_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=verbose, n_jobs=-1)
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

# pullm_gridsearch = HalvingGridSearchCV(XGBRFClassifier(tree_method='hist', grow_policy='depthwise', subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pullm_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), pullm_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=verbose, n_jobs=-1)
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
# Vancouver Random Forest tuning
# vanco_class_weight = vanco_y_train[vanco_y_train == 0].count() / vanco_y_train[vanco_y_train == 1].count()
# vanco_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True)}]

# vanco_gridsearch = HalvingGridSearchCV(XGBRFClassifier(tree_method='hist', grow_policy='depthwise', subsample=0.8, colsample_bytree=0.8, scale_pos_weight=vanco_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), vanco_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=verbose, n_jobs=-1)
# vanco_best_model = vanco_gridsearch.fit(vanco_x_train, vanco_y_train)

# print(f'Best Vancouver Random Forest parameters: {vanco_gridsearch.best_params_}')

#%%
# Vancouver Random Forest
# vanco_class_weight = vanco_y_train[vanco_y_train == 0].count() / vanco_y_train[vanco_y_train == 1].count()
# vanco_rf = XGBRFClassifier(tree_method='hist', grow_policy='depthwise', subsample=0.8, colsample_bytree=0.8, scale_pos_weight=vanco_class_weight, 
# 								eval_metric='logloss', **vanco_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(vanco_x_train, vanco_y_train, verbose=False)

# vanco_rf_probs = vanco_rf.predict_proba(vanco_x_train)
# vanco_rf_probs = vanco_rf_probs[:, 1]
# vanco_rf_auc = roc_auc_score(vanco_y_train, vanco_rf_probs)

# print(f'Overall accuracy for Vancouver Random Forest model (training): {vanco_rf.score(vanco_x_train, vanco_y_train):.4f}')
# print(f'ROC AUC for Vancouver Random Forest model (training): {vanco_rf_auc:.4f}')
# print(f'Overall accuracy for Vancouver Random Forest model (validation): {vanco_rf.score(vanco_x_cv, vanco_y_cv):.4f}\n')

#%%
# Tri-Cities Random Forest tuning
# trici_class_weight = trici_y_cv[trici_y_cv == 0].count() / trici_y_cv[trici_y_cv == 1].count()
# trici_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True)}]

# trici_gridsearch = HalvingGridSearchCV(XGBRFClassifier(tree_method='hist', grow_policy='depthwise', subsample=0.8, colsample_bytree=0.8, scale_pos_weight=trici_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), trici_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=verbose, n_jobs=-1)
# trici_best_model = trici_gridsearch.fit(trici_x_cv, trici_y_cv)

# print(f'Best Tri-Cities Random Forest parameters: {trici_gridsearch.best_params_}')

#%%
# Tri-Cities Random Forest
# trici_class_weight = trici_y_train[trici_y_train == 0].count() / trici_y_train[trici_y_train == 1].count()
# trici_rf = XGBRFClassifier(tree_method='hist', grow_policy='depthwise', subsample=0.8, colsample_bytree=0.8, scale_pos_weight=trici_class_weight, 
# 								eval_metric='logloss', **trici_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(trici_x_train, trici_y_train, verbose=False)

# trici_rf_probs = trici_rf.predict_proba(trici_x_train)
# trici_rf_probs = trici_rf_probs[:, 1]
# trici_rf_auc = roc_auc_score(trici_y_train, trici_rf_probs)

# print(f'Overall accuracy for Tri-Cities Random Forest model (training): {trici_rf.score(trici_x_train, trici_y_train):.4f}')
# print(f'ROC AUC for Tri-Cities Random Forest model (training): {trici_rf_auc:.4f}')
# print(f'Overall accuracy for Tri-Cities Random Forest model (validation): {trici_rf.score(trici_x_cv, trici_y_cv):.4f}\n')

#%%
# University Random Forest tuning
# univr_class_weight = univr_y_cv[univr_y_cv == 0].count() / univr_y_cv[univr_y_cv == 1].count()
# univr_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
# 						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True)}]

# univr_gridsearch = HalvingGridSearchCV(XGBRFClassifier(tree_method='hist', grow_policy='depthwise', subsample=0.8, colsample_bytree=0.8, scale_pos_weight=univr_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), univr_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=verbose, n_jobs=-1)
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
if build_ft_ft_2yr_prod.DatasetBuilderProd.valid_pass == 0 and build_ft_ft_2yr_prod.DatasetBuilderProd.training_pass == 0:
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
# Vancouver XGBoost Random Forest model selection
if build_ft_ft_2yr_prod.DatasetBuilderProd.valid_pass == 0 and build_ft_ft_2yr_prod.DatasetBuilderProd.training_pass == 0:
	vanco_start = time.perf_counter()

	vanco_class_weight = vanco_y_train[vanco_y_train == 0].count() / vanco_y_train[vanco_y_train == 1].count()
	vanco_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
							'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
							'learning_rate': [0.01, 0.5, 1.0]}]

	vanco_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, colsample_bynode=colsample_bynode, scale_pos_weight=vanco_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), vanco_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=verbose, n_jobs=-1)
	vanco_best_model = vanco_gridsearch.fit(vanco_x_train, vanco_y_train, eval_set=[(vanco_x_cv, vanco_y_cv)], early_stopping_rounds=20, verbose=False)

	vanco_stop = time.perf_counter()

	print(f'Vancouver XGB Random Forest model trained in {(vanco_stop - vanco_start)/60:.1f} minutes')
	print(f'Best Vancouver XGB Random Forest parameters: {vanco_gridsearch.best_params_}')

	vanco_xgbrf = XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, colsample_bynode=colsample_bynode, scale_pos_weight=vanco_class_weight, 
									eval_metric='logloss', **vanco_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(vanco_x_train, vanco_y_train, eval_set=[(vanco_x_cv, vanco_y_cv)], early_stopping_rounds=20, verbose=False)

	joblib.dump(vanco_xgbrf, f'Z:\\Nathan\\Models\\student_risk\\models\\vanco_{model_descr}_model_v{sklearn.__version__}.pkl')

else:
	vanco_xgbrf = joblib.load(f'Z:\\Nathan\\Models\\student_risk\\models\\vanco_{model_descr}_model_v{sklearn.__version__}.pkl')

#%%
# Vancouver metrics
vanco_xgbrf_train_probs = vanco_xgbrf.predict_proba(vanco_x_train)
vanco_xgbrf_train_probs = vanco_xgbrf_train_probs[:, 1]
vanco_xgbrf_train_auc = roc_auc_score(vanco_y_train, vanco_xgbrf_train_probs)

vanco_xgbrf_cv_probs = vanco_xgbrf.predict_proba(vanco_x_cv)
vanco_xgbrf_cv_probs = vanco_xgbrf_cv_probs[:, 1]
vanco_xgbrf_cv_auc = roc_auc_score(vanco_y_cv, vanco_xgbrf_cv_probs)

print(f'Overall accuracy for Vancouver XGB Random Forest model (training): {vanco_xgbrf.score(vanco_x_train, vanco_y_train):.4f}')
print(f'ROC AUC for Vancouver XGB Random Forest model (training): {vanco_xgbrf_train_auc:.4f}')
print(f'Overall accuracy for Vancouver XGB Random Forest model (validation): {vanco_xgbrf.score(vanco_x_cv, vanco_y_cv):.4f}')
print(f'ROC AUC for Vancouver XGB Random Forest model (validation): {vanco_xgbrf_cv_auc:.4f}\n')

# Vancouver metrics by sensitive features
vanco_metrics = {
	'accuracy': accuracy_score,
    'tpr': true_positive_rate,
    'tnr': true_negative_rate,
    'balanced': balanced_accuracy_score,
    'matrix': confusion_matrix,
    'headcount': count
}

vanco_group_train = pd.DataFrame()
vanco_group_valid = pd.DataFrame()

vanco_group_train['male'] = vanco_x_train[:, vanco_feat_names.index('male')]
vanco_group_train['underrep_minority'] = vanco_x_train[:, vanco_feat_names.index('underrep_minority')]
vanco_group_valid['male'] = vanco_x_cv[:, vanco_feat_names.index('male')]
vanco_group_valid['underrep_minority'] = vanco_x_cv[:, vanco_feat_names.index('underrep_minority')]

vanco_metric_train_frame = MetricFrame(
    metrics=vanco_metrics, y_true=vanco_y_train, y_pred=vanco_xgbrf.predict(vanco_x_train), sensitive_features=vanco_group_train
)

vanco_metric_valid_frame = MetricFrame(
    metrics=vanco_metrics, y_true=vanco_y_cv, y_pred=vanco_xgbrf.predict(vanco_x_cv), sensitive_features=vanco_group_valid
)

print('Vancouver metrics by sensitive features (training)\n')
print(vanco_metric_train_frame.by_group)
print('\n')

print('Vancouver metrics by sensitive features (validation)\n')
print(vanco_metric_valid_frame.by_group)
print('\n')

helper_funcs.fairness_output(auto_engine, model_id, 'train', model_descr, vanco_metric_train_frame, run_date, vanco_campus_var)
helper_funcs.fairness_output(auto_engine, model_id, 'valid', model_descr, vanco_metric_valid_frame, run_date, vanco_campus_var)

#%%
# Tri-Cities XGBoost Random Forest model selection
if build_ft_ft_2yr_prod.DatasetBuilderProd.valid_pass == 0 and build_ft_ft_2yr_prod.DatasetBuilderProd.training_pass == 0:
	trici_start = time.perf_counter()

	trici_class_weight = trici_y_train[trici_y_train == 0].count() / trici_y_train[trici_y_train == 1].count()
	trici_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
							'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
							'learning_rate': [0.01, 0.5, 1.0]}]

	trici_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, colsample_bynode=colsample_bynode, scale_pos_weight=trici_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), trici_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=verbose, n_jobs=-1)
	trici_best_model = trici_gridsearch.fit(trici_x_train, trici_y_train, eval_set=[(trici_x_cv, trici_y_cv)], early_stopping_rounds=20, verbose=False)

	trici_stop = time.perf_counter()

	print(f'Tri-Cities XGB Random Forest model trained in {(trici_stop - trici_start)/60:.1f} minutes')
	print(f'Best Tri-Cities XGB Random Forest parameters: {trici_gridsearch.best_params_}')

	trici_xgbrf = XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, colsample_bynode=colsample_bynode, scale_pos_weight=trici_class_weight, 
									eval_metric='logloss', **trici_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(trici_x_train, trici_y_train, eval_set=[(trici_x_cv, trici_y_cv)], early_stopping_rounds=20, verbose=False)

	joblib.dump(trici_xgbrf, f'Z:\\Nathan\\Models\\student_risk\\models\\trici_{model_descr}_model_v{sklearn.__version__}.pkl')

else:
	trici_xgbrf = joblib.load(f'Z:\\Nathan\\Models\\student_risk\\models\\trici_{model_descr}_model_v{sklearn.__version__}.pkl')

#%%
# Tri-Cities metrics
trici_xgbrf_train_probs = trici_xgbrf.predict_proba(trici_x_train)
trici_xgbrf_train_probs = trici_xgbrf_train_probs[:, 1]
trici_xgbrf_train_auc = roc_auc_score(trici_y_train, trici_xgbrf_train_probs)

trici_xgbrf_cv_probs = trici_xgbrf.predict_proba(trici_x_cv)
trici_xgbrf_cv_probs = trici_xgbrf_cv_probs[:, 1]
trici_xgbrf_cv_auc = roc_auc_score(trici_y_cv, trici_xgbrf_cv_probs)

print(f'Overall accuracy for Tri-Cities XGB Random Forest model (training): {trici_xgbrf.score(trici_x_train, trici_y_train):.4f}')
print(f'ROC AUC for Tri-Cities XGB Random Forest model (training): {trici_xgbrf_train_auc:.4f}')
print(f'Overall accuracy for Tri-Cities XGB Random Forest model (validation): {trici_xgbrf.score(trici_x_cv, trici_y_cv):.4f}')
print(f'ROC AUC for Tri-Cities XGB Random Forest model (validation): {trici_xgbrf_cv_auc:.4f}\n')

# Tri-Cities metrics by sensitive features 
trici_metrics = {
	'accuracy': accuracy_score,
    'tpr': true_positive_rate,
    'tnr': true_negative_rate,
    'balanced': balanced_accuracy_score,
    'matrix': confusion_matrix,
    'headcount': count
}

trici_group_train = pd.DataFrame()
trici_group_valid = pd.DataFrame()

trici_group_train['male'] = trici_x_train[:, trici_feat_names.index('male')]
trici_group_train['underrep_minority'] = trici_x_train[:, trici_feat_names.index('underrep_minority')]
trici_group_valid['male'] = trici_x_cv[:, trici_feat_names.index('male')]
trici_group_valid['underrep_minority'] = trici_x_cv[:, trici_feat_names.index('underrep_minority')]

trici_metric_train_frame = MetricFrame(
    metrics=trici_metrics, y_true=trici_y_train, y_pred=trici_xgbrf.predict(trici_x_train), sensitive_features=trici_group_train
)

trici_metric_valid_frame = MetricFrame(
    metrics=trici_metrics, y_true=trici_y_cv, y_pred=trici_xgbrf.predict(trici_x_cv), sensitive_features=trici_group_valid
)

print('Tri-Cities metrics by sensitive features (training)\n')
print(trici_metric_train_frame.by_group)
print('\n')

print('Tri-Cities metrics by sensitive features (validation)\n')
print(trici_metric_valid_frame.by_group)
print('\n')

helper_funcs.fairness_output(auto_engine, model_id, 'train', model_descr, trici_metric_train_frame, run_date, trici_campus_var)
helper_funcs.fairness_output(auto_engine, model_id, 'valid', model_descr, trici_metric_valid_frame, run_date, trici_campus_var)

#%%
# University XGBoost Random Forest model selection
if build_ft_ft_2yr_prod.DatasetBuilderProd.valid_pass== 0 and build_ft_ft_2yr_prod.DatasetBuilderProd.training_pass == 0:
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
# Vancouver VCF
# vanco_vcf = VotingClassifier(estimators=[('lreg', vanco_lreg), ('sgd', vanco_sgd)], voting='soft', weights=[1, 1]).fit(vanco_x_train, vanco_y_train)

# vanco_vcf_probs = vanco_vcf.predict_proba(vanco_x_train)
# vanco_vcf_probs = vanco_vcf_probs[:, 1]
# vanco_vcf_auc = roc_auc_score(vanco_y_train, vanco_vcf_probs)

# print(f'\nOverall accuracy for Vancouver ensemble model (training): {vanco_vcf.score(vanco_x_train, vanco_y_train):.4f}')
# print(f'ROC AUC for Vancouver ensemble model (training): {vanco_vcf_auc:.4f}\n')

#%%
# Tri-Cities VCF
# trici_vcf = VotingClassifier(estimators=[('lreg', trici_lreg), ('sgd', trici_sgd)], voting='soft', weights=[1, 1]).fit(trici_x_train, trici_y_train)

# trici_vcf_probs = trici_vcf.predict_proba(trici_x_train)
# trici_vcf_probs = trici_vcf_probs[:, 1]
# trici_vcf_auc = roc_auc_score(trici_y_train, trici_vcf_probs)

# print(f'\nOverall accuracy for Tri-Cities ensemble model (training): {trici_vcf.score(trici_x_train, trici_y_train):.4f}')
# print(f'ROC AUC for Tri-Cities ensemble model (training): {trici_vcf_auc:.4f}\n')

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
# Vancouver SHAP training
vanco_explainer = shap.TreeExplainer(model=vanco_xgbrf, data=vanco_x_train, model_output='predict_proba')

#%%
# Vancouver SHAP prediction
vanco_shap_values = vanco_explainer.shap_values(X=vanco_x_test)

#%%
# Vancouver SHAP plots
# 	for index in range(len(vanco_shap_values[0])):
# 		shap.plots._waterfall.waterfall_legacy(vanco_explainer.expected_value[0], vanco_shap_values[0][index], vanco_x_test[index], feature_names=vanco_feat_names, max_display=4)

#%%
vanco_shap_results = []

for index in range(len(vanco_shap_values[0])):
	vanco_shap_results.extend(pd.DataFrame(data=vanco_shap_values[0][index].reshape(1, len(vanco_feat_names)), columns=vanco_feat_names).sort_values(by=0, axis=1, key=abs, ascending=False).to_dict(orient='records'))

vanco_shap_zip = dict(zip(vanco_shap_outcome, vanco_shap_results))

#%%
# Tri-Cities SHAP training
trici_explainer = shap.TreeExplainer(model=trici_xgbrf, data=trici_x_train, model_output='predict_proba')

#%%
# Tri-Cities SHAP prediction
trici_shap_values = trici_explainer.shap_values(X=trici_x_test)

#%%
# Tri-Cities SHAP plots
# 	for index in range(len(trici_shap_values[0])):
# 		shap.plots._waterfall.waterfall_legacy(trici_explainer.expected_value[0], trici_shap_values[0][index], trici_x_test[index], feature_names=trici_feat_names, max_display=4)

#%%
trici_shap_results = []

for index in range(len(trici_shap_values[0])):
	trici_shap_results.extend(pd.DataFrame(data=trici_shap_values[0][index].reshape(1, len(trici_feat_names)), columns=trici_feat_names).sort_values(by=0, axis=1, key=abs, ascending=False).to_dict(orient='records'))

trici_shap_zip = dict(zip(trici_shap_outcome, trici_shap_results))

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
# Vancouver probabilites
# vanco_lreg_pred_probs = vanco_lreg.predict_proba(vanco_x_test)
# vanco_lreg_pred_probs = vanco_lreg_pred_probs[:, 1]
# vanco_sgd_pred_probs = vanco_sgd.predict_proba(vanco_x_test)
# vanco_sgd_pred_probs = vanco_sgd_pred_probs[:, 1]
# vanco_xgb_pred_probs = vanco_xgb.predict_proba(vanco_x_test)
# vanco_xgb_pred_probs = vanco_xgb_pred_probs[:, 1]
# vanco_rf_pred_probs = vanco_rf.predict_proba(vanco_x_test)
# vanco_rf_pred_probs = vanco_rf_pred_probs[:, 1]
vanco_xgbrf_pred_probs = vanco_xgbrf.predict_proba(vanco_x_test)
vanco_xgbrf_pred_probs = vanco_xgbrf_pred_probs[:, 1]
# vanco_mlp_pred_probs = vanco_mlp.predict_proba(vanco_x_test)
# vanco_mlp_pred_probs = vanco_mlp_pred_probs[:, 1]
# vanco_vcf_pred_probs = vanco_vcf.predict_proba(vanco_x_test)
# vanco_vcf_pred_probs = vanco_vcf_pred_probs[:, 1]

#%%
# Tri-Cities probabilities
# trici_lreg_pred_probs = trici_lreg.predict_proba(trici_x_test)
# trici_lreg_pred_probs = trici_lreg_pred_probs[:, 1]
# trici_sgd_pred_probs = trici_sgd.predict_proba(trici_x_test)
# trici_sgd_pred_probs = trici_sgd_pred_probs[:, 1]
# trici_xgb_pred_probs = trici_xgb.predict_proba(trici_x_test)
# trici_xgb_pred_probs = trici_xgb_pred_probs[:, 1]
# trici_rf_pred_probs = trici_rf.predict_proba(trici_x_test)
# trici_rf_pred_probs = trici_rf_pred_probs[:, 1]
trici_xgbrf_pred_probs = trici_xgbrf.predict_proba(trici_x_test)
trici_xgbrf_pred_probs = trici_xgbrf_pred_probs[:, 1]
# trici_mlp_pred_probs = trici_mlp.predict_proba(trici_x_test)
# trici_mlp_pred_probs = trici_mlp_pred_probs[:, 1]
# trici_vcf_pred_probs = trici_vcf.predict_proba(trici_x_test)
# trici_vcf_pred_probs = trici_vcf_pred_probs[:, 1]

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
# Vancouver predicted outcome
# vanco_pred_outcome['lr_prob'] = pd.DataFrame(vanco_lreg_pred_probs)
# vanco_pred_outcome['lr_pred'] = vanco_lreg.predict(vanco_x_test)
# vanco_pred_outcome['sgd_prob'] = pd.DataFrame(vanco_sgd_pred_probs)
# vanco_pred_outcome['sgd_pred'] = vanco_sgd.predict(vanco_x_test)
# vanco_pred_outcome['xgb_prob'] = pd.DataFrame(vanco_xgb_pred_probs)
# vanco_pred_outcome['xgb_pred'] = vanco_xgb.predict(vanco_x_test)
# vanco_pred_outcome['rf_prob'] = pd.DataFrame(vanco_rf_pred_probs)
# vanco_pred_outcome['rf_pred'] = vanco_rf.predict(vanco_x_test)
vanco_pred_outcome['xgbrf_prob'] = pd.DataFrame(vanco_xgbrf_pred_probs)
vanco_pred_outcome['xgbrf_pred'] = vanco_xgbrf.predict(vanco_x_test)
# vanco_pred_outcome['mlp_prob'] = pd.DataFrame(vanco_mlp_pred_probs)
# vanco_pred_outcome['mlp_pred'] = vanco_mlp.predict(vanco_x_test)
# vanco_pred_outcome['vcf_prob'] = pd.DataFrame(vanco_vcf_pred_probs)
# vanco_pred_outcome['vcf_pred'] = vanco_vcf.predict(vanco_x_test)
vanco_pred_outcome.to_csv(f'Z:\\Nathan\\Models\\student_risk\\predictions\\vanco\\vanco_{model_descr}_pred_outcome.csv', encoding='utf-8', index=False)

#%%
# Tri-Cities predicted outcome
# trici_pred_outcome['lr_prob'] = pd.DataFrame(trici_lreg_pred_probs)
# trici_pred_outcome['lr_pred'] = trici_lreg.predict(trici_x_test)
# trici_pred_outcome['sgd_prob'] = pd.DataFrame(trici_sgd_pred_probs)
# trici_pred_outcome['sgd_pred'] = trici_sgd.predict(trici_x_test)
# trici_pred_outcome['xgb_prob'] = pd.DataFrame(trici_xgb_pred_probs)
# trici_pred_outcome['xgb_pred'] = trici_xgb.predict(trici_x_test)
# trici_pred_outcome['rf_prob'] = pd.DataFrame(trici_rf_pred_probs)
# trici_pred_outcome['rf_pred'] = trici_rf.predict(trici_x_test)
trici_pred_outcome['xgbrf_prob'] = pd.DataFrame(trici_xgbrf_pred_probs)
trici_pred_outcome['xgbrf_pred'] = trici_xgbrf.predict(trici_x_test)
# trici_pred_outcome['mlp_prob'] = pd.DataFrame(trici_mlp_pred_probs)
# trici_pred_outcome['mlp_pred'] = trici_mlp.predict(trici_x_test)
# trici_pred_outcome['vcf_prob'] = pd.DataFrame(trici_vcf_pred_probs)
# trici_pred_outcome['vcf_pred'] = trici_vcf.predict(trici_x_test)
trici_pred_outcome.to_csv(f'Z:\\Nathan\\Models\\student_risk\\predictions\\trici\\trici_{model_descr}_pred_outcome.csv', encoding='utf-8', index=False)

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

#%%
# Vancouver output
helper_funcs.aggregate_outcome(vanco_aggregate_outcome, vanco_xgbrf_pred_probs, 'vanco', model_descr)
helper_funcs.results_output(auto_engine, model_id, run_date, vanco_current_outcome, vanco_xgbrf_pred_probs, 'vanco', model_descr)
helper_funcs.shap_output(engine, student_shap, top_N, model_id, run_date, vanco_shap_zip, 'vanco', model_descr)

#%%
# Tri-Cities output
helper_funcs.aggregate_outcome(trici_aggregate_outcome, trici_xgbrf_pred_probs, 'trici', model_descr)
helper_funcs.results_output(auto_engine, model_id, run_date, trici_current_outcome, trici_xgbrf_pred_probs, 'trici', model_descr)
helper_funcs.shap_output(engine, student_shap, top_N, model_id, run_date, trici_shap_zip, 'trici', model_descr)

#%%
# University output
helper_funcs.aggregate_outcome(univr_aggregate_outcome, univr_xgbrf_pred_probs, 'univr', model_descr)
helper_funcs.results_output(auto_engine, model_id, run_date, univr_current_outcome, univr_xgbrf_pred_probs, 'univr', model_descr)
helper_funcs.shap_output(engine, student_shap, top_N, model_id, run_date, univr_shap_zip, 'univr', model_descr)

print('Done\n')