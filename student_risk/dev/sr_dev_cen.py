#%%
import gower
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fairlearn.metrics import MetricFrame, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate, selection_rate, count
from imblearn.under_sampling import NearMiss, TomekLinks
from matplotlib.legend_handler import HandlerLine2D
from patsy.highlevel import dmatrices
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import make_column_transformer
from sklearn.ensemble import VotingClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  SGDClassifier)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import HalvingGridSearchCV, cross_val_predict
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from statsmodels.api import OLS
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier

import shap
from student_risk import build_dev

#%%
# Global variable initialization
strm: str = None
outcome: str = 'term'
top_N: int = 5
model_descr: str = 'ft_ft_1yr'
unwanted_vars: list = ['emplid','enrl_ind']

#%%
# Development variable initialization
full_acad_year: int = 2024
aid_snapshot: str = 'usnews'
snapshot: str = 'eot'
term_type: str = 'FAL'

#%%
# WSU branding
wsu_color: tuple = (0.596,0.117,0.196)
wsu_cmap = sns.light_palette('#981e32',as_cmap=True)
plt.rcParams['figure.figsize'] = [15, 10]
plt.rcParams['font.size'] = '24'
plt.rcParams['axes.labelsize'] = '24'

#%%
# Global XGBoost hyperparameter initialization
min_child_weight = 8
max_bin = 32
num_parallel_tree = 64
subsample = 0.8
colsample_bytree = 0.8
colsample_bynode = 0.8
verbose = True

#%%
# SAS dataset builder
build_dev.DatasetBuilderDev.build_census_dev(outcome, full_acad_year, aid_snapshot, snapshot, term_type)

#%%
# Import pre-split data
validation_set = pd.read_sas('Z:\\Nathan\\Models\\student_risk\\datasets\\ft_ft_1yr_validation_set.sas7bdat', encoding='latin1')
training_set = pd.read_sas('Z:\\Nathan\\Models\\student_risk\\datasets\\ft_ft_1yr_training_set.sas7bdat', encoding='latin1')
testing_set = pd.read_sas('Z:\\Nathan\\Models\\student_risk\\datasets\\ft_ft_1yr_testing_set.sas7bdat', encoding='latin1')

#%%
# Training AWE instrumental variable
training_awe = training_set[[
                            'emplid',
                            'high_school_gpa',
                            'underrep_minority',
                            'male',
                            'sat_erws',
                            'sat_mss',
                            'educ_rate',
                            'gini_indx',
                            'median_inc'                
                            ]].dropna()

awe_x_train = training_awe[[
                            'sat_erws',
                            'sat_mss',
                            'underrep_minority',
                            'male',
                            'educ_rate',
                            'gini_indx',
                            'median_inc'
                            ]]

awe_y_train = training_awe[[
                            'high_school_gpa'
                            ]]

y, x = dmatrices('high_school_gpa ~ sat_erws + sat_mss + underrep_minority + male + educ_rate + gini_indx + median_inc', data=training_awe, return_type='dataframe')
reg_mod = OLS(y, x)
reg_res = reg_mod.fit()
print(reg_res.summary())

reg = LinearRegression()
reg.fit(awe_x_train, awe_y_train)

training_awe_pred = pd.DataFrame()
training_awe_pred['emplid'] = training_awe['emplid']
training_awe_pred['actual'] = training_awe['high_school_gpa']
training_awe_pred['predicted'] = reg.predict(awe_x_train)
training_awe_pred['awe_instrument'] = training_awe_pred['actual'] - training_awe_pred['predicted']

training_set = training_set.join(training_awe_pred.set_index('emplid'), on='emplid')

#%%
# Testing AWE instrumental variable
testing_awe = testing_set[[
                            'emplid',
                            'high_school_gpa',
                            'underrep_minority',
                            'male',
                            'sat_erws',
                            'sat_mss',
                            'educ_rate',
                            'gini_indx',
                            'median_inc'
                            ]].dropna()

awe_x_test = testing_awe[[
                            'sat_erws',
                            'sat_mss',
                            'underrep_minority',
                            'male',
                            'educ_rate',
                            'gini_indx',
                            'median_inc'                        
                            ]]

awe_y_test = testing_awe[[
                            'high_school_gpa'
                            ]]

y, x = dmatrices('high_school_gpa ~ sat_erws + sat_mss + underrep_minority + male + educ_rate + gini_indx + median_inc', data=testing_awe, return_type='dataframe')
reg_mod = OLS(y, x)
reg_res = reg_mod.fit()
print(reg_res.summary())

reg = LinearRegression()
reg.fit(awe_x_test, awe_y_test)

testing_awe_pred = pd.DataFrame()
testing_awe_pred['emplid'] = testing_awe['emplid']
testing_awe_pred['awe_actual'] = testing_awe['high_school_gpa']
testing_awe_pred['awe_predicted'] = reg.predict(awe_x_test)
testing_awe_pred['awe_instrument'] = testing_awe_pred['awe_actual'] - testing_awe_pred['awe_predicted']

testing_set = testing_set.join(testing_awe_pred.set_index('emplid'), on='emplid')

#%%
# Training CDI instrumental variable
training_cdi = training_set[[
                            'emplid',
                            'high_school_gpa',
                            'fall_lec_count',
                            'sat_erws',
                            'sat_mss',
                            'underrep_minority',
                            'male',
                            'median_inc',
                            'fall_avg_pct_withdrawn',
                            'fall_avg_difficulty'                
                            ]].dropna()

cdi_x_train = training_cdi[[
                            'high_school_gpa',
                            'fall_lec_count',
                            'sat_erws',
                            'sat_mss',
                            'underrep_minority',
                            'male',
                            'median_inc',
                            'fall_avg_pct_withdrawn'
                            ]]

cdi_y_train = training_cdi[[
                            'fall_avg_difficulty'
                            ]]

y, x = dmatrices('fall_avg_difficulty ~ high_school_gpa + fall_lec_count + fall_avg_pct_withdrawn + sat_erws + sat_mss + underrep_minority + male + median_inc', data=training_cdi, return_type='dataframe')
reg_mod = OLS(y, x)
reg_res = reg_mod.fit()
print(reg_res.summary())

reg = LinearRegression()
reg.fit(cdi_x_train, cdi_y_train)

training_cdi_pred = pd.DataFrame()
training_cdi_pred['emplid'] = training_cdi['emplid']
training_cdi_pred['cdi_actual'] = training_cdi['fall_avg_difficulty']
training_cdi_pred['cdi_predicted'] = reg.predict(cdi_x_train)
training_cdi_pred['cdi_instrument'] = training_cdi_pred['cdi_actual'] - training_cdi_pred['cdi_predicted']

training_set = training_set.join(training_cdi_pred.set_index('emplid'), on='emplid')

#%%
# Testing CDI instrumental variable
testing_cdi = testing_set[[
                            'emplid',
                            'high_school_gpa',
                            'fall_lec_count',
                            'sat_erws',
                            'sat_mss',
                            'underrep_minority',
                            'male',
                            'median_inc',
                            'fall_avg_pct_withdrawn',
                            'fall_avg_difficulty'                 
                            ]].dropna()

cdi_x_test = testing_cdi[[
                            'high_school_gpa',
                            'fall_lec_count',
                            'sat_erws',
                            'sat_mss',
                            'underrep_minority',
                            'male',
                            'median_inc',
                            'fall_avg_pct_withdrawn' 
                            ]]

cdi_y_test = testing_cdi[[
                            'avg_difficulty'
                            ]]

y, x = dmatrices('fall_avg_difficulty ~ high_school_gpa + fall_lec_count + fall_avg_pct_withdrawn + sat_erws + sat_mss + underrep_minority + male + median_inc', data=training_cdi, return_type='dataframe')
reg_mod = OLS(y, x)
reg_res = reg_mod.fit()
print(reg_res.summary())

reg = LinearRegression()
reg.fit(cdi_x_test, cdi_y_test)

testing_cdi_pred = pd.DataFrame()
testing_cdi_pred['emplid'] = testing_cdi['emplid']
testing_cdi_pred['cdi_actual'] = testing_cdi['avg_difficulty']
testing_cdi_pred['cdi_predicted'] = reg.predict(cdi_x_test)
testing_cdi_pred['cdi_instrument'] = testing_cdi_pred['cdi_actual'] - testing_cdi_pred['cdi_predicted']

testing_set = testing_set.join(testing_cdi_pred.set_index('emplid'), on='emplid')

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
'fall_term_gpa',
'fall_term_gpa_mi',
'fall_term_D_grade_count',
'fall_term_F_grade_count',
# 'fall_term_S_grade_count',
# 'fall_term_W_grade_count',
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
# 'spring_lec_count',
# 'spring_lab_count',
# 'spring_stu_count',
# 'spring_oth_count',
# 'spring_enrl_avg',
# 'spring_enrl_avg_mi',
# 'spring_class_time_early',
# 'spring_class_time_early_mi',
# 'spring_class_time_late',
# 'spring_class_time_late_mi',
# 'spring_sun_class',
# 'spring_mon_class',
# 'spring_tues_class',
# 'spring_wed_class',
# 'spring_thurs_class',
# 'spring_fri_class',
# 'spring_sat_class',
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
# 'spring_credit_hours',
# 'total_spring_units',
# 'spring_withdrawn_hours',
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

pullm_x_vars = [x for x in pullm_data_vars if x not in unwanted_vars]

# Pullman dataframes
pullm_logit_df = training_set[(training_set['adj_acad_prog_primary_campus'] == 'PULLM')][pullm_data_vars].dropna().drop(columns=['emplid'])

pullm_validation_set = validation_set[(validation_set['adj_acad_prog_primary_campus'] == 'PULLM')][pullm_data_vars].dropna()

pullm_training_set = training_set[(training_set['adj_acad_prog_primary_campus'] == 'PULLM')][pullm_data_vars].dropna()

pullm_testing_set = testing_set[(testing_set['adj_acad_prog_primary_campus'] == 'PULLM')][pullm_data_vars].dropna()

pullm_testing_set = pullm_testing_set.reset_index()

pullm_shap_outcome = pullm_testing_set['emplid'].copy(deep=True).values.tolist()

pullm_pred_outcome = pullm_testing_set[[ 
                            'emplid',
                            # 'enrl_ind'
                            ]].copy(deep=True)

pullm_aggregate_outcome = pullm_testing_set[[ 
                            'emplid',
							'male',
							'underrep_minority',
							'first_gen_flag',
							'resident'
                            # 'enrl_ind'
                            ]].copy(deep=True)

pullm_current_outcome = pullm_testing_set[[ 
                            'emplid',
                            # 'enrl_ind'
                            ]].copy(deep=True)

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
'high_school_gpa',
'high_school_gpa_mi',
'fall_midterm_gpa_avg',
'fall_midterm_gpa_avg_mi',
'fall_midterm_grade_count',
# 'fall_midterm_F_grade_count',
# 'fall_midterm_S_grade_count',
# 'fall_midterm_W_grade_count',
'fall_term_gpa',
'fall_term_gpa_mi',
# 'fall_term_no_letter_count',
'fall_term_F_grade_count',
# 'fall_term_S_grade_count',
# 'fall_term_W_grade_count',
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
'fall_credit_hours',
# 'total_fall_units',
'fall_withdrawn_hours',
# 'fall_withdrawn_ind',
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
'unmet_need_acpt',
'unmet_need_acpt_mi'
]

vanco_x_vars = [x for x in vanco_data_vars if x not in unwanted_vars]

# Vancouver dataframes
vanco_logit_df = training_set[(training_set['adj_acad_prog_primary_campus'] == 'VANCO')][vanco_data_vars].dropna().drop(columns=['emplid'])

vanco_validation_set = validation_set[(validation_set['adj_acad_prog_primary_campus'] == 'VANCO')][vanco_data_vars].dropna()

vanco_training_set = training_set[(training_set['adj_acad_prog_primary_campus'] == 'VANCO')][vanco_data_vars].dropna()

vanco_testing_set = testing_set[(testing_set['adj_acad_prog_primary_campus'] == 'VANCO')][vanco_data_vars].dropna()

vanco_testing_set = vanco_testing_set.reset_index()

vanco_shap_outcome = vanco_testing_set['emplid'].copy(deep=True).values.tolist()

vanco_pred_outcome = vanco_testing_set[[ 
                            'emplid',
                            # 'enrl_ind'
                            ]].copy(deep=True)

vanco_aggregate_outcome = vanco_testing_set[[ 
                            'emplid',
							'male',
							'underrep_minority',
							'first_gen_flag',
							'resident'
                            # 'enrl_ind'
                            ]].copy(deep=True)

vanco_current_outcome = vanco_testing_set[[ 
                            'emplid',
                            # 'enrl_ind'
                            ]].copy(deep=True)

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
'high_school_gpa',
'high_school_gpa_mi',
'fall_midterm_gpa_avg',
'fall_midterm_gpa_avg_mi',
'fall_midterm_grade_count',
# 'fall_midterm_F_grade_count',
# 'fall_midterm_S_grade_count',
# 'fall_midterm_W_grade_count',
'fall_term_gpa',
'fall_term_gpa_mi',
# 'fall_term_no_letter_count',
'fall_term_F_grade_count',
# 'fall_term_S_grade_count',
# 'fall_term_W_grade_count',
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
'fall_credit_hours',
# 'total_fall_units',
'fall_withdrawn_hours',
# 'fall_withdrawn_ind',
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
'unmet_need_acpt',
'unmet_need_acpt_mi'
]

trici_x_vars = [x for x in trici_data_vars if x not in unwanted_vars]

# Tri-Cities dataframes
trici_logit_df = training_set[(training_set['adj_acad_prog_primary_campus'] == 'TRICI')][trici_data_vars].dropna().drop(columns=['emplid'])

trici_validation_set = validation_set[(validation_set['adj_acad_prog_primary_campus'] == 'TRICI')][trici_data_vars].dropna()

trici_training_set = training_set[(training_set['adj_acad_prog_primary_campus'] == 'TRICI')][trici_data_vars].dropna()

trici_testing_set = testing_set[(testing_set['adj_acad_prog_primary_campus'] == 'TRICI')][trici_data_vars].dropna()
								
trici_testing_set = trici_testing_set.reset_index()

trici_shap_outcome = trici_testing_set['emplid'].copy(deep=True).values.tolist()

trici_pred_outcome = trici_testing_set[[ 
                            'emplid',
                            # 'enrl_ind'
                            ]].copy(deep=True)

trici_aggregate_outcome = trici_testing_set[[ 
                            'emplid',
							'male',
							'underrep_minority',
							'first_gen_flag',
							'resident'
                            # 'enrl_ind'
                            ]].copy(deep=True)

trici_current_outcome = trici_testing_set[[ 
                            'emplid',
                            # 'enrl_ind'
                            ]].copy(deep=True)

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
'high_school_gpa',
'high_school_gpa_mi',
'fall_midterm_gpa_avg',
'fall_midterm_gpa_avg_mi',
'fall_midterm_grade_count',
# 'fall_midterm_F_grade_count',
'fall_midterm_S_grade_count',
'fall_midterm_W_grade_count',
'fall_term_gpa',
'fall_term_gpa_mi',
# 'fall_term_no_letter_count',
'fall_term_F_grade_count',
# 'fall_term_S_grade_count',
# 'fall_term_W_grade_count',
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
'fall_credit_hours',
# 'total_fall_units',
'fall_withdrawn_hours',
# 'fall_withdrawn_ind',
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
'unmet_need_acpt',
'unmet_need_acpt_mi'
]

univr_x_vars = [x for x in univr_data_vars if x not in unwanted_vars]

# University dataframes
univr_logit_df = training_set[univr_data_vars].dropna().drop(columns=['emplid'])

univr_validation_set = validation_set[univr_data_vars].dropna()

univr_training_set = training_set[univr_data_vars].dropna()

univr_testing_set = testing_set[(testing_set['adj_acad_prog_primary_campus'] == 'EVERE')
								| (testing_set['adj_acad_prog_primary_campus'] == 'SPOKA') 
								| (testing_set['adj_acad_prog_primary_campus'] == 'ONLIN')][univr_data_vars].dropna()

univr_testing_set = univr_testing_set.reset_index()

univr_shap_outcome = univr_testing_set['emplid'].copy(deep=True).values.tolist()

univr_pred_outcome = univr_testing_set[[ 
                            'emplid',
                            # 'enrl_ind'
                            ]].copy(deep=True)

univr_aggregate_outcome = univr_testing_set[[ 
                            'emplid',
							'male',
							'underrep_minority',
							'first_gen_flag',
							'resident'
                            # 'enrl_ind'
                            ]].copy(deep=True)

univr_current_outcome = univr_testing_set[[ 
                            'emplid',
                            # 'enrl_ind'
                            ]].copy(deep=True)

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

pullm_x_training_gower = gower.gower_matrix(pullm_x_training_outlier)
pullm_x_validation_gower = gower.gower_matrix(pullm_x_validation_outlier)

pullm_training_set['mask'] = LocalOutlierFactor(metric='precomputed', n_jobs=-1).fit_predict(pullm_x_training_gower)
pullm_validation_set['mask'] = LocalOutlierFactor(metric='precomputed', n_jobs=-1).fit_predict(pullm_x_validation_gower)

pullm_training_outlier_set = pullm_training_set.drop(pullm_training_set[pullm_training_set['mask'] == 1].index)
pullm_training_outlier_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\pullm_ft_ft_1yr_training_outlier_set.csv', encoding='utf-8', index=False)
pullm_validation_outlier_set = pullm_validation_set.drop(pullm_validation_set[pullm_validation_set['mask'] == 1].index)
pullm_validation_outlier_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\pullm_ft_ft_1yr_validation_outlier_set.csv', encoding='utf-8', index=False)

pullm_training_set = pullm_training_set.drop(pullm_training_set[pullm_training_set['mask'] == -1].index)
pullm_training_set = pullm_training_set.drop(columns='mask')
pullm_validation_set = pullm_validation_set.drop(pullm_validation_set[pullm_validation_set['mask'] == -1].index)
pullm_validation_set = pullm_validation_set.drop(columns='mask')

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

vanco_x_training_gower = gower.gower_matrix(vanco_x_training_outlier)
vanco_x_validation_gower = gower.gower_matrix(vanco_x_validation_outlier)

vanco_training_set['mask'] = LocalOutlierFactor(metric='precomputed', n_jobs=-1).fit_predict(vanco_x_training_gower)
vanco_validation_set['mask'] = LocalOutlierFactor(metric='precomputed', n_jobs=-1).fit_predict(vanco_x_validation_gower)

vanco_training_outlier_set = vanco_training_set.drop(vanco_training_set[vanco_training_set['mask'] == 1].index)
vanco_training_outlier_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\vanco_ft_ft_1yr_training_outlier_set.csv', encoding='utf-8', index=False)
vanco_validation_outlier_set = vanco_validation_set.drop(vanco_validation_set[vanco_validation_set['mask'] == 1].index)
vanco_validation_outlier_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\vanco_ft_ft_1yr_validation_outlier_set.csv', encoding='utf-8', index=False)

vanco_training_set = vanco_training_set.drop(vanco_training_set[vanco_training_set['mask'] == -1].index)
vanco_training_set = vanco_training_set.drop(columns='mask')
vanco_validation_set = vanco_validation_set.drop(vanco_validation_set[vanco_validation_set['mask'] == -1].index)
vanco_validation_set = vanco_validation_set.drop(columns='mask')

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

trici_x_training_gower = gower.gower_matrix(trici_x_training_outlier)
trici_x_validation_gower = gower.gower_matrix(trici_x_validation_outlier)

trici_training_set['mask'] = LocalOutlierFactor(metric='precomputed', n_jobs=-1).fit_predict(trici_x_training_gower)
trici_validation_set['mask'] = LocalOutlierFactor(metric='precomputed', n_jobs=-1).fit_predict(trici_x_validation_gower)

trici_training_outlier_set = trici_training_set.drop(trici_training_set[trici_training_set['mask'] == 1].index)
trici_training_outlier_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\trici_ft_ft_1yr_training_outlier_set.csv', encoding='utf-8', index=False)
trici_validation_outlier_set = trici_validation_set.drop(trici_validation_set[trici_validation_set['mask'] == 1].index)
trici_validation_outlier_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\trici_ft_ft_1yr_validation_outlier_set.csv', encoding='utf-8', index=False)

trici_training_set = trici_training_set.drop(trici_training_set[trici_training_set['mask'] == -1].index)
trici_training_set = trici_training_set.drop(columns='mask')
trici_validation_set = trici_validation_set.drop(trici_validation_set[trici_validation_set['mask'] == -1].index)
trici_validation_set = trici_validation_set.drop(columns='mask')

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

univr_x_training_gower = gower.gower_matrix(univr_x_training_outlier)
univr_x_validation_gower = gower.gower_matrix(univr_x_validation_outlier)

univr_training_set['mask'] = LocalOutlierFactor(metric='precomputed', n_jobs=-1).fit_predict(univr_x_training_gower)
univr_validation_set['mask'] = LocalOutlierFactor(metric='precomputed', n_jobs=-1).fit_predict(univr_x_validation_gower)

univr_training_outlier_set = univr_training_set.drop(univr_training_set[univr_training_set['mask'] == 1].index)
univr_training_outlier_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\univr_ft_ft_1yr_training_outlier_set.csv', encoding='utf-8', index=False)
univr_validation_outlier_set = univr_validation_set.drop(univr_validation_set[univr_validation_set['mask'] == 1].index)
univr_validation_outlier_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\univr_ft_ft_1yr_validation_outlier_set.csv', encoding='utf-8', index=False)

univr_training_set = univr_training_set.drop(univr_training_set[univr_training_set['mask'] == -1].index)
univr_training_set = univr_training_set.drop(columns='mask')
univr_validation_set = univr_validation_set.drop(univr_validation_set[univr_validation_set['mask'] == -1].index)
univr_validation_set = univr_validation_set.drop(columns='mask')

#%%
# Create Tomek Link undersampled validation and training sets
# https://imbalanced-learn.org/stable/under_sampling.html#tomek-s-links

# Pullman undersample
pullm_x_train = pullm_training_set.drop(columns=['enrl_ind','emplid'])
pullm_x_cv = pullm_validation_set.drop(columns=['enrl_ind','emplid'])

pullm_x_test = pullm_testing_set[pullm_x_vars]

pullm_y_train = pullm_training_set['enrl_ind']
pullm_y_cv = pullm_validation_set['enrl_ind']
pullm_y_test = pullm_testing_set['enrl_ind']

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

pullm_feat_names = []

for name, transformer, features, _ in pullm_tomek_prep._iter(fitted=True):

	if transformer != 'passthrough':
		try:
			pullm_feat_names.extend(pullm_tomek_prep.named_transformers_[name].get_feature_names())
		except AttributeError:
			pullm_feat_names.extend(features)

	if transformer == 'passthrough':
		pullm_feat_names.extend(pullm_tomek_prep._feature_names_in[features])

pullm_under_train = TomekLinks(sampling_strategy='auto', n_jobs=-1)
pullm_under_valid = TomekLinks(sampling_strategy='auto', n_jobs=-1)

pullm_x_train, pullm_y_train = pullm_under_train.fit_resample(pullm_x_train, pullm_y_train)
pullm_x_cv, pullm_y_cv = pullm_under_valid.fit_resample(pullm_x_cv, pullm_y_cv)

pullm_tomek_train_index = pullm_under_train.sample_indices_
pullm_tomek_valid_index = pullm_under_valid.sample_indices_
pullm_training_set = pullm_training_set.reset_index(drop=True)
pullm_validation_set = pullm_validation_set.reset_index(drop=True)

pullm_tomek_train_set = pullm_training_set.drop(pullm_tomek_train_index)
pullm_tomek_train_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\pullm_ft_ft_1yr_tomek_training_set.csv', encoding='utf-8', index=False)
pullm_tomek_valid_set = pullm_validation_set.drop(pullm_tomek_valid_index)
pullm_tomek_valid_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\pullm_ft_ft_1yr_tomek_validation_set.csv', encoding='utf-8', index=False)

#%%
# Vancouver undersample
vanco_x_train = vanco_training_set.drop(columns=['enrl_ind','emplid'])
vanco_x_cv = vanco_validation_set.drop(columns=['enrl_ind','emplid'])

vanco_x_test = vanco_testing_set[vanco_x_vars]

vanco_y_train = vanco_training_set['enrl_ind']
vanco_y_cv = vanco_validation_set['enrl_ind']
vanco_y_test = vanco_testing_set['enrl_ind']

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

vanco_feat_names = []

for name, transformer, features, _ in vanco_tomek_prep._iter(fitted=True):

	if transformer != 'passthrough':
		try:
			vanco_feat_names.extend(vanco_tomek_prep.named_transformers_[name].get_feature_names())
		except AttributeError:
			vanco_feat_names.extend(features)

	if transformer == 'passthrough':
		vanco_feat_names.extend(vanco_tomek_prep._feature_names_in[features])

vanco_under_train = TomekLinks(sampling_strategy='auto', n_jobs=-1)
vanco_under_valid = TomekLinks(sampling_strategy='auto', n_jobs=-1)

vanco_x_train, vanco_y_train = vanco_under_train.fit_resample(vanco_x_train, vanco_y_train)
vanco_x_cv, vanco_y_cv = vanco_under_valid.fit_resample(vanco_x_cv, vanco_y_cv)

vanco_tomek_train_index = vanco_under_train.sample_indices_
vanco_tomek_valid_index = vanco_under_valid.sample_indices_
vanco_training_set = vanco_training_set.reset_index(drop=True)
vanco_validation_set = vanco_validation_set.reset_index(drop=True)

vanco_tomek_train_set = vanco_training_set.drop(vanco_tomek_train_index)
vanco_tomek_train_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\vanco_ft_ft_1yr_tomek_training_set.csv', encoding='utf-8', index=False)
vanco_tomek_valid_set = vanco_validation_set.drop(vanco_tomek_valid_index)
vanco_tomek_valid_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\vanco_ft_ft_1yr_tomek_validation_set.csv', encoding='utf-8', index=False)

#%%
# Tri-Cities undersample
trici_x_train = trici_training_set.drop(columns=['enrl_ind','emplid'])
trici_x_cv = trici_validation_set.drop(columns=['enrl_ind','emplid'])

trici_x_test = trici_testing_set[trici_x_vars]

trici_y_train = trici_training_set['enrl_ind']
trici_y_cv = trici_validation_set['enrl_ind']
trici_y_test = trici_testing_set['enrl_ind']

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

trici_feat_names = []

for name, transformer, features, _ in trici_tomek_prep._iter(fitted=True):

	if transformer != 'passthrough':
		try:
			trici_feat_names.extend(trici_tomek_prep.named_transformers_[name].get_feature_names())
		except AttributeError:
			trici_feat_names.extend(features)

	if transformer == 'passthrough':
		trici_feat_names.extend(trici_tomek_prep._feature_names_in[features])

trici_under_train = TomekLinks(sampling_strategy='auto', n_jobs=-1)
trici_under_valid = TomekLinks(sampling_strategy='auto', n_jobs=-1)

trici_x_train, trici_y_train = trici_under_train.fit_resample(trici_x_train, trici_y_train)
trici_x_cv, trici_y_cv = trici_under_valid.fit_resample(trici_x_cv, trici_y_cv)

trici_tomek_train_index = trici_under_train.sample_indices_
trici_tomek_valid_index = trici_under_valid.sample_indices_
trici_training_set = trici_training_set.reset_index(drop=True)
trici_validation_set = trici_validation_set.reset_index(drop=True)

trici_tomek_train_set = trici_training_set.drop(trici_tomek_train_index)
trici_tomek_train_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\trici_ft_ft_1yr_tomek_training_set.csv', encoding='utf-8', index=False)
trici_tomek_valid_set = trici_validation_set.drop(trici_tomek_valid_index)
trici_tomek_valid_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\trici_ft_ft_1yr_tomek_validation_set.csv', encoding='utf-8', index=False)

#%%
# University undersample
univr_x_train = univr_training_set.drop(columns=['enrl_ind','emplid'])
univr_x_cv = univr_validation_set.drop(columns=['enrl_ind','emplid'])

univr_x_test = univr_testing_set[univr_x_vars]

univr_y_train = univr_training_set['enrl_ind']
univr_y_cv = univr_validation_set['enrl_ind']
univr_y_test = univr_testing_set['enrl_ind']

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

univr_feat_names = []

for name, transformer, features, _ in univr_tomek_prep._iter(fitted=True):

	if transformer != 'passthrough':
		try:
			univr_feat_names.extend(univr_tomek_prep.named_transformers_[name].get_feature_names())
		except AttributeError:
			univr_feat_names.extend(features)

	if transformer == 'passthrough':
		univr_feat_names.extend(univr_tomek_prep._feature_names_in[features])

univr_under_train = TomekLinks(sampling_strategy='auto', n_jobs=-1)
univr_under_valid = TomekLinks(sampling_strategy='auto', n_jobs=-1)

univr_x_train, univr_y_train = univr_under_train.fit_resample(univr_x_train, univr_y_train)
univr_x_cv, univr_y_cv = univr_under_valid.fit_resample(univr_x_cv, univr_y_cv)

univr_tomek_train_index = univr_under_train.sample_indices_
univr_tomek_valid_index = univr_under_valid.sample_indices_
univr_training_set = univr_training_set.reset_index(drop=True)
univr_validation_set = univr_validation_set.reset_index(drop=True)

univr_tomek_train_set = univr_training_set.drop(univr_tomek_train_index)
univr_tomek_train_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\univr_ft_ft_1yr_tomek_training_set.csv', encoding='utf-8', index=False)
univr_tomek_valid_set = univr_validation_set.drop(univr_tomek_valid_index)
univr_tomek_valid_set.to_csv('Z:\\Nathan\\Models\\student_risk\\outliers\\univr_ft_ft_1yr_tomek_validation_set.csv', encoding='utf-8', index=False)

#%%
# Standard logistic model

# Pullman standard model
print('\nStandard logistic model for Pullman freshmen...\n')

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
print('\nStandard logistic model for Vancouver freshmen...\n')

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
print('\nStandard logistic model for Tri-Cities freshmen...\n')

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
print('\nStandard logistic model for University freshmen...\n')

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
# Logistic model

# Pullman logistic tuning
pullm_hyperparameters = [{'penalty': ['elasticnet'],
                    'l1_ratio': np.linspace(0, 1, 11, endpoint=True),
                    'C': np.logspace(0, 4, 20, endpoint=True)}]

pullm_gridsearch = HalvingGridSearchCV(LogisticRegression(solver='saga', class_weight='balanced'), pullm_hyperparameters, cv=5, verbose=0, n_jobs=-1)
pullm_best_model = pullm_gridsearch.fit(pullm_x_train, pullm_y_train)

print(f'Best parameters: {pullm_gridsearch.best_params_}')

#%%
# Pullman logistic
pullm_lreg_ccv = LogisticRegression(penalty='elasticnet', class_weight='balanced', solver='saga', max_iter=5000, l1_ratio=0.0, C=1.0, n_jobs=-1, verbose=False).fit(pullm_x_train, pullm_y_train)

# Pullman logistic calibration
# pullm_lreg = LogisticRegression(penalty='elasticnet', class_weight='balanced', solver='saga', max_iter=5000, l1_ratio=0.0, C=1.0, n_jobs=-1, verbose=False)
# pullm_lreg_ccv = CalibratedClassifierCV(pullm_lreg, method='isotonic', cv=5).fit(pullm_x_train, pullm_y_train)

pullm_lreg_probs = pullm_lreg_ccv.predict_proba(pullm_x_train)
pullm_lreg_probs = pullm_lreg_probs[:, 1]
pullm_lreg_auc = roc_auc_score(pullm_y_train, pullm_lreg_probs)

print(f'Overall accuracy for Pullman logistic model (training): {pullm_lreg_ccv.score(pullm_x_train, pullm_y_train):.4f}')
print(f'ROC AUC for Pullman logistic model (training): {pullm_lreg_auc:.4f}')
print(f'Overall accuracy for Pullman logistic model (testing): {pullm_lreg_ccv.score(pullm_x_test, pullm_y_test):.4f}')

pullm_lreg_fpr, pullm_lreg_tpr, pullm_thresholds = roc_curve(pullm_y_train, pullm_lreg_probs, drop_intermediate=False)

plt.plot(pullm_lreg_fpr, pullm_lreg_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('LOGISTIC ROC CURVE (TRAINING)')
plt.show()

pullm_lreg_y, pullm_lreg_x = calibration_curve(pullm_y_train, pullm_lreg_probs, n_bins=10)

plt.plot(pullm_lreg_y, pullm_lreg_x, marker = '.', color=wsu_color, lw=6, label = 'Logistic Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('LOGISTIC CALIBRATION PLOT (TRAINING)')
plt.show()

#%%
# Pullman confusion matrix
pullm_lreg_matrix = confusion_matrix(pullm_y_test, pullm_lreg_ccv.predict(pullm_x_test))
pullm_lreg_df = pd.DataFrame(pullm_lreg_matrix)

sns.heatmap(pullm_lreg_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('LOGISTIC CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# Vancouver logistic tuning
vanco_hyperparameters = [{'penalty': ['elasticnet'],
                    'l1_ratio': np.linspace(0, 1, 11, endpoint=True),
                    'C': np.logspace(0, 4, 20, endpoint=True)}]

vanco_gridsearch = HalvingGridSearchCV(LogisticRegression(solver='saga', class_weight='balanced'), vanco_hyperparameters, cv=5, verbose=0, n_jobs=-1)
vanco_best_model = vanco_gridsearch.fit(vanco_x_train, vanco_y_train)

print(f'Best parameters: {vanco_gridsearch.best_params_}')

#%%
# Vancouver logistic
vanco_lreg_ccv = LogisticRegression(penalty='elasticnet', class_weight='balanced', solver='saga', max_iter=5000, l1_ratio=0.0, C=1.0, n_jobs=-1, verbose=False).fit(vanco_x_train, vanco_y_train)

# Vancouver logistic calibration
# vanco_lreg_ccv = LogisticRegression(penalty='elasticnet', class_weight='balanced', solver='saga', max_iter=5000, l1_ratio=0.0, C=1.0, n_jobs=-1, verbose=False)
# vanco_lreg_ccv = CalibratedClassifierCV(vanco_lreg, method='isotonic', cv=5).fit(vanco_x_train, vanco_y_train)

vanco_lreg_probs = vanco_lreg_ccv.predict_proba(vanco_x_train)
vanco_lreg_probs = vanco_lreg_probs[:, 1]
vanco_lreg_auc = roc_auc_score(vanco_y_train, vanco_lreg_probs)

print(f'Overall accuracy for Vancouver logistic model (training): {vanco_lreg_ccv.score(vanco_x_train, vanco_y_train):.4f}')
print(f'ROC AUC for Vancouver logistic model (training): {vanco_lreg_auc:.4f}')
print(f'Overall accuracy for Vancouver logistic model (testing): {vanco_lreg_ccv.score(vanco_x_test, vanco_y_test):.4f}')

vanco_lreg_fpr, vanco_lreg_tpr, vanco_thresholds = roc_curve(vanco_y_train, vanco_lreg_probs, drop_intermediate=False)

plt.plot(vanco_lreg_fpr, vanco_lreg_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('LOGISTIC ROC CURVE (TRAINING)')
plt.show()

vanco_lreg_y, vanco_lreg_x = calibration_curve(vanco_y_train, vanco_lreg_probs, n_bins=10)

plt.plot(vanco_lreg_y, vanco_lreg_x, marker = '.', color=wsu_color, lw=6, label = 'Logistic Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('LOGISTIC CALIBRATION PLOT')
plt.show()

#%%
# Vancouver confusion matrix
vanco_lreg_matrix = confusion_matrix(vanco_y_test, vanco_lreg_ccv.predict(vanco_x_test))
vanco_lreg_df = pd.DataFrame(vanco_lreg_matrix)

sns.heatmap(vanco_lreg_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('LOGISTIC CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# Tri-Cities logistic tuning
trici_hyperparameters = [{'penalty': ['elasticnet'],
                    'l1_ratio': np.linspace(0, 1, 11, endpoint=True),
                    'C': np.logspace(0, 4, 20, endpoint=True)}]

trici_gridsearch = HalvingGridSearchCV(LogisticRegression(solver='saga', class_weight='balanced'), trici_hyperparameters, cv=5, verbose=0, n_jobs=-1)
trici_best_model = trici_gridsearch.fit(trici_x_train, trici_y_train)

print(f'Best parameters: {trici_gridsearch.best_params_}')

#%%
# Tri-Cities logistic
trici_lreg_ccv = LogisticRegression(penalty='elasticnet', class_weight='balanced', solver='saga', max_iter=5000, l1_ratio=0.0, C=1.0, n_jobs=-1, verbose=False).fit(trici_x_train, trici_y_train)

# Tri-Cities logistic calibration
# trici_lreg_ccv = LogisticRegression(penalty='elasticnet', class_weight='balanced', solver='saga', max_iter=5000, l1_ratio=0.0, C=1.0, n_jobs=-1, verbose=False)
# trici_lreg_ccv = CalibratedClassifierCV(trici_lreg, method='isotonic', cv=5).fit(trici_x_train, trici_y_train)

trici_lreg_probs = trici_lreg_ccv.predict_proba(trici_x_train)
trici_lreg_probs = trici_lreg_probs[:, 1]
trici_lreg_auc = roc_auc_score(trici_y_train, trici_lreg_probs)

print(f'Overall accuracy for Tri-Cities logistic model (training): {trici_lreg_ccv.score(trici_x_train, trici_y_train):.4f}')
print(f'ROC AUC for Tri-Cities logistic model (training): {trici_lreg_auc:.4f}')
print(f'Overall accuracy for Tri-Cities logistic model (testing): {trici_lreg_ccv.score(trici_x_test, trici_y_test):.4f}')

trici_lreg_fpr, trici_lreg_tpr, trici_thresholds = roc_curve(trici_y_train, trici_lreg_probs, drop_intermediate=False)

plt.plot(trici_lreg_fpr, trici_lreg_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('LOGISTIC ROC CURVE (TRAINING)')
plt.show()

trici_lreg_y, trici_lreg_x = calibration_curve(trici_y_train, trici_lreg_probs, n_bins=10)

plt.plot(trici_lreg_y, trici_lreg_x, marker = '.', color=wsu_color, lw=6, label = 'Logistic Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('LOGISTIC CALIBRATION PLOT')
plt.show()

#%%
# Tri-Cities confusion matrix
trici_lreg_matrix = confusion_matrix(trici_y_test, trici_lreg_ccv.predict(trici_x_test))
trici_lreg_df = pd.DataFrame(trici_lreg_matrix)

sns.heatmap(trici_lreg_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('LOGISTIC CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# Stochastic gradient descent model

# Pullman SGD
pullm_sgd_ccv = SGDClassifier(loss='modified_huber', penalty='elasticnet', class_weight='balanced', early_stopping=False, max_iter=5000, l1_ratio=0.0, learning_rate='adaptive', eta0=0.0001, tol=0.0001, n_iter_no_change=100, n_jobs=-1, verbose=False).fit(pullm_x_train, pullm_y_train)

# Pullman SGD calibration
# pullm_sgd = SGDClassifier(loss='modified_huber', penalty='elasticnet', class_weight='balanced', early_stopping=False, max_iter=5000, l1_ratio=0.0, learning_rate='adaptive', eta0=0.0001, tol=0.0001, n_iter_no_change=100, n_jobs=-1, verbose=False).fit(pullm_x_train, pullm_y_train)
# pullm_sgd_ccv = CalibratedClassifierCV(pullm_sgd, method='isotonic', cv=5).fit(pullm_x_train, pullm_y_train)

pullm_sgd_probs = pullm_sgd_ccv.predict_proba(pullm_x_train)
pullm_sgd_probs = pullm_sgd_probs[:, 1]
pullm_sgd_auc = roc_auc_score(pullm_y_train, pullm_sgd_probs)

print(f'\nOverall accuracy for Pullman SGD model (training): {pullm_sgd_ccv.score(pullm_x_train, pullm_y_train):.4f}')
print(f'ROC AUC for Pullman SGD model (training): {pullm_sgd_auc:.4f}')
print(f'Overall accuracy for Pullman SGD model (testing): {pullm_sgd_ccv.score(pullm_x_test, pullm_y_test):.4f}')

pullm_sgd_fpr, pullm_sgd_tpr, pullm_thresholds = roc_curve(pullm_y_train, pullm_sgd_probs, drop_intermediate=False)

plt.plot(pullm_sgd_fpr, pullm_sgd_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('SGD ROC CURVE (TRAINING)')
plt.show()

pullm_sgd_y, pullm_sgd_x = calibration_curve(pullm_y_train, pullm_sgd_probs, n_bins=10)

plt.plot(pullm_sgd_y, pullm_sgd_x, marker = '.', color=wsu_color, lw=6, label = 'SGD Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('SGD CALIBRATION PLOT')
plt.show()

#%%
# Pullman SGD confusion matrix
pullm_sgd_matrix = confusion_matrix(pullm_y_test, pullm_sgd_ccv.predict(pullm_x_test))
pullm_sgd_df = pd.DataFrame(pullm_sgd_matrix)

sns.heatmap(pullm_sgd_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('SGD CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# Vancouver SGD
vanco_sgd_ccv = SGDClassifier(loss='modified_huber', penalty='elasticnet', class_weight='balanced', early_stopping=False, max_iter=5000, l1_ratio=0.0, learning_rate='adaptive', eta0=0.0001, tol=0.0001, n_iter_no_change=100, n_jobs=-1, verbose=False).fit(vanco_x_train, vanco_y_train)

# Vancouver SGD calibration
# vanco_sgd = SGDClassifier(loss='modified_huber', penalty='elasticnet', class_weight='balanced', early_stopping=False, max_iter=5000, l1_ratio=0.0, learning_rate='adaptive', eta0=0.0001, tol=0.0001, n_iter_no_change=100, n_jobs=-1, verbose=False).fit(vanco_x_train, vanco_y_train)
# vanco_sgd_ccv = CalibratedClassifierCV(vanco_sgd, method='isotonic', cv=5).fit(vanco_x_train, vanco_y_train)

vanco_sgd_probs = vanco_sgd_ccv.predict_proba(vanco_x_train)
vanco_sgd_probs = vanco_sgd_probs[:, 1]
vanco_sgd_auc = roc_auc_score(vanco_y_train, vanco_sgd_probs)

print(f'\nOverall accuracy for Vancouver SGD model (training): {vanco_sgd_ccv.score(vanco_x_train, vanco_y_train):.4f}')
print(f'ROC AUC for Vancouver SGD model (training): {vanco_sgd_auc:.4f}')
print(f'Overall accuracy for Vancouver SGD model (testing): {vanco_sgd_ccv.score(vanco_x_test, vanco_y_test):.4f}')

vanco_sgd_fpr, vanco_sgd_tpr, vanco_thresholds = roc_curve(vanco_y_train, vanco_sgd_probs, drop_intermediate=False)

plt.plot(vanco_sgd_fpr, vanco_sgd_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('SGD ROC CURVE (TRAINING)')
plt.show()

vanco_sgd_y, vanco_sgd_x = calibration_curve(vanco_y_train, vanco_sgd_probs, n_bins=10)

plt.plot(vanco_sgd_y, vanco_sgd_x, marker = '.', color=wsu_color, lw=6, label = 'SGD Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('SGD CALIBRATION PLOT')
plt.show()

#%%
# Vancouver SGD confusion matrix
vanco_sgd_matrix = confusion_matrix(vanco_y_test, vanco_sgd_ccv.predict(vanco_x_test))
vanco_sgd_df = pd.DataFrame(vanco_sgd_matrix)

sns.heatmap(vanco_sgd_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('SGD CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# Tri-Cities SGD
trici_sgd_ccv = SGDClassifier(loss='modified_huber', penalty='elasticnet', class_weight='balanced', early_stopping=False, max_iter=5000, l1_ratio=0.0, learning_rate='adaptive', eta0=0.0001, tol=0.0001, n_iter_no_change=100, n_jobs=-1, verbose=False).fit(trici_x_train, trici_y_train)

# Tri-Cities SGD calibration
# trici_sgd = SGDClassifier(loss='modified_huber', penalty='elasticnet', class_weight='balanced', early_stopping=False, max_iter=5000, l1_ratio=0.0, learning_rate='adaptive', eta0=0.0001, tol=0.0001, n_iter_no_change=100, n_jobs=-1, verbose=False).fit(trici_x_train, trici_y_train)
# trici_sgd_ccv = CalibratedClassifierCV(trici_sgd, method='isotonic', cv=5).fit(trici_x_train, trici_y_train)

trici_sgd_probs = trici_sgd_ccv.predict_proba(trici_x_train)
trici_sgd_probs = trici_sgd_probs[:, 1]
trici_sgd_auc = roc_auc_score(trici_y_train, trici_sgd_probs)

print(f'\nOverall accuracy for Tri-Cities SGD model (training): {trici_sgd_ccv.score(trici_x_train, trici_y_train):.4f}')
print(f'ROC AUC for Tri-Cities SGD model (training): {trici_sgd_auc:.4f}')
print(f'Overall accuracy for Tri-Cities SGD model (testing): {trici_sgd_ccv.score(trici_x_test, trici_y_test):.4f}')

trici_sgd_fpr, trici_sgd_tpr, trici_thresholds = roc_curve(trici_y_train, trici_sgd_probs, drop_intermediate=False)

plt.plot(trici_sgd_fpr, trici_sgd_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('SGD ROC CURVE (TRAINING)')
plt.show()

trici_sgd_y, trici_sgd_x = calibration_curve(trici_y_train, trici_sgd_probs, n_bins=10)

plt.plot(trici_sgd_y, trici_sgd_x, marker = '.', color=wsu_color, lw=6, label = 'SGD Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('SGD CALIBRATION PLOT')
plt.show()

#%%
# Tri-Cities SGD confusion matrix
trici_sgd_matrix = confusion_matrix(trici_y_test, trici_sgd_ccv.predict(trici_x_test))
trici_sgd_df = pd.DataFrame(trici_sgd_matrix)

sns.heatmap(trici_sgd_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('SGD CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# Multi-layer perceptron model

# Pullman MLP
pullm_mlp_ccv = MLPClassifier(hidden_layer_sizes=(75,50,25), activation='relu', solver='sgd', alpha=2.5, learning_rate_init=0.001, n_iter_no_change=25, max_iter=5000, verbose=False).fit(pullm_x_train, pullm_y_train)

# Pullman MLP calibration
# pullm_mlp = MLPClassifier(hidden_layer_sizes=(75,50,25), activation='relu', solver='sgd', alpha=2.5, learning_rate_init=0.001, n_iter_no_change=25, max_iter=5000, verbose=False)
# pullm_mlp_ccv = CalibratedClassifierCV(pullm_mlp, method='isotonic', cv=5).fit(pullm_x_train, pullm_y_train)

pullm_mlp_probs = pullm_mlp_ccv.predict_proba(pullm_x_train)
pullm_mlp_probs = pullm_mlp_probs[:, 1]
pullm_mlp_auc = roc_auc_score(pullm_y_train, pullm_mlp_probs)

print(f'\nOverall accuracy for Pullman multi-layer perceptron model (training): {pullm_mlp_ccv.score(pullm_x_train, pullm_y_train):.4f}')
print(f'ROC AUC for Pullman multi-layer perceptron model (training): {pullm_mlp_auc:.4f}')
print(f'Overall accuracy for Pullman multi-layer perceptron model (testing): {pullm_mlp_ccv.score(pullm_x_test, pullm_y_test):.4f}')

pullm_mlp_fpr, pullm_mlp_tpr, pullm_thresholds = roc_curve(pullm_y_train, pullm_mlp_probs, drop_intermediate=False)

plt.plot(pullm_mlp_fpr, pullm_mlp_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('NEURAL NETWORK ROC CURVE (TRAINING)')
plt.show()

pullm_mlp_y, pullm_mlp_x = calibration_curve(pullm_y_train, pullm_mlp_probs, n_bins=10)

plt.plot(pullm_mlp_y, pullm_mlp_x, marker = '.', color=wsu_color, lw=6, label = 'MLP Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('MLP CALIBRATION PLOT')
plt.show()

#%%
# Pullman MLP confusion matrix
pullm_mlp_matrix = confusion_matrix(pullm_y_test, pullm_mlp_ccv.predict(pullm_x_test))
pullm_mlp_df = pd.DataFrame(pullm_mlp_matrix)

sns.heatmap(pullm_mlp_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('NEURAL NETWORK CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# Vancouver MLP
vanco_mlp = MLPClassifier(hidden_layer_sizes=(75,50,25), activation='relu', solver='sgd', alpha=2.5, learning_rate_init=0.001, n_iter_no_change=25, max_iter=5000, verbose=False).fit(vanco_x_train, vanco_y_train)

vanco_mlp_probs = vanco_mlp.predict_proba(vanco_x_train)
vanco_mlp_probs = vanco_mlp_probs[:, 1]
vanco_mlp_auc = roc_auc_score(vanco_y_train, vanco_mlp_probs)

print(f'\nOverall accuracy for Vancouver multi-layer perceptron model (training): {vanco_mlp.score(vanco_x_train, vanco_y_train):.4f}')
print(f'ROC AUC for Vancouver multi-layer perceptron model (training): {vanco_mlp_auc:.4f}')
print(f'Overall accuracy for Vancouver multi-layer perceptron model (testing): {vanco_mlp.score(vanco_x_test, vanco_y_test):.4f}')

vanco_mlp_fpr, vanco_mlp_tpr, vanco_thresholds = roc_curve(vanco_y_train, vanco_mlp_probs, drop_intermediate=False)

plt.plot(vanco_mlp_fpr, vanco_mlp_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('NEURAL NETWORK ROC CURVE (TRAINING)')
plt.show()

#%%
# Vancouver MLP confusion matrix
vanco_mlp_matrix = confusion_matrix(vanco_y_test, vanco_mlp.predict(vanco_x_test))
vanco_mlp_df = pd.DataFrame(vanco_mlp_matrix)

sns.heatmap(vanco_mlp_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('NEURAL NETWORK CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# Tri-Cities MLP
trici_mlp = MLPClassifier(hidden_layer_sizes=(75,50,25), activation='relu', solver='sgd', alpha=2.5, learning_rate_init=0.001, n_iter_no_change=25, max_iter=5000, verbose=False).fit(trici_x_train, trici_y_train)

trici_mlp_probs = trici_mlp.predict_proba(trici_x_train)
trici_mlp_probs = trici_mlp_probs[:, 1]
trici_mlp_auc = roc_auc_score(trici_y_train, trici_mlp_probs)

print(f'\nOverall accuracy for Tri-Cities multi-layer perceptron model (training): {trici_mlp.score(trici_x_train, trici_y_train):.4f}')
print(f'ROC AUC for Tri-Cities multi-layer perceptron model (training): {trici_mlp_auc:.4f}')
print(f'Overall accuracy for Tri-Cities multi-layer perceptron model (testing): {trici_mlp.score(trici_x_test, trici_y_test):.4f}')

trici_mlp_fpr, trici_mlp_tpr, trici_thresholds = roc_curve(trici_y_train, trici_mlp_probs, drop_intermediate=False)

plt.plot(trici_mlp_fpr, trici_mlp_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('NEURAL NETWORK ROC CURVE (TRAINING)')
plt.show()

#%%
# Tri-Cities MLP confusion matrix
trici_mlp_matrix = confusion_matrix(trici_y_test, trici_mlp.predict(trici_x_test))
trici_mlp_df = pd.DataFrame(trici_mlp_matrix)

sns.heatmap(trici_mlp_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('NEURAL NETWORK CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# XGBoost model

# Pullman XGBoost tuning
pullm_class_weight = pullm_y_train[pullm_y_train == 0].count() / pullm_y_train[pullm_y_train == 1].count()
pullm_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True)}]

pullm_gridsearch = HalvingGridSearchCV(XGBClassifier(n_estimators=100, scale_pos_weight=pullm_class_weight, eval_metric='logloss', use_label_encoder=False), pullm_hyperparameters, scoring='roc_auc', cv=3, verbose=0, n_jobs=-1)
pullm_best_model = pullm_gridsearch.fit(pullm_x_train, pullm_y_train)

print(f'Best parameters: {pullm_gridsearch.best_params_}')

#%%
# Pullman XGB
pullm_class_weight = pullm_y_train[pullm_y_train == 0].count() / pullm_y_train[pullm_y_train == 1].count()

pullm_xgb_ccv = XGBClassifier(n_estimators=1000, scale_pos_weight=pullm_class_weight, 
								eval_metric='logloss', **pullm_gridsearch.best_params_, use_label_encoder=False).fit(pullm_x_train, pullm_y_train)

# Pullman XGB calibration
# pullm_xgb = XGBClassifier(n_estimators=1000, scale_pos_weight=pullm_class_weight, eval_metric='logloss', **pullm_gridsearch.best_params_, use_label_encoder=False)
# pullm_xgb_ccv = CalibratedClassifierCV(pullm_xgb, method='isotonic', cv=5).fit(pullm_x_train, pullm_y_train)

pullm_xgb_probs = pullm_xgb_ccv.predict_proba(pullm_x_train)
pullm_xgb_probs = pullm_xgb_probs[:, 1]
pullm_xgb_auc = roc_auc_score(pullm_y_train, pullm_xgb_probs)

print(f'\nOverall accuracy for Pullman XGB model (training): {pullm_xgb_ccv.score(pullm_x_train, pullm_y_train):.4f}')
print(f'ROC AUC for Pullman XGB model (training): {pullm_xgb_auc:.4f}')
print(f'Overall accuracy for Pullman XGB model (testing): {pullm_xgb_ccv.score(pullm_x_test, pullm_y_test):.4f}')

pullm_xgb_fpr, pullm_xgb_tpr, pullm_thresholds = roc_curve(pullm_y_train, pullm_xgb_probs, drop_intermediate=False)

plt.plot(pullm_xgb_fpr, pullm_xgb_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('XGBOOST ROC CURVE (TRAINING)')
plt.show()

pullm_xgb_y, pullm_xgb_x = calibration_curve(pullm_y_train, pullm_xgb_probs, n_bins=10)

plt.plot(pullm_xgb_y, pullm_xgb_x, marker = '.', color=wsu_color, lw=6, label = 'XGBoost Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('XGBOOST CALIBRATION PLOT (TRAINING)')
plt.show()

pullm_xgb_matrix = confusion_matrix(pullm_y_test, pullm_xgb_ccv.predict_proba(pullm_x_test)[:, 1] > 0.5)
pullm_xgb_df = pd.DataFrame(pullm_xgb_matrix)

sns.heatmap(pullm_xgb_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('XGBOOST CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# Vancouver XGBoost tuning
vanco_class_weight = vanco_y_train[vanco_y_train == 0].count() / vanco_y_train[vanco_y_train == 1].count()
vanco_hyperparameters = [{'max_depth': np.linspace(5, 15, 11, dtype=int, endpoint=True),
						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True)}]

vanco_gridsearch = HalvingGridSearchCV(XGBClassifier(n_estimators=100, scale_pos_weight=vanco_class_weight, eval_metric='logloss', use_label_encoder=False), vanco_hyperparameters, scoring='roc_auc', cv=3, verbose=0, n_jobs=-1)
vanco_best_model = vanco_gridsearch.fit(vanco_x_train, vanco_y_train)

print(f'Best parameters: {vanco_gridsearch.best_params_}')

#%%
# Vancouver XGB
vanco_class_weight = vanco_y_train[vanco_y_train == 0].count() / vanco_y_train[vanco_y_train == 1].count()

vanco_xgb_ccv = XGBClassifier(n_estimators=1000, scale_pos_weight=vanco_class_weight, 
								eval_metric='logloss', **vanco_gridsearch.best_params_, use_label_encoder=False).fit(vanco_x_train, vanco_y_train)

# Vancouver XGB calibration
# vanco_xgb = XGBClassifier(n_estimators=1000, scale_pos_weight=vanco_class_weight, eval_metric='logloss', **vanco_gridsearch.best_params_, use_label_encoder=False)
# vanco_xgb_ccv = CalibratedClassifierCV(vanco_xgb, method='isotonic', cv=5).fit(vanco_x_train, vanco_y_train)

vanco_xgb_probs = vanco_xgb_ccv.predict_proba(vanco_x_train)
vanco_xgb_probs = vanco_xgb_probs[:, 1]
vanco_xgb_auc = roc_auc_score(vanco_y_train, vanco_xgb_probs)

print(f'\nOverall accuracy for Vancouver XGB model (training): {vanco_xgb_ccv.score(vanco_x_train, vanco_y_train):.4f}')
print(f'ROC AUC for Vancouver XGB model (training): {vanco_xgb_auc:.4f}')
print(f'Overall accuracy for Vancouver XGB model (testing): {vanco_xgb_ccv.score(vanco_x_test, vanco_y_test):.4f}')

vanco_xgb_fpr, vanco_xgb_tpr, vanco_thresholds = roc_curve(vanco_y_train, vanco_xgb_probs, drop_intermediate=False)

plt.plot(vanco_xgb_fpr, vanco_xgb_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('XGBOOST ROC CURVE (TRAINING)')
plt.show()

vanco_xgb_y, vanco_xgb_x = calibration_curve(vanco_y_train, vanco_xgb_probs, n_bins=10)

plt.plot(vanco_xgb_y, vanco_xgb_x, marker = '.', color=wsu_color, lw=6, label = 'XGBoost Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('XGBOOST CALIBRATION PLOT (TRAINING)')
plt.show()

vanco_xgb_matrix = confusion_matrix(vanco_y_test, vanco_xgb_ccv.predict_proba(vanco_x_test)[:, 1] > 0.5)
vanco_xgb_df = pd.DataFrame(vanco_xgb_matrix)

sns.heatmap(vanco_xgb_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('XGBOOST CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# Tri-Cities XGBoost tuning
trici_class_weight = trici_y_train[trici_y_train == 0].count() / trici_y_train[trici_y_train == 1].count()
trici_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True)}]

trici_gridsearch = HalvingGridSearchCV(XGBClassifier(n_estimators=100, scale_pos_weight=trici_class_weight, eval_metric='logloss', use_label_encoder=False), trici_hyperparameters, scoring='roc_auc', cv=3, verbose=0, n_jobs=-1)
trici_best_model = trici_gridsearch.fit(trici_x_train, trici_y_train)

print(f'Best parameters: {trici_gridsearch.best_params_}')

#%%
# Tri-Cities XGB
trici_class_weight = trici_y_train[trici_y_train == 0].count() / trici_y_train[trici_y_train == 1].count()

trici_xgb_ccv = XGBClassifier(n_estimators=1000, scale_pos_weight=trici_class_weight, 
								eval_metric='logloss', **trici_gridsearch.best_params_, use_label_encoder=False).fit(trici_x_train, trici_y_train)

# Tri-Cities XGB calibration
# trici_xgb = XGBClassifier(n_estimators=1000, scale_pos_weight=trici_class_weight, eval_metric='logloss', **trici_gridsearch.best_params_, use_label_encoder=False)
# trici_xgb_ccv = CalibratedClassifierCV(trici_xgb, method='isotonic', cv=5).fit(trici_x_train, trici_y_train)

trici_xgb_probs = trici_xgb_ccv.predict_proba(trici_x_train)
trici_xgb_probs = trici_xgb_probs[:, 1]
trici_xgb_auc = roc_auc_score(trici_y_train, trici_xgb_probs)

print(f'\nOverall accuracy for Tri-Cities XGB model (training): {trici_xgb_ccv.score(trici_x_train, trici_y_train):.4f}')
print(f'ROC AUC for Tri-Cities XGB model (training): {trici_xgb_auc:.4f}')
print(f'Overall accuracy for Tri-Cities XGB model (testing): {trici_xgb_ccv.score(trici_x_test, trici_y_test):.4f}')

trici_xgb_fpr, trici_xgb_tpr, trici_thresholds = roc_curve(trici_y_train, trici_xgb_probs, drop_intermediate=False)

plt.plot(trici_xgb_fpr, trici_xgb_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('XGBOOST ROC CURVE (TRAINING)')
plt.show()

trici_xgb_y, trici_xgb_x = calibration_curve(trici_y_train, trici_xgb_probs, n_bins=10)

plt.plot(trici_xgb_y, trici_xgb_x, marker = '.', color=wsu_color, lw=6, label = 'XGBoost Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('XGBOOST CALIBRATION PLOT (TRAINING)')
plt.show()

trici_xgb_matrix = confusion_matrix(trici_y_test, trici_xgb_ccv.predict_proba(trici_x_test)[:, 1] > 0.5)
trici_xgb_df = pd.DataFrame(trici_xgb_matrix)

sns.heatmap(trici_xgb_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('XGBOOST CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# University XGBoost tuning
univr_class_weight = univr_y_train[univr_y_train == 0].count() / univr_y_train[univr_y_train == 1].count()
univr_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True)}]

univr_gridsearch = HalvingGridSearchCV(XGBClassifier(n_estimators=100, scale_pos_weight=univr_class_weight, eval_metric='logloss', use_label_encoder=False), univr_hyperparameters, scoring='roc_auc', cv=3, verbose=0, n_jobs=-1)
univr_best_model = univr_gridsearch.fit(univr_x_train, univr_y_train)

print(f'Best parameters: {univr_gridsearch.best_params_}')

#%%
# University XGB
class_weight = univr_y_train[univr_y_train == 0].count() / univr_y_train[univr_y_train == 1].count()

univr_xgb_ccv = XGBClassifier(n_estimators=1000, scale_pos_weight=univr_class_weight, 
								eval_metric='logloss', **univr_gridsearch.best_params_, use_label_encoder=False).fit(univr_x_train, univr_y_train)

# University XGB calibration
# univr_xgb = XGBClassifier(n_estimators=1000, scale_pos_weight=univr_class_weight, eval_metric='logloss', **univr_gridsearch.best_params_, use_label_encoder=False)
# univr_xgb_ccv = CalibratedClassifierCV(univr_xgb, method='isotonic', cv=5).fit(univr_x_train, univr_y_train)

univr_xgb_probs = univr_xgb_ccv.predict_proba(univr_x_train)
univr_xgb_probs = univr_xgb_probs[:, 1]
univr_xgb_auc = roc_auc_score(univr_y_train, univr_xgb_probs)

print(f'\nOverall accuracy for University XGB model (training): {univr_xgb_ccv.score(univr_x_train, univr_y_train):.4f}')
print(f'ROC AUC for University XGB model (training): {univr_xgb_auc:.4f}')
print(f'Overall accuracy for University XGB model (testing): {univr_xgb_ccv.score(univr_x_test, univr_y_test):.4f}')

univr_xgb_fpr, univr_xgb_tpr, univr_thresholds = roc_curve(univr_y_train, univr_xgb_probs, drop_intermediate=False)

plt.plot(univr_xgb_fpr, univr_xgb_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('XGBOOST ROC CURVE (TRAINING)')
plt.show()

univr_xgb_y, univr_xgb_x = calibration_curve(univr_y_train, univr_xgb_probs, n_bins=10)

plt.plot(univr_xgb_y, univr_xgb_x, marker = '.', color=wsu_color, lw=6, label = 'XGBoost Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('XGBOOST CALIBRATION PLOT (TRAINING)')
plt.show()

univr_xgb_matrix = confusion_matrix(univr_y_test, univr_xgb_ccv.predict_proba(univr_x_test)[:, 1] > 0.5)
univr_xgb_df = pd.DataFrame(univr_xgb_matrix)

sns.heatmap(univr_xgb_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('XGBOOST CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# XGBoost Random Forest model

# Pullman XGBoost Random Forest tuning
pullm_class_weight = pullm_y_train[pullm_y_train == 0].count() / pullm_y_train[pullm_y_train == 1].count()
pullm_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
						'learning_rate': [0.01, 0.5, 1.0]}]

pullm_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, colsample_bynode=colsample_bynode, scale_pos_weight=pullm_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), pullm_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=verbose, n_jobs=-1)
pullm_best_model = pullm_gridsearch.fit(pullm_x_train, pullm_y_train)

print(f'Best parameters: {pullm_gridsearch.best_params_}')

#%%
# Pullman XGB Random Forest
pullm_class_weight = pullm_y_train[pullm_y_train == 0].count() / pullm_y_train[pullm_y_train == 1].count()
pullm_xgbrf_ccv = XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, colsample_bynode=colsample_bynode, scale_pos_weight=pullm_class_weight, 
								eval_metric='logloss', **pullm_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(pullm_x_train, pullm_y_train, eval_set=[(pullm_x_cv, pullm_y_cv)], early_stopping_rounds=20, verbose=False)

# Pullman XGB Random Forest calibration
# pullm_xgbrf = XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, scale_pos_weight=pullm_class_weight, 
# 								eval_metric='logloss', **pullm_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(pullm_x_train, pullm_y_train, eval_set=[(pullm_x_cv, pullm_y_cv)], early_stopping_rounds=20, verbose=False)
# pullm_xgbrf_ccv = CalibratedClassifierCV(pullm_xgbrf, method='isotonic', cv=5).fit(pullm_x_train, pullm_y_train)

pullm_xgbrf_probs = pullm_xgbrf_ccv.predict_proba(pullm_x_train)
pullm_xgbrf_probs = pullm_xgbrf_probs[:, 1]
pullm_xgbrf_auc = roc_auc_score(pullm_y_train, pullm_xgbrf_probs)

print(f'\nOverall accuracy for Pullman XGB Random Forest model (training): {pullm_xgbrf_ccv.score(pullm_x_train, pullm_y_train):.4f}')
print(f'ROC AUC for Pullman XGB Random Forest model (training): {pullm_xgbrf_auc:.4f}')
print(f'Overall accuracy for Pullman XGB Random Forest model (testing): {pullm_xgbrf_ccv.score(pullm_x_test, pullm_y_test):.4f}')

pullm_xgbrf_fpr, pullm_xgbrf_tpr, pullm_thresholds = roc_curve(pullm_y_train, pullm_xgbrf_probs, drop_intermediate=False)

plt.plot(pullm_xgbrf_fpr, pullm_xgbrf_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=6, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('XGBOOST ROC CURVE (TRAINING)')
plt.show()

pullm_xgbrf_y, pullm_xgbrf_x = calibration_curve(pullm_y_train, pullm_xgbrf_probs, n_bins=10)

plt.plot(pullm_xgbrf_y, pullm_xgbrf_x, marker = '.', color=wsu_color, lw=6, label = 'XGBoost Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=6, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('XGBOOST CALIBRATION PLOT (TRAINING)')
plt.show()

pullm_xgbrf_matrix = confusion_matrix(pullm_y_test, pullm_xgbrf_ccv.predict_proba(pullm_x_test)[:, 1] > 0.5)
pullm_xgbrf_df = pd.DataFrame(pullm_xgbrf_matrix)

sns.heatmap(pullm_xgbrf_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('XGBOOST CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

pullm_metrics = {
	'accuracy': accuracy_score,
    'tpr': true_positive_rate,
    'tnr': true_negative_rate,
    'balanced': balanced_accuracy_score,
    'matrix': confusion_matrix,
    'headcount': count
}

pullm_group = pd.DataFrame()

pullm_group['male'] = pullm_x_test[:, 7]
pullm_group['underrep_minority'] = pullm_x_test[:, 8]

pullm_metric_frame = MetricFrame(
    metrics=pullm_metrics, y_true=pullm_y_test, y_pred=pullm_xgbrf_ccv.predict(pullm_x_test), sensitive_features=pullm_group
)

print('Pullman metric differences by sex indicator...\n')
print(pullm_metric_frame.by_group)
print('\n')

#%%
# Vancouver XGBoost Random Forest tuning
vanco_class_weight = vanco_y_train[vanco_y_train == 0].count() / vanco_y_train[vanco_y_train == 1].count()
vanco_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
						'learning_rate': [0.01, 0.5, 1.0]}]

vanco_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, scale_pos_weight=vanco_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), vanco_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=False, n_jobs=-1)
vanco_best_model = vanco_gridsearch.fit(vanco_x_train, vanco_y_train)

print(f'Best parameters: {vanco_gridsearch.best_params_}')

#%%
# Vancouver XGB Random Forest
vanco_class_weight = vanco_y_train[vanco_y_train == 0].count() / vanco_y_train[vanco_y_train == 1].count()

vanco_xgbrf_ccv = XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, scale_pos_weight=vanco_class_weight, 
								eval_metric='logloss', **vanco_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(vanco_x_train, vanco_y_train, eval_set=[(vanco_x_cv, vanco_y_cv)], early_stopping_rounds=20, verbose=False)

# Vancouver XGB Random Forest calibration
# vanco_xgbrf = XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, scale_pos_weight=vanco_class_weight, 
# 								eval_metric='logloss', **vanco_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(vanco_x_train, vanco_y_train, eval_set=[(vanco_x_cv, vanco_y_cv)], early_stopping_rounds=20, verbose=False)
# vanco_xgbrf_ccv = CalibratedClassifierCV(vanco_xgbrf, method='isotonic', cv=5).fit(vanco_x_train, vanco_y_train)

vanco_xgbrf_probs = vanco_xgbrf_ccv.predict_proba(vanco_x_train)
vanco_xgbrf_probs = vanco_xgbrf_probs[:, 1]
vanco_xgbrf_auc = roc_auc_score(vanco_y_train, vanco_xgbrf_probs)

print(f'\nOverall accuracy for Vancouver XGB Random Forest model (training): {vanco_xgbrf_ccv.score(vanco_x_train, vanco_y_train):.4f}')
print(f'ROC AUC for Vancouver XGB Random Forest model (training): {vanco_xgbrf_auc:.4f}')
print(f'Overall accuracy for Vancouver XGB Random Forest model (testing): {vanco_xgbrf_ccv.score(vanco_x_test, vanco_y_test):.4f}')

vanco_xgbrf_fpr, vanco_xgbrf_tpr, vanco_thresholds = roc_curve(vanco_y_train, vanco_xgbrf_probs, drop_intermediate=False)

plt.plot(vanco_xgbrf_fpr, vanco_xgbrf_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=6, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('XGBOOST ROC CURVE (TRAINING)')
plt.show()

vanco_xgbrf_y, vanco_xgbrf_x = calibration_curve(vanco_y_train, vanco_xgbrf_probs, n_bins=10)

plt.plot(vanco_xgbrf_y, vanco_xgbrf_x, marker = '.', color=wsu_color, lw=6, label = 'XGBoost Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=6, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('XGBOOST CALIBRATION PLOT (TRAINING)')
plt.show()

vanco_xgbrf_matrix = confusion_matrix(vanco_y_test, vanco_xgbrf_ccv.predict_proba(vanco_x_test)[:, 1] > 0.5)
vanco_xgbrf_df = pd.DataFrame(vanco_xgbrf_matrix)

sns.heatmap(vanco_xgbrf_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('XGBOOST CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

vanco_metrics = {
	'accuracy': accuracy_score,
    'tpr': true_positive_rate,
    'tnr': true_negative_rate,
    'balanced': balanced_accuracy_score,
    'matrix': confusion_matrix,
    'headcount': count
}

vanco_group = pd.DataFrame()

vanco_group['male'] = vanco_x_test[:, 7]
vanco_group['underrep_minority'] = vanco_x_test[:, 8]

vanco_metric_frame = MetricFrame(
    metrics=vanco_metrics, y_true=vanco_y_test, y_pred=vanco_xgbrf_ccv.predict(vanco_x_test), sensitive_features=vanco_group
)

print('Vancouver metric differences by sex indicator...\n')
print(vanco_metric_frame.by_group)
print('\n')

#%%
# Tri-Cities XGBoost Random Forest tuning
trici_class_weight = trici_y_train[trici_y_train == 0].count() / trici_y_train[trici_y_train == 1].count()
trici_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
						'learning_rate': [0.01, 0.5, 1.0]}]

trici_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, scale_pos_weight=trici_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), trici_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=False, n_jobs=-1)
trici_best_model = trici_gridsearch.fit(trici_x_train, trici_y_train)

print(f'Best parameters: {trici_gridsearch.best_params_}')

#%%
# Tri-Cities XGB Random Forest
trici_class_weight = trici_y_train[trici_y_train == 0].count() / trici_y_train[trici_y_train == 1].count()

trici_xgbrf_ccv = XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, scale_pos_weight=trici_class_weight, 
								eval_metric='logloss', **trici_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(trici_x_train, trici_y_train, eval_set=[(trici_x_cv, trici_y_cv)], early_stopping_rounds=20, verbose=False)

# Tri-Cities XGB Random Forest calibration
# trici_xgbrf = XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, scale_pos_weight=trici_class_weight, 
# 								eval_metric='logloss', **trici_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(trici_x_train, trici_y_train, eval_set=[(trici_x_cv, trici_y_cv)], early_stopping_rounds=20, verbose=False)
# trici_xgbrf_ccv = CalibratedClassifierCV(trici_xgbrf, method='isotonic', cv=5).fit(trici_x_train, trici_y_train)

trici_xgbrf_probs = trici_xgbrf_ccv.predict_proba(trici_x_train)
trici_xgbrf_probs = trici_xgbrf_probs[:, 1]
trici_xgbrf_auc = roc_auc_score(trici_y_train, trici_xgbrf_probs)

print(f'\nOverall accuracy for Tri-Cities XGB Random Forest model (training): {trici_xgbrf_ccv.score(trici_x_train, trici_y_train):.4f}')
print(f'ROC AUC for Tri-Cities XGB Random Forest model (training): {trici_xgbrf_auc:.4f}')
print(f'Overall accuracy for Tri-Cities XGB Random Forest model (testing): {trici_xgbrf_ccv.score(trici_x_test, trici_y_test):.4f}')

trici_xgbrf_fpr, trici_xgbrf_tpr, trici_thresholds = roc_curve(trici_y_train, trici_xgbrf_probs, drop_intermediate=False)

plt.plot(trici_xgbrf_fpr, trici_xgbrf_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=6, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('XGBOOST ROC CURVE (TRAINING)')
plt.show()

trici_xgbrf_y, trici_xgbrf_x = calibration_curve(trici_y_train, trici_xgbrf_probs, n_bins=10)

plt.plot(trici_xgbrf_y, trici_xgbrf_x, marker = '.', color=wsu_color, lw=6, label = 'XGBoost Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=6, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('XGBOOST CALIBRATION PLOT (TRAINING)')
plt.show()

trici_xgbrf_matrix = confusion_matrix(trici_y_test, trici_xgbrf_ccv.predict_proba(trici_x_test)[:, 1] > 0.5)
trici_xgbrf_df = pd.DataFrame(trici_xgbrf_matrix)

sns.heatmap(trici_xgbrf_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('XGBOOST CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

trici_metrics = {
	'accuracy': accuracy_score,
    'tpr': true_positive_rate,
    'tnr': true_negative_rate,
    'balanced': balanced_accuracy_score,
    'matrix': confusion_matrix,
    'headcount': count
}

trici_group = pd.DataFrame()

trici_group['male'] = trici_x_test[:, 7]
trici_group['underrep_minority'] = trici_x_test[:, 8]

trici_metric_frame = MetricFrame(
    metrics=trici_metrics, y_true=trici_y_test, y_pred=trici_xgbrf_ccv.predict(trici_x_test), sensitive_features=trici_group
)

print('Tri-Cities metric differences by sex indicator...\n')
print(trici_metric_frame.by_group)
print('\n')

#%%
# University XGBoost Random Forest tuning
univr_class_weight = univr_y_train[univr_y_train == 0].count() / univr_y_train[univr_y_train == 1].count()
univr_hyperparameters = [{'max_depth': np.linspace(1, 16, 16, dtype=int, endpoint=True),
						'gamma': np.linspace(1, 16, 16, dtype=int, endpoint=True),
						'learning_rate': [0.01, 0.5, 1.0]}]

univr_gridsearch = HalvingGridSearchCV(XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, scale_pos_weight=univr_class_weight, eval_metric='logloss', use_label_encoder=False, n_jobs=-1), univr_hyperparameters, resource='n_estimators', factor=3, min_resources=2, max_resources=500, scoring='roc_auc', cv=3, aggressive_elimination=True, verbose=False, n_jobs=-1)
univr_best_model = univr_gridsearch.fit(univr_x_train, univr_y_train)

print(f'Best parameters: {univr_gridsearch.best_params_}')

#%%
# University XGB Random Forest
class_weight = univr_y_train[univr_y_train == 0].count() / univr_y_train[univr_y_train == 1].count()

univr_xgbrf_ccv = XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, scale_pos_weight=univr_class_weight, 
								eval_metric='logloss', **univr_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(univr_x_train, univr_y_train, eval_set=[(univr_x_cv, univr_y_cv)], early_stopping_rounds=20, verbose=False)

# University XGB Random Forest calibration
# univr_xgbrf = XGBClassifier(tree_method='hist', grow_policy='depthwise', min_child_weight=min_child_weight, max_bin=max_bin, num_parallel_tree=num_parallel_tree, subsample=subsample, colsample_bytree=colsample_bytree, scale_pos_weight=univr_class_weight, 
# 								eval_metric='logloss', **univr_gridsearch.best_params_, use_label_encoder=False, n_jobs=-1).fit(univr_x_train, univr_y_train, eval_set=[(univr_x_cv, univr_y_cv)], early_stopping_rounds=20, verbose=False)
# univr_xgbrf_ccv = CalibratedClassifierCV(univr_xgbrf, method='isotonic', cv=5).fit(univr_x_train, univr_y_train)

univr_xgbrf_probs = univr_xgbrf_ccv.predict_proba(univr_x_train)
univr_xgbrf_probs = univr_xgbrf_probs[:, 1]
univr_xgbrf_auc = roc_auc_score(univr_y_train, univr_xgbrf_probs)

print(f'\nOverall accuracy for University XGB Random Forest model (training): {univr_xgbrf_ccv.score(univr_x_train, univr_y_train):.4f}')
print(f'ROC AUC for University XGB Random Forest model (training): {univr_xgbrf_auc:.4f}')
print(f'Overall accuracy for University XGB Random Forest model (testing): {univr_xgbrf_ccv.score(univr_x_test, univr_y_test):.4f}')

univr_xgbrf_fpr, univr_xgbrf_tpr, univr_thresholds = roc_curve(univr_y_train, univr_xgbrf_probs, drop_intermediate=False)

plt.plot(univr_xgbrf_fpr, univr_xgbrf_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=6, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('XGBOOST ROC CURVE (TRAINING)')
plt.show()

univr_xgbrf_y, univr_xgbrf_x = calibration_curve(univr_y_train, univr_xgbrf_probs, n_bins=10)

plt.plot(univr_xgbrf_y, univr_xgbrf_x, marker = '.', color=wsu_color, lw=6, label = 'XGBoost Classifier')
plt.plot([0, 1], [0, 1], color='black', lw=6, linestyle = '--', label = 'Calibrated')

leg = plt.legend(loc = 'upper left')
plt.xlabel('AVERAGE PREDICTED PROBABILITY')
plt.ylabel('RATIO OF POSITIVES')
plt.title('XGBOOST CALIBRATION PLOT (TRAINING)')
plt.show()

univr_xgbrf_matrix = confusion_matrix(univr_y_test, univr_xgbrf_ccv.predict_proba(univr_x_test)[:, 1] > 0.5)
univr_xgbrf_df = pd.DataFrame(univr_xgbrf_matrix)

sns.heatmap(univr_xgbrf_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('XGBOOST CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

univr_metrics = {
	'accuracy': accuracy_score,
    'tpr': true_positive_rate,
    'tnr': true_negative_rate,
    'balanced': balanced_accuracy_score,
    'matrix': confusion_matrix,
    'headcount': count
}

univr_group = pd.DataFrame()

univr_group['male'] = univr_x_test[:, 7]
univr_group['underrep_minority'] = univr_x_test[:, 8]

univr_metric_frame = MetricFrame(
    metrics=univr_metrics, y_true=univr_y_test, y_pred=univr_xgbrf_ccv.predict(univr_x_test), sensitive_features=univr_group
)

print('University metric differences by sex indicator...\n')
print(univr_metric_frame.by_group)
print('\n')

#%%
# Ensemble model

# Pullman VCF
pullm_vcf = VotingClassifier(estimators=[('lreg', pullm_lreg_ccv), ('sgd', pullm_sgd_ccv)], voting='soft', weights=[1, 1]).fit(pullm_x_train, pullm_y_train)

pullm_vcf_probs_train = pullm_vcf.predict_proba(pullm_x_train)
pullm_vcf_probs_train = pullm_vcf_probs_train[:, 1]
pullm_vcf_auc_train = roc_auc_score(pullm_y_train, pullm_vcf_probs_train)

pullm_vcf_probs_test = pullm_vcf.predict_proba(pullm_x_test)
pullm_vcf_probs_test = pullm_vcf_probs_test[:, 1]
pullm_vcf_auc_test = roc_auc_score(pullm_y_test, pullm_vcf_probs_test)

print(f'\nOverall accuracy for Pullman ensemble model (training): {pullm_vcf.score(pullm_x_train, pullm_y_train):.4f}')
print(f'ROC AUC for Pullman ensemble model (training): {pullm_vcf_auc_train:.4f}')

pullm_vcf_fpr_train, pullm_vcf_tpr_train, pullm_thresholds_train = roc_curve(pullm_y_train, pullm_vcf_probs_train, drop_intermediate=False)

plt.plot(pullm_vcf_fpr_train, pullm_vcf_tpr_train, color=wsu_color, lw=4, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=4, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('ENSEMBLE ROC CURVE (TRAINING)')
plt.show()

print(f'Overall accuracy for Pullman ensemble model (testing): {pullm_vcf.score(pullm_x_test, pullm_y_test):.4f}')
print(f'ROC AUC for Pullman ensemble model (testing): {pullm_vcf_auc_test:.4f}')

pullm_vcf_fpr_test, pullm_vcf_tpr_test, pullm_thresholds_test = roc_curve(pullm_y_test, pullm_vcf_probs_test, drop_intermediate=False)

plt.plot(pullm_vcf_fpr_test, pullm_vcf_tpr_test, color=wsu_color, lw=4, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=4, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('ENSEMBLE ROC CURVE (TESTING)')
plt.show()

#%%
# Pullman VCF confusion matrix
pullm_vcf_matrix = confusion_matrix(pullm_y_test, pullm_vcf.predict(pullm_x_test))
pullm_vcf_df = pd.DataFrame(pullm_vcf_matrix)

sns.heatmap(pullm_vcf_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('ENSEMBLE CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# Vancouver VCF
vanco_vcf = VotingClassifier(estimators=[('lreg', vanco_lreg_ccv), ('sgd', vanco_sgd_ccv)], voting='soft', weights=[1, 1]).fit(vanco_x_train, vanco_y_train)

vanco_vcf_probs = vanco_vcf.predict_proba(vanco_x_train)
vanco_vcf_probs = vanco_vcf_probs[:, 1]
vanco_vcf_auc = roc_auc_score(vanco_y_train, vanco_vcf_probs)

print(f'\nOverall accuracy for Vancouver ensemble model (training): {vanco_vcf.score(vanco_x_train, vanco_y_train):.4f}')
print(f'ROC AUC for Vancouver ensemble model (training): {vanco_vcf_auc:.4f}')
print(f'Overall accuracy for Vancouver ensemble model (testing): {vanco_vcf.score(vanco_x_test, vanco_y_test):.4f}')

vanco_vcf_fpr, vanco_vcf_tpr, vanco_thresholds = roc_curve(vanco_y_train, vanco_vcf_probs, drop_intermediate=False)

plt.plot(vanco_vcf_fpr, vanco_vcf_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('ENSEMBLE ROC CURVE (TRAINING)')
plt.show()

#%%
# Vancouver VCF confusion matrix
vanco_vcf_matrix = confusion_matrix(vanco_y_test, vanco_vcf.predict(vanco_x_test))
vanco_vcf_df = pd.DataFrame(vanco_vcf_matrix)

sns.heatmap(vanco_vcf_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('ENSEMBLE CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# Tri-Cities VCF
trici_vcf = VotingClassifier(estimators=[('lreg', trici_lreg_ccv), ('sgd', trici_sgd_ccv)], voting='soft', weights=[1, 1]).fit(trici_x_train, trici_y_train)

trici_vcf_probs = trici_vcf.predict_proba(trici_x_train)
trici_vcf_probs = trici_vcf_probs[:, 1]
trici_vcf_auc = roc_auc_score(trici_y_train, trici_vcf_probs)

print(f'\nOverall accuracy for Tri-Cities ensemble model (training): {trici_vcf.score(trici_x_train, trici_y_train):.4f}')
print(f'ROC AUC for Tri-Cities ensemble model (training): {trici_vcf_auc:.4f}')
print(f'Overall accuracy for Tri-Cities ensemble model (testing): {trici_vcf.score(trici_x_test, trici_y_test):.4f}')

trici_vcf_fpr, trici_vcf_tpr, trici_thresholds = roc_curve(trici_y_train, trici_vcf_probs, drop_intermediate=False)

plt.plot(trici_vcf_fpr, trici_vcf_tpr, color=wsu_color, lw=6, label='ROC CURVE')
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('FALSE-POSITIVE RATE (1 - SPECIFICITY)')
plt.ylabel('TRUE-POSITIVE RATE (SENSITIVITY)')
plt.title('ENSEMBLE ROC CURVE (TRAINING)')
plt.show()

#%%
# Tri-Cities VCF confusion matrix
trici_vcf_matrix = confusion_matrix(trici_y_test, trici_vcf.predict(trici_x_test))
trici_vcf_df = pd.DataFrame(trici_vcf_matrix)

sns.heatmap(trici_vcf_df, annot=True, fmt='d', cbar=None, cmap=wsu_cmap)
plt.title('ENSEMBLE CONFUSION MATRIX'), plt.tight_layout()
plt.ylabel('TRUE CLASS'), plt.xlabel('PREDICTED CLASS')
plt.show()

#%%
# Pullman SHAP undersample
# pullm_under_shap = NearMiss(sampling_strategy={0:(pullm_y_train[pullm_y_train == 0].count()//5), 1:(pullm_y_train[pullm_y_train == 1].count()//5)}, version=3, n_jobs=-1)
# pullm_x_shap, pullm_y_shap = pullm_under_shap.fit_resample(pullm_x_train, pullm_y_train)

#%%
# Pullman SHAP training
pullm_explainer = shap.TreeExplainer(model=pullm_xgb_ccv, data=pullm_x_train, model_output='predict_proba')

#%%
# Pullman SHAP prediction
pullm_shap_values = pullm_explainer.shap_values(X=pullm_x_test)

#%%
# Pullman SHAP plots
# for index in range(len(pullm_shap_values[0])):
# 	shap.plots._waterfall.waterfall_legacy(pullm_explainer.expected_value[0], pullm_shap_values[0][index], pullm_x_test[index], feature_names=pullm_feat_names, max_display=4)

#%%
pullm_shap_results = []

for index in range(len(pullm_shap_values[0])):
	pullm_shap_results.extend(pd.DataFrame(data=pullm_shap_values[0][index].reshape(1, len(pullm_feat_names)), columns=pullm_feat_names).sort_values(by=0, axis=1, key=abs, ascending=False).to_dict(orient='records'))

pullm_shap_zip = dict(zip(pullm_shap_outcome, pullm_shap_results))

#%%
# Vancouver SHAP undersample
# vanco_under_shap = NearMiss(sampling_strategy={0:(vanco_y_train[vanco_y_train == 0].count()//2), 1:(vanco_y_train[vanco_y_train == 1].count()//2)}, version=3, n_jobs=-1)
# vanco_x_shap, vanco_y_shap = vanco_under_shap.fit_resample(vanco_x_train, vanco_y_train)

#%%
# Vancouver SHAP training
vanco_explainer = shap.TreeExplainer(model=vanco_xgb_ccv, data=vanco_x_train, model_output='predict_proba')

#%%
# Vancouver SHAP prediction
vanco_shap_values = vanco_explainer.shap_values(X=vanco_x_test)

#%%
# Vancouver SHAP plots
# for index in range(len(vanco_shap_values[0])):
# 	shap.plots._waterfall.waterfall_legacy(vanco_explainer.expected_value[0], vanco_shap_values[0][index], vanco_x_test[index], feature_names=vanco_feat_names, max_display=4)

#%%
vanco_shap_results = []

for index in range(len(vanco_shap_values[0])):
	vanco_shap_results.extend(pd.DataFrame(data=vanco_shap_values[0][index].reshape(1, len(vanco_feat_names)), columns=vanco_feat_names).sort_values(by=0, axis=1, key=abs, ascending=False).to_dict(orient='records'))

vanco_shap_zip = dict(zip(vanco_shap_outcome, vanco_shap_results))

#%%
# Tri-Cities SHAP undersample
# trici_under_shap = NearMiss(sampling_strategy={0:(trici_y_train[trici_y_train == 0].count()//2), 1:(trici_y_train[trici_y_train == 1].count()//2)}, version=3, n_jobs=-1)
# trici_x_shap, trici_y_shap = trici_under_shap.fit_resample(trici_x_train, trici_y_train)

#%%
# Tri-Cities SHAP training
trici_explainer = shap.TreeExplainer(model=trici_xgb_ccv, data=trici_x_train, model_output='predict_proba')

#%%
# Tri-Cities SHAP prediction
trici_shap_values = trici_explainer.shap_values(X=trici_x_test)

#%%
# Tri-Cities SHAP plots
# for index in range(len(trici_shap_values[0])):
# 	shap.plots._waterfall.waterfall_legacy(trici_explainer.expected_value[0], trici_shap_values[0][index], trici_x_test[index], feature_names=trici_feat_names, max_display=4)

#%%
trici_shap_results = []

for index in range(len(trici_shap_values[0])):
	trici_shap_results.extend(pd.DataFrame(data=trici_shap_values[0][index].reshape(1, len(trici_feat_names)), columns=trici_feat_names).sort_values(by=0, axis=1, key=abs, ascending=False).to_dict(orient='records'))

trici_shap_zip = dict(zip(trici_shap_outcome, trici_shap_results))

#%%
# Prepare model predictions

# Pullman probabilites
pullm_lreg_pred_probs = pullm_lreg_ccv.predict_proba(pullm_x_test)
pullm_lreg_pred_probs = pullm_lreg_pred_probs[:, 1]
pullm_sgd_pred_probs = pullm_sgd_ccv.predict_proba(pullm_x_test)
pullm_sgd_pred_probs = pullm_sgd_pred_probs[:, 1]
pullm_xgb_pred_probs = pullm_xgb_ccv.predict_proba(pullm_x_test)
pullm_xgb_pred_probs = pullm_xgb_pred_probs[:, 1]
pullm_mlp_pred_probs = pullm_mlp_ccv.predict_proba(pullm_x_test)
pullm_mlp_pred_probs = pullm_mlp_pred_probs[:, 1]
pullm_vcf_pred_probs = pullm_vcf.predict_proba(pullm_x_test)
pullm_vcf_pred_probs = pullm_vcf_pred_probs[:, 1]

#%%
# Vancouver probabilites
vanco_lreg_pred_probs = vanco_lreg_ccv.predict_proba(vanco_x_test)
vanco_lreg_pred_probs = vanco_lreg_pred_probs[:, 1]
vanco_sgd_pred_probs = vanco_sgd_ccv.predict_proba(vanco_x_test)
vanco_sgd_pred_probs = vanco_sgd_pred_probs[:, 1]
vanco_xgb_pred_probs = vanco_xgb_ccv.predict_proba(vanco_x_test)
vanco_xgb_pred_probs = vanco_xgb_pred_probs[:, 1]
vanco_mlp_pred_probs = vanco_mlp.predict_proba(vanco_x_test)
vanco_mlp_pred_probs = vanco_mlp_pred_probs[:, 1]
vanco_vcf_pred_probs = vanco_vcf.predict_proba(vanco_x_test)
vanco_vcf_pred_probs = vanco_vcf_pred_probs[:, 1]

#%%
# Tri-Cities probabilities
trici_lreg_pred_probs = trici_lreg_ccv.predict_proba(trici_x_test)
trici_lreg_pred_probs = trici_lreg_pred_probs[:, 1]
trici_sgd_pred_probs = trici_sgd_ccv.predict_proba(trici_x_test)
trici_sgd_pred_probs = trici_sgd_pred_probs[:, 1]
trici_xgb_pred_probs = trici_xgb_ccv.predict_proba(trici_x_test)
trici_xgb_pred_probs = trici_xgb_pred_probs[:, 1]
trici_mlp_pred_probs = trici_mlp.predict_proba(trici_x_test)
trici_mlp_pred_probs = trici_mlp_pred_probs[:, 1]
trici_vcf_pred_probs = trici_vcf.predict_proba(trici_x_test)
trici_vcf_pred_probs = trici_vcf_pred_probs[:, 1]

#%%
# Output model predictions to file

# Pullman predicted outcome
pullm_pred_outcome['lr_prob'] = pd.DataFrame(pullm_lreg_pred_probs)
pullm_pred_outcome['lr_pred'] = pullm_lreg_ccv.predict(pullm_x_test)
pullm_pred_outcome['sgd_prob'] = pd.DataFrame(pullm_sgd_pred_probs)
pullm_pred_outcome['sgd_pred'] = pullm_sgd_ccv.predict(pullm_x_test)
pullm_pred_outcome['xgb_prob'] = pd.DataFrame(pullm_xgb_pred_probs)
pullm_pred_outcome['xgb_pred'] = pullm_xgb_ccv.predict(pullm_x_test)
pullm_pred_outcome['mlp_prob'] = pd.DataFrame(pullm_mlp_pred_probs)
pullm_pred_outcome['mlp_pred'] = pullm_mlp_ccv.predict(pullm_x_test)
pullm_pred_outcome['vcf_prob'] = pd.DataFrame(pullm_vcf_pred_probs)
pullm_pred_outcome['vcf_pred'] = pullm_vcf.predict(pullm_x_test)

#%%
# Vancouver predicted outcome
vanco_pred_outcome['lr_prob'] = pd.DataFrame(vanco_lreg_pred_probs)
vanco_pred_outcome['lr_pred'] = vanco_lreg_ccv.predict(vanco_x_test)
vanco_pred_outcome['sgd_prob'] = pd.DataFrame(vanco_sgd_pred_probs)
vanco_pred_outcome['sgd_pred'] = vanco_sgd_ccv.predict(vanco_x_test)
vanco_pred_outcome['xgb_prob'] = pd.DataFrame(vanco_xgb_pred_probs)
vanco_pred_outcome['xgb_pred'] = vanco_xgb_ccv.predict(vanco_x_test)
vanco_pred_outcome['mlp_prob'] = pd.DataFrame(vanco_mlp_pred_probs)
vanco_pred_outcome['mlp_pred'] = vanco_mlp.predict(vanco_x_test)
vanco_pred_outcome['vcf_prob'] = pd.DataFrame(vanco_vcf_pred_probs)
vanco_pred_outcome['vcf_pred'] = vanco_vcf.predict(vanco_x_test)

#%%
# Tri-Cities predicted outcome
trici_pred_outcome['lr_prob'] = pd.DataFrame(trici_lreg_pred_probs)
trici_pred_outcome['lr_pred'] = trici_lreg_ccv.predict(trici_x_test)
trici_pred_outcome['sgd_prob'] = pd.DataFrame(trici_sgd_pred_probs)
trici_pred_outcome['sgd_pred'] = trici_sgd_ccv.predict(trici_x_test)
trici_pred_outcome['xgb_prob'] = pd.DataFrame(trici_xgb_pred_probs)
trici_pred_outcome['xgb_pred'] = trici_xgb_ccv.predict(trici_x_test)
trici_pred_outcome['mlp_prob'] = pd.DataFrame(trici_mlp_pred_probs)
trici_pred_outcome['mlp_pred'] = trici_mlp.predict(trici_x_test)
trici_pred_outcome['vcf_prob'] = pd.DataFrame(trici_vcf_pred_probs)
trici_pred_outcome['vcf_pred'] = trici_vcf.predict(trici_x_test)
