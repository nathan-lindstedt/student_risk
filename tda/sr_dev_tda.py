#%%
from datetime import date
import kmapper as km
import numpy as np
import pandas as pd
import sklearn
import umap
from patsy.highlevel import dmatrices
from sklearn import cluster
from sklearn import ensemble
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier

from student_risk import build_dev, config, helper_funcs

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
'high_school_gpa',
'high_school_gpa_mi',
'fall_midterm_gpa_avg',
'fall_midterm_gpa_avg_mi',
'fall_midterm_grade_count',
# 'fall_midterm_F_grade_count',
'fall_midterm_S_grade_count',
'fall_midterm_W_grade_count',
# 'fall_term_gpa',
# 'fall_term_gpa_mi',
# 'fall_term_no_letter_count',
# 'fall_term_F_grade_count',
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
'AP',
'RS',
'CHS',
# 'IB',
# 'AICE',
'IB_AICE', 
'fall_credit_hours',
# 'total_fall_units',
'fall_withdrawn_hours',
# 'fall_withdrawn_ind',
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
'unmet_need_acpt',
'unmet_need_acpt_mi'
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
'high_school_gpa',
'high_school_gpa_mi',
'fall_midterm_gpa_avg',
'fall_midterm_gpa_avg_mi',
'fall_midterm_grade_count',
# 'fall_midterm_F_grade_count',
# 'fall_midterm_S_grade_count',
# 'fall_midterm_W_grade_count',
# 'fall_term_gpa',
# 'fall_term_gpa_mi',
# 'fall_term_no_letter_count',
# 'fall_term_F_grade_count',
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
'unmet_need_acpt',
'unmet_need_acpt_mi'
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
'high_school_gpa',
'high_school_gpa_mi',
'fall_midterm_gpa_avg',
'fall_midterm_gpa_avg_mi',
'fall_midterm_grade_count',
# 'fall_midterm_F_grade_count',
# 'fall_midterm_S_grade_count',
# 'fall_midterm_W_grade_count',
# 'fall_term_gpa',
# 'fall_term_gpa_mi',
# 'fall_term_no_letter_count',
# 'fall_term_F_grade_count',
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
'unmet_need_acpt',
'unmet_need_acpt_mi'
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
pullm_id = np.array(pullm_training_set['emplid'].astype('int').to_list())
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
vanco_id = np.array(vanco_training_set['emplid'].astype('int').to_list())
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
# University outliers
univr_id = np.array(univr_training_set['emplid'].astype('int').to_list())
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
# Topological Data Analysis

# Pullman Isolation Forest Lens
projector = ensemble.IsolationForest(random_state=0, n_jobs=-1)
projector.fit(pullm_x_train)
pullm_proj = projector.decision_function(pullm_x_train)

#%%
# Pullman TDA Mapping
pullm_mapper = km.KeplerMapper(verbose=2)
pullm_cluster = pullm_mapper.fit_transform(pullm_x_train, projection='l2norm')
pullm_lens = np.c_[pullm_proj, pullm_cluster]
pullm_graph = pullm_mapper.map(pullm_lens, pullm_x_train, cover=km.Cover(n_cubes=20, perc_overlap=0.50), clusterer=cluster.AgglomerativeClustering(2))
pullm_mapper.visualize(pullm_graph, custom_tooltips=pullm_id, color_values=pullm_y_train, color_function_name="target", X=pullm_x_train, X_names=pullm_feat_names, path_html=f'pullm_mapper_visualization_output_{date.today()}.html')

#%%
# Vanouver Isolation Forest Lens
projector = ensemble.IsolationForest(random_state=0, n_jobs=-1)
projector.fit(vanco_x_train)
vanco_proj = projector.decision_function(vanco_x_train)

#%%
# Vanouver TDA Mapping
vanco_mapper = km.KeplerMapper(verbose=2)
vanco_cluster = vanco_mapper.fit_transform(vanco_x_train, projection='l2norm')
vanco_lens = np.c_[vanco_proj, vanco_cluster]
vanco_graph = vanco_mapper.map(vanco_lens, vanco_x_train, cover=km.Cover(n_cubes=20, perc_overlap=0.50), clusterer=cluster.AgglomerativeClustering(2))
vanco_mapper.visualize(vanco_graph, custom_tooltips=vanco_id, color_values=vanco_y_train, color_function_name="target", X=vanco_x_train, X_names=vanco_feat_names, path_html=f'vanco_mapper_visualization_output_{date.today()}.html')

#%%
# University Isolation Forest Lens
projector = ensemble.IsolationForest(random_state=0, n_jobs=-1)
projector.fit(univr_x_train)
univr_proj = projector.decision_function(univr_x_train)

#%%
# University TDA Mapping
univr_mapper = km.KeplerMapper(verbose=2)
univr_cluster = univr_mapper.fit_transform(univr_x_train, projection='l2norm')
univr_lens = np.c_[univr_proj, univr_cluster]
univr_graph = univr_mapper.map(univr_lens, univr_x_train, cover=km.Cover(n_cubes=20, perc_overlap=0.50), clusterer=cluster.AgglomerativeClustering(2))
univr_mapper.visualize(univr_graph, custom_tooltips=univr_id, color_values=univr_y_train, color_function_name="target", X=univr_x_train, X_names=univr_feat_names, path_html=f'univr_mapper_visualization_output_{date.today()}.html')
