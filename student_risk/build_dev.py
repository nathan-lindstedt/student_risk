#%%
import time

import saspy
from halo import HaloNotebook
from IPython.display import HTML


#%%
class DatasetBuilderDev:

	@staticmethod
	def build_admissions_dev():
		
		# Start SAS session
		print('\nStart SAS session...')

		sas = saspy.SASsession()

		sas.submit("""
		options sqlreduceput=all sqlremerge;
		run;
		""")

		# Set libname statements
		print('Set libname statements...')

		sas.submit("""
		%let dsn = census;
		%let adm = adm;
		""")

		sas.submit("""
		libname &dsn. odbc dsn=&dsn. schema=dbo;
		libname &adm. odbc dsn=&adm. schema=dbo;
		libname acs \"Z:\\Nathan\\Models\\student_risk\\supplemental_files\\\";
		""")

		print('Done\n')

		# Set macro variables
		print('Set macro variables...')

		sas.submit("""
		%let acs_lag = 2;
		%let lag_year = 1;
		%let end_cohort = %eval(&full_acad_year. - (2 * &lag_year.));
		%let start_cohort = %eval(&end_cohort. - 9);
		""")

		print('Done\n')

		# Import supplemental files
		print('Import supplemental files...')
		start = time.perf_counter()

		sas.submit("""
		proc import out=act_to_sat_engl_read
			datafile=\"Z:\\Nathan\\Models\\student_risk\\supplemental_files\\act_to_sat_engl_read.xlsx\"
			dbms=XLSX REPLACE;
			getnames=YES;
			run;
		""")

		sas.submit("""
		proc import out=act_to_sat_math
			datafile=\"Z:\\Nathan\\Models\\student_risk\\supplemental_files\\act_to_sat_math.xlsx\"
			dbms=XLSX REPLACE;
			getnames=YES;
			run;
		""")

		sas.submit("""
		proc import out=cpi
			datafile=\"Z:\\Nathan\\Models\\student_risk\\supplemental_files\\cpi.xlsx\"
			dbms=XLSX REPLACE;
			getnames=YES;
		run;
		""")

		stop = time.perf_counter()
		print(f'Done in {stop - start:.1f} seconds\n')

		# Create SAS macro
		print('Create SAS macro...')

		sas.submit("""
		%macro loop;

			%do cohort_year=&start_cohort. %to &end_cohort.;

			proc sql;
				create table cohort_&cohort_year. as
				select distinct a.*,
					substr(a.last_sch_postal,1,5) as targetid,
					case when a.sex = 'M' then 1 
						else 0
					end as male,
					case when a.age < 18.25 then 'Q1'
						when 18.25 <= a.age < 18.5 then 'Q2'
						when 18.5 <= a.age < 18.75 then 'Q3'
						when 18.75 <= a.age then 'Q4'
						else 'missing'
					end as age_group,
					case when a.father_attended_wsu_flag = 'Y' then 1 
						else 0
					end as father_wsu_flag,
					case when a.mother_attended_wsu_flag = 'Y' then 1 
						else 0
					end as mother_wsu_flag,
					case when a.ipeds_ethnic_group in ('2', '3', '5', '7', 'Z') then 1 
						else 0
					end as underrep_minority,
					case when a.WA_residency = 'RES' then 1
						else 0
					end as resident,
					case when a.adm_parent1_highest_educ_lvl in ('B','C','D','E','F') then '< bach'
						when a.adm_parent1_highest_educ_lvl = 'G' then 'bach'
						when a.adm_parent1_highest_educ_lvl in ('H','I','J','K','L') then '> bach'
							else 'missing'
					end as parent1_highest_educ_lvl,
					case when a.adm_parent2_highest_educ_lvl in ('B','C','D','E','F') then '< bach'
						when a.adm_parent2_highest_educ_lvl = 'G' then 'bach'
						when a.adm_parent2_highest_educ_lvl in ('H','I','J','K','L') then '> bach'
							else 'missing'
					end as parent2_highest_educ_lvl,
					b.distance as distance,
					l.cpi_adj,
					c.median_inc as median_inc_wo_cpi,
					c.median_inc*l.cpi_adj as median_inc,
					c.gini_indx,
					d.pvrt_total/d.pvrt_base as pvrt_rate,
					e.educ_total/e.educ_base as educ_rate,
					f.pop/(g.area*3.861E-7) as pop_dens,
					h.median_value as median_value_wo_cpi,
					h.median_value*l.cpi_adj as median_value,
					i.race_blk/i.race_tot as pct_blk,
					i.race_ai/i.race_tot as pct_ai,
					i.race_asn/i.race_tot as pct_asn,
					i.race_hawi/i.race_tot as pct_hawi,
					i.race_oth/i.race_tot as pct_oth,
					i.race_two/i.race_tot as pct_two,
					(i.race_blk + i.race_ai + i.race_asn + i.race_hawi + i.race_oth + i.race_two)/i.race_tot as pct_non,
					j.ethnic_hisp/j.ethnic_tot as pct_hisp,
					case when k.locale = '11' then 1 else 0 end as city_large,
					case when k.locale = '12' then 1 else 0 end as city_mid,
					case when k.locale = '13' then 1 else 0 end as city_small,
					case when k.locale = '21' then 1 else 0 end as suburb_large,
					case when k.locale = '22' then 1 else 0 end as suburb_mid,
					case when k.locale = '23' then 1 else 0 end as suburb_small,
					case when k.locale = '31' then 1 else 0 end as town_fringe,
					case when k.locale = '32' then 1 else 0 end as town_distant,
					case when k.locale = '33' then 1 else 0 end as town_remote,
					case when k.locale = '41' then 1 else 0 end as rural_fringe,
					case when k.locale = '42' then 1 else 0 end as rural_distant,
					case when k.locale = '43' then 1 else 0 end as rural_remote
				from &dsn..new_student_enrolled_vw as a
				left join acs.distance as b
					on substr(a.last_sch_postal,1,5) = b.targetid
				left join acs.acs_income_%eval(&cohort_year. - &acs_lag.) as c
					on substr(a.last_sch_postal,1,5) = c.geoid
				left join acs.acs_poverty_%eval(&cohort_year. - &acs_lag.) as d
					on substr(a.last_sch_postal,1,5) = d.geoid
				left join acs.acs_education_%eval(&cohort_year. - &acs_lag.) as e
					on substr(a.last_sch_postal,1,5) = e.geoid
				left join acs.acs_demo_%eval(&cohort_year. - &acs_lag.) as f
					on substr(a.last_sch_postal,1,5) = f.geoid
				left join acs.acs_area_%eval(&cohort_year. - &acs_lag.) as g
					on substr(a.last_sch_postal,1,5) = g.geoid
				left join acs.acs_housing_%eval(&cohort_year. - &acs_lag.) as h
					on substr(a.last_sch_postal,1,5) = h.geoid
				left join acs.acs_race_%eval(&cohort_year. - &acs_lag.) as i
					on substr(a.last_sch_postal,1,5) = i.geoid
				left join acs.acs_ethnicity_%eval(&cohort_year. - &acs_lag.) as j
					on substr(a.last_sch_postal,1,5) = j.geoid
				left join acs.edge_locale14_zcta_table as k
					on substr(a.last_sch_postal,1,5) = k.zcta5ce10
				left join cpi as l
					on input(a.full_acad_year, 4.) = l.acs_lag
				where a.full_acad_year = "&cohort_year"
					and substr(a.strm,4,1) = '7'
					and a.acad_career = 'UGRD'
					and a.adj_admit_type_cat in ('FRSH')
					and a.ipeds_full_part_time = 'F'
					and a.ipeds_ind = 1
					and a.term_credit_hours > 0
					and a.WA_residency ^= 'NON-I'
				order by a.emplid
			;quit;
			
			proc sql;
				create table new_student_&cohort_year. as
				select distinct
					emplid,
					pell_recipient_ind,
					eot_term_gpa,
					eot_term_gpa_hours
				from &dsn..new_student_profile_ugrd_cs
				where substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and adj_admit_type in ('FRS','IFR','IPF','TRN','ITR','IPT')
					and ipeds_full_part_time = 'F'
					and WA_residency ^= 'NON-I'
			;quit;

			%if &cohort_year. < &end_cohort. %then %do;
				proc sql;
					create table enrolled_&cohort_year. as
					select distinct 
						emplid, 
						term_code as cont_term,
						enrl_ind as enrl_ind
					from &dsn..student_enrolled_vw
					where snapshot = 'census'
						and full_acad_year = put(%eval(&cohort_year. + &lag_year.), 4.)
						and substr(strm,4,1) = '7'
						and acad_career = 'UGRD'
						and new_continue_status = 'CTU'
						and term_credit_hours > 0
					order by emplid
				;quit;
			%end;

			%if &cohort_year. = &end_cohort. %then %do;
				proc sql;
					create table enrolled_&cohort_year. as
					select distinct 
						emplid, 
						input(substr(strm, 1, 1) || '0' || substr(strm, 2, 2) || '3', 5.) as cont_term,
						enrl_ind as enrl_ind
					from acs.enrl_data
					where substr(strm,4,1) = '7'
						and acad_career = 'UGRD'
					order by emplid
				;quit;
			%end;

			proc sql;
				create table race_detail_&cohort_year. as
				select 
					a.emplid,
					case when hispc.emplid is not null 	then 'Y'
														else 'N'
														end as race_hispanic,
					case when amind.emplid is not null then 'Y'
													else 'N'
													end as race_american_indian,
					case when alask.emplid is not null then 'Y'
													else 'N'
													end as race_alaska,
					case when asian.emplid is not null then 'Y'
													else 'N'
													end as race_asian,
					case when black.emplid is not null then 'Y'
													else 'N'
													end as race_black,
					case when hawai.emplid is not null then 'Y'
													else 'N'
													end as race_native_hawaiian,
					case when white.emplid is not null then 'Y'
													else 'N'
													end as race_white
				from cohort_&cohort_year. as a
				left join (select distinct e4.emplid from &dsn..student_ethnic_detail as e4
							left join &dsn..xw_ethnic_detail_to_group_vw as xe4
								on e4.ethnic_cd = xe4.ethnic_cd
							where e4.snapshot = 'census'
								and e4.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe4.ethnic_group = '4') as asian
					on a.emplid = asian.emplid
				left join (select distinct e2.emplid from &dsn..student_ethnic_detail as e2
							left join &dsn..xw_ethnic_detail_to_group_vw as xe2
								on e2.ethnic_cd = xe2.ethnic_cd
							where e2.snapshot = 'census'
								and e2.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe2.ethnic_group = '2') as black
					on a.emplid = black.emplid
				left join (select distinct e7.emplid from &dsn..student_ethnic_detail as e7
							left join &dsn..xw_ethnic_detail_to_group_vw as xe7
								on e7.ethnic_cd = xe7.ethnic_cd
							where e7.snapshot = 'census'
								and e7.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe7.ethnic_group = '7') as hawai
					on a.emplid = hawai.emplid
				left join (select distinct e1.emplid from &dsn..student_ethnic_detail as e1
							left join &dsn..xw_ethnic_detail_to_group_vw as xe1
								on e1.ethnic_cd = xe1.ethnic_cd
							where e1.snapshot = 'census'
								and e1.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe1.ethnic_group = '1') as white
					on a.emplid = white.emplid
				left join (select distinct e5a.emplid from &dsn..student_ethnic_detail as e5a
							left join &dsn..xw_ethnic_detail_to_group_vw as xe5a
								on e5a.ethnic_cd = xe5a.ethnic_cd
							where e5a.snapshot = 'census' 
								and e5a.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe5a.ethnic_group = '5'
								and e5a.ethnic_cd in ('014','016','017','018',
														'935','941','942','943',
														'950','R10','R14')) as alask
					on a.emplid = alask.emplid
				left join (select distinct e5b.emplid from &dsn..student_ethnic_detail as e5b
							left join &dsn..xw_ethnic_detail_to_group_vw as xe5b
								on e5b.ethnic_cd = xe5b.ethnic_cd
							where e5b.snapshot = 'census'
								and e5b.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe5b.ethnic_group = '5'
								and e5b.ethnic_cd not in ('014','016','017','018',
															'935','941','942','943',
															'950','R14')) as amind
					on a.emplid = amind.emplid
				left join (select distinct e6.emplid from &dsn..student_ethnic_detail as e6
							left join &dsn..xw_ethnic_detail_to_group_vw as xe6
								on e6.ethnic_cd = xe6.ethnic_cd
							where e6.snapshot = 'census'
								and e6.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe6.ethnic_group = '3') as hispc
					on a.emplid = hispc.emplid
			;quit;

			proc sql;
				create table need_&cohort_year. as
				select distinct
					emplid,
					snapshot as need_snap,
					aid_year,
					fed_efc,
					fed_need
				from &dsn..fa_award_period
				where snapshot = "&aid_snapshot."
					and aid_year = "&cohort_year."	
					and award_period = 'A'
					and efc_status = 'O'
			;quit;

			proc sql;
				create table aid_&cohort_year. as
				select distinct
					emplid,
					snapshot as aid_snap,
					aid_year,
					sum(disbursed_amt) as total_disb,
					sum(offer_amt) as total_offer,
					sum(accept_amt) as total_accept
				from &dsn..fa_award_aid_year_vw
				where snapshot = "&aid_snapshot."
					and aid_year = "&cohort_year."
					and award_period in ('A','B')
					and award_status in ('A','O')
					and acad_career = 'UGRD'
				group by emplid
			;quit;

			proc sql;
				create table date_&cohort_year. as
				select distinct
					min(emplid) as emplid,
					min(week_from_term_begin_dt) as min_week_from_term_begin_dt,
					max(week_from_term_begin_dt) as max_week_from_term_begin_dt,
					count(week_from_term_begin_dt) as count_week_from_term_begin_dt
				from &adm..UGRD_shortened_vw
				where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and ugrd_applicant_counting_ind = 1
				group by emplid
				order by emplid;
			;quit;

			proc sql;
				create table class_registration_&cohort_year. as
				select distinct
					strm,
					emplid,
					class_nbr,
					crse_id,
					subject_catalog_nbr,
					ssr_component,
					unt_taken,
					credit_hours_earned,
					class_gpa,
					crse_grade_off
				from &dsn..class_registration_vw
				where snapshot = 'census'
					and strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and subject_catalog_nbr ^= 'NURS 399'
					and enrl_ind = 1
			;quit;
		
			proc sql;
				create table class_difficulty_&cohort_year. as
				select distinct
					a.subject_catalog_nbr,
					a.ssr_component,
					coalesce(b.total_grade_A, 0) + coalesce(c.total_grade_A, 0) + coalesce(d.total_grade_A, 0)
						+ coalesce(e.total_grade_A, 0) + coalesce(f.total_grade_A, 0) + coalesce(g.total_grade_A, 0) as total_grade_A,
					(calculated total_grade_A * 4.0) as total_grade_A_GPA,
					coalesce(b.total_grade_A_minus, 0) + coalesce(c.total_grade_A_minus, 0) + coalesce(d.total_grade_A_minus, 0)
						+ coalesce(e.total_grade_A_minus, 0) + coalesce(f.total_grade_A_minus, 0) + coalesce(g.total_grade_A_minus, 0) as total_grade_A_minus,
					(calculated total_grade_A_minus * 3.7) as total_grade_A_minus_GPA,
					coalesce(b.total_grade_B_plus, 0) + coalesce(c.total_grade_B_plus, 0) + coalesce(d.total_grade_B_plus, 0)
						+ coalesce(e.total_grade_B_plus, 0) + coalesce(f.total_grade_B_plus, 0) + coalesce(g.total_grade_B_plus, 0) as total_grade_B_plus,
					(calculated total_grade_B_plus * 3.3) as total_grade_B_plus_GPA,
					coalesce(b.total_grade_B, 0) + coalesce(c.total_grade_B, 0) + coalesce(d.total_grade_B, 0)
						+ coalesce(e.total_grade_B, 0) + coalesce(f.total_grade_B, 0) + coalesce(g.total_grade_B, 0) as total_grade_B,
					(calculated total_grade_B * 3.0) as total_grade_B_GPA,
					coalesce(b.total_grade_B_minus, 0) + coalesce(c.total_grade_B_minus, 0) + coalesce(d.total_grade_B_minus, 0) 
						+ coalesce(e.total_grade_B_minus, 0) + coalesce(f.total_grade_B_minus, 0) + coalesce(g.total_grade_B_minus, 0) as total_grade_B_minus,
					(calculated total_grade_B_minus * 2.7) as total_grade_B_minus_GPA,
					coalesce(b.total_grade_C_plus, 0) + coalesce(c.total_grade_C_plus, 0) + coalesce(d.total_grade_C_plus, 0) 
						+ coalesce(e.total_grade_C_plus, 0) + coalesce(f.total_grade_C_plus, 0) + coalesce(g.total_grade_C_plus, 0) as total_grade_C_plus,
					(calculated total_grade_C_plus * 2.3) as total_grade_C_plus_GPA,
					coalesce(b.total_grade_C, 0) + coalesce(c.total_grade_C, 0) + coalesce(d.total_grade_C, 0) 
						+ coalesce(e.total_grade_C, 0) + coalesce(f.total_grade_C, 0) + coalesce(g.total_grade_C, 0) as total_grade_C,
					(calculated total_grade_C * 2.0) as total_grade_C_GPA,
					coalesce(b.total_grade_C_minus, 0) + coalesce(c.total_grade_C_minus, 0) + coalesce(d.total_grade_C_minus, 0)
						+ coalesce(e.total_grade_C_minus, 0) + coalesce(f.total_grade_C_minus, 0) + coalesce(g.total_grade_C_minus, 0) as total_grade_C_minus,
					(calculated total_grade_C_minus * 1.7) as total_grade_C_minus_GPA,
					coalesce(b.total_grade_D_plus, 0) + coalesce(c.total_grade_D_plus, 0) + coalesce(d.total_grade_D_plus, 0)
						+ coalesce(e.total_grade_D_plus, 0) + coalesce(f.total_grade_D_plus, 0) + coalesce(g.total_grade_D_plus, 0) as total_grade_D_plus,
					(calculated total_grade_D_plus * 1.3) as total_grade_D_plus_GPA,
					coalesce(b.total_grade_D, 0) + coalesce(c.total_grade_D, 0) + coalesce(d.total_grade_D, 0) 
						+ coalesce(e.total_grade_D, 0) + coalesce(f.total_grade_D, 0) + coalesce(g.total_grade_D, 0) as total_grade_D,
					(calculated total_grade_D * 1.0) as total_grade_D_GPA,
					coalesce(b.total_grade_F, 0) + coalesce(c.total_grade_F, 0) + coalesce(d.total_grade_F, 0) 
						+ coalesce(e.total_grade_F, 0) + coalesce(f.total_grade_F, 0) + coalesce(g.total_grade_F, 0) as total_grade_F,
					coalesce(b.total_withdrawn, 0) + coalesce(c.total_withdrawn, 0) + coalesce(d.total_withdrawn, 0) 
						+ coalesce(e.total_withdrawn, 0) + coalesce(f.total_withdrawn, 0) + coalesce(g.total_withdrawn, 0) as total_withdrawn,
					coalesce(b.total_dropped, 0) + coalesce(c.total_dropped, 0) + coalesce(d.total_dropped, 0)
						+ coalesce(e.total_dropped, 0) + coalesce(f.total_dropped, 0) + coalesce(g.total_dropped, 0) as total_dropped,
					coalesce(b.total_grade_I, 0) + coalesce(c.total_grade_I, 0) + coalesce(d.total_grade_I, 0)
						+ coalesce(e.total_grade_I, 0) + coalesce(f.total_grade_I, 0) + coalesce(g.total_grade_I, 0) as total_grade_I,
					coalesce(b.total_grade_X, 0) + coalesce(c.total_grade_X, 0) + coalesce(d.total_grade_X, 0)
						+ coalesce(e.total_grade_X, 0) + coalesce(f.total_grade_X, 0) + coalesce(g.total_grade_X, 0) as total_grade_X,
					coalesce(b.total_grade_U, 0) + coalesce(c.total_grade_U, 0) + coalesce(d.total_grade_U, 0)
						+ coalesce(e.total_grade_U, 0) + coalesce(f.total_grade_U, 0) + coalesce(g.total_grade_U, 0) as total_grade_U,
					coalesce(b.total_grade_S, 0) + coalesce(c.total_grade_S, 0) + coalesce(d.total_grade_S, 0)
						+ coalesce(e.total_grade_S, 0) + coalesce(f.total_grade_S, 0) + coalesce(g.total_grade_S, 0) as total_grade_S,
					coalesce(b.total_grade_P, 0) + coalesce(c.total_grade_P, 0) + coalesce(d.total_grade_P, 0)
						+ coalesce(e.total_grade_P, 0) + coalesce(f.total_grade_P, 0) + coalesce(g.total_grade_P, 0) as total_grade_P,
					coalesce(b.total_no_grade, 0) + coalesce(c.total_no_grade, 0) + coalesce(d.total_no_grade, 0)
						+ coalesce(e.total_no_grade, 0) + coalesce(f.total_no_grade, 0) + coalesce(g.total_no_grade, 0) as total_no_grade,
					(calculated total_grade_A + calculated total_grade_A_minus 
						+ calculated total_grade_B_plus + calculated total_grade_B + calculated total_grade_B_minus
						+ calculated total_grade_C_plus + calculated total_grade_C + calculated total_grade_C_minus
						+ calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F) as total_grades,
					(calculated total_grade_A + calculated total_grade_A_minus 
						+ calculated total_grade_B_plus + calculated total_grade_B + calculated total_grade_B_minus
						+ calculated total_grade_C_plus + calculated total_grade_C + calculated total_grade_C_minus
						+ calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F + calculated total_withdrawn) as total_students,
					(calculated total_grade_A_GPA + calculated total_grade_A_minus_GPA 
						+ calculated total_grade_B_plus_GPA + calculated total_grade_B_GPA + calculated total_grade_B_minus_GPA
						+ calculated total_grade_C_plus_GPA + calculated total_grade_C_GPA + calculated total_grade_C_minus_GPA
						+ calculated total_grade_D_plus_GPA + calculated total_grade_D_GPA) as total_grades_GPA,
					(calculated total_grades_GPA / calculated total_grades) as class_average,
					(calculated total_withdrawn / calculated total_students) as pct_withdrawn,
					(calculated total_grade_C_minus + calculated total_grade_D_plus + calculated total_grade_D 
						+ calculated total_grade_F + calculated total_withdrawn) as CDFW,
					(calculated CDFW / calculated total_students) as pct_CDFW,
					(calculated total_grade_C_minus + calculated total_grade_D_plus + calculated total_grade_D 
						+ calculated total_grade_F) as CDF,
					(calculated CDF / calculated total_students) as pct_CDF,
					(calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F 
						+ calculated total_withdrawn) as DFW,
					(calculated DFW / calculated total_students) as pct_DFW,
					(calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F) as DF,
					(calculated DF / calculated total_students) as pct_DF
				from &dsn..class_vw as a
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'LEC'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as b
					on a.subject_catalog_nbr = b.subject_catalog_nbr
						and a.ssr_component = b.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'LAB'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as c
					on a.subject_catalog_nbr = c.subject_catalog_nbr
						and a.ssr_component = c.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'INT'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as d
					on a.subject_catalog_nbr = d.subject_catalog_nbr
						and a.ssr_component = d.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'STU'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as e
					on a.subject_catalog_nbr = e.subject_catalog_nbr
						and a.ssr_component = e.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'SEM'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as f
					on a.subject_catalog_nbr = f.subject_catalog_nbr
						and a.ssr_component = f.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as g
					on a.subject_catalog_nbr = g.subject_catalog_nbr
						and a.ssr_component = g.ssr_component
				where a.snapshot = 'eot'
					and a.full_acad_year = "&cohort_year."
					and a.grading_basis = 'GRD'
				order by a.subject_catalog_nbr
			;quit;

			proc sql;
				create table coursework_difficulty_&cohort_year. as
				select distinct
					a.emplid,
					avg(b.class_average) as fall_avg_difficulty,
					avg(b.pct_withdrawn) as fall_avg_pct_withdrawn,
					avg(b.pct_CDFW) as fall_avg_pct_CDFW,
					avg(b.pct_CDF) as fall_avg_pct_CDF,
					avg(b.pct_DFW) as fall_avg_pct_DFW,
					avg(b.pct_DF) as fall_avg_pct_DF
				from class_registration_&cohort_year. as a
				left join class_difficulty_&cohort_year. as b
					on a.subject_catalog_nbr = b.subject_catalog_nbr
						and a.ssr_component = b.ssr_component
						and a.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
				group by a.emplid
			;quit;

			proc sql;
				create table class_count_&cohort_year. as
				select distinct
					a.emplid,
					count(b.class_nbr) as fall_lec_count,
					count(c.class_nbr) as fall_lab_count,
					count(d.class_nbr) as fall_int_count,
					count(e.class_nbr) as fall_stu_count,
					count(f.class_nbr) as fall_sem_count,
					count(g.class_nbr) as fall_oth_count,
					sum(h.unt_taken) as fall_lec_units,
					sum(i.unt_taken) as fall_lab_units,
					sum(j.unt_taken) as fall_int_units,
					sum(k.unt_taken) as fall_stu_units,
					sum(l.unt_taken) as fall_sem_units,
					sum(m.unt_taken) as fall_oth_units,
					coalesce(calculated fall_lec_units, 0) + coalesce(calculated fall_lab_units, 0) + coalesce(calculated fall_int_units, 0) 
						+ coalesce(calculated fall_stu_units, 0) + coalesce(calculated fall_sem_units, 0) + coalesce(calculated fall_oth_units, 0) as total_fall_units
				from class_registration_&cohort_year. as a
				left join (select distinct emplid, 
								class_nbr
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LEC') as b
					on a.emplid = b.emplid
						and a.class_nbr = b.class_nbr
				left join (select distinct emplid, 
								class_nbr
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LAB') as c
					on a.emplid = c.emplid
						and a.class_nbr = c.class_nbr
				left join (select distinct emplid, 
								class_nbr
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'INT') as d
					on a.emplid = d.emplid
						and a.class_nbr = d.class_nbr
				left join (select distinct emplid, 
								class_nbr
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'STU') as e
					on a.emplid = e.emplid
						and a.class_nbr = e.class_nbr
				left join (select distinct emplid, 
								class_nbr
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'SEM') as f
					on a.emplid = f.emplid
						and a.class_nbr = f.class_nbr
				left join (select distinct emplid, 
								class_nbr
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')) as g
					on a.emplid = g.emplid
						and a.class_nbr = g.class_nbr
				left join (select distinct emplid, 
								class_nbr,
								unt_taken
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LEC') as h
					on a.emplid = h.emplid
						and a.class_nbr = h.class_nbr
				left join (select distinct emplid, 
								class_nbr,
								unt_taken
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LAB') as i
					on a.emplid = i.emplid
						and a.class_nbr = i.class_nbr
				left join (select distinct emplid, 
								class_nbr,
								unt_taken
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'INT') as j
					on a.emplid = j.emplid
						and a.class_nbr = j.class_nbr
				left join (select distinct emplid, 
								class_nbr,
								unt_taken
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'STU') as k
					on a.emplid = k.emplid
						and a.class_nbr = k.class_nbr
				left join (select distinct emplid, 
								class_nbr,
								unt_taken
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'SEM') as l
					on a.emplid = l.emplid
						and a.class_nbr = l.class_nbr
				left join (select distinct emplid, 
								class_nbr,
								unt_taken
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')) as m
					on a.emplid = m.emplid
						and a.class_nbr = m.class_nbr
				group by a.emplid
			;quit;

			proc sql;
				create table term_contact_hrs_&cohort_year. as
				select distinct
					a.emplid,
					sum(b.lec_contact_hrs) as fall_lec_contact_hrs,
					sum(c.lab_contact_hrs) as fall_lab_contact_hrs,
					sum(d.int_contact_hrs) as fall_int_contact_hrs,
					sum(e.stu_contact_hrs) as fall_stu_contact_hrs,
					sum(f.sem_contact_hrs) as fall_sem_contact_hrs,
					sum(g.oth_contact_hrs) as fall_oth_contact_hrs,
					coalesce(calculated fall_lec_contact_hrs, 0) + coalesce(calculated fall_lab_contact_hrs, 0) + coalesce(calculated fall_int_contact_hrs, 0) 
						+ coalesce(calculated fall_stu_contact_hrs, 0) + coalesce(calculated fall_sem_contact_hrs, 0) + coalesce(calculated fall_oth_contact_hrs, 0) as total_fall_contact_hrs
				from class_registration_&cohort_year. as a
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as lec_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'LEC'
							group by subject_catalog_nbr) as b
					on a.subject_catalog_nbr = b.subject_catalog_nbr
						and a.ssr_component = b.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as lab_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'LAB'
							group by subject_catalog_nbr) as c
					on a.subject_catalog_nbr = c.subject_catalog_nbr
						and a.ssr_component = c.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as int_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'INT'
							group by subject_catalog_nbr) as d
					on a.subject_catalog_nbr = d.subject_catalog_nbr
						and a.ssr_component = d.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as stu_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'STU'
							group by subject_catalog_nbr) as e
					on a.subject_catalog_nbr = e.subject_catalog_nbr
						and a.ssr_component = e.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as sem_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'SEM'
							group by subject_catalog_nbr) as f
					on a.subject_catalog_nbr = f.subject_catalog_nbr
						and a.ssr_component = f.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as oth_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')
							group by subject_catalog_nbr) as g
					on a.subject_catalog_nbr = g.subject_catalog_nbr
						and a.ssr_component = g.ssr_component
						and substr(a.strm,4,1) = '7'
				group by a.emplid
			;quit;

			proc sql;
				create table dataset_&cohort_year. as
				select 
					a.*,
					b.pell_recipient_ind,
					b.eot_term_gpa,
					b.eot_term_gpa_hours,
					c.cont_term,
					c.enrl_ind,
					e.need_snap,
					e.fed_efc,
					e.fed_need,
					f.aid_snap,
					f.total_disb,
					f.total_offer,
					f.total_accept,
					m.min_week_from_term_begin_dt,
					m.max_week_from_term_begin_dt,
					m.count_week_from_term_begin_dt,
					(4.0 - n.fall_avg_difficulty) as fall_avg_difficulty,
					n.fall_avg_pct_withdrawn,
					n.fall_avg_pct_CDFW,
					n.fall_avg_pct_CDF,
					n.fall_avg_pct_DFW,
					n.fall_avg_pct_DF,
					s.fall_lec_count,
					s.fall_lab_count,
					s.fall_int_count,
					s.fall_stu_count,
					s.fall_sem_count,
					s.fall_oth_count,
					s.total_fall_units,
					o.fall_lec_contact_hrs,
					o.fall_lab_contact_hrs,
					o.fall_int_contact_hrs,
					o.fall_stu_contact_hrs,
					o.fall_sem_contact_hrs,
					o.fall_oth_contact_hrs,
					o.total_fall_contact_hrs,
					t.race_american_indian,
					t.race_alaska,
					t.race_asian,
					t.race_black,
					t.race_native_hawaiian,
					t.race_white
				from cohort_&cohort_year. as a
				left join new_student_&cohort_year. as b
					on a.emplid = b.emplid
				left join enrolled_&cohort_year. as c
					on a.emplid = c.emplid
				left join need_&cohort_year. as e
					on a.emplid = e.emplid
						and a.aid_year = e.aid_year
				left join aid_&cohort_year. as f
					on a.emplid = f.emplid
						and a.aid_year = f.aid_year
				left join date_&cohort_year. as m
					on a.emplid = m.emplid
				left join coursework_difficulty_&cohort_year. as n
					on a.emplid = n.emplid
				left join term_contact_hrs_&cohort_year. as o
					on a.emplid = o.emplid
				left join class_count_&cohort_year. as s
					on a.emplid = s.emplid
				left join race_detail_&cohort_year. as t
					on a.emplid = t.emplid
			;quit;
			
			%end;

			proc sql;
				create table cohort_&cohort_year. as
				select distinct 
					a.*,
					p.admit_type,
					q.adj_admit_type_cat,
					case when a.sex = 'M' then 1 
							else 0
					end as male,
					b.*,
					case when b.WA_residency = 'RES' then 1
						else 0
					end as resident,
					case when b.adm_parent1_highest_educ_lvl in ('B','C','D','E','F') then '< bach'
						when b.adm_parent1_highest_educ_lvl = 'G' then 'bach'
						when b.adm_parent1_highest_educ_lvl in ('H','I','J','K','L') then '> bach'
							else 'missing'
					end as parent1_highest_educ_lvl,
					case when b.adm_parent2_highest_educ_lvl in ('B','C','D','E','F') then '< bach'
						when b.adm_parent2_highest_educ_lvl = 'G' then 'bach'
						when b.adm_parent2_highest_educ_lvl in ('H','I','J','K','L') then '> bach'
							else 'missing'
					end as parent2_highest_educ_lvl,
					d.*,
					case when d.ipeds_ethnic_group in ('2', '3', '5', '7', 'Z') then 1 
						else 0
					end as underrep_minority,
					substr(e.ext_org_postal,1,5) as targetid,
					f.distance as distance,
					g.median_inc,
					g.gini_indx,
					h.pvrt_total/h.pvrt_base as pvrt_rate,
					i.educ_total/i.educ_base as educ_rate,
					j.pop/(k.area*3.861E-7) as pop_dens,
					l.median_value,
					m.race_blk/m.race_tot as pct_blk,
					m.race_ai/m.race_tot as pct_ai,
					m.race_asn/m.race_tot as pct_asn,
					m.race_hawi/m.race_tot as pct_hawi,
					m.race_oth/m.race_tot as pct_oth,
					m.race_two/m.race_tot as pct_two,
					(m.race_blk + m.race_ai + m.race_asn + m.race_hawi + m.race_oth + m.race_two)/m.race_tot as pct_non,
					n.ethnic_hisp/n.ethnic_tot as pct_hisp,
					case when o.locale = '11' then 1 else 0 end as city_large,
					case when o.locale = '12' then 1 else 0 end as city_mid,
					case when o.locale = '13' then 1 else 0 end as city_small,
					case when o.locale = '21' then 1 else 0 end as suburb_large,
					case when o.locale = '22' then 1 else 0 end as suburb_mid,
					case when o.locale = '23' then 1 else 0 end as suburb_small,
					case when o.locale = '31' then 1 else 0 end as town_fringe,
					case when o.locale = '32' then 1 else 0 end as town_distant,
					case when o.locale = '33' then 1 else 0 end as town_remote,
					case when o.locale = '41' then 1 else 0 end as rural_fringe,
					case when o.locale = '42' then 1 else 0 end as rural_distant,
					case when o.locale = '43' then 1 else 0 end as rural_remote
				from &adm..fact_u as a
				left join &adm..xd_person_demo as b
					on a.sid_per_demo = b.sid_per_demo
				left join &adm..xd_admit_type as c
					on a.sid_admit_type = c.sid_admit_type
						and c.admit_type in ('FRS','IFR','IPF','TRN','ITR','IPT')
				left join &adm..xd_ipeds_ethnic_group as d
					on a.sid_ipeds_ethnic_group = d.sid_ipeds_ethnic_group
				left join &adm..xd_school as e
					on a.sid_ext_org_id = e.sid_ext_org_id
				left join acs.distance as f
					on substr(e.ext_org_postal,1,5) = f.targetid
				left join acs.acs_income_%eval(&cohort_year. - &acs_lag.) as g
					on substr(e.ext_org_postal,1,5) = g.geoid
				left join acs.acs_poverty_%eval(&cohort_year. - &acs_lag.) as h
					on substr(e.ext_org_postal,1,5) = h.geoid
				left join acs.acs_education_%eval(&cohort_year. - &acs_lag.) as i
					on substr(e.ext_org_postal,1,5) = i.geoid
				left join acs.acs_demo_%eval(&cohort_year. - &acs_lag.) as j
					on substr(e.ext_org_postal,1,5) = j.geoid
				left join acs.acs_area_%eval(&cohort_year. - &acs_lag.) as k
					on substr(e.ext_org_postal,1,5) = k.geoid
				left join acs.acs_housing_%eval(&cohort_year. - &acs_lag.) as l
					on substr(e.ext_org_postal,1,5) = l.geoid
				left join acs.acs_race_%eval(&cohort_year. - &acs_lag.) as m
					on substr(e.ext_org_postal,1,5) = m.geoid
				left join acs.acs_ethnicity_%eval(&cohort_year. - &acs_lag.) as n
					on substr(e.ext_org_postal,1,5) = n.geoid
				left join acs.edge_locale14_zcta_table as o
					on substr(e.ext_org_postal,1,5) = o.zcta5ce10
				left join &adm..xd_admit_type as p
					on a.sid_admit_type = p.sid_admit_type
				left join &dsn..xw_admit_type as q
					on p.admit_type = q.admit_type
				left join &adm..xd_person_demo as r
					on a.sid_per_demo = r.sid_per_demo
				where a.sid_snapshot = (select max(sid_snapshot) as sid_snapshot 
										from &adm..fact_u where strm = (substr(put(%eval(&cohort_year. - &lag_year.), z4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), z4.), 3, 2) || '7'))
					and a.acad_career = 'UGRD' 
					and a.enrolled = 1
					and q.adj_admit_type_cat in ('FRSH')
					and r.wa_residency ^= 'NON-I'
			;quit;

			proc sql;
				create table race_detail_&cohort_year. as
				select 
					a.emplid,
					case when hispc.emplid is not null 	then 'Y'
														else 'N'
														end as race_hispanic,
					case when amind.emplid is not null 	then 'Y'
														else 'N'
														end as race_american_indian,
					case when alask.emplid is not null 	then 'Y'
														else 'N'
														end as race_alaska,
					case when asian.emplid is not null 	then 'Y'
														else 'N'
														end as race_asian,
					case when black.emplid is not null 	then 'Y'
														else 'N'
														end as race_black,
					case when hawai.emplid is not null 	then 'Y'
														else 'N'
														end as race_native_hawaiian,
					case when white.emplid is not null 	then 'Y'
														else 'N'
														end as race_white
				from cohort_&cohort_year. as a
				left join (select distinct e4.emplid from &dsn..student_ethnic_detail as e4
							left join &dsn..xw_ethnic_detail_to_group_vw as xe4
								on e4.ethnic_cd = xe4.ethnic_cd
							where e4.snapshot = 'census'
								and e4.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe4.ethnic_group = '4') as asian
					on a.emplid = asian.emplid
				left join (select distinct e2.emplid from &dsn..student_ethnic_detail as e2
							left join &dsn..xw_ethnic_detail_to_group_vw as xe2
								on e2.ethnic_cd = xe2.ethnic_cd
							where e2.snapshot = 'census'
								and e2.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe2.ethnic_group = '2') as black
					on a.emplid = black.emplid
				left join (select distinct e7.emplid from &dsn..student_ethnic_detail as e7
							left join &dsn..xw_ethnic_detail_to_group_vw as xe7
								on e7.ethnic_cd = xe7.ethnic_cd
							where e7.snapshot = 'census'
								and e7.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe7.ethnic_group = '7') as hawai
					on a.emplid = hawai.emplid
				left join (select distinct e1.emplid from &dsn..student_ethnic_detail as e1
							left join &dsn..xw_ethnic_detail_to_group_vw as xe1
								on e1.ethnic_cd = xe1.ethnic_cd
							where e1.snapshot = 'census'
								and e1.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe1.ethnic_group = '1') as white
					on a.emplid = white.emplid
				left join (select distinct e5a.emplid from &dsn..student_ethnic_detail as e5a
							left join &dsn..xw_ethnic_detail_to_group_vw as xe5a
								on e5a.ethnic_cd = xe5a.ethnic_cd
							where e5a.snapshot = 'census' 
								and e5a.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe5a.ethnic_group = '5'
								and e5a.ethnic_cd in ('014','016','017','018',
														'935','941','942','943',
														'950','R10','R14')) as alask
					on a.emplid = alask.emplid
				left join (select distinct e5b.emplid from &dsn..student_ethnic_detail as e5b
							left join &dsn..xw_ethnic_detail_to_group_vw as xe5b
								on e5b.ethnic_cd = xe5b.ethnic_cd
							where e5b.snapshot = 'census'
								and e5b.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe5b.ethnic_group = '5'
								and e5b.ethnic_cd not in ('014','016','017','018',
															'935','941','942','943',
															'950','R14')) as amind
					on a.emplid = amind.emplid
				left join (select distinct e6.emplid from &dsn..student_ethnic_detail as e6
							left join &dsn..xw_ethnic_detail_to_group_vw as xe6
								on e6.ethnic_cd = xe6.ethnic_cd
							where e6.snapshot = 'census'
								and e6.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe6.ethnic_group = '3') as hispc
					on a.emplid = hispc.emplid
			;quit;
		
			proc sql;
				create table need_&cohort_year. as
				select distinct
					emplid,
					aid_year,
					max(fed_need) as fed_need
				from acs.finaid_data
					where aid_year = "&cohort_year."
				group by emplid, aid_year
			;quit;
		
			proc sql;
				create table aid_&cohort_year. as
				select distinct
					emplid,
					aid_year,
					sum(total_offer) as total_offer,
					sum(total_accept) as total_accept
				from acs.finaid_data
					where aid_year = "&cohort_year."
				group by emplid, aid_year
			;quit;
		
			proc sql;
				create table date_&cohort_year. as
				select distinct
					min(emplid) as emplid,
					min(week_from_term_begin_dt) as min_week_from_term_begin_dt,
					max(week_from_term_begin_dt) as max_week_from_term_begin_dt,
					count(week_from_term_begin_dt) as count_week_from_term_begin_dt
				from &adm..UGRD_shortened_vw
				where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and ugrd_applicant_counting_ind = 1
				group by emplid
				order by emplid;
			;quit;
		
			proc sql;
				create table class_registration_&cohort_year. as
				select distinct
					strm,
					emplid,
					class_nbr,
					crse_id,
					unt_taken,
					strip(subject) || ' ' || strip(catalog_nbr) as subject_catalog_nbr,
					ssr_component
				from acs.subcatnbr_data
				where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
			;quit;

			proc sql;
				create table class_difficulty_&cohort_year. as
				select distinct
					a.subject_catalog_nbr,
					a.ssr_component,
					coalesce(b.total_grade_A, 0) + coalesce(c.total_grade_A, 0) + coalesce(d.total_grade_A, 0)
						+ coalesce(e.total_grade_A, 0) + coalesce(f.total_grade_A, 0) + coalesce(g.total_grade_A, 0) as total_grade_A,
					(calculated total_grade_A * 4.0) as total_grade_A_GPA,
					coalesce(b.total_grade_A_minus, 0) + coalesce(c.total_grade_A_minus, 0) + coalesce(d.total_grade_A_minus, 0)
						+ coalesce(e.total_grade_A_minus, 0) + coalesce(f.total_grade_A_minus, 0) + coalesce(g.total_grade_A_minus, 0) as total_grade_A_minus,
					(calculated total_grade_A_minus * 3.7) as total_grade_A_minus_GPA,
					coalesce(b.total_grade_B_plus, 0) + coalesce(c.total_grade_B_plus, 0) + coalesce(d.total_grade_B_plus, 0)
						+ coalesce(e.total_grade_B_plus, 0) + coalesce(f.total_grade_B_plus, 0) + coalesce(g.total_grade_B_plus, 0) as total_grade_B_plus,
					(calculated total_grade_B_plus * 3.3) as total_grade_B_plus_GPA,
					coalesce(b.total_grade_B, 0) + coalesce(c.total_grade_B, 0) + coalesce(d.total_grade_B, 0)
						+ coalesce(e.total_grade_B, 0) + coalesce(f.total_grade_B, 0) + coalesce(g.total_grade_B, 0) as total_grade_B,
					(calculated total_grade_B * 3.0) as total_grade_B_GPA,
					coalesce(b.total_grade_B_minus, 0) + coalesce(c.total_grade_B_minus, 0) + coalesce(d.total_grade_B_minus, 0) 
						+ coalesce(e.total_grade_B_minus, 0) + coalesce(f.total_grade_B_minus, 0) + coalesce(g.total_grade_B_minus, 0) as total_grade_B_minus,
					(calculated total_grade_B_minus * 2.7) as total_grade_B_minus_GPA,
					coalesce(b.total_grade_C_plus, 0) + coalesce(c.total_grade_C_plus, 0) + coalesce(d.total_grade_C_plus, 0) 
						+ coalesce(e.total_grade_C_plus, 0) + coalesce(f.total_grade_C_plus, 0) + coalesce(g.total_grade_C_plus, 0) as total_grade_C_plus,
					(calculated total_grade_C_plus * 2.3) as total_grade_C_plus_GPA,
					coalesce(b.total_grade_C, 0) + coalesce(c.total_grade_C, 0) + coalesce(d.total_grade_C, 0) 
						+ coalesce(e.total_grade_C, 0) + coalesce(f.total_grade_C, 0) + coalesce(g.total_grade_C, 0) as total_grade_C,
					(calculated total_grade_C * 2.0) as total_grade_C_GPA,
					coalesce(b.total_grade_C_minus, 0) + coalesce(c.total_grade_C_minus, 0) + coalesce(d.total_grade_C_minus, 0)
						+ coalesce(e.total_grade_C_minus, 0) + coalesce(f.total_grade_C_minus, 0) + coalesce(g.total_grade_C_minus, 0) as total_grade_C_minus,
					(calculated total_grade_C_minus * 1.7) as total_grade_C_minus_GPA,
					coalesce(b.total_grade_D_plus, 0) + coalesce(c.total_grade_D_plus, 0) + coalesce(d.total_grade_D_plus, 0)
						+ coalesce(e.total_grade_D_plus, 0) + coalesce(f.total_grade_D_plus, 0) + coalesce(g.total_grade_D_plus, 0) as total_grade_D_plus,
					(calculated total_grade_D_plus * 1.3) as total_grade_D_plus_GPA,
					coalesce(b.total_grade_D, 0) + coalesce(c.total_grade_D, 0) + coalesce(d.total_grade_D, 0) 
						+ coalesce(e.total_grade_D, 0) + coalesce(f.total_grade_D, 0) + coalesce(g.total_grade_D, 0) as total_grade_D,
					(calculated total_grade_D * 1.0) as total_grade_D_GPA,
					coalesce(b.total_grade_F, 0) + coalesce(c.total_grade_F, 0) + coalesce(d.total_grade_F, 0) 
						+ coalesce(e.total_grade_F, 0) + coalesce(f.total_grade_F, 0) + coalesce(g.total_grade_F, 0) as total_grade_F,
					coalesce(b.total_withdrawn, 0) + coalesce(c.total_withdrawn, 0) + coalesce(d.total_withdrawn, 0) 
						+ coalesce(e.total_withdrawn, 0) + coalesce(f.total_withdrawn, 0) + coalesce(g.total_withdrawn, 0) as total_withdrawn,
					coalesce(b.total_dropped, 0) + coalesce(c.total_dropped, 0) + coalesce(d.total_dropped, 0)
						+ coalesce(e.total_dropped, 0) + coalesce(f.total_dropped, 0) + coalesce(g.total_dropped, 0) as total_dropped,
					coalesce(b.total_grade_I, 0) + coalesce(c.total_grade_I, 0) + coalesce(d.total_grade_I, 0)
						+ coalesce(e.total_grade_I, 0) + coalesce(f.total_grade_I, 0) + coalesce(g.total_grade_I, 0) as total_grade_I,
					coalesce(b.total_grade_X, 0) + coalesce(c.total_grade_X, 0) + coalesce(d.total_grade_X, 0)
						+ coalesce(e.total_grade_X, 0) + coalesce(f.total_grade_X, 0) + coalesce(g.total_grade_X, 0) as total_grade_X,
					coalesce(b.total_grade_U, 0) + coalesce(c.total_grade_U, 0) + coalesce(d.total_grade_U, 0)
						+ coalesce(e.total_grade_U, 0) + coalesce(f.total_grade_U, 0) + coalesce(g.total_grade_U, 0) as total_grade_U,
					coalesce(b.total_grade_S, 0) + coalesce(c.total_grade_S, 0) + coalesce(d.total_grade_S, 0)
						+ coalesce(e.total_grade_S, 0) + coalesce(f.total_grade_S, 0) + coalesce(g.total_grade_S, 0) as total_grade_S,
					coalesce(b.total_grade_P, 0) + coalesce(c.total_grade_P, 0) + coalesce(d.total_grade_P, 0)
						+ coalesce(e.total_grade_P, 0) + coalesce(f.total_grade_P, 0) + coalesce(g.total_grade_P, 0) as total_grade_P,
					coalesce(b.total_no_grade, 0) + coalesce(c.total_no_grade, 0) + coalesce(d.total_no_grade, 0)
						+ coalesce(e.total_no_grade, 0) + coalesce(f.total_no_grade, 0) + coalesce(g.total_no_grade, 0) as total_no_grade,
					(calculated total_grade_A + calculated total_grade_A_minus 
						+ calculated total_grade_B_plus + calculated total_grade_B + calculated total_grade_B_minus
						+ calculated total_grade_C_plus + calculated total_grade_C + calculated total_grade_C_minus
						+ calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F) as total_grades,
					(calculated total_grade_A + calculated total_grade_A_minus 
						+ calculated total_grade_B_plus + calculated total_grade_B + calculated total_grade_B_minus
						+ calculated total_grade_C_plus + calculated total_grade_C + calculated total_grade_C_minus
						+ calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F + calculated total_withdrawn) as total_students,
					(calculated total_grade_A_GPA + calculated total_grade_A_minus_GPA 
						+ calculated total_grade_B_plus_GPA + calculated total_grade_B_GPA + calculated total_grade_B_minus_GPA
						+ calculated total_grade_C_plus_GPA + calculated total_grade_C_GPA + calculated total_grade_C_minus_GPA
						+ calculated total_grade_D_plus_GPA + calculated total_grade_D_GPA) as total_grades_GPA,
					(calculated total_grades_GPA / calculated total_grades) as class_average,
					(calculated total_withdrawn / calculated total_students) as pct_withdrawn,
					(calculated total_grade_C_minus + calculated total_grade_D_plus + calculated total_grade_D 
						+ calculated total_grade_F + calculated total_withdrawn) as CDFW,
					(calculated CDFW / calculated total_students) as pct_CDFW,
					(calculated total_grade_C_minus + calculated total_grade_D_plus + calculated total_grade_D 
						+ calculated total_grade_F) as CDF,
					(calculated CDF / calculated total_students) as pct_CDF,
					(calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F 
						+ calculated total_withdrawn) as DFW,
					(calculated DFW / calculated total_students) as pct_DFW,
					(calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F) as DF,
					(calculated DF / calculated total_students) as pct_DF
				from &dsn..class_vw as a
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'LEC'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as b
					on a.subject_catalog_nbr = b.subject_catalog_nbr
						and a.ssr_component = b.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'LAB'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as c
					on a.subject_catalog_nbr = c.subject_catalog_nbr
						and a.ssr_component = c.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'INT'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as d
					on a.subject_catalog_nbr = d.subject_catalog_nbr
						and a.ssr_component = d.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'STU'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as e
					on a.subject_catalog_nbr = e.subject_catalog_nbr
						and a.ssr_component = e.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'SEM'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as f
					on a.subject_catalog_nbr = f.subject_catalog_nbr
						and a.ssr_component = f.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as g
					on a.subject_catalog_nbr = g.subject_catalog_nbr
						and a.ssr_component = g.ssr_component
				where a.snapshot = 'eot'
					and a.full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
					and a.grading_basis = 'GRD'
				order by a.subject_catalog_nbr
			;quit;

			proc sql;
				create table coursework_difficulty_&cohort_year. as
				select distinct
					a.emplid,
					avg(b.class_average) as fall_avg_difficulty,
					avg(b.pct_withdrawn) as fall_avg_pct_withdrawn,
					avg(b.pct_CDFW) as fall_avg_pct_CDFW,
					avg(b.pct_CDF) as fall_avg_pct_CDF,
					avg(b.pct_DFW) as fall_avg_pct_DFW,
					avg(b.pct_DF) as fall_avg_pct_DF
				from class_registration_&cohort_year. as a
				left join class_difficulty_&cohort_year. as b
					on a.subject_catalog_nbr = b.subject_catalog_nbr
						and a.ssr_component = b.ssr_component
						and a.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
				group by a.emplid
			;quit;

			proc sql;
				create table class_count_&cohort_year. as
				select distinct
					a.emplid,
					count(b.class_nbr) as fall_lec_count,
					count(c.class_nbr) as fall_lab_count,
					count(d.class_nbr) as fall_int_count,
					count(e.class_nbr) as fall_stu_count,
					count(f.class_nbr) as fall_sem_count,
					count(g.class_nbr) as fall_oth_count,
					sum(h.unt_taken) as fall_lec_units,
					sum(i.unt_taken) as fall_lab_units,
					sum(j.unt_taken) as fall_int_units,
					sum(k.unt_taken) as fall_stu_units,
					sum(l.unt_taken) as fall_sem_units,
					sum(m.unt_taken) as fall_oth_units,
					coalesce(calculated fall_lec_units, 0) + coalesce(calculated fall_lab_units, 0) + coalesce(calculated fall_int_units, 0) 
						+ coalesce(calculated fall_stu_units, 0) + coalesce(calculated fall_sem_units, 0) + coalesce(calculated fall_oth_units, 0) as total_fall_units
				from class_registration_&cohort_year. as a
				left join (select distinct emplid, 
								class_nbr
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LEC') as b
					on a.emplid = b.emplid
						and a.class_nbr = b.class_nbr
				left join (select distinct emplid, 
								class_nbr
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LAB') as c
					on a.emplid = c.emplid
						and a.class_nbr = c.class_nbr
				left join (select distinct emplid, 
								class_nbr
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'INT') as d
					on a.emplid = d.emplid
						and a.class_nbr = d.class_nbr
				left join (select distinct emplid, 
								class_nbr
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'STU') as e
					on a.emplid = e.emplid
						and a.class_nbr = e.class_nbr
				left join (select distinct emplid, 
								class_nbr
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'SEM') as f
					on a.emplid = f.emplid
						and a.class_nbr = f.class_nbr
				left join (select distinct emplid, 
								class_nbr
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')) as g
					on a.emplid = g.emplid
						and a.class_nbr = g.class_nbr
				left join (select distinct emplid, 
								class_nbr,
								unt_taken
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LEC') as h
					on a.emplid = h.emplid
						and a.class_nbr = h.class_nbr
				left join (select distinct emplid, 
								class_nbr,
								unt_taken
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LAB') as i
					on a.emplid = i.emplid
						and a.class_nbr = i.class_nbr
				left join (select distinct emplid, 
								class_nbr,
								unt_taken
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'INT') as j
					on a.emplid = j.emplid
						and a.class_nbr = j.class_nbr
				left join (select distinct emplid, 
								class_nbr,
								unt_taken
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'STU') as k
					on a.emplid = k.emplid
						and a.class_nbr = k.class_nbr
				left join (select distinct emplid, 
								class_nbr,
								unt_taken
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'SEM') as l
					on a.emplid = l.emplid
						and a.class_nbr = l.class_nbr
				left join (select distinct emplid, 
								class_nbr,
								unt_taken
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')) as m
					on a.emplid = m.emplid
						and a.class_nbr = m.class_nbr
				group by a.emplid
			;quit;

			proc sql;
				create table term_contact_hrs_&cohort_year. as
				select distinct
					a.emplid,
					sum(b.lec_contact_hrs) as fall_lec_contact_hrs,
					sum(c.lab_contact_hrs) as fall_lab_contact_hrs,
					sum(d.int_contact_hrs) as fall_int_contact_hrs,
					sum(e.stu_contact_hrs) as fall_stu_contact_hrs,
					sum(f.sem_contact_hrs) as fall_sem_contact_hrs,
					sum(g.oth_contact_hrs) as fall_oth_contact_hrs,
					coalesce(calculated fall_lec_contact_hrs, 0) + coalesce(calculated fall_lab_contact_hrs, 0) + coalesce(calculated fall_int_contact_hrs, 0) 
						+ coalesce(calculated fall_stu_contact_hrs, 0) + coalesce(calculated fall_sem_contact_hrs, 0) + coalesce(calculated fall_oth_contact_hrs, 0) as total_fall_contact_hrs
				from class_registration_&cohort_year. as a
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as lec_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'LEC'
							group by subject_catalog_nbr) as b
					on a.subject_catalog_nbr = b.subject_catalog_nbr
						and a.ssr_component = b.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as lab_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'LAB'
							group by subject_catalog_nbr) as c
					on a.subject_catalog_nbr = c.subject_catalog_nbr
						and a.ssr_component = c.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as int_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'INT'
							group by subject_catalog_nbr) as d
					on a.subject_catalog_nbr = d.subject_catalog_nbr
						and a.ssr_component = d.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as stu_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'STU'
							group by subject_catalog_nbr) as e
					on a.subject_catalog_nbr = e.subject_catalog_nbr
						and a.ssr_component = e.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as sem_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'SEM'
							group by subject_catalog_nbr) as f
					on a.subject_catalog_nbr = f.subject_catalog_nbr
						and a.ssr_component = f.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as oth_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')
							group by subject_catalog_nbr) as g
					on a.subject_catalog_nbr = g.subject_catalog_nbr
						and a.ssr_component = g.ssr_component
						and substr(a.strm,4,1) = '7'
				group by a.emplid
			;quit;

			proc sql;
				create table dataset_&cohort_year. as
				select distinct 
					a.*,
					w.min_week_from_term_begin_dt,
					w.max_week_from_term_begin_dt,
					w.count_week_from_term_begin_dt,
					(4.0 - q.fall_avg_difficulty) as fall_avg_difficulty,
					q.fall_avg_pct_withdrawn,
					q.fall_avg_pct_CDFW,
					q.fall_avg_pct_CDF,
					q.fall_avg_pct_DFW,
					q.fall_avg_pct_DF,
					u.fall_lec_count,
					u.fall_lab_count,
					u.fall_int_count,
					u.fall_stu_count,
					u.fall_sem_count,
					u.fall_oth_count,
					u.total_fall_units,
					r.fall_lec_contact_hrs,
					r.fall_lab_contact_hrs,
					r.fall_int_contact_hrs,
					r.fall_stu_contact_hrs,
					r.fall_sem_contact_hrs,
					r.fall_oth_contact_hrs,
					r.total_fall_contact_hrs,
					s.fed_need,
					x.total_offer,
					v.race_american_indian,
					v.race_alaska,
					v.race_asian,
					v.race_black,
					v.race_native_hawaiian,
					v.race_white
				from cohort_&cohort_year. as a
				left join coursework_difficulty_&cohort_year. as q
					on a.emplid = q.emplid
				left join term_contact_hrs_&cohort_year. as r
					on a.emplid = r.emplid
				left join need_&cohort_year. as s
					on a.emplid = s.emplid
						and s.aid_year = "&cohort_year."
				left join aid_&cohort_year. as x
					on a.emplid = x.emplid
						and x.aid_year = "&cohort_year."
				left join class_count_&cohort_year. as u
					on a.emplid = u.emplid
				left join race_detail_&cohort_year. as v
					on a.emplid = v.emplid
				left join date_&cohort_year. as w
					on a.emplid = w.emplid
				where u.total_fall_units >= 12
			;quit;
			
		%mend loop;
		""")

		print('Done\n')

		# Run SAS macro program to prepare data from precensus
		print('Run SAS macro program...')
		start = time.perf_counter()

		sas_log = sas.submit("""
		%loop;
		""")

		HTML(sas_log['LOG'])

		stop = time.perf_counter()
		print(f'Done in {(stop - start)/60:.1f} minutes\n')

		# Prepare data
		print('Prepare data...')

		sas.submit("""
		data validation_set;
			set dataset_&start_cohort.;
			if enrl_ind = . then enrl_ind = 0;
			if distance = . then acs_mi = 1; else acs_mi = 0;
			if distance = . then distance = 0;
			if pop_dens = . then pop_dens = 0;
			if educ_rate = . then educ_rate = 0;	
			if pct_blk = . then pct_blk = 0;	
			if pct_ai = . then pct_ai = 0;	
			if pct_asn = .	then pct_asn = 0;
			if pct_hawi = . then pct_hawi = 0;
			if pct_two = . then pct_two = 0;
			if pct_hisp = . then pct_hisp = 0;
			if pct_oth = . then pct_oth = 0;
			if pct_non = . then pct_non = 0;
			if median_inc = . then median_inc = 0;
			if median_value = . then median_value = 0;
			if gini_indx = . then gini_indx = 0;
			if pvrt_rate = . then pvrt_rate = 0;
			if educ_rate = . then educ_rate = 0;
			if ad_dta = . then ad_dta = 0;
			if ad_ast = . then ad_ast = 0;
			if ad_hsdip = . then ad_hsdip = 0;
			if ad_ged = . then ad_ged = 0;
			if ad_ger = . then ad_ger = 0;
			if ad_gens = . then ad_gens = 0;
			if ap = . then ap = 0;
			if rs = . then rs = 0;
			if chs = . then chs = 0;
			if ib = . then ib = 0;
			if aice = . then aice = 0;
			if ib_aice = . then ib_aice = 0;
			if athlete = . then athlete = 0;
			if remedial = . then remedial = 0;
			if sat_mss = . then sat_mss = 0;
			if sat_erws = . then sat_erws = 0;
			if high_school_gpa = . then high_school_gpa_mi = 1; else high_school_gpa_mi = 0;
			if high_school_gpa = . then high_school_gpa = 0;
			if transfer_gpa = . then transfer_gpa_mi = 1; else transfer_gpa_mi = 0;
			if transfer_gpa = . then transfer_gpa = 0;
			if last_sch_proprietorship = '' then last_sch_proprietorship = 'UNKN';
			if ipeds_ethnic_group_descrshort = '' then ipeds_ethnic_group_descrshort = 'NS';
			if fall_avg_pct_withdrawn = . then fall_avg_pct_withdrawn = 0;
			if fall_lec_count = . then fall_lec_count = 0;
			if fall_lab_count = . then fall_lab_count = 0;
			if fall_int_count = . then fall_int_count = 0;
			if fall_stu_count = . then fall_stu_count = 0;
			if fall_sem_count = . then fall_sem_count = 0;
			if fall_oth_count = . then fall_oth_count = 0;
			if fall_lec_contact_hrs = . then fall_lec_contact_hrs = 0;
			if fall_lab_contact_hrs = . then fall_lab_contact_hrs = 0;
			if fall_int_contact_hrs = . then fall_int_contact_hrs = 0;
			if fall_stu_contact_hrs = . then fall_stu_contact_hrs = 0;
			if fall_sem_contact_hrs = . then fall_sem_contact_hrs = 0;
			if fall_oth_contact_hrs = . then fall_oth_contact_hrs = 0;
			if total_fall_contact_hrs = . then total_fall_contact_hrs = 0;
			if fall_avg_pct_CDFW = . then fall_avg_pct_CDFW = 0;
			if fall_avg_pct_CDF = . then fall_avg_pct_CDF = 0;
			if fall_avg_pct_DFW = . then fall_avg_pct_DFW = 0;
			if fall_avg_pct_DF = . then fall_avg_pct_DF = 0;
			if fall_avg_difficulty = . then fall_crse_mi = 1; else fall_crse_mi = 0; 
			if fall_avg_difficulty = . then fall_avg_difficulty = 0;
			if spring_avg_pct_withdrawn = . then spring_avg_pct_withdrawn = 0;
			if spring_avg_pct_CDFW = . then spring_avg_pct_CDFW = 0;
			if spring_avg_pct_CDF = . then spring_avg_pct_CDF = 0;
			if spring_avg_pct_DFW = . then spring_avg_pct_DFW = 0;
			if spring_avg_pct_DF = . then spring_avg_pct_DF = 0;
			if spring_avg_difficulty = . then spring_crse_mi = 1; else spring_crse_mi = 0; 
			if spring_avg_difficulty = . then spring_avg_difficulty = 0;
			if spring_lec_count = . then spring_lec_count = 0;
			if spring_lab_count = . then spring_lab_count = 0;
			if spring_int_count = . then spring_int_count = 0;
			if spring_stu_count = . then spring_stu_count = 0;
			if spring_sem_count = . then spring_sem_count = 0;
			if spring_oth_count = . then spring_oth_count = 0;
			if spring_lec_contact_hrs = . then spring_lec_contact_hrs = 0;
			if spring_lab_contact_hrs = . then spring_lab_contact_hrs = 0;
			if spring_int_contact_hrs = . then spring_int_contact_hrs = 0;
			if spring_stu_contact_hrs = . then spring_stu_contact_hrs = 0;
			if spring_sem_contact_hrs = . then spring_sem_contact_hrs = 0;
			if spring_oth_contact_hrs = . then spring_oth_contact_hrs = 0;
			if total_spring_contact_hrs = . then total_spring_contact_hrs = 0;
			if total_fall_units = . then total_fall_units = 0;
			if total_spring_units = . then total_spring_units = 0;
			if fall_credit_hours = . then fall_credit_hours = 0;
			if spring_credit_hours = . then spring_credit_hours = 0;
			if fall_lec_contact_hrs = . then fall_lec_contact_hrs = 0;
			if fall_lab_contact_hrs = . then fall_lab_contact_hrs = 0;
			if spring_lec_contact_hrs = . then spring_lec_contact_hrs = 0;
			if spring_lab_contact_hrs = . then spring_lab_contact_hrs = 0;
			if total_fall_contact_hrs = . then total_fall_contact_hrs = 0;
			if total_spring_contact_hrs = . then total_spring_contact_hrs = 0;
			if fall_midterm_gpa_avg = . then fall_midterm_gpa_avg_mi = 1; else fall_midterm_gpa_avg_mi = 0;
			if fall_midterm_gpa_avg = . then fall_midterm_gpa_avg = 0;
			if fall_midterm_grade_count = . then fall_midterm_grade_count = 0;
			if fall_midterm_S_grade_count = . then fall_midterm_S_grade_count = 0;
			if fall_midterm_W_grade_count = . then fall_midterm_W_grade_count = 0;
			if spring_midterm_gpa_avg = . then spring_midterm_gpa_avg_mi = 1; else spring_midterm_gpa_avg_mi = 0;
			if spring_midterm_gpa_avg = . then spring_midterm_gpa_avg = 0;
			if spring_midterm_grade_count = . then spring_midterm_grade_count = 0;
			if spring_midterm_S_grade_count = . then spring_midterm_S_grade_count = 0;
			if spring_midterm_W_grade_count = . then spring_midterm_W_grade_count = 0;
			if fall_term_gpa = . then fall_term_gpa_mi = 1; else fall_term_gpa_mi = 0;
			if fall_term_gpa = . then fall_term_gpa = 0;
			if spring_term_gpa = . then spring_term_gpa_mi = 1; else spring_term_gpa_mi = 0;
			if spring_term_gpa = . then spring_term_gpa = 0;
			if fall_term_D_grade_count = . then fall_term_D_grade_count_mi = 1; else fall_term_D_grade_count_mi = 0;
			if fall_term_D_grade_count = . then fall_term_D_grade_count = 0;
			if fall_term_F_grade_count = . then fall_term_F_grade_count_mi = 1; else fall_term_F_grade_count_mi = 0;
			if fall_term_F_grade_count = . then fall_term_F_grade_count = 0;
			if fall_term_W_grade_count = . then fall_term_W_grade_count_mi = 1; else fall_term_W_grade_count_mi = 0;
			if fall_term_W_grade_count = . then fall_term_W_grade_count = 0;
			if fall_term_I_grade_count = . then fall_term_I_grade_count_mi = 1; else fall_term_I_grade_count_mi = 0;
			if fall_term_I_grade_count = . then fall_term_I_grade_count = 0;
			if fall_term_X_grade_count = . then fall_term_X_grade_count_mi = 1; else fall_term_X_grade_count_mi = 0;
			if fall_term_X_grade_count = . then fall_term_X_grade_count = 0;
			if fall_term_U_grade_count = . then fall_term_U_grade_count_mi = 1; else fall_term_U_grade_count_mi = 0;
			if fall_term_U_grade_count = . then fall_term_U_grade_count = 0;
			if fall_term_S_grade_count = . then fall_term_S_grade_count_mi = 1; else fall_term_S_grade_count_mi = 0;
			if fall_term_S_grade_count = . then fall_term_S_grade_count = 0;
			if fall_term_P_grade_count = . then fall_term_P_grade_count_mi = 1; else fall_term_P_grade_count_mi = 0;
			if fall_term_P_grade_count = . then fall_term_P_grade_count = 0;
			if fall_term_Z_grade_count = . then fall_term_Z_grade_count_mi = 1; else fall_term_Z_grade_count_mi = 0;
			if fall_term_Z_grade_count = . then fall_term_Z_grade_count = 0;
			if fall_term_letter_count = . then fall_term_letter_count_mi = 1; else fall_term_letter_count_mi = 0;
			if fall_term_letter_count = . then fall_term_letter_count = 0;
			if fall_term_grade_count = . then fall_term_grade_count_mi = 1; else fall_term_grade_count_mi = 0;
			if fall_term_grade_count = . then fall_term_grade_count = 0;
			fall_term_no_letter_count = fall_term_grade_count - fall_term_letter_count;
			if spring_term_D_grade_count = . then spring_term_D_grade_count_mi = 1; else spring_term_D_grade_count_mi = 0;
			if spring_term_D_grade_count = . then spring_term_D_grade_count = 0;
			if spring_term_F_grade_count = . then spring_term_F_grade_count_mi = 1; else spring_term_F_grade_count_mi = 0;
			if spring_term_F_grade_count = . then spring_term_F_grade_count = 0;
			if spring_term_W_grade_count = . then spring_term_W_grade_count_mi = 1; else spring_term_W_grade_count_mi = 0;
			if spring_term_W_grade_count = . then spring_term_W_grade_count = 0;
			if spring_term_I_grade_count = . then spring_term_I_grade_count_mi = 1; else spring_term_I_grade_count_mi = 0;
			if spring_term_I_grade_count = . then spring_term_I_grade_count = 0;
			if spring_term_X_grade_count = . then spring_term_X_grade_count_mi = 1; else spring_term_X_grade_count_mi = 0;
			if spring_term_X_grade_count = . then spring_term_X_grade_count = 0;
			if spring_term_U_grade_count = . then spring_term_U_grade_count_mi = 1; else spring_term_U_grade_count_mi = 0;
			if spring_term_U_grade_count = . then spring_term_U_grade_count = 0;
			if spring_term_S_grade_count = . then spring_term_S_grade_count_mi = 1; else spring_term_S_grade_count_mi = 0;
			if spring_term_S_grade_count = . then spring_term_S_grade_count = 0;
			if spring_term_P_grade_count = . then spring_term_P_grade_count_mi = 1; else spring_term_P_grade_count_mi = 0;
			if spring_term_P_grade_count = . then spring_term_P_grade_count = 0;
			if spring_term_Z_grade_count = . then spring_term_Z_grade_count_mi = 1; else spring_term_Z_grade_count_mi = 0;
			if spring_term_Z_grade_count = . then spring_term_Z_grade_count = 0;
			if spring_term_letter_count = . then spring_term_leter_count_mi = 1; else spring_term_leter_count_mi = 0;
			if spring_term_letter_count = . then spring_term_letter_count = 0;
			if spring_term_grade_count = . then spring_term_grade_count_mi = 1; else spring_term_grade_count_mi = 0;
			if spring_term_grade_count = . then spring_term_grade_count = 0;
			spring_term_no_letter_count = spring_term_grade_count - spring_term_letter_count;
			if first_gen_flag = '' then first_gen_flag_mi = 1; else first_gen_flag_mi = 0;
			if first_gen_flag = '' then first_gen_flag = 'N';
			if camp_addr_indicator ^= 'Y' then camp_addr_indicator = 'N';
			if housing_reshall_indicator ^= 'Y' then housing_reshall_indicator = 'N';
			if housing_ssa_indicator ^= 'Y' then housing_ssa_indicator = 'N';
			if housing_family_indicator ^= 'Y' then housing_family_indicator = 'N';
			if afl_reshall_indicator ^= 'Y' then afl_reshall_indicator = 'N';
			if afl_ssa_indicator ^= 'Y' then afl_ssa_indicator = 'N';
			if afl_family_indicator ^= 'Y' then afl_family_indicator = 'N';
			if afl_greek_indicator ^= 'Y' then afl_greek_indicator = 'N';
			if afl_greek_life_indicator ^= 'Y' then afl_greek_life_indicator = 'N';
			fall_withdrawn_hours = (total_fall_units - fall_credit_hours) * -1;
			if total_fall_units = 0 then fall_withdrawn_ind = 1; else fall_withdrawn_ind = 0;
			spring_withdrawn_hours = (total_spring_units - spring_credit_hours) * -1;
			if total_spring_units = 0 then spring_withdrawn = 1; else spring_withdrawn = 0;
			spring_midterm_gpa_change = spring_midterm_gpa_avg - fall_cum_gpa;
			unmet_need_disb = fed_need - total_disb;
			unmet_need_acpt = fed_need - total_accept;
			if unmet_need_acpt = . then unmet_need_acpt_mi = 1; else unmet_need_acpt_mi = 0;
			if unmet_need_acpt < 0 then unmet_need_acpt = 0;
			unmet_need_ofr = fed_need - total_offer;
			if unmet_need_ofr = . then unmet_need_ofr_mi = 1; else unmet_need_ofr_mi = 0;
			if unmet_need_ofr < 0 then unmet_need_ofr = 0;
			if fed_efc = . then fed_efc = 0;
			if fed_need = . then fed_need = 0;
			if total_disb = . then total_disb = 0;
			if total_offer = . then total_offer = 0;
			if total_accept = . then total_accept = 0;
		run;

		data training_set;
			set dataset_%eval(&start_cohort. + &lag_year.)-dataset_%eval(&end_cohort. - &lag_year.);
			if enrl_ind = . then enrl_ind = 0;
			if distance = . then acs_mi = 1; else acs_mi = 0;
			if distance = . then distance = 0;
			if pop_dens = . then pop_dens = 0;
			if educ_rate = . then educ_rate = 0;	
			if pct_blk = . then pct_blk = 0;	
			if pct_ai = . then pct_ai = 0;	
			if pct_asn = .	then pct_asn = 0;
			if pct_hawi = . then pct_hawi = 0;
			if pct_two = . then pct_two = 0;
			if pct_hisp = . then pct_hisp = 0;
			if pct_oth = . then pct_oth = 0;
			if pct_non = . then pct_non = 0;
			if median_inc = . then median_inc = 0;
			if median_value = . then median_value = 0;
			if gini_indx = . then gini_indx = 0;
			if pvrt_rate = . then pvrt_rate = 0;
			if educ_rate = . then educ_rate = 0;
			if ad_dta = . then ad_dta = 0;
			if ad_ast = . then ad_ast = 0;
			if ad_hsdip = . then ad_hsdip = 0;
			if ad_ged = . then ad_ged = 0;
			if ad_ger = . then ad_ger = 0;
			if ad_gens = . then ad_gens = 0;
			if ap = . then ap = 0;
			if rs = . then rs = 0;
			if chs = . then chs = 0;
			if ib = . then ib = 0;
			if aice = . then aice = 0;
			if ib_aice = . then ib_aice = 0;
			if athlete = . then athlete = 0;
			if remedial = . then remedial = 0;
			if sat_mss = . then sat_mss = 0;
			if sat_erws = . then sat_erws = 0;
			if high_school_gpa = . then high_school_gpa_mi = 1; else high_school_gpa_mi = 0;
			if high_school_gpa = . then high_school_gpa = 0;
			if transfer_gpa = . then transfer_gpa_mi = 1; else transfer_gpa_mi = 0;
			if transfer_gpa = . then transfer_gpa = 0;
			if last_sch_proprietorship = '' then last_sch_proprietorship = 'UNKN';
			if ipeds_ethnic_group_descrshort = '' then ipeds_ethnic_group_descrshort = 'NS';
			if fall_avg_pct_withdrawn = . then fall_avg_pct_withdrawn = 0;
			if fall_lec_count = . then fall_lec_count = 0;
			if fall_lab_count = . then fall_lab_count = 0;
			if fall_int_count = . then fall_int_count = 0;
			if fall_stu_count = . then fall_stu_count = 0;
			if fall_sem_count = . then fall_sem_count = 0;
			if fall_oth_count = . then fall_oth_count = 0;
			if fall_lec_contact_hrs = . then fall_lec_contact_hrs = 0;
			if fall_lab_contact_hrs = . then fall_lab_contact_hrs = 0;
			if fall_int_contact_hrs = . then fall_int_contact_hrs = 0;
			if fall_stu_contact_hrs = . then fall_stu_contact_hrs = 0;
			if fall_sem_contact_hrs = . then fall_sem_contact_hrs = 0;
			if fall_oth_contact_hrs = . then fall_oth_contact_hrs = 0;
			if total_fall_contact_hrs = . then total_fall_contact_hrs = 0;
			if fall_avg_pct_CDFW = . then fall_avg_pct_CDFW = 0;
			if fall_avg_pct_CDF = . then fall_avg_pct_CDF = 0;
			if fall_avg_pct_DFW = . then fall_avg_pct_DFW = 0;
			if fall_avg_pct_DF = . then fall_avg_pct_DF = 0;
			if fall_avg_difficulty = . then fall_crse_mi = 1; else fall_crse_mi = 0; 
			if fall_avg_difficulty = . then fall_avg_difficulty = 0;
			if spring_avg_pct_withdrawn = . then spring_avg_pct_withdrawn = 0;
			if spring_avg_pct_CDFW = . then spring_avg_pct_CDFW = 0;
			if spring_avg_pct_CDF = . then spring_avg_pct_CDF = 0;
			if spring_avg_pct_DFW = . then spring_avg_pct_DFW = 0;
			if spring_avg_pct_DF = . then spring_avg_pct_DF = 0;
			if spring_avg_difficulty = . then spring_crse_mi = 1; else spring_crse_mi = 0; 
			if spring_avg_difficulty = . then spring_avg_difficulty = 0;
			if spring_lec_count = . then spring_lec_count = 0;
			if spring_lab_count = . then spring_lab_count = 0;
			if spring_int_count = . then spring_int_count = 0;
			if spring_stu_count = . then spring_stu_count = 0;
			if spring_sem_count = . then spring_sem_count = 0;
			if spring_oth_count = . then spring_oth_count = 0;
			if spring_lec_contact_hrs = . then spring_lec_contact_hrs = 0;
			if spring_lab_contact_hrs = . then spring_lab_contact_hrs = 0;
			if spring_int_contact_hrs = . then spring_int_contact_hrs = 0;
			if spring_stu_contact_hrs = . then spring_stu_contact_hrs = 0;
			if spring_sem_contact_hrs = . then spring_sem_contact_hrs = 0;
			if spring_oth_contact_hrs = . then spring_oth_contact_hrs = 0;
			if total_spring_contact_hrs = . then total_spring_contact_hrs = 0;
			if total_fall_units = . then total_fall_units = 0;
			if total_spring_units = . then total_spring_units = 0;
			if fall_credit_hours = . then fall_credit_hours = 0;
			if spring_credit_hours = . then spring_credit_hours = 0;
			if fall_lec_contact_hrs = . then fall_lec_contact_hrs = 0;
			if fall_lab_contact_hrs = . then fall_lab_contact_hrs = 0;
			if spring_lec_contact_hrs = . then spring_lec_contact_hrs = 0;
			if spring_lab_contact_hrs = . then spring_lab_contact_hrs = 0;
			if total_fall_contact_hrs = . then total_fall_contact_hrs = 0;
			if total_spring_contact_hrs = . then total_spring_contact_hrs = 0;
			if fall_midterm_gpa_avg = . then fall_midterm_gpa_avg_mi = 1; else fall_midterm_gpa_avg_mi = 0;
			if fall_midterm_gpa_avg = . then fall_midterm_gpa_avg = 0;
			if fall_midterm_grade_count = . then fall_midterm_grade_count = 0;
			if fall_midterm_S_grade_count = . then fall_midterm_S_grade_count = 0;
			if fall_midterm_W_grade_count = . then fall_midterm_W_grade_count = 0;
			if spring_midterm_gpa_avg = . then spring_midterm_gpa_avg_mi = 1; else spring_midterm_gpa_avg_mi = 0;
			if spring_midterm_gpa_avg = . then spring_midterm_gpa_avg = 0;
			if spring_midterm_grade_count = . then spring_midterm_grade_count = 0;
			if spring_midterm_S_grade_count = . then spring_midterm_S_grade_count = 0;
			if spring_midterm_W_grade_count = . then spring_midterm_W_grade_count = 0;
			if fall_term_gpa = . then fall_term_gpa_mi = 1; else fall_term_gpa_mi = 0;
			if fall_term_gpa = . then fall_term_gpa = 0;
			if spring_term_gpa = . then spring_term_gpa_mi = 1; else spring_term_gpa_mi = 0;
			if spring_term_gpa = . then spring_term_gpa = 0;
			if fall_term_D_grade_count = . then fall_term_D_grade_count_mi = 1; else fall_term_D_grade_count_mi = 0;
			if fall_term_D_grade_count = . then fall_term_D_grade_count = 0;
			if fall_term_F_grade_count = . then fall_term_F_grade_count_mi = 1; else fall_term_F_grade_count_mi = 0;
			if fall_term_F_grade_count = . then fall_term_F_grade_count = 0;
			if fall_term_W_grade_count = . then fall_term_W_grade_count_mi = 1; else fall_term_W_grade_count_mi = 0;
			if fall_term_W_grade_count = . then fall_term_W_grade_count = 0;
			if fall_term_I_grade_count = . then fall_term_I_grade_count_mi = 1; else fall_term_I_grade_count_mi = 0;
			if fall_term_I_grade_count = . then fall_term_I_grade_count = 0;
			if fall_term_X_grade_count = . then fall_term_X_grade_count_mi = 1; else fall_term_X_grade_count_mi = 0;
			if fall_term_X_grade_count = . then fall_term_X_grade_count = 0;
			if fall_term_U_grade_count = . then fall_term_U_grade_count_mi = 1; else fall_term_U_grade_count_mi = 0;
			if fall_term_U_grade_count = . then fall_term_U_grade_count = 0;
			if fall_term_S_grade_count = . then fall_term_S_grade_count_mi = 1; else fall_term_S_grade_count_mi = 0;
			if fall_term_S_grade_count = . then fall_term_S_grade_count = 0;
			if fall_term_P_grade_count = . then fall_term_P_grade_count_mi = 1; else fall_term_P_grade_count_mi = 0;
			if fall_term_P_grade_count = . then fall_term_P_grade_count = 0;
			if fall_term_Z_grade_count = . then fall_term_Z_grade_count_mi = 1; else fall_term_Z_grade_count_mi = 0;
			if fall_term_Z_grade_count = . then fall_term_Z_grade_count = 0;
			if fall_term_letter_count = . then fall_term_letter_count_mi = 1; else fall_term_letter_count_mi = 0;
			if fall_term_letter_count = . then fall_term_letter_count = 0;
			if fall_term_grade_count = . then fall_term_grade_count_mi = 1; else fall_term_grade_count_mi = 0;
			if fall_term_grade_count = . then fall_term_grade_count = 0;
			fall_term_no_letter_count = fall_term_grade_count - fall_term_letter_count;
			if spring_term_D_grade_count = . then spring_term_D_grade_count_mi = 1; else spring_term_D_grade_count_mi = 0;
			if spring_term_D_grade_count = . then spring_term_D_grade_count = 0;
			if spring_term_F_grade_count = . then spring_term_F_grade_count_mi = 1; else spring_term_F_grade_count_mi = 0;
			if spring_term_F_grade_count = . then spring_term_F_grade_count = 0;
			if spring_term_W_grade_count = . then spring_term_W_grade_count_mi = 1; else spring_term_W_grade_count_mi = 0;
			if spring_term_W_grade_count = . then spring_term_W_grade_count = 0;
			if spring_term_I_grade_count = . then spring_term_I_grade_count_mi = 1; else spring_term_I_grade_count_mi = 0;
			if spring_term_I_grade_count = . then spring_term_I_grade_count = 0;
			if spring_term_X_grade_count = . then spring_term_X_grade_count_mi = 1; else spring_term_X_grade_count_mi = 0;
			if spring_term_X_grade_count = . then spring_term_X_grade_count = 0;
			if spring_term_U_grade_count = . then spring_term_U_grade_count_mi = 1; else spring_term_U_grade_count_mi = 0;
			if spring_term_U_grade_count = . then spring_term_U_grade_count = 0;
			if spring_term_S_grade_count = . then spring_term_S_grade_count_mi = 1; else spring_term_S_grade_count_mi = 0;
			if spring_term_S_grade_count = . then spring_term_S_grade_count = 0;
			if spring_term_P_grade_count = . then spring_term_P_grade_count_mi = 1; else spring_term_P_grade_count_mi = 0;
			if spring_term_P_grade_count = . then spring_term_P_grade_count = 0;
			if spring_term_Z_grade_count = . then spring_term_Z_grade_count_mi = 1; else spring_term_Z_grade_count_mi = 0;
			if spring_term_Z_grade_count = . then spring_term_Z_grade_count = 0;
			if spring_term_letter_count = . then spring_term_leter_count_mi = 1; else spring_term_leter_count_mi = 0;
			if spring_term_letter_count = . then spring_term_letter_count = 0;
			if spring_term_grade_count = . then spring_term_grade_count_mi = 1; else spring_term_grade_count_mi = 0;
			if spring_term_grade_count = . then spring_term_grade_count = 0;
			spring_term_no_letter_count = spring_term_grade_count - spring_term_letter_count;
			if first_gen_flag = '' then first_gen_flag_mi = 1; else first_gen_flag_mi = 0;
			if first_gen_flag = '' then first_gen_flag = 'N';
			if camp_addr_indicator ^= 'Y' then camp_addr_indicator = 'N';
			if housing_reshall_indicator ^= 'Y' then housing_reshall_indicator = 'N';
			if housing_ssa_indicator ^= 'Y' then housing_ssa_indicator = 'N';
			if housing_family_indicator ^= 'Y' then housing_family_indicator = 'N';
			if afl_reshall_indicator ^= 'Y' then afl_reshall_indicator = 'N';
			if afl_ssa_indicator ^= 'Y' then afl_ssa_indicator = 'N';
			if afl_family_indicator ^= 'Y' then afl_family_indicator = 'N';
			if afl_greek_indicator ^= 'Y' then afl_greek_indicator = 'N';
			if afl_greek_life_indicator ^= 'Y' then afl_greek_life_indicator = 'N';
			fall_withdrawn_hours = (total_fall_units - fall_credit_hours) * -1;
			if total_fall_units = 0 then fall_withdrawn_ind = 1; else fall_withdrawn_ind = 0;
			spring_withdrawn_hours = (total_spring_units - spring_credit_hours) * -1;
			if total_spring_units = 0 then spring_withdrawn = 1; else spring_withdrawn = 0;
			spring_midterm_gpa_change = spring_midterm_gpa_avg - fall_cum_gpa;
			unmet_need_disb = fed_need - total_disb;
			unmet_need_acpt = fed_need - total_accept;
			if unmet_need_acpt = . then unmet_need_acpt_mi = 1; else unmet_need_acpt_mi = 0;
			if unmet_need_acpt < 0 then unmet_need_acpt = 0;
			unmet_need_ofr = fed_need - total_offer;
			if unmet_need_ofr = . then unmet_need_ofr_mi = 1; else unmet_need_ofr_mi = 0;
			if unmet_need_ofr < 0 then unmet_need_ofr = 0;
			if fed_efc = . then fed_efc = 0;
			if fed_need = . then fed_need = 0;
			if total_disb = . then total_disb = 0;
			if total_offer = . then total_offer = 0;
			if total_accept = . then total_accept = 0;
		run;

		data testing_set;
			set dataset_&end_cohort.;
			if enrl_ind = . then enrl_ind = 0;
			if distance = . then acs_mi = 1; else acs_mi = 0;
			if distance = . then distance = 0;
			if pop_dens = . then pop_dens = 0;
			if educ_rate = . then educ_rate = 0;	
			if pct_blk = . then pct_blk = 0;	
			if pct_ai = . then pct_ai = 0;	
			if pct_asn = .	then pct_asn = 0;
			if pct_hawi = . then pct_hawi = 0;
			if pct_two = . then pct_two = 0;
			if pct_hisp = . then pct_hisp = 0;
			if pct_oth = . then pct_oth = 0;
			if pct_non = . then pct_non = 0;
			if median_inc = . then median_inc = 0;
			if median_value = . then median_value = 0;
			if gini_indx = . then gini_indx = 0;
			if pvrt_rate = . then pvrt_rate = 0;
			if educ_rate = . then educ_rate = 0;
			if ad_dta = . then ad_dta = 0;
			if ad_ast = . then ad_ast = 0;
			if ad_hsdip = . then ad_hsdip = 0;
			if ad_ged = . then ad_ged = 0;
			if ad_ger = . then ad_ger = 0;
			if ad_gens = . then ad_gens = 0;
			if ap = . then ap = 0;
			if rs = . then rs = 0;
			if chs = . then chs = 0;
			if ib = . then ib = 0;
			if aice = . then aice = 0;
			if ib_aice = . then ib_aice = 0;
			if athlete = . then athlete = 0;
			if remedial = . then remedial = 0;
			if sat_mss = . then sat_mss = 0;
			if sat_erws = . then sat_erws = 0;
			if high_school_gpa = . then high_school_gpa_mi = 1; else high_school_gpa_mi = 0;
			if high_school_gpa = . then high_school_gpa = 0;
			if transfer_gpa = . then transfer_gpa_mi = 1; else transfer_gpa_mi = 0;
			if transfer_gpa = . then transfer_gpa = 0;
			if last_sch_proprietorship = '' then last_sch_proprietorship = 'UNKN';
			if ipeds_ethnic_group_descrshort = '' then ipeds_ethnic_group_descrshort = 'NS';
			if fall_avg_pct_withdrawn = . then fall_avg_pct_withdrawn = 0;
			if fall_lec_count = . then fall_lec_count = 0;
			if fall_lab_count = . then fall_lab_count = 0;
			if fall_int_count = . then fall_int_count = 0;
			if fall_stu_count = . then fall_stu_count = 0;
			if fall_sem_count = . then fall_sem_count = 0;
			if fall_oth_count = . then fall_oth_count = 0;
			if fall_lec_contact_hrs = . then fall_lec_contact_hrs = 0;
			if fall_lab_contact_hrs = . then fall_lab_contact_hrs = 0;
			if fall_int_contact_hrs = . then fall_int_contact_hrs = 0;
			if fall_stu_contact_hrs = . then fall_stu_contact_hrs = 0;
			if fall_sem_contact_hrs = . then fall_sem_contact_hrs = 0;
			if fall_oth_contact_hrs = . then fall_oth_contact_hrs = 0;
			if total_fall_contact_hrs = . then total_fall_contact_hrs = 0;
			if fall_avg_pct_CDFW = . then fall_avg_pct_CDFW = 0;
			if fall_avg_pct_CDF = . then fall_avg_pct_CDF = 0;
			if fall_avg_pct_DFW = . then fall_avg_pct_DFW = 0;
			if fall_avg_pct_DF = . then fall_avg_pct_DF = 0;
			if fall_avg_difficulty = . then fall_crse_mi = 1; else fall_crse_mi = 0; 
			if fall_avg_difficulty = . then fall_avg_difficulty = 0;
			if spring_avg_pct_withdrawn = . then spring_avg_pct_withdrawn = 0;
			if spring_avg_pct_CDFW = . then spring_avg_pct_CDFW = 0;
			if spring_avg_pct_CDF = . then spring_avg_pct_CDF = 0;
			if spring_avg_pct_DFW = . then spring_avg_pct_DFW = 0;
			if spring_avg_pct_DF = . then spring_avg_pct_DF = 0;
			if spring_avg_difficulty = . then spring_crse_mi = 1; else spring_crse_mi = 0; 
			if spring_avg_difficulty = . then spring_avg_difficulty = 0;
			if spring_lec_count = . then spring_lec_count = 0;
			if spring_lab_count = . then spring_lab_count = 0;
			if spring_int_count = . then spring_int_count = 0;
			if spring_stu_count = . then spring_stu_count = 0;
			if spring_sem_count = . then spring_sem_count = 0;
			if spring_oth_count = . then spring_oth_count = 0;
			if spring_lec_contact_hrs = . then spring_lec_contact_hrs = 0;
			if spring_lab_contact_hrs = . then spring_lab_contact_hrs = 0;
			if spring_int_contact_hrs = . then spring_int_contact_hrs = 0;
			if spring_stu_contact_hrs = . then spring_stu_contact_hrs = 0;
			if spring_sem_contact_hrs = . then spring_sem_contact_hrs = 0;
			if spring_oth_contact_hrs = . then spring_oth_contact_hrs = 0;
			if total_spring_contact_hrs = . then total_spring_contact_hrs = 0;
			if total_fall_units = . then total_fall_units = 0;
			if total_spring_units = . then total_spring_units = 0;
			if fall_credit_hours = . then fall_credit_hours = 0;
			if spring_credit_hours = . then spring_credit_hours = 0;
			if fall_lec_contact_hrs = . then fall_lec_contact_hrs = 0;
			if fall_lab_contact_hrs = . then fall_lab_contact_hrs = 0;
			if spring_lec_contact_hrs = . then spring_lec_contact_hrs = 0;
			if spring_lab_contact_hrs = . then spring_lab_contact_hrs = 0;
			if total_fall_contact_hrs = . then total_fall_contact_hrs = 0;
			if total_spring_contact_hrs = . then total_spring_contact_hrs = 0;
			if fall_midterm_gpa_avg = . then fall_midterm_gpa_avg_mi = 1; else fall_midterm_gpa_avg_mi = 0;
			if fall_midterm_gpa_avg = . then fall_midterm_gpa_avg = 0;
			if fall_midterm_grade_count = . then fall_midterm_grade_count = 0;
			if fall_midterm_S_grade_count = . then fall_midterm_S_grade_count = 0;
			if fall_midterm_W_grade_count = . then fall_midterm_W_grade_count = 0;
			if spring_midterm_gpa_avg = . then spring_midterm_gpa_avg_mi = 1; else spring_midterm_gpa_avg_mi = 0;
			if spring_midterm_gpa_avg = . then spring_midterm_gpa_avg = 0;
			if spring_midterm_grade_count = . then spring_midterm_grade_count = 0;
			if spring_midterm_S_grade_count = . then spring_midterm_S_grade_count = 0;
			if spring_midterm_W_grade_count = . then spring_midterm_W_grade_count = 0;
			if fall_term_gpa = . then fall_term_gpa_mi = 1; else fall_term_gpa_mi = 0;
			if fall_term_gpa = . then fall_term_gpa = 0;
			if spring_term_gpa = . then spring_term_gpa_mi = 1; else spring_term_gpa_mi = 0;
			if spring_term_gpa = . then spring_term_gpa = 0;
			if fall_term_D_grade_count = . then fall_term_D_grade_count_mi = 1; else fall_term_D_grade_count_mi = 0;
			if fall_term_D_grade_count = . then fall_term_D_grade_count = 0;
			if fall_term_F_grade_count = . then fall_term_F_grade_count_mi = 1; else fall_term_F_grade_count_mi = 0;
			if fall_term_F_grade_count = . then fall_term_F_grade_count = 0;
			if fall_term_W_grade_count = . then fall_term_W_grade_count_mi = 1; else fall_term_W_grade_count_mi = 0;
			if fall_term_W_grade_count = . then fall_term_W_grade_count = 0;
			if fall_term_I_grade_count = . then fall_term_I_grade_count_mi = 1; else fall_term_I_grade_count_mi = 0;
			if fall_term_I_grade_count = . then fall_term_I_grade_count = 0;
			if fall_term_X_grade_count = . then fall_term_X_grade_count_mi = 1; else fall_term_X_grade_count_mi = 0;
			if fall_term_X_grade_count = . then fall_term_X_grade_count = 0;
			if fall_term_U_grade_count = . then fall_term_U_grade_count_mi = 1; else fall_term_U_grade_count_mi = 0;
			if fall_term_U_grade_count = . then fall_term_U_grade_count = 0;
			if fall_term_S_grade_count = . then fall_term_S_grade_count_mi = 1; else fall_term_S_grade_count_mi = 0;
			if fall_term_S_grade_count = . then fall_term_S_grade_count = 0;
			if fall_term_P_grade_count = . then fall_term_P_grade_count_mi = 1; else fall_term_P_grade_count_mi = 0;
			if fall_term_P_grade_count = . then fall_term_P_grade_count = 0;
			if fall_term_Z_grade_count = . then fall_term_Z_grade_count_mi = 1; else fall_term_Z_grade_count_mi = 0;
			if fall_term_Z_grade_count = . then fall_term_Z_grade_count = 0;
			if fall_term_letter_count = . then fall_term_letter_count_mi = 1; else fall_term_letter_count_mi = 0;
			if fall_term_letter_count = . then fall_term_letter_count = 0;
			if fall_term_grade_count = . then fall_term_grade_count_mi = 1; else fall_term_grade_count_mi = 0;
			if fall_term_grade_count = . then fall_term_grade_count = 0;
			fall_term_no_letter_count = fall_term_grade_count - fall_term_letter_count;
			if spring_term_D_grade_count = . then spring_term_D_grade_count_mi = 1; else spring_term_D_grade_count_mi = 0;
			if spring_term_D_grade_count = . then spring_term_D_grade_count = 0;
			if spring_term_F_grade_count = . then spring_term_F_grade_count_mi = 1; else spring_term_F_grade_count_mi = 0;
			if spring_term_F_grade_count = . then spring_term_F_grade_count = 0;
			if spring_term_W_grade_count = . then spring_term_W_grade_count_mi = 1; else spring_term_W_grade_count_mi = 0;
			if spring_term_W_grade_count = . then spring_term_W_grade_count = 0;
			if spring_term_I_grade_count = . then spring_term_I_grade_count_mi = 1; else spring_term_I_grade_count_mi = 0;
			if spring_term_I_grade_count = . then spring_term_I_grade_count = 0;
			if spring_term_X_grade_count = . then spring_term_X_grade_count_mi = 1; else spring_term_X_grade_count_mi = 0;
			if spring_term_X_grade_count = . then spring_term_X_grade_count = 0;
			if spring_term_U_grade_count = . then spring_term_U_grade_count_mi = 1; else spring_term_U_grade_count_mi = 0;
			if spring_term_U_grade_count = . then spring_term_U_grade_count = 0;
			if spring_term_S_grade_count = . then spring_term_S_grade_count_mi = 1; else spring_term_S_grade_count_mi = 0;
			if spring_term_S_grade_count = . then spring_term_S_grade_count = 0;
			if spring_term_P_grade_count = . then spring_term_P_grade_count_mi = 1; else spring_term_P_grade_count_mi = 0;
			if spring_term_P_grade_count = . then spring_term_P_grade_count = 0;
			if spring_term_Z_grade_count = . then spring_term_Z_grade_count_mi = 1; else spring_term_Z_grade_count_mi = 0;
			if spring_term_Z_grade_count = . then spring_term_Z_grade_count = 0;
			if spring_term_letter_count = . then spring_term_leter_count_mi = 1; else spring_term_leter_count_mi = 0;
			if spring_term_letter_count = . then spring_term_letter_count = 0;
			if spring_term_grade_count = . then spring_term_grade_count_mi = 1; else spring_term_grade_count_mi = 0;
			if spring_term_grade_count = . then spring_term_grade_count = 0;
			spring_term_no_letter_count = spring_term_grade_count - spring_term_letter_count;
			if first_gen_flag = '' then first_gen_flag_mi = 1; else first_gen_flag_mi = 0;
			if first_gen_flag = '' then first_gen_flag = 'N';
			if camp_addr_indicator ^= 'Y' then camp_addr_indicator = 'N';
			if housing_reshall_indicator ^= 'Y' then housing_reshall_indicator = 'N';
			if housing_ssa_indicator ^= 'Y' then housing_ssa_indicator = 'N';
			if housing_family_indicator ^= 'Y' then housing_family_indicator = 'N';
			if afl_reshall_indicator ^= 'Y' then afl_reshall_indicator = 'N';
			if afl_ssa_indicator ^= 'Y' then afl_ssa_indicator = 'N';
			if afl_family_indicator ^= 'Y' then afl_family_indicator = 'N';
			if afl_greek_indicator ^= 'Y' then afl_greek_indicator = 'N';
			if afl_greek_life_indicator ^= 'Y' then afl_greek_life_indicator = 'N';
			fall_withdrawn_hours = (total_fall_units - fall_credit_hours) * -1;
			if total_fall_units = 0 then fall_withdrawn_ind = 1; else fall_withdrawn_ind = 0;
			spring_withdrawn_hours = (total_spring_units - spring_credit_hours) * -1;
			if total_spring_units = 0 then spring_withdrawn = 1; else spring_withdrawn = 0;
			spring_midterm_gpa_change = spring_midterm_gpa_avg - fall_cum_gpa;
			unmet_need_disb = fed_need - total_disb;
			unmet_need_acpt = fed_need - total_accept;
			if unmet_need_acpt = . then unmet_need_acpt_mi = 1; else unmet_need_acpt_mi = 0;
			if unmet_need_acpt < 0 then unmet_need_acpt = 0;
			unmet_need_ofr = fed_need - total_offer;
			if unmet_need_ofr = . then unmet_need_ofr_mi = 1; else unmet_need_ofr_mi = 0;
			if unmet_need_ofr < 0 then unmet_need_ofr = 0;
			if fed_efc = . then fed_efc = 0;
			if fed_need = . then fed_need = 0;
			if total_disb = . then total_disb = 0;
			if total_offer = . then total_offer = 0;
			if total_accept = . then total_accept = 0;
		run;
		""")

		print('Done\n')

		# Export data from SAS
		print('Export data from SAS...')

		sas_log = sas.submit("""
		libname valid \"Z:\\Nathan\\Models\\student_risk\\datasets\\\" outencoding=\'UTF-8\';

		%let valid_pass = 0;

		%if %sysfunc(exist(valid.ft_ft_1yr_validation_set)) 
			%then %do;
				data work.validation_set_compare;
					set valid.ft_ft_1yr_validation_set;
				run;
			%end;
			
			%else %do;
				data work.validation_set_compare;
					set work.validation_set;
					stop;
				run;
				
				proc compare data=validation_set compare=validation_set_compare method=absolute;
				run;
			%end;

		proc compare data=validation_set compare=validation_set_compare method=absolute;
		run;

		%if &sysinfo ^= 0
			 
			%then %do;
				data valid.ft_ft_1yr_validation_set;
					set work.validation_set;
				run;
			%end;
			
			%else %do;
				%let valid_pass = 1;
			%end;

		libname training \"Z:\\Nathan\\Models\\student_risk\\datasets\\\" outencoding=\'UTF-8\';

		%let training_pass = 0;

		%if %sysfunc(exist(training.ft_ft_1yr_training_set)) 
			%then %do;
				data work.training_set_compare;
					set training.ft_ft_1yr_training_set;
				run;
			%end;
			
			%else %do;
				data work.training_set_compare;
					set work.training_set;
					stop;
				run;
				
				proc compare data=training_set compare=training_set_compare method=absolute;
				run;
			%end;

		proc compare data=training_set compare=training_set_compare method=absolute;
		run;

		%if &sysinfo ^= 0
			 
			%then %do;
				data training.ft_ft_1yr_training_set;
					set work.training_set;
				run;
			%end;
			
			%else %do;
				%let training_pass = 1;
			%end;
			
		libname testing \"Z:\\Nathan\\Models\\student_risk\\datasets\\\" outencoding=\'UTF-8\';

		%let testing_pass = 0;

		%if %sysfunc(exist(testing.ft_ft_1yr_testing_set)) 
			%then %do;
				data work.testing_set_compare;
					set testing.ft_ft_1yr_testing_set;
				run;
			%end;
			
			%else %do;
				data work.testing_set_compare;
					set work.testing_set;
					stop;
				run;
				
				proc compare data=testing_set compare=testing_set_compare method=absolute;
				run;
			%end;

		proc compare data=testing_set compare=testing_set_compare method=absolute;
		run;
		
		%if &sysinfo ^= 0
			 
			%then %do;
				data testing.ft_ft_1yr_testing_set;
					set work.testing_set;
				run;
			%end;
			
			%else %do;
				%let testing_pass = 1;
			%end;
		""")

		HTML(sas_log['LOG'])

		print('Done\n')

		# End SAS session
		sas.endsas()

	@staticmethod
	def build_census_dev(outcome: str, full_acad_year: int, aid_snapshot: str, snapshot: str, term_type: str) -> None:

		# Start SAS session
		print('\nStart SAS session...')

		sas = saspy.SASsession()

		sas.submit("""
		options sqlreduceput=all sqlremerge;
		run;
		""")
		
		# Set libname statements
		print('Set libname statements...')

		sas.submit("""
		%let dsn = census;
		%let adm = adm;
		""")

		sas.submit("""
		libname &dsn. odbc dsn=&dsn. schema=dbo;
		libname &dev. odbc dsn=&dev. schema=dbo;
		libname &adm. odbc dsn=&adm. schema=dbo;
		libname acs \"Z:\\Nathan\\Models\\student_risk\\supplemental_files\\\";
		""")

		print('Done\n')

		# Set macro variables
		print('Set macro variables...')

		sas.symput('outcome', outcome)
		sas.symput('full_acad_year', full_acad_year)
		sas.symput('aid_snapshot', aid_snapshot)
		sas.symput('snapshot', snapshot)
		sas.symput('term_type', term_type)

		sas.submit("""
		%let acs_lag = 2;
		%let lag_year = 1;
		%let end_cohort = %eval(&full_acad_year. - (2 * &lag_year.));
		%let start_cohort = %eval(&end_cohort. - 8);
		""")

		print('Done\n')

		# Import supplemental files
		print('Import supplemental files...')
		start = time.perf_counter()

		sas.submit("""
		proc import out=act_to_sat_engl_read
			datafile=\"Z:\\Nathan\\Models\\student_risk\\supplemental_files\\act_to_sat_engl_read.xlsx\"
			dbms=XLSX REPLACE;
			getnames=YES;
			run;
		""")

		sas.submit("""
		proc import out=act_to_sat_math
			datafile=\"Z:\\Nathan\\Models\\student_risk\\supplemental_files\\act_to_sat_math.xlsx\"
			dbms=XLSX REPLACE;
			getnames=YES;
			run;
		""")

		sas.submit("""
		proc import out=cpi
			datafile=\"Z:\\Nathan\\Models\\student_risk\\supplemental_files\\cpi.xlsx\"
			dbms=XLSX REPLACE;
			getnames=YES;
		run;
		""")

		stop = time.perf_counter()
		print(f'Done in {stop - start:.1f} seconds\n')

		# Create SAS macro
		print('Create SAS macro...')

		sas.submit("""
		%macro loop;

			%do cohort_year=&start_cohort. %to &end_cohort.;
			
			proc sql;
				create table cohort_&cohort_year. as
				select distinct a.*,
					substr(a.last_sch_postal,1,5) as targetid,
					case when a.sex = 'M' then 1 
						else 0
					end as male,
					case when a.age < 18.25 then 'Q1'
						when 18.25 <= a.age < 18.5 then 'Q2'
						when 18.5 <= a.age < 18.75 then 'Q3'
						when 18.75 <= a.age then 'Q4'
						else 'missing'
					end as age_group,
					case when a.father_attended_wsu_flag = 'Y' then 1 
						else 0
					end as father_wsu_flag,
					case when a.mother_attended_wsu_flag = 'Y' then 1 
						else 0
					end as mother_wsu_flag,
					case when a.ipeds_ethnic_group in ('2', '3', '5', '7', 'Z') then 1 
						else 0
					end as underrep_minority,
					case when a.WA_residency = 'RES' then 1
						else 0
					end as resident,
					case when a.adm_parent1_highest_educ_lvl in ('B','C','D','E','F') then '< bach'
						when a.adm_parent1_highest_educ_lvl = 'G' then 'bach'
						when a.adm_parent1_highest_educ_lvl in ('H','I','J','K','L') then '> bach'
							else 'missing'
					end as parent1_highest_educ_lvl,
					case when a.adm_parent2_highest_educ_lvl in ('B','C','D','E','F') then '< bach'
						when a.adm_parent2_highest_educ_lvl = 'G' then 'bach'
						when a.adm_parent2_highest_educ_lvl in ('H','I','J','K','L') then '> bach'
							else 'missing'
					end as parent2_highest_educ_lvl,
					coalesce(b.PULLM_distance_km, b2.VANCO_distance_km, b3.TRICI_distance_km, b4.EVERE_distance_km, b5.SPOKA_distance_km, b6.PULLM_distance_km) as distance,
					l.cpi_adj,
					c.median_inc as median_inc_wo_cpi,
					c.median_inc*l.cpi_adj as median_inc,
					c.gini_indx,
					d.pvrt_total/d.pvrt_base as pvrt_rate,
					e.educ_total/e.educ_base as educ_rate,
					f.pop/(g.area*3.861E-7) as pop_dens,
					h.median_value as median_value_wo_cpi,
					h.median_value*l.cpi_adj as median_value,
					i.race_blk/i.race_tot as pct_blk,
					i.race_ai/i.race_tot as pct_ai,
					i.race_asn/i.race_tot as pct_asn,
					i.race_hawi/i.race_tot as pct_hawi,
					i.race_oth/i.race_tot as pct_oth,
					i.race_two/i.race_tot as pct_two,
					(i.race_blk + i.race_ai + i.race_asn + i.race_hawi + i.race_oth + i.race_two)/i.race_tot as pct_non,
					j.ethnic_hisp/j.ethnic_tot as pct_hisp,
					case when k.locale = '11' then 1 else 0 end as city_large,
					case when k.locale = '12' then 1 else 0 end as city_mid,
					case when k.locale = '13' then 1 else 0 end as city_small,
					case when k.locale = '21' then 1 else 0 end as suburb_large,
					case when k.locale = '22' then 1 else 0 end as suburb_mid,
					case when k.locale = '23' then 1 else 0 end as suburb_small,
					case when k.locale = '31' then 1 else 0 end as town_fringe,
					case when k.locale = '32' then 1 else 0 end as town_distant,
					case when k.locale = '33' then 1 else 0 end as town_remote,
					case when k.locale = '41' then 1 else 0 end as rural_fringe,
					case when k.locale = '42' then 1 else 0 end as rural_distant,
					case when k.locale = '43' then 1 else 0 end as rural_remote
				from &dsn..new_student_enrolled_vw as a
				left join acs.distance_km as b
					on substr(a.last_sch_postal,1,5) = b.inputid
						and a.adj_acad_prog_primary_campus = 'PULLM'
				left join acs.distance_km as b2
					on substr(a.last_sch_postal,1,5) = b2.inputid
						and a.adj_acad_prog_primary_campus = 'VANCO'
				left join acs.distance_km as b3
					on substr(a.last_sch_postal,1,5) = b3.inputid
						and a.adj_acad_prog_primary_campus = 'TRICI'
				left join acs.distance_km as b4
					on substr(a.last_sch_postal,1,5) = b4.inputid
						and a.adj_acad_prog_primary_campus = 'EVERE'
				left join acs.distance_km as b5
					on substr(a.last_sch_postal,1,5) = b5.inputid
						and a.adj_acad_prog_primary_campus = 'SPOKA'
				left join acs.distance_km as b6
					on substr(a.last_sch_postal,1,5) = b6.inputid
						and a.adj_acad_prog_primary_campus = 'ONLIN'
				left join acs.acs_income_%eval(&cohort_year. - &acs_lag.) as c
					on substr(a.last_sch_postal,1,5) = c.geoid
				left join acs.acs_poverty_%eval(&cohort_year. - &acs_lag.) as d
					on substr(a.last_sch_postal,1,5) = d.geoid
				left join acs.acs_education_%eval(&cohort_year. - &acs_lag.) as e
					on substr(a.last_sch_postal,1,5) = e.geoid
				left join acs.acs_demo_%eval(&cohort_year. - &acs_lag.) as f
					on substr(a.last_sch_postal,1,5) = f.geoid
				left join acs.acs_area_%eval(&cohort_year. - &acs_lag.) as g
					on substr(a.last_sch_postal,1,5) = g.geoid
				left join acs.acs_housing_%eval(&cohort_year. - &acs_lag.) as h
					on substr(a.last_sch_postal,1,5) = h.geoid
				left join acs.acs_race_%eval(&cohort_year. - &acs_lag.) as i
					on substr(a.last_sch_postal,1,5) = i.geoid
				left join acs.acs_ethnicity_%eval(&cohort_year. - &acs_lag.) as j
					on substr(a.last_sch_postal,1,5) = j.geoid
				left join acs.edge_locale14_zcta_table as k
					on substr(a.last_sch_postal,1,5) = k.zcta5ce10
				left join cpi as l
					on input(a.full_acad_year,4.) = l.acs_lag
				where a.full_acad_year = "&cohort_year."
					and substr(a.strm,4,1) = '7'
					and a.acad_career = 'UGRD'
					and a.adj_admit_type_cat in ('FRSH')
					and a.ipeds_full_part_time = 'F'
					and a.ipeds_ind = 1
					and a.term_credit_hours > 0
					and a.WA_residency ^= 'NON-I'
			;quit;
			
			proc sql;
				create table pell_&cohort_year. as
				select distinct
					emplid,
					pell_recipient_ind
				from &dsn..new_student_profile_ugrd_cs
				where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and adj_admit_type_cat in ('FRSH')
					and ipeds_full_part_time = 'F'
					and WA_residency ^= 'NON-I'
			;quit;
			
			proc sql;
				create table eot_term_gpa_&cohort_year. as
				select distinct
					a.emplid,
					b.term_gpa as fall_term_gpa,
					b.term_gpa_hours as fall_term_gpa_hours,
					b.cum_gpa as fall_cum_gpa,
					b.cum_gpa_hours as fall_cum_gpa_hours,
					c.term_gpa as spring_term_gpa,
					c.term_gpa_hours as spring_term_gpa_hours,
					c.cum_gpa as spring_cum_gpa,
					c.cum_gpa_hours as spring_cum_gpa_hours
				from &dsn..student_enrolled_vw as a
				left join &dsn..student_enrolled_vw as b
					on a.emplid = b.emplid
						and b.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
						and b.snapshot = 'eot'
						and b.ipeds_full_part_time = 'F'
				left join &dsn..student_enrolled_vw as c
					on a.emplid = c.emplid
						and c.strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
						and c.snapshot = 'eot'
						and c.ipeds_full_part_time = 'F'
				where a.snapshot = 'eot'
					and a.full_acad_year = "&cohort_year."
					and a.ipeds_full_part_time = 'F'
			;quit;
			
			%if &outcome. = term %then %do;
				proc sql;
					create table enrolled_&cohort_year. as
					select distinct 
						a.emplid, 
						b.cont_term,
						c.grad_term,
						case when c.emplid is not null	then 1
														else 0
														end as deg_ind,
						case when b.emplid is not null 	then 1
							when c.emplid is not null	then 1
														else 0
														end as enrl_ind
					from &dsn..student_enrolled_vw as a
					left join (select distinct 
									emplid 
									,term_code as cont_term
									,enrl_ind
								from &dsn..student_enrolled_vw 
								where snapshot = 'census'
									and full_acad_year = put(&cohort_year., 4.)
									and substr(strm,4,1) = '3'
									and acad_career = 'UGRD'
									and new_continue_status = 'CTU'
									and term_credit_hours > 0) as b
						on a.emplid = b.emplid
					left join (select distinct 
									emplid
									,term_code as grad_term
								from &dsn..student_degree_vw 
								where snapshot = 'degree'
									and full_acad_year = put(&cohort_year., 4.)
									and substr(strm,4,1) = '3'
									and acad_career = 'UGRD'
									and ipeds_award_lvl = 5) as c
						on a.emplid = c.emplid
					where a.snapshot = 'census'
						and a.full_acad_year = "&cohort_year."
						and substr(a.strm,4,1) = '7'
						and a.acad_career = 'UGRD'
						and a.term_credit_hours > 0
				;quit;
			%end;

			%if &outcome. = year %then %do;
				proc sql;
					create table enrolled_&cohort_year. as
					select distinct 
						a.emplid, 
						b.cont_term,
						c.grad_term,
						case when c.emplid is not null	then 1
														else 0
														end as deg_ind,
						case when b.emplid is not null 	then 1
							when c.emplid is not null	then 1
														else 0
														end as enrl_ind
					from &dsn..student_enrolled_vw as a
					left join (select distinct 
									emplid 
									,term_code as cont_term
									,enrl_ind
								from &dsn..student_enrolled_vw 
								where snapshot = 'census'
									and full_acad_year = put(%eval(&cohort_year. + &lag_year.), 4.)
									and substr(strm,4,1) = '7'
									and acad_career = 'UGRD'
									and new_continue_status = 'CTU'
									and term_credit_hours > 0) as b
						on a.emplid = b.emplid
					left join (select distinct 
									emplid
									,term_code as grad_term
								from &dsn..student_degree_vw 
								where snapshot = 'degree'
									and put(&cohort_year., 4.) <= full_acad_year <= put(%eval(&cohort_year. + &lag_year.), 4.)
									and acad_career = 'UGRD'
									and ipeds_award_lvl = 5) as c
						on a.emplid = c.emplid
					where a.snapshot = 'census'
						and a.full_acad_year = "&cohort_year."
						and substr(a.strm,4,1) = '7'
						and a.acad_career = 'UGRD'
						and a.term_credit_hours > 0
				;quit;
			%end;

			proc sql;
				create table race_detail_&cohort_year. as
				select 
					a.emplid,
					case when hispc.emplid is not null 	then 'Y'
														else 'N'
														end as race_hispanic,
					case when amind.emplid is not null 	then 'Y'
														else 'N'
														end as race_american_indian,
					case when alask.emplid is not null 	then 'Y'
														else 'N'
														end as race_alaska,
					case when asian.emplid is not null 	then 'Y'
														else 'N'
														end as race_asian,
					case when black.emplid is not null 	then 'Y'
														else 'N'
														end as race_black,
					case when hawai.emplid is not null 	then 'Y'
														else 'N'
														end as race_native_hawaiian,
					case when white.emplid is not null 	then 'Y'
														else 'N'
														end as race_white
				from cohort_&cohort_year. as a
				left join (select distinct e4.emplid from &dsn..student_ethnic_detail as e4
							left join &dsn..xw_ethnic_detail_to_group_vw as xe4
								on e4.ethnic_cd = xe4.ethnic_cd
							where e4.snapshot = 'census'
								and e4.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe4.ethnic_group = '4') as asian
					on a.emplid = asian.emplid
				left join (select distinct e2.emplid from &dsn..student_ethnic_detail as e2
							left join &dsn..xw_ethnic_detail_to_group_vw as xe2
								on e2.ethnic_cd = xe2.ethnic_cd
							where e2.snapshot = 'census'
								and e2.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe2.ethnic_group = '2') as black
					on a.emplid = black.emplid
				left join (select distinct e7.emplid from &dsn..student_ethnic_detail as e7
							left join &dsn..xw_ethnic_detail_to_group_vw as xe7
								on e7.ethnic_cd = xe7.ethnic_cd
							where e7.snapshot = 'census'
								and e7.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe7.ethnic_group = '7') as hawai
					on a.emplid = hawai.emplid
				left join (select distinct e1.emplid from &dsn..student_ethnic_detail as e1
							left join &dsn..xw_ethnic_detail_to_group_vw as xe1
								on e1.ethnic_cd = xe1.ethnic_cd
							where e1.snapshot = 'census'
								and e1.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe1.ethnic_group = '1') as white
					on a.emplid = white.emplid
				left join (select distinct e5a.emplid from &dsn..student_ethnic_detail as e5a
							left join &dsn..xw_ethnic_detail_to_group_vw as xe5a
								on e5a.ethnic_cd = xe5a.ethnic_cd
							where e5a.snapshot = 'census' 
								and e5a.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe5a.ethnic_group = '5'
								and e5a.ethnic_cd in ('014','016','017','018',
														'935','941','942','943',
														'950','R10','R14')) as alask
					on a.emplid = alask.emplid
				left join (select distinct e5b.emplid from &dsn..student_ethnic_detail as e5b
							left join &dsn..xw_ethnic_detail_to_group_vw as xe5b
								on e5b.ethnic_cd = xe5b.ethnic_cd
							where e5b.snapshot = 'census'
								and e5b.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe5b.ethnic_group = '5'
								and e5b.ethnic_cd not in ('014','016','017','018',
															'935','941','942','943',
															'950','R14')) as amind
					on a.emplid = amind.emplid
				left join (select distinct e6.emplid from &dsn..student_ethnic_detail as e6
							left join &dsn..xw_ethnic_detail_to_group_vw as xe6
								on e6.ethnic_cd = xe6.ethnic_cd
							where e6.snapshot = 'census'
								and e6.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe6.ethnic_group = '3') as hispc
					on a.emplid = hispc.emplid
			;quit;
			
			proc sql;
				create table plan_&cohort_year. as 
				select distinct 
					emplid,
					acad_plan,
					acad_plan_descr,
					plan_owner_org,
					plan_owner_org_descr,
					plan_owner_group_descrshort,
					case when plan_owner_group_descrshort = 'Business' then 1 else 0 end as business,
					case when plan_owner_group_descrshort = 'CAHNREXT' 
						and plan_owner_org = '03_1240' then 1 else 0 end as cahnrs_anml,
					case when plan_owner_group_descrshort = 'CAHNREXT' 
						and plan_owner_org = '03_1990' then 1 else 0 end as cahnrs_envr,
					case when plan_owner_group_descrshort = 'CAHNREXT' 
						and plan_owner_org = '03_1150' then 1 else 0 end as cahnrs_econ,	
					case when plan_owner_group_descrshort = 'CAHNREXT'
						and plan_owner_org not in ('03_1240','03_1990','03_1150') then 1 else 0 end as cahnrext,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_1540' then 1 else 0 end as cas_chem,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_1710' then 1 else 0 end as cas_crim,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_2530' then 1 else 0 end as cas_math,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_2900' then 1 else 0 end as cas_psyc,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_8434' then 1 else 0 end as cas_biol,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_1830' then 1 else 0 end as cas_engl,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_2790' then 1 else 0 end as cas_phys,	
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org not in ('31_1540','31_1710','31_2530','31_2900','31_8434','31_1830','31_2790') then 1 else 0 end as cas,
					case when plan_owner_group_descrshort = 'Comm' then 1 else 0 end as comm,
					case when plan_owner_group_descrshort = 'Education' then 1 else 0 end as education,
					case when plan_owner_group_descrshort in ('Med Sci','Medicine') then 1 else 0 end as medicine,
					case when plan_owner_group_descrshort = 'Nursing' then 1 else 0 end as nursing,
					case when plan_owner_group_descrshort = 'Pharmacy' then 1 else 0 end as pharmacy,
					case when plan_owner_group_descrshort = 'Provost' then 1 else 0 end as provost,
					case when plan_owner_group_descrshort = 'VCEA' 
						and plan_owner_org = '05_1520' then 1 else 0 end as vcea_bioe,
					case when plan_owner_group_descrshort = 'VCEA' 
						and plan_owner_org = '05_1590' then 1 else 0 end as vcea_cive,
					case when plan_owner_group_descrshort = 'VCEA' 
						and plan_owner_org = '05_1260' then 1 else 0 end as vcea_desn,
					case when plan_owner_group_descrshort = 'VCEA' 
						and plan_owner_org = '05_1770' then 1 else 0 end as vcea_eecs,
					case when plan_owner_group_descrshort = 'VCEA' 
						and plan_owner_org = '05_2540' then 1 else 0 end as vcea_mech,
					case when plan_owner_group_descrshort = 'VCEA' 
						and plan_owner_org not in ('05_1520','05_1590','05_1260','05_1770','05_2540') then 1 else 0 end as vcea,				
					case when plan_owner_group_descrshort = 'Vet Med' then 1 else 0 end as vet_med,
					case when plan_owner_group_descrshort not in ('Business','CAHNREXT','CAS','Comm',
																'Education','Med Sci','Medicine','Nursing',
																'Pharmacy','Provost','VCEA','Vet Med') then 1 else 0
					end as groupless,
					case when plan_owner_percent_owned = 50 and plan_owner_org in ('05_1770','03_1990','12_8595','31_8434') then 1 else 0
					end as split_plan,
					lsamp_stem_flag,
					anywhere_stem_flag
				from &dsn..student_acad_prog_plan_vw
				where snapshot = 'census'
					and full_acad_year = "&cohort_year."
					and substr(strm, 4, 1) = '7'
					and acad_career = 'UGRD'
					and adj_admit_type_cat in ('FRSH')
					and WA_residency ^= 'NON-I'
					and primary_plan_flag = 'Y'
					and primary_prog_flag = 'Y'
					and calculated split_plan = 0
			;quit;
			
			proc sql;
				create table need_&cohort_year. as
				select distinct
					emplid,
					snapshot as need_snap,
					aid_year,
					fed_efc,
					fed_need
				from &dsn..fa_award_period
				where snapshot = "&aid_snapshot."
					and aid_year = "&cohort_year."	
					and award_period = 'A'
					and efc_status = 'O'
			;quit;
			
			proc sql;
				create table aid_&cohort_year. as
				select distinct
					emplid,
					snapshot as aid_snap,
					aid_year,
					sum(disbursed_amt) as total_disb,
					sum(offer_amt) as total_offer,
					sum(accept_amt) as total_accept
				from &dsn..fa_award_aid_year_vw
				where snapshot = "&aid_snapshot."
					and aid_year = "&cohort_year."
					and award_period in ('A','B')
					and award_status in ('A','O')
					and acad_career = 'UGRD'
				group by emplid
			;quit;
			
			proc sql;
				create table dependent_&cohort_year. as
				select distinct
					a.emplid,
					b.snapshot as dependent_snap,
					a.num_in_family,
					a.stdnt_have_dependents,
					a.stdnt_have_children_to_support,
					a.stdnt_agi,
					a.stdnt_agi_blank
				from &dsn..fa_isir as a
				inner join (select distinct emplid, aid_year, min(snapshot) as snapshot from &dsn..fa_award_aid_year_vw where aid_year = "&cohort_year.") as b
					on a.emplid = b.emplid
						and a.aid_year = b.aid_year
						and a.snapshot = b.snapshot
				where a.aid_year = "&cohort_year."
			;quit;
			
			proc sql;
				create table exams_&cohort_year. as 
				select distinct
					a.emplid,
					a.best,
					a.bestr,
					a.qvalue,
					a.act_engl,
					a.act_read,
					a.act_math,
					largest(1, a.sat_erws, xw_one.sat_erws, xw_three.sat_erws) as sat_erws,
					largest(1, a.sat_mss, xw_two.sat_mss, xw_four.sat_mss) as sat_mss,
					largest(1, (a.sat_erws + a.sat_mss), (xw_one.sat_erws + xw_two.sat_mss), (xw_three.sat_erws + xw_four.sat_mss)) as sat_comp
				from &dsn..new_freshmen_test_score_vw as a
				left join &dsn..xw_sat_i_to_sat_erws as xw_one
					on (a.sat_i_verb + a.sat_i_wr) = xw_one.sat_i_verb_plus_wr
				left join &dsn..xw_sat_i_to_sat_mss as xw_two
					on a.sat_i_math = xw_two.sat_i_math
				left join act_to_sat_engl_read as xw_three
					on (a.act_engl + a.act_read) = xw_three.act_engl_read
				left join act_to_sat_math as xw_four
					on a.act_math = xw_four.act_math
				where snapshot = 'census'
			;quit;

			proc sql;
				create table degrees_&cohort_year. as
				select distinct
					emplid,
					case when degree in ('AD_AAS_T','AD_AS-T','AD_AS-T1','AD_AS-T2','AD_AS-T2B','AD_AST2C','AD_AST2M') 	then 'AD_AST' 
						when substr(degree,1,6) = 'AD_DTA' 																then 'AD_DTA' 																						
																														else degree end as degree,
					1 as ind
				from &dsn..student_ext_degree
				where floor(degree_term_code / 10) <= &cohort_year.
					and degree in ('AD_AAS_T','AD_AS-T','AD_AS-T1','AD_AS-T2','AD_AS-T2B',
									'AD_AST2C','AD_AST2M','AD_DTA','AD_GED','AD_GENS','AD_GER',
									'AD_HSDIP')
				order by emplid
			;quit;
			
			proc transpose data=degrees_&cohort_year. let out=degrees_&cohort_year. (drop=_name_);
				by emplid;
				id degree;
			run;
			
			proc sql;
				create table preparatory_&cohort_year. as
				select distinct
					emplid,
					ext_subject_area,
					1 as ind
				from &dsn..student_ext_acad_subj
				where snapshot = 'census'
					and ext_subject_area in ('CHS','RS','AP','IB','AICE')
				union
				select distinct
					emplid,
					'RS' as ext_subject_area,
					 1 as ind
				from &dsn..student_acad_prog_plan_vw
				where snapshot = 'census'
					and tuition_group in ('1RS','1TRS')
				order by emplid
			;quit;
			
			proc transpose data=preparatory_&cohort_year. let out=preparatory_&cohort_year. (drop=_name_);
				by emplid;
				id ext_subject_area;
			run;
			
			proc sql;
				create table visitation_&cohort_year. as
				select distinct a.emplid,
					b.snap_date,
					a.attendee_afr_am_scholars_visit,
					a.attendee_alive,
					a.attendee_campus_visit,
					a.attendee_cashe,
					a.attendee_destination,
					a.attendee_experience,
					a.attendee_fcd_pullman,
					a.attendee_fced,
					a.attendee_fcoc,
					a.attendee_fcod,
					a.attendee_group_visit,
					a.attendee_honors_visit,
					a.attendee_imagine_tomorrow,
					a.attendee_imagine_u,
					a.attendee_la_bienvenida,
					a.attendee_lvp_camp,
					a.attendee_oos_destination,
					a.attendee_oos_experience,
					a.attendee_preview,
					a.attendee_preview_jrs,
					a.attendee_shaping,
					a.attendee_top_scholars,
					a.attendee_transfer_day,
					a.attendee_vibes,
					a.attendee_welcome_center,
					a.attendee_any_visitation_ind,
					a.attendee_total_visits
				from &adm..UGRD_visitation_attendee as a
				inner join (select distinct emplid, max(snap_date) as snap_date 
							from &adm..UGRD_visitation_attendee 
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
							group by emplid) as b
					on a.emplid = b.emplid
						and a.snap_date = b.snap_date
				where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
			;quit;
			
			proc sql;
				create table visitation_detail_&cohort_year. as
				select distinct a.emplid,
					a.snap_date,
					a.go2,
					a.ocv_dt,
					a.ocv_fcd,
					a.ocv_fprv,
					a.ocv_gdt,
					a.ocv_jprv,
					a.ri_col,
					a.ri_fair,
					a.ri_hsv,
					a.ri_nac,
					a.ri_wac,
					a.ri_other,
					a.tap,
					a.tst,
					a.vi_chegg,
					a.vi_crn,
					a.vi_cxc,
					a.vi_mco,
					a.np_group,
					a.out_group,
					a.ref_group,
					a.ocv_da,
					a.ocv_ea,
					a.ocv_fced,
					a.ocv_fcoc,
					a.ocv_fcod,
					a.ocv_oosd,
					a.ocv_oose,
					a.ocv_ve
				from &adm..UGRD_visitation as a
				inner join (select distinct emplid, max(snap_date) as snap_date 
							from &adm..UGRD_visitation 
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
							group by emplid) as b
					on a.emplid = b.emplid
						and a.snap_date = b.snap_date
				where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
			;quit;
					
			proc sql;
				create table athlete_&cohort_year. as
				select distinct 
					emplid,
					case when (mbaseball = 'Y' 
						or mbasketball = 'Y'
						or mfootball = 'Y'
						or mgolf = 'Y'
						or mitrack = 'Y'
						or motrack = 'Y'
						or mxcountry = 'Y'
						or wbasketball = 'Y'
						or wgolf = 'Y'
						or witrack = 'Y'
						or wotrack = 'Y'
						or wsoccer = 'Y'
						or wswimming = 'Y'
						or wtennis = 'Y'
						or wvolleyball = 'Y'
						or wvrowing = 'Y'
						or wxcountry = 'Y') then 1 else 0
					end as athlete
				from &dsn..student_athlete_vw
				where snapshot = 'eot'
					and strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and ugrd_adj_admit_type in ('FRS','IFR','IPF','TRN','ITR','IPT')
			;quit;
			
			proc sql;
				create table remedial_&cohort_year. as
				select distinct
					emplid,
					case when grading_basis_enrl in ('REM','RMS','RMP') 	then 1
																			else 0
																			end as remedial
				from &dsn..class_registration_vw
				where snapshot = "&snapshot."
					and aid_year = "&cohort_year."
					and grading_basis_enrl in ('REM','RMS','RMP')
			;quit;

			proc sql;
				create table date_&cohort_year. as
				select distinct
					emplid,
					min(week_from_term_begin_dt) as min_week_from_term_begin_dt,
					max(week_from_term_begin_dt) as max_week_from_term_begin_dt,
					count(week_from_term_begin_dt) as count_week_from_term_begin_dt
				from &adm..UGRD_shortened_vw
				where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and ugrd_applicant_counting_ind = 1
				group by emplid
			;quit;
			
			proc sql;
				create table term_credit_hours_&cohort_year. as
				select distinct
					a.emplid,
					coalesce(a.term_credit_hours, 0) as fall_credit_hours,
					coalesce(b.term_credit_hours, 0) as spring_credit_hours
				from &dsn..student_enrolled_vw as a
				left join &dsn..student_enrolled_vw as b
					on a.emplid = b.emplid
						and a.acad_career = b.acad_career
						and b.snapshot = 'census'
						and b.strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
						and b.enrl_ind = 1
						and a.ipeds_full_part_time = 'F'
				where a.snapshot = 'census'
					and a.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and a.enrl_ind = 1
					and a.ipeds_full_part_time = 'F'
			;quit;
			
			proc sql;
				create table class_registration_&cohort_year. as
				select distinct
					strm,
					emplid,
					class_nbr,
					crse_id,
					ssr_component,
					unt_taken,
					grading_basis_enrl,
					enrl_status_reason,
					enrl_ind,
					class_grade_points as grade_points,
					class_grade_points_per_unit as grd_pts_per_unit,
					subject_catalog_nbr,
					crse_grade_off as crse_grade,
					case when crse_grade_off = 'A' 	then 4.0
						when crse_grade_off = 'A-'	then 3.7
						when crse_grade_off = 'B+'	then 3.3
						when crse_grade_off = 'B'	then 3.0
						when crse_grade_off = 'B-'	then 2.7
						when crse_grade_off = 'C+'	then 2.3
						when crse_grade_off = 'C'	then 2.0
						when crse_grade_off = 'C-'	then 1.7
						when crse_grade_off = 'D+'	then 1.3
						when crse_grade_off = 'D'	then 1.0
						when crse_grade_off = 'F'	then 0.0
													else .
													end as class_gpa,
					case when crse_grade_off = 'D' 	then 1
													else 0
													end as D_grade_ind,
					case when crse_grade_off = 'F' 	then 1
													else 0
													end as F_grade_ind,
					case when crse_grade_off = 'W' 	then 1
													else 0
													end as W_grade_ind,
					case when crse_grade_off = 'I' 	then 1
													else 0
													end as I_grade_ind,
					case when crse_grade_off = 'X' 	then 1
													else 0
													end as X_grade_ind,
					case when crse_grade_off = 'U' 	then 1
													else 0
													end as U_grade_ind,
					case when crse_grade_off = 'S' 	then 1
													else 0
													end as S_grade_ind,
					case when crse_grade_off = 'P' 	then 1
													else 0
													end as P_grade_ind,
					case when crse_grade_input = 'Z'	then 1
														else 0
														end as Z_grade_ind,
					case when unt_taken is not null and enrl_status_reason ^= 'WDRW'	then 1
																						else 0
																						end as term_grade_ind
				from &dsn..class_registration_vw
				where snapshot = 'eot'
					and full_acad_year = "&cohort_year."
					and subject_catalog_nbr ^= 'NURS 399'
					and stdnt_enrl_status = 'E'
			;quit;

			proc sql;
				create table midterm_class_registration_&cohort_year. as
				select distinct
					strm,
					emplid,
					class_nbr,
					crse_id,
					ssr_component,
					unt_taken,
					grading_basis_enrl,
					enrl_status_reason,
					enrl_ind,
					class_grade_points as grade_points,
					class_grade_points_per_unit as grd_pts_per_unit,
					subject_catalog_nbr,
					crse_grade_off as crse_grade,
					case when crse_grade_off = 'A' 	then 4.0
						when crse_grade_off = 'A-'	then 3.7
						when crse_grade_off = 'B+'	then 3.3
						when crse_grade_off = 'B'	then 3.0
						when crse_grade_off = 'B-'	then 2.7
						when crse_grade_off = 'C+'	then 2.3
						when crse_grade_off = 'C'	then 2.0
						when crse_grade_off = 'C-'	then 1.7
						when crse_grade_off = 'D+'	then 1.3
						when crse_grade_off = 'D'	then 1.0
						when crse_grade_off = 'F'	then 0.0
													else .
													end as class_gpa,
					case when crse_grade_off = 'D' 	then 1
													else 0
													end as D_grade_ind,
					case when crse_grade_off = 'F' 	then 1
													else 0
													end as F_grade_ind,
					case when crse_grade_off = 'W' 	then 1
													else 0
													end as W_grade_ind,
					case when crse_grade_off = 'I' 	then 1
													else 0
													end as I_grade_ind,
					case when crse_grade_off = 'X' 	then 1
													else 0
													end as X_grade_ind,
					case when crse_grade_off = 'U' 	then 1
													else 0
													end as U_grade_ind,
					case when crse_grade_off = 'S' 	then 1
													else 0
													end as S_grade_ind,
					case when crse_grade_off = 'P' 	then 1
													else 0
													end as P_grade_ind,
					case when crse_grade_input = 'Z'	then 1
														else 0
														end as Z_grade_ind,
					case when unt_taken is not null and enrl_status_reason ^= 'WDRW'	then 1
																						else 0
																						end as term_grade_ind
				from &dsn..class_registration_vw
				where snapshot = 'midterm'
					and full_acad_year = "&cohort_year."
					and subject_catalog_nbr ^= 'NURS 399'
					and stdnt_enrl_status = 'E'
			;quit;
			
			proc sql;
				create table eot_class_registration_&cohort_year. as
				select distinct
					strm,
					emplid,
					class_nbr,
					crse_id,
					ssr_component,
					unt_taken,
					grading_basis_enrl,
					enrl_status_reason,
					enrl_ind,
					class_grade_points as grade_points,
					class_grade_points_per_unit as grd_pts_per_unit,
					subject_catalog_nbr,
					crse_grade_off as crse_grade,
					case when crse_grade_off = 'A' 	then 4.0
						when crse_grade_off = 'A-'	then 3.7
						when crse_grade_off = 'B+'	then 3.3
						when crse_grade_off = 'B'	then 3.0
						when crse_grade_off = 'B-'	then 2.7
						when crse_grade_off = 'C+'	then 2.3
						when crse_grade_off = 'C'	then 2.0
						when crse_grade_off = 'C-'	then 1.7
						when crse_grade_off = 'D+'	then 1.3
						when crse_grade_off = 'D'	then 1.0
						when crse_grade_off = 'F'	then 0.0
													else .
													end as class_gpa,
					case when crse_grade_off = 'D' 	then 1
													else 0
													end as D_grade_ind,
					case when crse_grade_off = 'F' 	then 1
													else 0
													end as F_grade_ind,
					case when crse_grade_off = 'W' 	then 1
													else 0
													end as W_grade_ind,
					case when crse_grade_off = 'I' 	then 1
													else 0
													end as I_grade_ind,
					case when crse_grade_off = 'X' 	then 1
													else 0
													end as X_grade_ind,
					case when crse_grade_off = 'U' 	then 1
													else 0
													end as U_grade_ind,
					case when crse_grade_off = 'S' 	then 1
													else 0
													end as S_grade_ind,
					case when crse_grade_off = 'P' 	then 1
													else 0
													end as P_grade_ind,
					case when crse_grade_input = 'Z'	then 1
														else 0
														end as Z_grade_ind,
					case when unt_taken is not null and enrl_status_reason ^= 'WDRW'	then 1
																						else 0
																						end as term_grade_ind
				from &dsn..class_registration_vw
				where snapshot = 'eot'
					and full_acad_year = "&cohort_year."
					and subject_catalog_nbr ^= 'NURS 399'
					and stdnt_enrl_status = 'E'
			;quit;
			
			proc sql;
				create table eot_fall_term_grades_&cohort_year. as
				select distinct
					a.emplid,
					b.fall_term_gpa_hours,
					b.fall_term_gpa,
					c.fall_term_D_grade_count,
					c.fall_term_F_grade_count,
					c.fall_term_W_grade_count,
					c.fall_term_I_grade_count,
					c.fall_term_X_grade_count,
					c.fall_term_U_grade_count,
					c.fall_term_S_grade_count,
					c.fall_term_P_grade_count,
					c.fall_term_Z_grade_count,
					c.fall_term_letter_count,
					c.fall_term_grade_count
				from eot_class_registration_&cohort_year. as a
				left join (select distinct
								emplid,
								sum(unt_taken) as fall_term_gpa_hours,
								round(sum(class_gpa * unt_taken) / sum(unt_taken), .01) as fall_term_gpa
							from eot_class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and grading_basis_enrl = 'GRD'
								and crse_grade in ('A','A-','B+','B','B-','C+','C','C-','D+','D','F')
							group by emplid) as b
					on a.emplid = b.emplid
				left join (select distinct 
								emplid,
								sum(D_grade_ind) as fall_term_D_grade_count,
								sum(F_grade_ind) as fall_term_F_grade_count,
								sum(W_grade_ind) as fall_term_W_grade_count,
								sum(I_grade_ind) as fall_term_I_grade_count,
								sum(X_grade_ind) as fall_term_X_grade_count,
								sum(U_grade_ind) as fall_term_U_grade_count,
								sum(S_grade_ind) as fall_term_S_grade_count,
								sum(P_grade_ind) as fall_term_P_grade_count,
								sum(Z_grade_ind) as fall_term_Z_grade_count,
								count(class_gpa) as fall_term_letter_count,
								sum(term_grade_ind) as fall_term_grade_count
							from eot_class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
							group by emplid) as c
					on a.emplid = c.emplid
				where a.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
			;quit;

			proc sql;
				create table eot_spring_term_grades_&cohort_year. as
				select distinct
					a.emplid,
					b.spring_term_gpa_hours,
					b.spring_term_gpa,
					c.spring_term_D_grade_count,
					c.spring_term_F_grade_count,
					c.spring_term_W_grade_count,
					c.spring_term_I_grade_count,
					c.spring_term_X_grade_count,
					c.spring_term_U_grade_count,
					c.spring_term_S_grade_count,
					c.spring_term_P_grade_count,
					c.spring_term_Z_grade_count,
					c.spring_term_letter_count,
					c.spring_term_grade_count
				from eot_class_registration_&cohort_year. as a
				left join (select distinct
								emplid,
								sum(unt_taken) as spring_term_gpa_hours,
								round(sum(class_gpa * unt_taken) / sum(unt_taken), .01) as spring_term_gpa
							from eot_class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and grading_basis_enrl = 'GRD'
								and crse_grade in ('A','A-','B+','B','B-','C+','C','C-','D+','D','F')
								group by emplid) as b
					on a.emplid = b.emplid
				left join (select distinct
								emplid,
								sum(D_grade_ind) as spring_term_D_grade_count,
								sum(F_grade_ind) as spring_term_F_grade_count,
								sum(W_grade_ind) as spring_term_W_grade_count,
								sum(I_grade_ind) as spring_term_I_grade_count,
								sum(X_grade_ind) as spring_term_X_grade_count,
								sum(U_grade_ind) as spring_term_U_grade_count,
								sum(S_grade_ind) as spring_term_S_grade_count,
								sum(P_grade_ind) as spring_term_P_grade_count,
								sum(Z_grade_ind) as spring_term_Z_grade_count,
								count(class_gpa) as spring_term_letter_count,
								sum(term_grade_ind) as spring_term_grade_count
							from eot_class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
							group by emplid) as c
					on a.emplid = c.emplid
				where a.strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
			;quit;

			proc sql;
				create table eot_cum_grades_&cohort_year. as
				select distinct
					emplid,
					sum(unt_taken) as cum_gpa_hours,
					round(sum(class_gpa * unt_taken) / sum(unt_taken), .01) as cum_gpa
				from eot_class_registration_&cohort_year.
				where (strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7' 
					or strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3')
					and grading_basis_enrl = 'GRD'
					and crse_grade in ('A','A-','B+','B','B-','C+','C','C-','D+','D','F')
				group by emplid
			;quit;
			
			proc sql;
				create table class_difficulty_&cohort_year. as
				select distinct
					a.subject_catalog_nbr,
					a.ssr_component,
					coalesce(b.total_grade_A, 0) + coalesce(c.total_grade_A, 0) + coalesce(d.total_grade_A, 0)
						+ coalesce(e.total_grade_A, 0) + coalesce(f.total_grade_A, 0) + coalesce(g.total_grade_A, 0) as total_grade_A,
					(calculated total_grade_A * 4.0) as total_grade_A_GPA,
					coalesce(b.total_grade_A_minus, 0) + coalesce(c.total_grade_A_minus, 0) + coalesce(d.total_grade_A_minus, 0)
						+ coalesce(e.total_grade_A_minus, 0) + coalesce(f.total_grade_A_minus, 0) + coalesce(g.total_grade_A_minus, 0) as total_grade_A_minus,
					(calculated total_grade_A_minus * 3.7) as total_grade_A_minus_GPA,
					coalesce(b.total_grade_B_plus, 0) + coalesce(c.total_grade_B_plus, 0) + coalesce(d.total_grade_B_plus, 0)
						+ coalesce(e.total_grade_B_plus, 0) + coalesce(f.total_grade_B_plus, 0) + coalesce(g.total_grade_B_plus, 0) as total_grade_B_plus,
					(calculated total_grade_B_plus * 3.3) as total_grade_B_plus_GPA,
					coalesce(b.total_grade_B, 0) + coalesce(c.total_grade_B, 0) + coalesce(d.total_grade_B, 0)
						+ coalesce(e.total_grade_B, 0) + coalesce(f.total_grade_B, 0) + coalesce(g.total_grade_B, 0) as total_grade_B,
					(calculated total_grade_B * 3.0) as total_grade_B_GPA,
					coalesce(b.total_grade_B_minus, 0) + coalesce(c.total_grade_B_minus, 0) + coalesce(d.total_grade_B_minus, 0) 
						+ coalesce(e.total_grade_B_minus, 0) + coalesce(f.total_grade_B_minus, 0) + coalesce(g.total_grade_B_minus, 0) as total_grade_B_minus,
					(calculated total_grade_B_minus * 2.7) as total_grade_B_minus_GPA,
					coalesce(b.total_grade_C_plus, 0) + coalesce(c.total_grade_C_plus, 0) + coalesce(d.total_grade_C_plus, 0) 
						+ coalesce(e.total_grade_C_plus, 0) + coalesce(f.total_grade_C_plus, 0) + coalesce(g.total_grade_C_plus, 0) as total_grade_C_plus,
					(calculated total_grade_C_plus * 2.3) as total_grade_C_plus_GPA,
					coalesce(b.total_grade_C, 0) + coalesce(c.total_grade_C, 0) + coalesce(d.total_grade_C, 0) 
						+ coalesce(e.total_grade_C, 0) + coalesce(f.total_grade_C, 0) + coalesce(g.total_grade_C, 0) as total_grade_C,
					(calculated total_grade_C * 2.0) as total_grade_C_GPA,
					coalesce(b.total_grade_C_minus, 0) + coalesce(c.total_grade_C_minus, 0) + coalesce(d.total_grade_C_minus, 0)
						+ coalesce(e.total_grade_C_minus, 0) + coalesce(f.total_grade_C_minus, 0) + coalesce(g.total_grade_C_minus, 0) as total_grade_C_minus,
					(calculated total_grade_C_minus * 1.7) as total_grade_C_minus_GPA,
					coalesce(b.total_grade_D_plus, 0) + coalesce(c.total_grade_D_plus, 0) + coalesce(d.total_grade_D_plus, 0)
						+ coalesce(e.total_grade_D_plus, 0) + coalesce(f.total_grade_D_plus, 0) + coalesce(g.total_grade_D_plus, 0) as total_grade_D_plus,
					(calculated total_grade_D_plus * 1.3) as total_grade_D_plus_GPA,
					coalesce(b.total_grade_D, 0) + coalesce(c.total_grade_D, 0) + coalesce(d.total_grade_D, 0) 
						+ coalesce(e.total_grade_D, 0) + coalesce(f.total_grade_D, 0) + coalesce(g.total_grade_D, 0) as total_grade_D,
					(calculated total_grade_D * 1.0) as total_grade_D_GPA,
					coalesce(b.total_grade_F, 0) + coalesce(c.total_grade_F, 0) + coalesce(d.total_grade_F, 0) 
						+ coalesce(e.total_grade_F, 0) + coalesce(f.total_grade_F, 0) + coalesce(g.total_grade_F, 0) as total_grade_F,
					coalesce(b.total_withdrawn, 0) + coalesce(c.total_withdrawn, 0) + coalesce(d.total_withdrawn, 0) 
						+ coalesce(e.total_withdrawn, 0) + coalesce(f.total_withdrawn, 0) + coalesce(g.total_withdrawn, 0) as total_withdrawn,
					coalesce(b.total_dropped, 0) + coalesce(c.total_dropped, 0) + coalesce(d.total_dropped, 0)
						+ coalesce(e.total_dropped, 0) + coalesce(f.total_dropped, 0) + coalesce(g.total_dropped, 0) as total_dropped,
					coalesce(b.total_grade_I, 0) + coalesce(c.total_grade_I, 0) + coalesce(d.total_grade_I, 0)
						+ coalesce(e.total_grade_I, 0) + coalesce(f.total_grade_I, 0) + coalesce(g.total_grade_I, 0) as total_grade_I,
					coalesce(b.total_grade_X, 0) + coalesce(c.total_grade_X, 0) + coalesce(d.total_grade_X, 0)
						+ coalesce(e.total_grade_X, 0) + coalesce(f.total_grade_X, 0) + coalesce(g.total_grade_X, 0) as total_grade_X,
					coalesce(b.total_grade_U, 0) + coalesce(c.total_grade_U, 0) + coalesce(d.total_grade_U, 0)
						+ coalesce(e.total_grade_U, 0) + coalesce(f.total_grade_U, 0) + coalesce(g.total_grade_U, 0) as total_grade_U,
					coalesce(b.total_grade_S, 0) + coalesce(c.total_grade_S, 0) + coalesce(d.total_grade_S, 0)
						+ coalesce(e.total_grade_S, 0) + coalesce(f.total_grade_S, 0) + coalesce(g.total_grade_S, 0) as total_grade_S,
					coalesce(b.total_grade_P, 0) + coalesce(c.total_grade_P, 0) + coalesce(d.total_grade_P, 0)
						+ coalesce(e.total_grade_P, 0) + coalesce(f.total_grade_P, 0) + coalesce(g.total_grade_P, 0) as total_grade_P,
					coalesce(b.total_no_grade, 0) + coalesce(c.total_no_grade, 0) + coalesce(d.total_no_grade, 0)
						+ coalesce(e.total_no_grade, 0) + coalesce(f.total_no_grade, 0) + coalesce(g.total_no_grade, 0) as total_no_grade,
					(calculated total_grade_A + calculated total_grade_A_minus 
						+ calculated total_grade_B_plus + calculated total_grade_B + calculated total_grade_B_minus
						+ calculated total_grade_C_plus + calculated total_grade_C + calculated total_grade_C_minus
						+ calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F) as total_grades,
					(calculated total_grade_A + calculated total_grade_A_minus 
						+ calculated total_grade_B_plus + calculated total_grade_B + calculated total_grade_B_minus
						+ calculated total_grade_C_plus + calculated total_grade_C + calculated total_grade_C_minus
						+ calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F + calculated total_withdrawn) as total_students,
					(calculated total_grade_A_GPA + calculated total_grade_A_minus_GPA 
						+ calculated total_grade_B_plus_GPA + calculated total_grade_B_GPA + calculated total_grade_B_minus_GPA
						+ calculated total_grade_C_plus_GPA + calculated total_grade_C_GPA + calculated total_grade_C_minus_GPA
						+ calculated total_grade_D_plus_GPA + calculated total_grade_D_GPA) as total_grades_GPA,
					(calculated total_grades_GPA / calculated total_grades) as class_average,
					(calculated total_withdrawn / calculated total_students) as pct_withdrawn,
					(calculated total_grade_C_minus + calculated total_grade_D_plus + calculated total_grade_D 
						+ calculated total_grade_F + calculated total_withdrawn) as CDFW,
					(calculated CDFW / calculated total_students) as pct_CDFW,
					(calculated total_grade_C_minus + calculated total_grade_D_plus + calculated total_grade_D 
						+ calculated total_grade_F) as CDF,
					(calculated CDF / calculated total_students) as pct_CDF,
					(calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F 
						+ calculated total_withdrawn) as DFW,
					(calculated DFW / calculated total_students) as pct_DFW,
					(calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F) as DF,
					(calculated DF / calculated total_students) as pct_DF
				from &dsn..class_vw as a
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'LEC'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as b
					on a.subject_catalog_nbr = b.subject_catalog_nbr
						and a.ssr_component = b.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'LAB'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as c
					on a.subject_catalog_nbr = c.subject_catalog_nbr
						and a.ssr_component = c.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'INT'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as d
					on a.subject_catalog_nbr = d.subject_catalog_nbr
						and a.ssr_component = d.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'STU'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as e
					on a.subject_catalog_nbr = e.subject_catalog_nbr
						and a.ssr_component = e.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'SEM'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as f
					on a.subject_catalog_nbr = f.subject_catalog_nbr
						and a.ssr_component = f.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as g
					on a.subject_catalog_nbr = g.subject_catalog_nbr
						and a.ssr_component = g.ssr_component
				where a.snapshot = 'eot'
					and a.full_acad_year = "&cohort_year."
					and a.grading_basis = 'GRD'
			;quit;
			
			proc sql;
				create table coursework_difficulty_&cohort_year. as
				select distinct
					a.emplid,
					avg(b.class_average) as fall_avg_difficulty,
					avg(b.pct_withdrawn) as fall_avg_pct_withdrawn,
					avg(b.pct_CDFW) as fall_avg_pct_CDFW,
					avg(b.pct_CDF) as fall_avg_pct_CDF,
					avg(b.pct_DFW) as fall_avg_pct_DFW,
					avg(b.pct_DF) as fall_avg_pct_DF,
					avg(c.class_average) as spring_avg_difficulty,
					avg(c.pct_withdrawn) as spring_avg_pct_withdrawn,
					avg(c.pct_CDFW) as spring_avg_pct_CDFW,
					avg(c.pct_CDF) as spring_avg_pct_CDF,
					avg(c.pct_DFW) as spring_avg_pct_DFW,
					avg(c.pct_DF) as spring_avg_pct_DF
				from class_registration_&cohort_year. as a
				left join class_difficulty_&cohort_year. as b
					on a.subject_catalog_nbr = b.subject_catalog_nbr
						and a.ssr_component = b.ssr_component
						and a.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
				left join class_difficulty_&cohort_year. as c
					on a.subject_catalog_nbr = c.subject_catalog_nbr
						and a.ssr_component = c.ssr_component
						and a.strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
				where a.enrl_status_reason ^= 'WDRW'
				group by a.emplid
			;quit;
			
proc sql;
				create table class_count_&cohort_year. as
				select distinct
					a.emplid,
					count(b.class_nbr) as fall_lec_count,
					count(c.class_nbr) as fall_lab_count,
					count(d.class_nbr) as fall_int_count,
					count(e.class_nbr) as fall_stu_count,
					count(f.class_nbr) as fall_sem_count,
					count(g.class_nbr) as fall_oth_count,
					sum(h.unt_taken) as fall_lec_units,
					sum(i.unt_taken) as fall_lab_units,
					sum(j.unt_taken) as fall_int_units,
					sum(k.unt_taken) as fall_stu_units,
					sum(l.unt_taken) as fall_sem_units,
					sum(m.unt_taken) as fall_oth_units,
					coalesce(calculated fall_lec_units, 0) + coalesce(calculated fall_lab_units, 0) + coalesce(calculated fall_int_units, 0) 
						+ coalesce(calculated fall_stu_units, 0) + coalesce(calculated fall_sem_units, 0) + coalesce(calculated fall_oth_units, 0) as total_fall_units,
					count(n.class_nbr) as spring_lec_count,
					count(o.class_nbr) as spring_lab_count,
					count(p.class_nbr) as spring_int_count,
					count(q.class_nbr) as spring_stu_count,
					count(r.class_nbr) as spring_sem_count,
					count(s.class_nbr) as spring_oth_count,
					sum(t.unt_taken) as spring_lec_units,
					sum(u.unt_taken) as spring_lab_units,
					sum(v.unt_taken) as spring_int_units,
					sum(w.unt_taken) as spring_stu_units,
					sum(x.unt_taken) as spring_sem_units,
					sum(y.unt_taken) as spring_oth_units,
					coalesce(calculated spring_lec_units, 0) + coalesce(calculated spring_lab_units, 0) + coalesce(calculated spring_int_units, 0) 
						+ coalesce(calculated spring_stu_units, 0) + coalesce(calculated spring_sem_units, 0) + coalesce(calculated spring_oth_units, 0) as total_spring_units
				from class_registration_&cohort_year. as a
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LEC' and enrl_status_reason ^= 'WDRW') as b
					on a.emplid = b.emplid
						and a.class_nbr = b.class_nbr
						and a.strm = b.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LAB' and enrl_status_reason ^= 'WDRW') as c
					on a.emplid = c.emplid
						and a.class_nbr = c.class_nbr
						and a.strm = c.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'INT' and enrl_status_reason ^= 'WDRW') as d
					on a.emplid = d.emplid
						and a.class_nbr = d.class_nbr
						and a.strm = d.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'STU' and enrl_status_reason ^= 'WDRW') as e
					on a.emplid = e.emplid
						and a.class_nbr = e.class_nbr
						and a.strm = e.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'SEM' and enrl_status_reason ^= 'WDRW') as f
					on a.emplid = f.emplid
						and a.class_nbr = f.class_nbr
						and a.strm = f.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component not in ('LAB','LEC','INT','STU','SEM') and enrl_status_reason ^= 'WDRW') as g
					on a.emplid = g.emplid
						and a.class_nbr = g.class_nbr
						and a.strm = g.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LEC' and enrl_status_reason ^= 'WDRW') as h
					on a.emplid = h.emplid
						and a.class_nbr = h.class_nbr
						and a.strm = h.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LAB' and enrl_status_reason ^= 'WDRW') as i
					on a.emplid = i.emplid
						and a.class_nbr = i.class_nbr
						and a.strm = i.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'INT' and enrl_status_reason ^= 'WDRW') as j
					on a.emplid = j.emplid
						and a.class_nbr = j.class_nbr
						and a.strm = j.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'STU' and enrl_status_reason ^= 'WDRW') as k
					on a.emplid = k.emplid
						and a.class_nbr = k.class_nbr
						and a.strm = k.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'SEM' and enrl_status_reason ^= 'WDRW') as l
					on a.emplid = l.emplid
						and a.class_nbr = l.class_nbr
						and a.strm = l.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component not in ('LAB','LEC','INT','STU','SEM') and enrl_status_reason ^= 'WDRW') as m
					on a.emplid = m.emplid
						and a.class_nbr = m.class_nbr
						and a.strm = m.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'LEC' and enrl_status_reason ^= 'WDRW') as n
					on a.emplid = n.emplid
						and a.class_nbr = n.class_nbr
						and a.strm = n.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'LAB' and enrl_status_reason ^= 'WDRW') as o
					on a.emplid = o.emplid
						and a.class_nbr = o.class_nbr
						and a.strm = o.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'INT' and enrl_status_reason ^= 'WDRW') as p
					on a.emplid = p.emplid
						and a.class_nbr = p.class_nbr
						and a.strm = p.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'STU' and enrl_status_reason ^= 'WDRW') as q
					on a.emplid = q.emplid
						and a.class_nbr = q.class_nbr
						and a.strm = q.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'SEM' and enrl_status_reason ^= 'WDRW') as r
					on a.emplid = r.emplid
						and a.class_nbr = r.class_nbr
						and a.strm = r.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component not in ('LAB','LEC','INT','STU','SEM') and enrl_status_reason ^= 'WDRW') as s
					on a.emplid = s.emplid
						and a.class_nbr = s.class_nbr
						and a.strm = s.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'LEC' and enrl_status_reason ^= 'WDRW') as t
					on a.emplid = t.emplid
						and a.class_nbr = t.class_nbr
						and a.strm = t.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'LAB' and enrl_status_reason ^= 'WDRW') as u
					on a.emplid = u.emplid
						and a.class_nbr = u.class_nbr
						and a.strm = u.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'INT' and enrl_status_reason ^= 'WDRW') as v
					on a.emplid = v.emplid
						and a.class_nbr = v.class_nbr
						and a.strm = v.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'STU' and enrl_status_reason ^= 'WDRW') as w
					on a.emplid = w.emplid
						and a.class_nbr = w.class_nbr
						and a.strm = w.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'SEM' and enrl_status_reason ^= 'WDRW') as x
					on a.emplid = x.emplid
						and a.class_nbr = x.class_nbr
						and a.strm = x.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component not in ('LAB','LEC','INT','STU','SEM') and enrl_status_reason ^= 'WDRW') as y
					on a.emplid = y.emplid
						and a.class_nbr = y.class_nbr
						and a.strm = y.strm
				group by a.emplid
			;quit;
			
			proc sql;       
				create table class_size_&cohort_year. as
				select distinct
					a.emplid
					,sum(b.total_enrl_hc) as fall_enrl_sum
					,round(avg(b.total_enrl_hc), .01) as fall_enrl_avg
					,sum(c.total_enrl_hc) as spring_enrl_sum
					,round(avg(c.total_enrl_hc), .01) as spring_enrl_avg
				from class_registration_&cohort_year. as a
				left join &dsn..class_vw as b
					on a.class_nbr = b.class_nbr
						and b.snapshot = 'census'
						and b.full_acad_year = "&cohort_year."
						and b.class_acad_career = 'UGRD'
						and substr(b.strm, 4, 1) = '7'
				left join &dsn..class_vw as c
					on a.class_nbr = c.class_nbr
						and c.snapshot = 'census'
						and c.full_acad_year = "&cohort_year."
						and c.class_acad_career = 'UGRD'
						and substr(c.strm, 4, 1) = '3'
				group by a.emplid
			;quit;
			
			proc sql;
				create table class_time_&cohort_year. as
				select distinct
					a.emplid
					,case when min(timepart(b.meeting_time_start)) < '10:00:00't then 1 else 0 end as fall_class_time_early
					,case when max(timepart(b.meeting_time_start)) > '16:00:00't then 1 else 0 end as fall_class_time_late
					,case when min(timepart(c.meeting_time_start)) < '10:00:00't then 1 else 0 end as spring_class_time_early
					,case when max(timepart(c.meeting_time_start)) > '16:00:00't then 1 else 0 end as spring_class_time_late
				from class_registration_&cohort_year. as a
				left join &dsn..class_mtg_pat_d_vw as b
					on a.class_nbr = b.class_nbr
						and b.snapshot = 'census'
						and b.full_acad_year = "&cohort_year."
						and b.class_acad_career = 'UGRD'
						and substr(b.strm, 4, 1) = '7'
				left join &dsn..class_mtg_pat_d_vw as c
					on a.class_nbr = c.class_nbr
						and c.snapshot = 'census'
						and c.full_acad_year = "&cohort_year."
						and c.class_acad_career = 'UGRD'
						and substr(c.strm, 4, 1) = '3'
				group by a.emplid
			;quit;
			
			proc sql;
				create table class_day_&cohort_year. as
				select distinct
					a.emplid
					,case when max(b.sun) = 'Y' then 1 else 0 end as fall_sun_class
					,case when max(b.mon) = 'Y' then 1 else 0 end as fall_mon_class
					,case when max(b.tues) = 'Y' then 1 else 0 end as fall_tues_class
					,case when max(b.wed) = 'Y' then 1 else 0 end as fall_wed_class
					,case when max(b.thurs) = 'Y' then 1 else 0 end as fall_thurs_class
					,case when max(b.fri) = 'Y' then 1 else 0 end as fall_fri_class
					,case when max(b.sat) = 'Y' then 1 else 0 end as fall_sat_class
					,case when max(c.sun) = 'Y' then 1 else 0 end as spring_sun_class
					,case when max(c.mon) = 'Y' then 1 else 0 end as spring_mon_class
					,case when max(c.tues) = 'Y' then 1 else 0 end as spring_tues_class
					,case when max(c.wed) = 'Y' then 1 else 0 end as spring_wed_class
					,case when max(c.thurs) = 'Y' then 1 else 0 end as spring_thurs_class
					,case when max(c.fri) = 'Y' then 1 else 0 end as spring_fri_class
					,case when max(c.sat) = 'Y' then 1 else 0 end as spring_sat_class
				from class_registration_&cohort_year. as a
				left join &dsn..class_mtg_pat_d_vw as b
					on a.class_nbr = b.class_nbr
						and b.snapshot = 'census'
						and b.full_acad_year = "&cohort_year."
						and b.class_acad_career = 'UGRD'
						and substr(b.strm, 4, 1) = '7'
				left join &dsn..class_mtg_pat_d_vw as c
					on a.class_nbr = c.class_nbr
						and c.snapshot = 'census'
						and c.full_acad_year = "&cohort_year."
						and c.class_acad_career = 'UGRD'
						and substr(c.strm, 4, 1) = '3'
				group by a.emplid
			;quit;
			
			proc sql;            
				create table term_contact_hrs_&cohort_year. as
				select distinct
					a.emplid,
					sum(b.lec_contact_hrs) as fall_lec_contact_hrs,
					sum(c.lab_contact_hrs) as fall_lab_contact_hrs,
					sum(d.int_contact_hrs) as fall_int_contact_hrs,
					sum(e.stu_contact_hrs) as fall_stu_contact_hrs,
					sum(f.sem_contact_hrs) as fall_sem_contact_hrs,
					sum(g.oth_contact_hrs) as fall_oth_contact_hrs,
					coalesce(calculated fall_lec_contact_hrs, 0) + coalesce(calculated fall_lab_contact_hrs, 0) + coalesce(calculated fall_int_contact_hrs, 0) 
						+ coalesce(calculated fall_stu_contact_hrs, 0) + coalesce(calculated fall_sem_contact_hrs, 0) + coalesce(calculated fall_oth_contact_hrs, 0) as total_fall_contact_hrs,
					sum(h.lec_contact_hrs) as spring_lec_contact_hrs,
					sum(i.lab_contact_hrs) as spring_lab_contact_hrs,
					sum(j.int_contact_hrs) as spring_int_contact_hrs,
					sum(k.stu_contact_hrs) as spring_stu_contact_hrs,
					sum(l.sem_contact_hrs) as spring_sem_contact_hrs,
					sum(m.oth_contact_hrs) as spring_oth_contact_hrs,
					coalesce(calculated spring_lec_contact_hrs, 0) + coalesce(calculated spring_lab_contact_hrs, 0) + coalesce(calculated spring_int_contact_hrs, 0) 
						+ coalesce(calculated spring_stu_contact_hrs, 0) + coalesce(calculated spring_sem_contact_hrs, 0) + coalesce(calculated spring_oth_contact_hrs, 0) as total_spring_contact_hrs
				from class_registration_&cohort_year. as a
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as lec_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'LEC'
							group by subject_catalog_nbr) as b
					on a.subject_catalog_nbr = b.subject_catalog_nbr
						and a.ssr_component = b.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as lab_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'LAB'
							group by subject_catalog_nbr) as c
					on a.subject_catalog_nbr = c.subject_catalog_nbr
						and a.ssr_component = c.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as int_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'INT'
							group by subject_catalog_nbr) as d
					on a.subject_catalog_nbr = d.subject_catalog_nbr
						and a.ssr_component = d.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as stu_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'STU'
							group by subject_catalog_nbr) as e
					on a.subject_catalog_nbr = e.subject_catalog_nbr
						and a.ssr_component = e.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as sem_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'SEM'
							group by subject_catalog_nbr) as f
					on a.subject_catalog_nbr = f.subject_catalog_nbr
						and a.ssr_component = f.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as oth_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')
							group by subject_catalog_nbr) as g
					on a.subject_catalog_nbr = g.subject_catalog_nbr
						and a.ssr_component = g.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as lec_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '3' 
								and ssr_component = 'LEC'
							group by subject_catalog_nbr) as h
					on a.subject_catalog_nbr = h.subject_catalog_nbr
						and a.ssr_component = h.ssr_component
						and substr(a.strm,4,1) = '3'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as lab_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '3' 
								and ssr_component = 'LAB'
							group by subject_catalog_nbr) as i
					on a.subject_catalog_nbr = i.subject_catalog_nbr
						and a.ssr_component = i.ssr_component
						and substr(a.strm,4,1) = '3'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as int_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '3' 
								and ssr_component = 'INT'
							group by subject_catalog_nbr) as j
					on a.subject_catalog_nbr = j.subject_catalog_nbr
						and a.ssr_component = j.ssr_component
						and substr(a.strm,4,1) = '3'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as stu_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '3' 
								and ssr_component = 'STU'
							group by subject_catalog_nbr) as k
					on a.subject_catalog_nbr = k.subject_catalog_nbr
						and a.ssr_component = k.ssr_component
						and substr(a.strm,4,1) = '3'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as sem_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '3' 
								and ssr_component = 'SEM'
							group by subject_catalog_nbr) as l
					on a.subject_catalog_nbr = l.subject_catalog_nbr
						and a.ssr_component = l.ssr_component
						and substr(a.strm,4,1) = '3'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as oth_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '3' 
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')
							group by subject_catalog_nbr) as m
					on a.subject_catalog_nbr = m.subject_catalog_nbr
						and a.ssr_component = m.ssr_component
						and substr(a.strm,4,1) = '3'
				group by a.emplid
			;quit;
			
			proc sql;
				create table fall_midterm_&cohort_year. as
				select distinct
					strm,
					emplid,
					class_nbr,
					crse_id,
					subject_catalog_nbr,
					case when crse_grade_input = 'A' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'A-' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'B+' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'B'	 and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'B-' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'C+' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'C'	 and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'C-' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'D+' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'D'	 and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'F'	 and enrl_status_reason ^= 'WDRW' 	then unt_taken
																						else .
																						end as unt_taken,
					case when crse_grade_input = 'A' and enrl_status_reason ^= 'WDRW' 	then 4.0
						when crse_grade_input = 'A-' and enrl_status_reason ^= 'WDRW' 	then 3.7
						when crse_grade_input = 'B+' and enrl_status_reason ^= 'WDRW' 	then 3.3
						when crse_grade_input = 'B'	 and enrl_status_reason ^= 'WDRW' 	then 3.0
						when crse_grade_input = 'B-' and enrl_status_reason ^= 'WDRW' 	then 2.7
						when crse_grade_input = 'C+' and enrl_status_reason ^= 'WDRW' 	then 2.3
						when crse_grade_input = 'C'	 and enrl_status_reason ^= 'WDRW' 	then 2.0
						when crse_grade_input = 'C-' and enrl_status_reason ^= 'WDRW' 	then 1.7
						when crse_grade_input = 'D+' and enrl_status_reason ^= 'WDRW' 	then 1.3
						when crse_grade_input = 'D'	 and enrl_status_reason ^= 'WDRW' 	then 1.0
						when crse_grade_input = 'F'	 and enrl_status_reason ^= 'WDRW' 	then 0.0
																						else .
																						end as fall_midterm_grade,
					case when calculated unt_taken is not null and enrl_status_reason ^= 'WDRW'		then 1
																									else 0
																									end as fall_midterm_grade_ind,
					case when crse_grade_input = 'S' and enrl_status_reason ^= 'WDRW'	then 1
																						else 0
																						end as fall_midterm_S_grade_ind,
					case when crse_grade_input = 'X'	then 1
														else 0
														end as fall_midterm_X_grade_ind,
					case when crse_grade_input = 'Z'	then 1
														else 0
														end as fall_midterm_Z_grade_ind,
					case when enrl_status_reason = 'WDRW'	then 1
															else 0
															end as fall_midterm_W_grade_ind
				from &dsn..class_registration_vw
				where snapshot = 'midterm'
					and substr(strm,4,1) = '7'
					and full_acad_year = "&cohort_year."
					and stdnt_enrl_status = 'E'
					and crse_grade_input ^= ''
			;quit;
			
			proc sql;
				create table spring_midterm_&cohort_year. as
				select distinct
					strm,
					emplid,
					class_nbr,
					crse_id,
					subject_catalog_nbr,
					case when crse_grade_input = 'A' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'A-' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'B+' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'B'	 and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'B-' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'C+' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'C'	 and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'C-' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'D+' and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'D'	 and enrl_status_reason ^= 'WDRW' 	then unt_taken
						when crse_grade_input = 'F'	 and enrl_status_reason ^= 'WDRW' 	then unt_taken
																						else .
																						end as unt_taken,
					case when crse_grade_input = 'A' and enrl_status_reason ^= 'WDRW' 	then 4.0
						when crse_grade_input = 'A-' and enrl_status_reason ^= 'WDRW' 	then 3.7
						when crse_grade_input = 'B+' and enrl_status_reason ^= 'WDRW' 	then 3.3
						when crse_grade_input = 'B'	 and enrl_status_reason ^= 'WDRW' 	then 3.0
						when crse_grade_input = 'B-' and enrl_status_reason ^= 'WDRW' 	then 2.7
						when crse_grade_input = 'C+' and enrl_status_reason ^= 'WDRW' 	then 2.3
						when crse_grade_input = 'C'	 and enrl_status_reason ^= 'WDRW' 	then 2.0
						when crse_grade_input = 'C-' and enrl_status_reason ^= 'WDRW' 	then 1.7
						when crse_grade_input = 'D+' and enrl_status_reason ^= 'WDRW' 	then 1.3
						when crse_grade_input = 'D'	 and enrl_status_reason ^= 'WDRW' 	then 1.0
						when crse_grade_input = 'F'	 and enrl_status_reason ^= 'WDRW' 	then 0.0
																						else .
																						end as spring_midterm_grade,
					case when calculated unt_taken is not null and enrl_status_reason ^= 'WDRW'		then 1
																									else 0
																									end as spring_midterm_grade_ind,
					case when crse_grade_input = 'S' and enrl_status_reason ^= 'WDRW'	then 1
																						else 0
																						end as spring_midterm_S_grade_ind,
					case when crse_grade_input = 'X'	then 1
														else 0
														end as spring_midterm_X_grade_ind,
					case when crse_grade_input = 'Z'	then 1
														else 0
														end as spring_midterm_Z_grade_ind,
					case when enrl_status_reason = 'WDRW'	then 1
															else 0
															end as spring_midterm_W_grade_ind
				from &dsn..class_registration_vw
				where snapshot = 'midterm'
					and substr(strm,4,1) = '3'
					and full_acad_year = "&cohort_year."
					and stdnt_enrl_status = 'E'
					and crse_grade_input ^= ''
			;quit;

			proc sql;
				create table midterm_grades_&cohort_year. as
				select distinct
					a.emplid,
					b.fall_midterm_gpa_avg,
					c.fall_midterm_grade_count,
					d.fall_midterm_S_grade_count,
					e.fall_midterm_X_grade_count,
					f.fall_midterm_Z_grade_count,
					g.fall_midterm_W_grade_count,
					h.spring_midterm_gpa_avg,
					i.spring_midterm_grade_count,
					j.spring_midterm_S_grade_count,
					k.spring_midterm_X_grade_count,
					l.spring_midterm_Z_grade_count,
					m.spring_midterm_W_grade_count
				from cohort_&cohort_year. as a
				left join (select distinct emplid, round(sum(fall_midterm_grade * unt_taken) / sum(unt_taken), .01) as fall_midterm_gpa_avg from fall_midterm_&cohort_year. group by emplid) as b
					on a.emplid = b.emplid
				left join (select distinct emplid, sum(fall_midterm_grade_ind) as fall_midterm_grade_count from fall_midterm_&cohort_year. group by emplid) as c 
					on a.emplid = c.emplid
				left join (select distinct emplid, sum(fall_midterm_S_grade_ind) as fall_midterm_S_grade_count from fall_midterm_&cohort_year. group by emplid) as d
					on a.emplid = d.emplid
				left join (select distinct emplid, sum(fall_midterm_X_grade_ind) as fall_midterm_X_grade_count from fall_midterm_&cohort_year. group by emplid) as e
					on a.emplid = e.emplid
				left join (select distinct emplid, sum(fall_midterm_Z_grade_ind) as fall_midterm_Z_grade_count from fall_midterm_&cohort_year. group by emplid) as f
					on a.emplid = f.emplid
				left join (select distinct emplid, sum(fall_midterm_W_grade_ind) as fall_midterm_W_grade_count from fall_midterm_&cohort_year. group by emplid) as g
					on a.emplid = g.emplid
				left join (select distinct emplid, round(sum(spring_midterm_grade * unt_taken) / sum(unt_taken), .01) as spring_midterm_gpa_avg from spring_midterm_&cohort_year. group by emplid) as h
					on a.emplid = h.emplid
				left join (select distinct emplid, sum(spring_midterm_grade_ind) as spring_midterm_grade_count from spring_midterm_&cohort_year. group by emplid) as i
					on a.emplid = i.emplid
				left join (select distinct emplid, sum(spring_midterm_S_grade_ind) as spring_midterm_S_grade_count from spring_midterm_&cohort_year. group by emplid) as j
					on a.emplid = j.emplid
				left join (select distinct emplid, sum(spring_midterm_X_grade_ind) as spring_midterm_X_grade_count from spring_midterm_&cohort_year. group by emplid) as k
					on a.emplid = k.emplid
				left join (select distinct emplid, sum(spring_midterm_Z_grade_ind) as spring_midterm_Z_grade_count from spring_midterm_&cohort_year. group by emplid) as l
					on a.emplid = l.emplid
				left join (select distinct emplid, sum(spring_midterm_W_grade_ind) as spring_midterm_W_grade_count from spring_midterm_&cohort_year. group by emplid) as m
					on a.emplid = m.emplid
			;quit;
			
			proc sql;
				create table exams_detail_&cohort_year. as
				select distinct
					emplid,
					max(sat_sup_rwc) as sat_sup_rwc,
					max(sat_sup_ce) as sat_sup_ce,
					max(sat_sup_ha) as sat_sup_ha,
					max(sat_sup_psda) as sat_sup_psda,
					max(sat_sup_ei) as sat_sup_ei,
					max(sat_sup_pam) as sat_sup_pam,
					max(sat_sup_sec) as sat_sup_sec
				from &dsn..student_test_comp_sat_w
				where snapshot = 'census'
				group by emplid
			;quit;
			
			proc sql;
				create table housing_&cohort_year. as
				select distinct
					emplid,
					camp_addr_indicator,
					housing_reshall_indicator,
					housing_ssa_indicator,
					housing_family_indicator,
					afl_reshall_indicator,
					afl_ssa_indicator,
					afl_family_indicator,
					afl_greek_indicator,
					afl_greek_life_indicator
				from &dsn..new_student_enrolled_housing_vw
				where snapshot = 'census'
					and strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and acad_career = 'UGRD'
					and adj_admit_type_cat in ('FRSH')
			;quit;
			
			proc sql;
				create table housing_detail_&cohort_year. as
				select distinct
					emplid,
					'#' || put(building_id, z2.) as building_id
				from &dsn..student_housing
				where snapshot = 'census'
					and strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
			;quit;
			
			proc sql;
				create table dataset_&cohort_year. as
				select 
					a.*,
					b.pell_recipient_ind,
					coalesce(y.fall_term_gpa, x.fall_term_gpa) as fall_term_gpa,
					coalesce(y.fall_term_gpa_hours, x.fall_term_gpa_hours) as fall_term_gpa_hours,
					y.fall_term_D_grade_count,
					y.fall_term_F_grade_count,
					y.fall_term_W_grade_count,
					y.fall_term_I_grade_count,
					y.fall_term_X_grade_count,
					y.fall_term_U_grade_count,
					y.fall_term_S_grade_count,
					y.fall_term_P_grade_count,
					y.fall_term_Z_grade_count,
					y.fall_term_letter_count,
					y.fall_term_grade_count,
					z.spring_term_gpa,
					z.spring_term_gpa_hours,
					z.spring_term_D_grade_count,
					z.spring_term_F_grade_count,
					z.spring_term_W_grade_count,
					z.spring_term_I_grade_count,
					z.spring_term_X_grade_count,
					z.spring_term_U_grade_count,
					z.spring_term_S_grade_count,
					z.spring_term_P_grade_count,
					z.spring_term_Z_grade_count,
					z.spring_term_letter_count,
					z.spring_term_grade_count,
					aa.cum_gpa,
					aa.cum_gpa_hours,
					c.cont_term,
					c.enrl_ind,
					d.acad_plan,
					d.acad_plan_descr,
					d.plan_owner_org,
					d.plan_owner_org_descr,
					d.plan_owner_group_descrshort,
					d.business,
					d.cahnrs_anml,
					d.cahnrs_envr,
					d.cahnrs_econ,
					d.cahnrext,
					d.cas_chem,
					d.cas_crim,
					d.cas_math,
					d.cas_psyc,
					d.cas_biol,
					d.cas_engl,
					d.cas_phys,
					d.cas,
					d.comm,
					d.education,
					d.medicine,
					d.nursing,
					d.pharmacy,
					d.provost,
					d.vcea_bioe,
					d.vcea_cive,
					d.vcea_desn,
					d.vcea_eecs,
					d.vcea_mech,
					d.vcea,
					d.vet_med,
					d.lsamp_stem_flag,
					d.anywhere_stem_flag,
					e.need_snap,
					e.fed_efc,
					e.fed_need,
					f.aid_snap,
					f.total_disb,
					f.total_offer,
					f.total_accept,
					v.dependent_snap,
					v.num_in_family,
					v.stdnt_have_dependents,
					v.stdnt_have_children_to_support,
					v.stdnt_agi,
					v.stdnt_agi_blank,
					g.best,
					g.bestr,
					g.qvalue,
					g.act_engl,
					g.act_read,
					g.act_math,
					g.sat_erws,
					g.sat_mss,
					g.sat_comp,
					h.ad_dta,
					h.ad_ast,
					h.ad_hsdip,
					h.ad_ged,
					h.ad_ger,
					h.ad_gens,
					i.ap,
					i.rs,
					i.chs,
					i.ib,
					i.aice,
					largest(1, i.ib, i.aice) as IB_AICE,
					j.attendee_alive,
					j.attendee_campus_visit,
					j.attendee_cashe,
					j.attendee_destination,
					j.attendee_experience,
					j.attendee_fcd_pullman,
					j.attendee_fced,
					j.attendee_fcoc,
					j.attendee_fcod,
					j.attendee_group_visit,
					j.attendee_honors_visit,
					j.attendee_imagine_tomorrow,
					j.attendee_imagine_u,
					j.attendee_la_bienvenida,
					j.attendee_lvp_camp,
					j.attendee_oos_destination,
					j.attendee_oos_experience,
					j.attendee_preview,
					j.attendee_preview_jrs,
					j.attendee_shaping,
					j.attendee_top_scholars,
					j.attendee_transfer_day,
					j.attendee_vibes,
					j.attendee_welcome_center,
					j.attendee_any_visitation_ind,
					j.attendee_total_visits,
					k.athlete,
					l.remedial,
					m.min_week_from_term_begin_dt,
					m.max_week_from_term_begin_dt,
					m.count_week_from_term_begin_dt,
					(4.0 - n.fall_avg_difficulty) as fall_avg_difficulty,
					n.fall_avg_pct_withdrawn,
					n.fall_avg_pct_CDFW,
					n.fall_avg_pct_CDF,
					n.fall_avg_pct_DFW,
					n.fall_avg_pct_DF,
					(4.0 - n.spring_avg_difficulty) as spring_avg_difficulty,
					n.spring_avg_pct_withdrawn,
					n.spring_avg_pct_CDFW,
					n.spring_avg_pct_CDF,
					n.spring_avg_pct_DFW,
					n.spring_avg_pct_DF,
					s.fall_lec_count,
					s.fall_lab_count,
					s.fall_int_count,
					s.fall_stu_count,
					s.fall_sem_count,
					s.fall_oth_count,
					s.total_fall_units,
					s.spring_lec_count,
					s.spring_lab_count,
					s.spring_int_count,
					s.spring_stu_count,
					s.spring_sem_count,
					s.spring_oth_count,
					s.total_spring_units,
					w.fall_credit_hours,
					w.spring_credit_hours,
					o.fall_lec_contact_hrs,
					o.fall_lab_contact_hrs,
					o.fall_int_contact_hrs,
					o.fall_stu_contact_hrs,
					o.fall_sem_contact_hrs,
					o.fall_oth_contact_hrs,
					o.total_fall_contact_hrs,
					o.spring_lec_contact_hrs,
					o.spring_lab_contact_hrs,
					o.spring_int_contact_hrs,
					o.spring_stu_contact_hrs,
					o.spring_sem_contact_hrs,
					o.spring_oth_contact_hrs,
					o.total_spring_contact_hrs,
					bb.fall_enrl_sum,
					bb.fall_enrl_avg,
					bb.spring_enrl_sum,
					bb.spring_enrl_avg,
					cc.fall_class_time_early,
					cc.fall_class_time_late,
					cc.spring_class_time_early,
					cc.spring_class_time_late,
					dd.fall_sun_class, 
					dd.fall_mon_class, 
					dd.fall_tues_class, 
					dd.fall_wed_class, 
					dd.fall_thurs_class, 
					dd.fall_fri_class, 
					dd.fall_sat_class, 
					dd.spring_sun_class, 
					dd.spring_mon_class, 
					dd.spring_tues_class, 
					dd.spring_wed_class, 
					dd.spring_thurs_class, 
					dd.spring_fri_class, 
					dd.spring_sat_class, 
					p.sat_sup_rwc,
					p.sat_sup_ce,
					p.sat_sup_ha,
					p.sat_sup_psda,
					p.sat_sup_ei,
					p.sat_sup_pam,
					p.sat_sup_sec,
					q.camp_addr_indicator,
					q.housing_reshall_indicator,
					q.housing_ssa_indicator,
					q.housing_family_indicator,
					q.afl_reshall_indicator,
					q.afl_ssa_indicator,
					q.afl_family_indicator,
					q.afl_greek_indicator,
					q.afl_greek_life_indicator,
					r.building_id,
					t.race_hispanic,
					t.race_american_indian,
					t.race_alaska,
					t.race_asian,
					t.race_black,
					t.race_native_hawaiian,
					t.race_white,
					u.fall_midterm_gpa_avg,
					u.fall_midterm_grade_count,
					u.fall_midterm_S_grade_count,
					u.fall_midterm_W_grade_count,
					u.spring_midterm_gpa_avg,
					u.spring_midterm_grade_count,
					u.spring_midterm_S_grade_count,
					u.spring_midterm_W_grade_count
				from cohort_&cohort_year. as a
				left join pell_&cohort_year. as b
					on a.emplid = b.emplid
				left join eot_term_gpa_&cohort_year. as x
					on a.emplid = x.emplid
				left join enrolled_&cohort_year. as c
					on a.emplid = c.emplid
				left join plan_&cohort_year. as d
					on a.emplid = d.emplid
				left join need_&cohort_year. as e
					on a.emplid = e.emplid
						and a.aid_year = e.aid_year
				left join aid_&cohort_year. as f
					on a.emplid = f.emplid
						and a.aid_year = f.aid_year
				left join exams_&cohort_year. as g
					on a.emplid = g.emplid
				left join degrees_&cohort_year. as h
					on a.emplid = h.emplid
				left join preparatory_&cohort_year. as i
					on a.emplid = i.emplid
				left join visitation_&cohort_year. as j
					on a.emplid = j.emplid
				left join athlete_&cohort_year. as k
					on a.emplid = k.emplid
				left join remedial_&cohort_year. as l
					on a.emplid = l.emplid
				left join date_&cohort_year. as m
					on a.emplid = m.emplid
				left join coursework_difficulty_&cohort_year. as n
					on a.emplid = n.emplid
				left join term_contact_hrs_&cohort_year. as o
					on a.emplid = o.emplid
				left join exams_detail_&cohort_year. as p
					on a.emplid = p.emplid
				left join housing_&cohort_year. as q
					on a.emplid = q.emplid
				left join housing_detail_&cohort_year. as r
					on a.emplid = r.emplid
				left join class_count_&cohort_year. as s
					on a.emplid = s.emplid
				left join race_detail_&cohort_year. as t
					on a.emplid = t.emplid
				left join midterm_grades_&cohort_year. as u
					on a.emplid = u.emplid
				left join dependent_&cohort_year. as v
					on a.emplid = v.emplid
				left join term_credit_hours_&cohort_year. as w
					on a.emplid = w.emplid
				left join eot_fall_term_grades_&cohort_year. as y
					on a.emplid = y.emplid
				left join eot_spring_term_grades_&cohort_year. as z
					on a.emplid = z.emplid
				left join eot_cum_grades_&cohort_year. as aa
					on a.emplid = aa.emplid
				left join class_size_&cohort_year. as bb
					on a.emplid = bb.emplid
				left join class_time_&cohort_year. as cc
					on a.emplid = cc.emplid 
				left join class_day_&cohort_year. as dd
					on a.emplid = dd.emplid           
			;quit;
				
			%end;

			proc sql;
				create table cohort_&cohort_year. as
				select distinct a.*,
					substr(a.last_sch_postal,1,5) as targetid,
					case when a.sex = 'M' then 1 
						else 0
					end as male,
					case when a.age < 18.25 then 'Q1'
						when 18.25 <= a.age < 18.5 then 'Q2'
						when 18.5 <= a.age < 18.75 then 'Q3'
						when 18.75 <= a.age then 'Q4'
						else 'missing'
					end as age_group,
					case when a.father_attended_wsu_flag = 'Y' then 1 
						else 0
					end as father_wsu_flag,
					case when a.mother_attended_wsu_flag = 'Y' then 1 
						else 0
					end as mother_wsu_flag,
					case when a.ipeds_ethnic_group in ('2', '3', '5', '7', 'Z') then 1 
						else 0
					end as underrep_minority,
					case when a.WA_residency = 'RES' then 1
						else 0
					end as resident,
					case when a.adm_parent1_highest_educ_lvl in ('B','C','D','E','F') then '< bach'
						when a.adm_parent1_highest_educ_lvl = 'G' then 'bach'
						when a.adm_parent1_highest_educ_lvl in ('H','I','J','K','L') then '> bach'
							else 'missing'
					end as parent1_highest_educ_lvl,
					case when a.adm_parent2_highest_educ_lvl in ('B','C','D','E','F') then '< bach'
						when a.adm_parent2_highest_educ_lvl = 'G' then 'bach'
						when a.adm_parent2_highest_educ_lvl in ('H','I','J','K','L') then '> bach'
							else 'missing'
					end as parent2_highest_educ_lvl,
					coalesce(b.PULLM_distance_km, b2.VANCO_distance_km, b3.TRICI_distance_km, b4.EVERE_distance_km, b5.SPOKA_distance_km, b6.PULLM_distance_km) as distance,
					c.median_inc,
					c.gini_indx,
					d.pvrt_total/d.pvrt_base as pvrt_rate,
					e.educ_total/e.educ_base as educ_rate,
					f.pop/(g.area*3.861E-7) as pop_dens,
					h.median_value,
					i.race_blk/i.race_tot as pct_blk,
					i.race_ai/i.race_tot as pct_ai,
					i.race_asn/i.race_tot as pct_asn,
					i.race_hawi/i.race_tot as pct_hawi,
					i.race_oth/i.race_tot as pct_oth,
					i.race_two/i.race_tot as pct_two,
					(i.race_blk + i.race_ai + i.race_asn + i.race_hawi + i.race_oth + i.race_two)/i.race_tot as pct_non,
					j.ethnic_hisp/j.ethnic_tot as pct_hisp,
					case when k.locale = '11' then 1 else 0 end as city_large,
					case when k.locale = '12' then 1 else 0 end as city_mid,
					case when k.locale = '13' then 1 else 0 end as city_small,
					case when k.locale = '21' then 1 else 0 end as suburb_large,
					case when k.locale = '22' then 1 else 0 end as suburb_mid,
					case when k.locale = '23' then 1 else 0 end as suburb_small,
					case when k.locale = '31' then 1 else 0 end as town_fringe,
					case when k.locale = '32' then 1 else 0 end as town_distant,
					case when k.locale = '33' then 1 else 0 end as town_remote,
					case when k.locale = '41' then 1 else 0 end as rural_fringe,
					case when k.locale = '42' then 1 else 0 end as rural_distant,
					case when k.locale = '43' then 1 else 0 end as rural_remote
				from &dsn..new_student_enrolled_vw as a
				left join acs.distance_km as b
					on substr(a.last_sch_postal,1,5) = b.inputid
						and a.adj_acad_prog_primary_campus = 'PULLM'
				left join acs.distance_km as b2
					on substr(a.last_sch_postal,1,5) = b2.inputid
						and a.adj_acad_prog_primary_campus = 'VANCO'
				left join acs.distance_km as b3
					on substr(a.last_sch_postal,1,5) = b3.inputid
						and a.adj_acad_prog_primary_campus = 'TRICI'
				left join acs.distance_km as b4
					on substr(a.last_sch_postal,1,5) = b4.inputid
						and a.adj_acad_prog_primary_campus = 'EVERE'
				left join acs.distance_km as b5
					on substr(a.last_sch_postal,1,5) = b5.inputid
						and a.adj_acad_prog_primary_campus = 'SPOKA'
				left join acs.distance_km as b6
					on substr(a.last_sch_postal,1,5) = b6.inputid
						and a.adj_acad_prog_primary_campus = 'ONLIN'
				left join acs.acs_income_%eval(&cohort_year. - &acs_lag.) as c
					on substr(a.last_sch_postal,1,5) = c.geoid
				left join acs.acs_poverty_%eval(&cohort_year. - &acs_lag.) as d
					on substr(a.last_sch_postal,1,5) = d.geoid
				left join acs.acs_education_%eval(&cohort_year. - &acs_lag.) as e
					on substr(a.last_sch_postal,1,5) = e.geoid
				left join acs.acs_demo_%eval(&cohort_year. - &acs_lag.) as f
					on substr(a.last_sch_postal,1,5) = f.geoid
				left join acs.acs_area_%eval(&cohort_year. - &acs_lag.) as g
					on substr(a.last_sch_postal,1,5) = g.geoid
				left join acs.acs_housing_%eval(&cohort_year. - &acs_lag.) as h
					on substr(a.last_sch_postal,1,5) = h.geoid
				left join acs.acs_race_%eval(&cohort_year. - &acs_lag.) as i
					on substr(a.last_sch_postal,1,5) = i.geoid
				left join acs.acs_ethnicity_%eval(&cohort_year. - &acs_lag.) as j
					on substr(a.last_sch_postal,1,5) = j.geoid
				left join acs.edge_locale14_zcta_table as k
					on substr(a.last_sch_postal,1,5) = k.zcta5ce10
				where a.full_acad_year = "&cohort_year"
					and substr(a.strm, 4 , 1) = '7'
					and a.acad_career = 'UGRD'
					and a.adj_admit_type_cat in ('FRSH')
					and a.ipeds_full_part_time = 'F'
					and a.ipeds_ind = 1
					and a.term_credit_hours > 0
					and a.WA_residency ^= 'NON-I'
			;quit;
			
			proc sql;
				create table pell_&cohort_year. as
				select distinct
					emplid,
					pell_recipient_ind
				from &dsn..new_student_profile_ugrd_cs
				where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and adj_admit_type_cat in ('FRSH')
					and ipeds_full_part_time = 'F'
					and WA_residency ^= 'NON-I'
			;quit;
			
			proc sql;
				create table eot_term_gpa_&cohort_year. as
				select distinct
					a.emplid,
					b.term_gpa as fall_term_gpa,
					b.term_gpa_hours as fall_term_gpa_hours,
					b.cum_gpa as fall_cum_gpa,
					b.cum_gpa_hours as fall_cum_gpa_hours,
					c.term_gpa as spring_term_gpa,
					c.term_gpa_hours as spring_term_gpa_hours,
					c.cum_gpa as spring_cum_gpa,
					c.cum_gpa_hours as spring_cum_gpa_hours
				from &dsn..student_enrolled_vw as a
				left join &dsn..student_enrolled_vw as b
					on a.emplid = b.emplid
						and b.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
						and b.snapshot = 'eot'
						and b.ipeds_full_part_time = 'F'
				left join &dsn..student_enrolled_vw as c
					on a.emplid = c.emplid
						and c.strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
						and c.snapshot = 'eot'
						and c.ipeds_full_part_time = 'F'
				where a.snapshot = 'eot'
					and a.full_acad_year = "&cohort_year."
					and a.ipeds_full_part_time = 'F'
			;quit;
			
			proc sql;
				create table race_detail_&cohort_year. as
				select 
					a.emplid,
					case when hispc.emplid is not null 	then 'Y'
														else 'N'
														end as race_hispanic,
					case when amind.emplid is not null 	then 'Y'
														else 'N'
														end as race_american_indian,
					case when alask.emplid is not null 	then 'Y'
														else 'N'
														end as race_alaska,
					case when asian.emplid is not null 	then 'Y'
														else 'N'
														end as race_asian,
					case when black.emplid is not null 	then 'Y'
														else 'N'
														end as race_black,
					case when hawai.emplid is not null 	then 'Y'
														else 'N'
														end as race_native_hawaiian,
					case when white.emplid is not null 	then 'Y'
														else 'N'
														end as race_white
				from cohort_&cohort_year. as a
				left join (select distinct e4.emplid from &dsn..student_ethnic_detail as e4
							left join &dsn..xw_ethnic_detail_to_group_vw as xe4
								on e4.ethnic_cd = xe4.ethnic_cd
							where e4.snapshot = 'census'
								and e4.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe4.ethnic_group = '4') as asian
					on a.emplid = asian.emplid
				left join (select distinct e2.emplid from &dsn..student_ethnic_detail as e2
							left join &dsn..xw_ethnic_detail_to_group_vw as xe2
								on e2.ethnic_cd = xe2.ethnic_cd
							where e2.snapshot = 'census'
								and e2.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe2.ethnic_group = '2') as black
					on a.emplid = black.emplid
				left join (select distinct e7.emplid from &dsn..student_ethnic_detail as e7
							left join &dsn..xw_ethnic_detail_to_group_vw as xe7
								on e7.ethnic_cd = xe7.ethnic_cd
							where e7.snapshot = 'census'
								and e7.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe7.ethnic_group = '7') as hawai
					on a.emplid = hawai.emplid
				left join (select distinct e1.emplid from &dsn..student_ethnic_detail as e1
							left join &dsn..xw_ethnic_detail_to_group_vw as xe1
								on e1.ethnic_cd = xe1.ethnic_cd
							where e1.snapshot = 'census'
								and e1.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe1.ethnic_group = '1') as white
					on a.emplid = white.emplid
				left join (select distinct e5a.emplid from &dsn..student_ethnic_detail as e5a
							left join &dsn..xw_ethnic_detail_to_group_vw as xe5a
								on e5a.ethnic_cd = xe5a.ethnic_cd
							where e5a.snapshot = 'census' 
								and e5a.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe5a.ethnic_group = '5'
								and e5a.ethnic_cd in ('014','016','017','018',
														'935','941','942','943',
														'950','R10','R14')) as alask
					on a.emplid = alask.emplid
				left join (select distinct e5b.emplid from &dsn..student_ethnic_detail as e5b
							left join &dsn..xw_ethnic_detail_to_group_vw as xe5b
								on e5b.ethnic_cd = xe5b.ethnic_cd
							where e5b.snapshot = 'census'
								and e5b.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe5b.ethnic_group = '5'
								and e5b.ethnic_cd not in ('014','016','017','018',
															'935','941','942','943',
															'950','R14')) as amind
					on a.emplid = amind.emplid
				left join (select distinct e6.emplid from &dsn..student_ethnic_detail as e6
							left join &dsn..xw_ethnic_detail_to_group_vw as xe6
								on e6.ethnic_cd = xe6.ethnic_cd
							where e6.snapshot = 'census'
								and e6.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and xe6.ethnic_group = '3') as hispc
					on a.emplid = hispc.emplid
			;quit;
			
			proc sql;
				create table plan_&cohort_year. as 
				select distinct 
					emplid,
					acad_plan,
					acad_plan_descr,
					plan_owner_org,
					plan_owner_org_descr,
					plan_owner_group_descrshort,
					case when plan_owner_group_descrshort = 'Business' then 1 else 0 end as business,
					case when plan_owner_group_descrshort = 'CAHNREXT' 
						and plan_owner_org = '03_1240' then 1 else 0 end as cahnrs_anml,
					case when plan_owner_group_descrshort = 'CAHNREXT' 
						and plan_owner_org = '03_1990' then 1 else 0 end as cahnrs_envr,
					case when plan_owner_group_descrshort = 'CAHNREXT' 
						and plan_owner_org = '03_1150' then 1 else 0 end as cahnrs_econ,	
					case when plan_owner_group_descrshort = 'CAHNREXT'
						and plan_owner_org not in ('03_1240','03_1990','03_1150') then 1 else 0 end as cahnrext,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_1540' then 1 else 0 end as cas_chem,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_1710' then 1 else 0 end as cas_crim,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_2530' then 1 else 0 end as cas_math,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_2900' then 1 else 0 end as cas_psyc,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_8434' then 1 else 0 end as cas_biol,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_1830' then 1 else 0 end as cas_engl,
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org = '31_2790' then 1 else 0 end as cas_phys,	
					case when plan_owner_group_descrshort = 'CAS'
						and plan_owner_org not in ('31_1540','31_1710','31_2530','31_2900','31_8434','31_1830','31_2790') then 1 else 0 end as cas,
					case when plan_owner_group_descrshort = 'Comm' then 1 else 0 end as comm,
					case when plan_owner_group_descrshort = 'Education' then 1 else 0 end as education,
					case when plan_owner_group_descrshort in ('Med Sci','Medicine') then 1 else 0 end as medicine,
					case when plan_owner_group_descrshort = 'Nursing' then 1 else 0 end as nursing,
					case when plan_owner_group_descrshort = 'Pharmacy' then 1 else 0 end as pharmacy,
					case when plan_owner_group_descrshort = 'Provost' then 1 else 0 end as provost,
					case when plan_owner_group_descrshort = 'VCEA' 
						and plan_owner_org = '05_1520' then 1 else 0 end as vcea_bioe,
					case when plan_owner_group_descrshort = 'VCEA' 
						and plan_owner_org = '05_1590' then 1 else 0 end as vcea_cive,
					case when plan_owner_group_descrshort = 'VCEA' 
						and plan_owner_org = '05_1260' then 1 else 0 end as vcea_desn,
					case when plan_owner_group_descrshort = 'VCEA' 
						and plan_owner_org = '05_1770' then 1 else 0 end as vcea_eecs,
					case when plan_owner_group_descrshort = 'VCEA' 
						and plan_owner_org = '05_2540' then 1 else 0 end as vcea_mech,
					case when plan_owner_group_descrshort = 'VCEA' 
						and plan_owner_org not in ('05_1520','05_1590','05_1260','05_1770','05_2540') then 1 else 0 end as vcea,				
					case when plan_owner_group_descrshort = 'Vet Med' then 1 else 0 end as vet_med,
					case when plan_owner_group_descrshort not in ('Business','CAHNREXT','CAS','Comm',
																'Education','Med Sci','Medicine','Nursing',
																'Pharmacy','Provost','VCEA','Vet Med') then 1 else 0
					end as groupless,
					case when plan_owner_percent_owned = 50 and plan_owner_org in ('05_1770','03_1990','12_8595','31_8434') then 1 else 0
					end as split_plan,
					lsamp_stem_flag,
					anywhere_stem_flag
				from &dsn..student_acad_prog_plan_vw
				where snapshot = 'census'
					and full_acad_year = "&cohort_year."
					and substr(strm, 4, 1) = '7'
					and acad_career = 'UGRD'
					and adj_admit_type_cat in ('FRSH')
					and WA_residency ^= 'NON-I'
					and primary_plan_flag = 'Y'
					and primary_prog_flag = 'Y'
					and calculated split_plan = 0
			;quit;
			
			proc sql;
				create table need_&cohort_year. as
				select distinct
					emplid,
					aid_year,
					max(fed_need) as fed_need
				from acs.finaid_data
					where aid_year = "&cohort_year."
				group by emplid, aid_year
			;quit;
			
			proc sql;
				create table aid_&cohort_year. as
				select distinct
					emplid,
					aid_year,
					sum(total_offer) as total_offer,
					sum(total_accept) as total_accept
				from acs.finaid_data
					where aid_year = "&cohort_year."
				group by emplid, aid_year
			;quit;
			
			proc sql;
				create table dependent_&cohort_year. as
				select distinct
					a.emplid,
					b.snapshot as dependent_snap,
					a.num_in_family,
					a.stdnt_have_dependents,
					a.stdnt_have_children_to_support,
					a.stdnt_agi,
					a.stdnt_agi_blank
				from &dsn..fa_isir as a
				inner join (select distinct emplid, aid_year, min(snapshot) as snapshot from &dsn..fa_award_aid_year_vw where aid_year = "&cohort_year.") as b
					on a.emplid = b.emplid
						and a.aid_year = b.aid_year
						and a.snapshot = b.snapshot
				where a.aid_year = "&cohort_year."
			;quit;
			
			proc sql;
				create table exams_&cohort_year. as 
				select distinct
					a.emplid,
					a.best,
					a.bestr,
					a.qvalue,
					a.act_engl,
					a.act_read,
					a.act_math,
					largest(1, a.sat_erws, xw_one.sat_erws, xw_three.sat_erws) as sat_erws,
					largest(1, a.sat_mss, xw_two.sat_mss, xw_four.sat_mss) as sat_mss,
					largest(1, (a.sat_erws + a.sat_mss), (xw_one.sat_erws + xw_two.sat_mss), (xw_three.sat_erws + xw_four.sat_mss)) as sat_comp
				from &dsn..new_freshmen_test_score_vw as a
				left join &dsn..xw_sat_i_to_sat_erws as xw_one
					on (a.sat_i_verb + a.sat_i_wr) = xw_one.sat_i_verb_plus_wr
				left join &dsn..xw_sat_i_to_sat_mss as xw_two
					on a.sat_i_math = xw_two.sat_i_math
				left join act_to_sat_engl_read as xw_three
					on (a.act_engl + a.act_read) = xw_three.act_engl_read
				left join act_to_sat_math as xw_four
					on a.act_math = xw_four.act_math
				where snapshot = 'census'
			;quit;

			proc sql;
				create table degrees_&cohort_year. as
				select distinct
					emplid,
					case when degree in ('AD_AAS_T','AD_AS-T','AD_AS-T1','AD_AS-T2','AD_AS-T2B','AD_AST2C','AD_AST2M') 	then 'AD_AST' 
						when substr(degree,1,6) = 'AD_DTA' 																then 'AD_DTA' 																						
																														else degree end as degree,
					1 as ind
				from &dsn..student_ext_degree
				where floor(degree_term_code / 10) <= &cohort_year.
					and degree in ('AD_AAS_T','AD_AS-T','AD_AS-T1','AD_AS-T2','AD_AS-T2B',
									'AD_AST2C','AD_AST2M','AD_DTA','AD_GED','AD_GENS','AD_GER',
									'AD_HSDIP')
				order by emplid
			;quit;
			
			proc transpose data=degrees_&cohort_year. let out=degrees_&cohort_year. (drop=_name_);
				by emplid;
				id degree;
			run;
			
			proc sql;
				create table preparatory_&cohort_year. as
				select distinct
					emplid,
					ext_subject_area,
					1 as ind
				from &dsn..student_ext_acad_subj
				where snapshot = 'census'
					and ext_subject_area in ('CHS','RS','AP','IB','AICE')
				union
				select distinct
					emplid,
					'RS' as ext_subject_area,
					 1 as ind
				from &dsn..student_acad_prog_plan_vw
				where snapshot = 'census'
					and tuition_group in ('1RS','1TRS')            
				order by emplid
			;quit;
			
			proc transpose data=preparatory_&cohort_year. let out=preparatory_&cohort_year. (drop=_name_);
				by emplid;
				id ext_subject_area;
			run;
			
			proc sql;
				create table visitation_&cohort_year. as
				select distinct a.emplid,
					b.snap_date,
					a.attendee_afr_am_scholars_visit,
					a.attendee_alive,
					a.attendee_campus_visit,
					a.attendee_cashe,
					a.attendee_destination,
					a.attendee_experience,
					a.attendee_fcd_pullman,
					a.attendee_fced,
					a.attendee_fcoc,
					a.attendee_fcod,
					a.attendee_group_visit,
					a.attendee_honors_visit,
					a.attendee_imagine_tomorrow,
					a.attendee_imagine_u,
					a.attendee_la_bienvenida,
					a.attendee_lvp_camp,
					a.attendee_oos_destination,
					a.attendee_oos_experience,
					a.attendee_preview,
					a.attendee_preview_jrs,
					a.attendee_shaping,
					a.attendee_top_scholars,
					a.attendee_transfer_day,
					a.attendee_vibes,
					a.attendee_welcome_center,
					a.attendee_any_visitation_ind,
					a.attendee_total_visits
				from &adm..UGRD_visitation_attendee as a
				inner join (select distinct emplid, max(snap_date) as snap_date 
							from &adm..UGRD_visitation_attendee 
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
							group by emplid) as b
					on a.emplid = b.emplid
						and a.snap_date = b.snap_date
				where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
			;quit;
			
			proc sql;
				create table visitation_detail_&cohort_year. as
				select distinct a.emplid,
					a.snap_date,
					a.go2,
					a.ocv_dt,
					a.ocv_fcd,
					a.ocv_fprv,
					a.ocv_gdt,
					a.ocv_jprv,
					a.ri_col,
					a.ri_fair,
					a.ri_hsv,
					a.ri_nac,
					a.ri_wac,
					a.ri_other,
					a.tap,
					a.tst,
					a.vi_chegg,
					a.vi_crn,
					a.vi_cxc,
					a.vi_mco,
					a.np_group,
					a.out_group,
					a.ref_group,
					a.ocv_da,
					a.ocv_ea,
					a.ocv_fced,
					a.ocv_fcoc,
					a.ocv_fcod,
					a.ocv_oosd,
					a.ocv_oose,
					a.ocv_ve
				from &adm..UGRD_visitation as a
				inner join (select distinct emplid, max(snap_date) as snap_date 
							from &adm..UGRD_visitation 
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
							group by emplid) as b
					on a.emplid = b.emplid
						and a.snap_date = b.snap_date
				where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
			;quit;
					
			proc sql;
				create table athlete_&cohort_year. as
				select distinct 
					emplid,
					case when (mbaseball = 'Y' 
						or mbasketball = 'Y'
						or mfootball = 'Y'
						or mgolf = 'Y'
						or mitrack = 'Y'
						or motrack = 'Y'
						or mxcountry = 'Y'
						or wbasketball = 'Y'
						or wgolf = 'Y'
						or witrack = 'Y'
						or wotrack = 'Y'
						or wsoccer = 'Y'
						or wswimming = 'Y'
						or wtennis = 'Y'
						or wvolleyball = 'Y'
						or wvrowing = 'Y'
						or wxcountry = 'Y') then 1 else 0
					end as athlete
				from &dsn..student_athlete_vw
				where snapshot = 'census'
					and strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and ugrd_adj_admit_type in ('FRS','IFR','IPF','TRN','ITR','IPT')
			;quit;
			
			proc sql;
				create table remedial_&cohort_year. as
				select distinct
					emplid,
					case when grading_basis_enrl in ('REM','RMS','RMP') 	then 1
																			else 0
																			end as remedial
				from &dsn..class_registration_vw
				where snapshot = 'census'
					and aid_year = "&cohort_year."
					and grading_basis_enrl in ('REM','RMS','RMP')
			;quit;
			
			proc sql;
				create table date_&cohort_year. as
				select distinct
					emplid,
					min(week_from_term_begin_dt) as min_week_from_term_begin_dt,
					max(week_from_term_begin_dt) as max_week_from_term_begin_dt,
					count(week_from_term_begin_dt) as count_week_from_term_begin_dt
				from &adm..UGRD_shortened_vw
				where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and ugrd_applicant_counting_ind = 1
				group by emplid
			;quit;
			
			proc sql;
				create table term_credit_hours_&cohort_year. as
				select distinct
					a.emplid,
					coalesce(a.term_credit_hours, 0) as fall_credit_hours,
					coalesce(b.term_credit_hours, 0) as spring_credit_hours
				from &dsn..student_enrolled_vw as a
				left join &dsn..student_enrolled_vw as b
					on a.emplid = b.emplid
						and a.acad_career = b.acad_career
						and b.snapshot = 'census'
						and b.strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
						and b.enrl_ind = 1
						and a.ipeds_full_part_time = 'F'
				where a.snapshot = 'census'
					and a.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and a.enrl_ind = 1
					and a.ipeds_full_part_time = 'F'
			;quit;
			
			%if &term_type. = SPR or &term_type. = SUM %then %do;
				proc sql;
					create table spring_class_registration_&cohort_year. as
					select distinct
						strm,
						emplid,
						class_nbr,
						crse_id,
						unt_taken,
						grading_basis_enrl,
						enrl_status_reason,
						grade_points,
						grd_pts_per_unit,
						strip(subject) || ' ' || strip(catalog_nbr) as subject_catalog_nbr,
						ssr_component,
						crse_grade_input_fin as crse_grade,
						case when crse_grade_input_fin = 'A' 	then 4.0
							when crse_grade_input_fin = 'A-'	then 3.7
							when crse_grade_input_fin = 'B+'	then 3.3
							when crse_grade_input_fin = 'B'		then 3.0
							when crse_grade_input_fin = 'B-'	then 2.7
							when crse_grade_input_fin = 'C+'	then 2.3
							when crse_grade_input_fin = 'C'		then 2.0
							when crse_grade_input_fin = 'C-'	then 1.7
							when crse_grade_input_fin = 'D+'	then 1.3
							when crse_grade_input_fin = 'D'		then 1.0
							when crse_grade_input_fin = 'F'		then 0.0
																else .
																end as class_gpa,
						case when crse_grade_off = 'D' 	then 1
														else 0
														end as D_grade_ind,	
						case when crse_grade_off = 'F' 	then 1
														else 0
														end as F_grade_ind,										
						case when crse_grade_off = 'W' 	then 1
														else 0
														end as W_grade_ind,
						case when crse_grade_off = 'I' 	then 1
														else 0
														end as I_grade_ind,
						case when crse_grade_off = 'X' 	then 1
														else 0
														end as X_grade_ind,
						case when crse_grade_off = 'U' 	then 1
														else 0
														end as U_grade_ind,
						case when crse_grade_off = 'S' 	then 1
														else 0
														end as S_grade_ind,
						case when crse_grade_off = 'P' 	then 1
														else 0
														end as P_grade_ind,
						case when crse_grade_input_fin = 'Z'	then 1
																else 0
																end as Z_grade_ind,
						case when unt_taken is not null and enrl_status_reason ^= 'WDRW'	then 1
																							else 0
																							end as term_grade_ind
					from acs.crse_grade_data
					where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
						and calculated subject_catalog_nbr ^= 'NURS 399'
						and stdnt_enrl_status = 'E'
				;quit;
				
				proc sql;
					create table fall_class_registration_&cohort_year. as
					select distinct
						strm,
						emplid,
						class_nbr,
						crse_id,
						unt_taken,
						grading_basis_enrl,
						enrl_status_reason,
						grade_points,
						grd_pts_per_unit,
						strip(subject) || ' ' || strip(catalog_nbr) as subject_catalog_nbr,
						ssr_component,
						crse_grade_input_fin as crse_grade,
						case when crse_grade_input_fin = 'A' 	then 4.0
							when crse_grade_input_fin = 'A-'	then 3.7
							when crse_grade_input_fin = 'B+'	then 3.3
							when crse_grade_input_fin = 'B'		then 3.0
							when crse_grade_input_fin = 'B-'	then 2.7
							when crse_grade_input_fin = 'C+'	then 2.3
							when crse_grade_input_fin = 'C'		then 2.0
							when crse_grade_input_fin = 'C-'	then 1.7
							when crse_grade_input_fin = 'D+'	then 1.3
							when crse_grade_input_fin = 'D'		then 1.0
							when crse_grade_input_fin = 'F'		then 0.0
																else .
																end as class_gpa,
						case when crse_grade_off = 'D' 	then 1
														else 0
														end as D_grade_ind,	
						case when crse_grade_off = 'F' 	then 1
														else 0
														end as F_grade_ind,	
						case when crse_grade_off = 'W' 	then 1
														else 0
														end as W_grade_ind,
						case when crse_grade_off = 'I' 	then 1
														else 0
														end as I_grade_ind,
						case when crse_grade_off = 'X' 	then 1
														else 0
														end as X_grade_ind,
						case when crse_grade_off = 'U' 	then 1
														else 0
														end as U_grade_ind,
						case when crse_grade_off = 'S' 	then 1
														else 0
														end as S_grade_ind,
						case when crse_grade_off = 'P' 	then 1
														else 0
														end as P_grade_ind,
						case when crse_grade_input_fin = 'Z'	then 1
																else 0
																end as Z_grade_ind,
						case when unt_taken is not null and enrl_status_reason ^= 'WDRW'	then 1
																							else 0
																							end as term_grade_ind
					from acs.crse_grade_data
					where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
						and calculated subject_catalog_nbr ^= 'NURS 399'
						and stdnt_enrl_status = 'E'
				;quit;
				
				data class_registration_&cohort_year.;
					set spring_class_registration_&cohort_year. fall_class_registration_&cohort_year.;
				run;
			%end;
			
			%if &term_type. = FAL %then %do;
				proc sql;
					create table class_registration_&cohort_year. as
					select distinct
						strm,
						emplid,
						class_nbr,
						crse_id,
						unt_taken,
						grading_basis_enrl,
						enrl_status_reason,
						grade_points,
						grd_pts_per_unit,
						strip(subject) || ' ' || strip(catalog_nbr) as subject_catalog_nbr,
						ssr_component,
						crse_grade_input_fin as crse_grade,
						case when crse_grade_input_fin = 'A' 	then 4.0
							when crse_grade_input_fin = 'A-'	then 3.7
							when crse_grade_input_fin = 'B+'	then 3.3
							when crse_grade_input_fin = 'B'		then 3.0
							when crse_grade_input_fin = 'B-'	then 2.7
							when crse_grade_input_fin = 'C+'	then 2.3
							when crse_grade_input_fin = 'C'		then 2.0
							when crse_grade_input_fin = 'C-'	then 1.7
							when crse_grade_input_fin = 'D+'	then 1.3
							when crse_grade_input_fin = 'D'		then 1.0
							when crse_grade_input_fin = 'F'		then 0.0
																else .
																end as class_gpa,
						case when crse_grade_off = 'D' 	then 1
														else 0
														end as D_grade_ind,	
						case when crse_grade_off = 'F' 	then 1
														else 0
														end as F_grade_ind,
						case when crse_grade_off = 'W' 	then 1
														else 0
														end as W_grade_ind,
						case when crse_grade_off = 'I' 	then 1
														else 0
														end as I_grade_ind,
						case when crse_grade_off = 'X' 	then 1
														else 0
														end as X_grade_ind,
						case when crse_grade_off = 'U' 	then 1
														else 0
														end as U_grade_ind,
						case when crse_grade_off = 'S' 	then 1
														else 0
														end as S_grade_ind,
						case when crse_grade_off = 'P' 	then 1
														else 0
														end as P_grade_ind,
						case when crse_grade_input_fin = 'Z'	then 1
																else 0
																end as Z_grade_ind,
						case when unt_taken is not null and enrl_status_reason ^= 'WDRW'	then 1
																							else 0
																							end as term_grade_ind
					from acs.crse_grade_data
					where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
						and calculated subject_catalog_nbr ^= 'NURS 399'
						and stdnt_enrl_status = 'E'
				;quit;
			%end;
			
			proc sql;
				create table eot_fall_term_grades_&cohort_year. as
				select distinct
					a.emplid,
					b.fall_term_gpa_hours,
					b.fall_term_gpa,
					c.fall_term_D_grade_count,
					c.fall_term_F_grade_count,
					c.fall_term_W_grade_count,
					c.fall_term_I_grade_count,
					c.fall_term_X_grade_count,
					c.fall_term_U_grade_count,
					c.fall_term_S_grade_count,
					c.fall_term_P_grade_count,
					c.fall_term_Z_grade_count,
					c.fall_term_letter_count,
					c.fall_term_grade_count
				from class_registration_&cohort_year. as a
				left join (select distinct
								emplid,
								sum(unt_taken) as fall_term_gpa_hours,
								round(sum(class_gpa * unt_taken) / sum(unt_taken), .01) as fall_term_gpa
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and grading_basis_enrl = 'GRD'
								and crse_grade in ('A','A-','B+','B','B-','C+','C','C-','D+','D','F')
							group by emplid) as b
					on a.emplid = b.emplid
				left join (select distinct 
								emplid,
								sum(D_grade_ind) as fall_term_D_grade_count,
								sum(F_grade_ind) as fall_term_F_grade_count,
								sum(W_grade_ind) as fall_term_W_grade_count,
								sum(I_grade_ind) as fall_term_I_grade_count,
								sum(X_grade_ind) as fall_term_X_grade_count,
								sum(U_grade_ind) as fall_term_U_grade_count,
								sum(S_grade_ind) as fall_term_S_grade_count,
								sum(P_grade_ind) as fall_term_P_grade_count,
								sum(Z_grade_ind) as fall_term_Z_grade_count,
								count(class_gpa) as fall_term_letter_count, 
								sum(term_grade_ind) as fall_term_grade_count
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
							group by emplid) as c
					on a.emplid = c.emplid
				where a.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
			;quit;

			proc sql;
				create table eot_spring_term_grades_&cohort_year. as
				select distinct
					a.emplid,
					b.spring_term_gpa_hours,
					b.spring_term_gpa,
					c.spring_term_D_grade_count,
					c.spring_term_F_grade_count,
					c.spring_term_W_grade_count,
					c.spring_term_I_grade_count,
					c.spring_term_X_grade_count,
					c.spring_term_U_grade_count,
					c.spring_term_S_grade_count,
					c.spring_term_P_grade_count,
					c.spring_term_Z_grade_count,
					c.spring_term_letter_count,
					c.spring_term_grade_count
				from class_registration_&cohort_year. as a
				left join (select distinct
								emplid,
								sum(unt_taken) as spring_term_gpa_hours,
								round(sum(class_gpa * unt_taken) / sum(unt_taken), .01) as spring_term_gpa
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and grading_basis_enrl = 'GRD'
								and crse_grade in ('A','A-','B+','B','B-','C+','C','C-','D+','D','F')
								group by emplid) as b
					on a.emplid = b.emplid
				left join (select distinct
								emplid,
								sum(D_grade_ind) as spring_term_D_grade_count,
								sum(F_grade_ind) as spring_term_F_grade_count,
								sum(W_grade_ind) as spring_term_W_grade_count,
								sum(I_grade_ind) as spring_term_I_grade_count,
								sum(X_grade_ind) as spring_term_X_grade_count,
								sum(U_grade_ind) as spring_term_U_grade_count,
								sum(S_grade_ind) as spring_term_S_grade_count,
								sum(P_grade_ind) as spring_term_P_grade_count,
								sum(Z_grade_ind) as spring_term_Z_grade_count,
								count(class_gpa) as spring_term_letter_count,
								sum(term_grade_ind) as spring_term_grade_count
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
							group by emplid) as c
					on a.emplid = c.emplid
				where a.strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
			;quit;
			
			proc sql;
				create table eot_cum_grades_&cohort_year. as
				select distinct
					emplid,
					sum(unt_taken) as cum_gpa_hours,
					round(sum(class_gpa * unt_taken) / sum(unt_taken), .01) as cum_gpa
				from class_registration_&cohort_year.
				where (strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7' 
					or strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3')
					and grading_basis_enrl = 'GRD'
					and crse_grade in ('A','A-','B+','B','B-','C+','C','C-','D+','D','F')
				group by emplid
			;quit;
			
			proc sql;
				create table class_difficulty_&cohort_year. as
				select distinct
					a.subject_catalog_nbr,
					a.ssr_component,
					coalesce(b.total_grade_A, 0) + coalesce(c.total_grade_A, 0) + coalesce(d.total_grade_A, 0)
						+ coalesce(e.total_grade_A, 0) + coalesce(f.total_grade_A, 0) + coalesce(g.total_grade_A, 0) as total_grade_A,
					(calculated total_grade_A * 4.0) as total_grade_A_GPA,
					coalesce(b.total_grade_A_minus, 0) + coalesce(c.total_grade_A_minus, 0) + coalesce(d.total_grade_A_minus, 0)
						+ coalesce(e.total_grade_A_minus, 0) + coalesce(f.total_grade_A_minus, 0) + coalesce(g.total_grade_A_minus, 0) as total_grade_A_minus,
					(calculated total_grade_A_minus * 3.7) as total_grade_A_minus_GPA,
					coalesce(b.total_grade_B_plus, 0) + coalesce(c.total_grade_B_plus, 0) + coalesce(d.total_grade_B_plus, 0)
						+ coalesce(e.total_grade_B_plus, 0) + coalesce(f.total_grade_B_plus, 0) + coalesce(g.total_grade_B_plus, 0) as total_grade_B_plus,
					(calculated total_grade_B_plus * 3.3) as total_grade_B_plus_GPA,
					coalesce(b.total_grade_B, 0) + coalesce(c.total_grade_B, 0) + coalesce(d.total_grade_B, 0)
						+ coalesce(e.total_grade_B, 0) + coalesce(f.total_grade_B, 0) + coalesce(g.total_grade_B, 0) as total_grade_B,
					(calculated total_grade_B * 3.0) as total_grade_B_GPA,
					coalesce(b.total_grade_B_minus, 0) + coalesce(c.total_grade_B_minus, 0) + coalesce(d.total_grade_B_minus, 0) 
						+ coalesce(e.total_grade_B_minus, 0) + coalesce(f.total_grade_B_minus, 0) + coalesce(g.total_grade_B_minus, 0) as total_grade_B_minus,
					(calculated total_grade_B_minus * 2.7) as total_grade_B_minus_GPA,
					coalesce(b.total_grade_C_plus, 0) + coalesce(c.total_grade_C_plus, 0) + coalesce(d.total_grade_C_plus, 0) 
						+ coalesce(e.total_grade_C_plus, 0) + coalesce(f.total_grade_C_plus, 0) + coalesce(g.total_grade_C_plus, 0) as total_grade_C_plus,
					(calculated total_grade_C_plus * 2.3) as total_grade_C_plus_GPA,
					coalesce(b.total_grade_C, 0) + coalesce(c.total_grade_C, 0) + coalesce(d.total_grade_C, 0) 
						+ coalesce(e.total_grade_C, 0) + coalesce(f.total_grade_C, 0) + coalesce(g.total_grade_C, 0) as total_grade_C,
					(calculated total_grade_C * 2.0) as total_grade_C_GPA,
					coalesce(b.total_grade_C_minus, 0) + coalesce(c.total_grade_C_minus, 0) + coalesce(d.total_grade_C_minus, 0)
						+ coalesce(e.total_grade_C_minus, 0) + coalesce(f.total_grade_C_minus, 0) + coalesce(g.total_grade_C_minus, 0) as total_grade_C_minus,
					(calculated total_grade_C_minus * 1.7) as total_grade_C_minus_GPA,
					coalesce(b.total_grade_D_plus, 0) + coalesce(c.total_grade_D_plus, 0) + coalesce(d.total_grade_D_plus, 0)
						+ coalesce(e.total_grade_D_plus, 0) + coalesce(f.total_grade_D_plus, 0) + coalesce(g.total_grade_D_plus, 0) as total_grade_D_plus,
					(calculated total_grade_D_plus * 1.3) as total_grade_D_plus_GPA,
					coalesce(b.total_grade_D, 0) + coalesce(c.total_grade_D, 0) + coalesce(d.total_grade_D, 0) 
						+ coalesce(e.total_grade_D, 0) + coalesce(f.total_grade_D, 0) + coalesce(g.total_grade_D, 0) as total_grade_D,
					(calculated total_grade_D * 1.0) as total_grade_D_GPA,
					coalesce(b.total_grade_F, 0) + coalesce(c.total_grade_F, 0) + coalesce(d.total_grade_F, 0) 
						+ coalesce(e.total_grade_F, 0) + coalesce(f.total_grade_F, 0) + coalesce(g.total_grade_F, 0) as total_grade_F,
					coalesce(b.total_withdrawn, 0) + coalesce(c.total_withdrawn, 0) + coalesce(d.total_withdrawn, 0) 
						+ coalesce(e.total_withdrawn, 0) + coalesce(f.total_withdrawn, 0) + coalesce(g.total_withdrawn, 0) as total_withdrawn,
					coalesce(b.total_dropped, 0) + coalesce(c.total_dropped, 0) + coalesce(d.total_dropped, 0)
						+ coalesce(e.total_dropped, 0) + coalesce(f.total_dropped, 0) + coalesce(g.total_dropped, 0) as total_dropped,
					coalesce(b.total_grade_I, 0) + coalesce(c.total_grade_I, 0) + coalesce(d.total_grade_I, 0)
						+ coalesce(e.total_grade_I, 0) + coalesce(f.total_grade_I, 0) + coalesce(g.total_grade_I, 0) as total_grade_I,
					coalesce(b.total_grade_X, 0) + coalesce(c.total_grade_X, 0) + coalesce(d.total_grade_X, 0)
						+ coalesce(e.total_grade_X, 0) + coalesce(f.total_grade_X, 0) + coalesce(g.total_grade_X, 0) as total_grade_X,
					coalesce(b.total_grade_U, 0) + coalesce(c.total_grade_U, 0) + coalesce(d.total_grade_U, 0)
						+ coalesce(e.total_grade_U, 0) + coalesce(f.total_grade_U, 0) + coalesce(g.total_grade_U, 0) as total_grade_U,
					coalesce(b.total_grade_S, 0) + coalesce(c.total_grade_S, 0) + coalesce(d.total_grade_S, 0)
						+ coalesce(e.total_grade_S, 0) + coalesce(f.total_grade_S, 0) + coalesce(g.total_grade_S, 0) as total_grade_S,
					coalesce(b.total_grade_P, 0) + coalesce(c.total_grade_P, 0) + coalesce(d.total_grade_P, 0)
						+ coalesce(e.total_grade_P, 0) + coalesce(f.total_grade_P, 0) + coalesce(g.total_grade_P, 0) as total_grade_P,
					coalesce(b.total_no_grade, 0) + coalesce(c.total_no_grade, 0) + coalesce(d.total_no_grade, 0)
						+ coalesce(e.total_no_grade, 0) + coalesce(f.total_no_grade, 0) + coalesce(g.total_no_grade, 0) as total_no_grade,
					(calculated total_grade_A + calculated total_grade_A_minus 
						+ calculated total_grade_B_plus + calculated total_grade_B + calculated total_grade_B_minus
						+ calculated total_grade_C_plus + calculated total_grade_C + calculated total_grade_C_minus
						+ calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F) as total_grades,
					(calculated total_grade_A + calculated total_grade_A_minus 
						+ calculated total_grade_B_plus + calculated total_grade_B + calculated total_grade_B_minus
						+ calculated total_grade_C_plus + calculated total_grade_C + calculated total_grade_C_minus
						+ calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F + calculated total_withdrawn) as total_students,
					(calculated total_grade_A_GPA + calculated total_grade_A_minus_GPA 
						+ calculated total_grade_B_plus_GPA + calculated total_grade_B_GPA + calculated total_grade_B_minus_GPA
						+ calculated total_grade_C_plus_GPA + calculated total_grade_C_GPA + calculated total_grade_C_minus_GPA
						+ calculated total_grade_D_plus_GPA + calculated total_grade_D_GPA) as total_grades_GPA,
					(calculated total_grades_GPA / calculated total_grades) as class_average,
					(calculated total_withdrawn / calculated total_students) as pct_withdrawn,
					(calculated total_grade_C_minus + calculated total_grade_D_plus + calculated total_grade_D 
						+ calculated total_grade_F + calculated total_withdrawn) as CDFW,
					(calculated CDFW / calculated total_students) as pct_CDFW,
					(calculated total_grade_C_minus + calculated total_grade_D_plus + calculated total_grade_D 
						+ calculated total_grade_F) as CDF,
					(calculated CDF / calculated total_students) as pct_CDF,
					(calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F 
						+ calculated total_withdrawn) as DFW,
					(calculated DFW / calculated total_students) as pct_DFW,
					(calculated total_grade_D_plus + calculated total_grade_D + calculated total_grade_F) as DF,
					(calculated DF / calculated total_students) as pct_DF
				from &dsn..class_vw as a
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'LEC'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as b
					on a.subject_catalog_nbr = b.subject_catalog_nbr
						and a.ssr_component = b.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'LAB'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as c
					on a.subject_catalog_nbr = c.subject_catalog_nbr
						and a.ssr_component = c.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'INT'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as d
					on a.subject_catalog_nbr = d.subject_catalog_nbr
						and a.ssr_component = d.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'STU'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as e
					on a.subject_catalog_nbr = e.subject_catalog_nbr
						and a.ssr_component = e.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component = 'SEM'
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as f
					on a.subject_catalog_nbr = f.subject_catalog_nbr
						and a.ssr_component = f.ssr_component
				left join (select distinct 
								subject_catalog_nbr,
								ssr_component,
								sum(total_grade_A) as total_grade_A,
								sum(total_grade_A_minus) as total_grade_A_minus,
								sum(total_grade_B_plus) as total_grade_B_plus,
								sum(total_grade_B) as total_grade_B,
								sum(total_grade_B_minus) as total_grade_B_minus,
								sum(total_grade_C_plus) as total_grade_C_plus,
								sum(total_grade_C) as total_grade_C,
								sum(total_grade_C_minus) as total_grade_C_minus,
								sum(total_grade_D_plus) as total_grade_D_plus,
								sum(total_grade_D) as total_grade_D,
								sum(total_grade_F) as total_grade_F,
								sum(total_withdrawn) as total_withdrawn,
								sum(total_dropped) as total_dropped,
								sum(total_grade_I) as total_grade_I,
								sum(total_grade_X) as total_grade_X,
								sum(total_grade_U) as total_grade_U,
								sum(total_grade_S) as total_grade_S,
								sum(total_grade_P) as total_grade_P,
								sum(total_no_grade) as total_no_grade
							from &dsn..class_vw
							where snapshot = 'eot'
								and full_acad_year = put(%eval(&cohort_year. - &lag_year.), 4.)
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')
								and grading_basis = 'GRD'
							group by subject_catalog_nbr) as g
					on a.subject_catalog_nbr = g.subject_catalog_nbr
						and a.ssr_component = g.ssr_component
				where a.snapshot = 'census'
					and a.full_acad_year = "&cohort_year."
					and a.grading_basis = 'GRD'
			;quit;
			
			proc sql;
				create table coursework_difficulty_&cohort_year. as
				select distinct
					a.emplid,
					avg(b.class_average) as fall_avg_difficulty,
					avg(b.pct_withdrawn) as fall_avg_pct_withdrawn,
					avg(b.pct_CDFW) as fall_avg_pct_CDFW,
					avg(b.pct_CDF) as fall_avg_pct_CDF,
					avg(b.pct_DFW) as fall_avg_pct_DFW,
					avg(b.pct_DF) as fall_avg_pct_DF,
					avg(c.class_average) as spring_avg_difficulty,
					avg(c.pct_withdrawn) as spring_avg_pct_withdrawn,
					avg(c.pct_CDFW) as spring_avg_pct_CDFW,
					avg(c.pct_CDF) as spring_avg_pct_CDF,
					avg(c.pct_DFW) as spring_avg_pct_DFW,
					avg(c.pct_DF) as spring_avg_pct_DF
				from class_registration_&cohort_year. as a
				left join class_difficulty_&cohort_year. as b
					on a.subject_catalog_nbr = b.subject_catalog_nbr
						and a.ssr_component = b.ssr_component
						and a.strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
				left join class_difficulty_&cohort_year. as c
					on a.subject_catalog_nbr = c.subject_catalog_nbr
						and a.ssr_component = c.ssr_component
						and a.strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
				where a.enrl_status_reason ^= 'WDRW'
				group by a.emplid
			;quit;
			
			proc sql;
				create table class_count_&cohort_year. as
				select distinct
					a.emplid,
					count(b.class_nbr) as fall_lec_count,
					count(c.class_nbr) as fall_lab_count,
					count(d.class_nbr) as fall_int_count,
					count(e.class_nbr) as fall_stu_count,
					count(f.class_nbr) as fall_sem_count,
					count(g.class_nbr) as fall_oth_count,
					sum(h.unt_taken) as fall_lec_units,
					sum(i.unt_taken) as fall_lab_units,
					sum(j.unt_taken) as fall_int_units,
					sum(k.unt_taken) as fall_stu_units,
					sum(l.unt_taken) as fall_sem_units,
					sum(m.unt_taken) as fall_oth_units,
					coalesce(calculated fall_lec_units, 0) + coalesce(calculated fall_lab_units, 0) + coalesce(calculated fall_int_units, 0) 
						+ coalesce(calculated fall_stu_units, 0) + coalesce(calculated fall_sem_units, 0) + coalesce(calculated fall_oth_units, 0) as total_fall_units,
					count(n.class_nbr) as spring_lec_count,
					count(o.class_nbr) as spring_lab_count,
					count(p.class_nbr) as spring_int_count,
					count(q.class_nbr) as spring_stu_count,
					count(r.class_nbr) as spring_sem_count,
					count(s.class_nbr) as spring_oth_count,
					sum(t.unt_taken) as spring_lec_units,
					sum(u.unt_taken) as spring_lab_units,
					sum(v.unt_taken) as spring_int_units,
					sum(w.unt_taken) as spring_stu_units,
					sum(x.unt_taken) as spring_sem_units,
					sum(y.unt_taken) as spring_oth_units,
					coalesce(calculated spring_lec_units, 0) + coalesce(calculated spring_lab_units, 0) + coalesce(calculated spring_int_units, 0) 
						+ coalesce(calculated spring_stu_units, 0) + coalesce(calculated spring_sem_units, 0) + coalesce(calculated spring_oth_units, 0) as total_spring_units
				from class_registration_&cohort_year. as a
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LEC' and enrl_status_reason ^= 'WDRW') as b
					on a.emplid = b.emplid
						and a.class_nbr = b.class_nbr
						and a.strm = b.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LAB' and enrl_status_reason ^= 'WDRW') as c
					on a.emplid = c.emplid
						and a.class_nbr = c.class_nbr
						and a.strm = c.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'INT' and enrl_status_reason ^= 'WDRW') as d
					on a.emplid = d.emplid
						and a.class_nbr = d.class_nbr
						and a.strm = d.strm
				left join (select distinct emplid, 
								class_nbr,
								strm,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'STU' and enrl_status_reason ^= 'WDRW') as e
					on a.emplid = e.emplid
						and a.class_nbr = e.class_nbr
						and a.strm = e.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'SEM' and enrl_status_reason ^= 'WDRW') as f
					on a.emplid = f.emplid
						and a.class_nbr = f.class_nbr
						and a.strm = f.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component not in ('LAB','LEC','INT','STU','SEM') and enrl_status_reason ^= 'WDRW') as g
					on a.emplid = g.emplid
						and a.class_nbr = g.class_nbr
						and a.strm = g.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LEC' and enrl_status_reason ^= 'WDRW') as h
					on a.emplid = h.emplid
						and a.class_nbr = h.class_nbr
						and a.strm = h.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'LAB' and enrl_status_reason ^= 'WDRW') as i
					on a.emplid = i.emplid
						and a.class_nbr = i.class_nbr
						and a.strm = i.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'INT' and enrl_status_reason ^= 'WDRW') as j
					on a.emplid = j.emplid
						and a.class_nbr = j.class_nbr
						and a.strm = j.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'STU' and enrl_status_reason ^= 'WDRW') as k
					on a.emplid = k.emplid
						and a.class_nbr = k.class_nbr
						and a.strm = k.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component = 'SEM' and enrl_status_reason ^= 'WDRW') as l
					on a.emplid = l.emplid
						and a.class_nbr = l.class_nbr
						and a.strm = l.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
								and ssr_component not in ('LAB','LEC','INT','STU','SEM') and enrl_status_reason ^= 'WDRW') as m
					on a.emplid = m.emplid
						and a.class_nbr = m.class_nbr
						and a.strm = m.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'LEC' and enrl_status_reason ^= 'WDRW') as n
					on a.emplid = n.emplid
						and a.class_nbr = n.class_nbr
						and a.strm = n.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'LAB' and enrl_status_reason ^= 'WDRW') as o
					on a.emplid = o.emplid
						and a.class_nbr = o.class_nbr
						and a.strm = o.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'INT' and enrl_status_reason ^= 'WDRW') as p
					on a.emplid = p.emplid
						and a.class_nbr = p.class_nbr
						and a.strm = p.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'STU' and enrl_status_reason ^= 'WDRW') as q
					on a.emplid = q.emplid
						and a.class_nbr = q.class_nbr
						and a.strm = q.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'SEM' and enrl_status_reason ^= 'WDRW') as r
					on a.emplid = r.emplid
						and a.class_nbr = r.class_nbr
						and a.strm = r.strm
				left join (select distinct emplid, 
								class_nbr,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component not in ('LAB','LEC','INT','STU','SEM') and enrl_status_reason ^= 'WDRW') as s
					on a.emplid = s.emplid
						and a.class_nbr = s.class_nbr
						and a.strm = s.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'LEC' and enrl_status_reason ^= 'WDRW') as t
					on a.emplid = t.emplid
						and a.class_nbr = t.class_nbr
						and a.strm = t.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'LAB' and enrl_status_reason ^= 'WDRW') as u
					on a.emplid = u.emplid
						and a.class_nbr = u.class_nbr
						and a.strm = u.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'INT' and enrl_status_reason ^= 'WDRW') as v
					on a.emplid = v.emplid
						and a.class_nbr = v.class_nbr
						and a.strm = v.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'STU' and enrl_status_reason ^= 'WDRW') as w
					on a.emplid = w.emplid
						and a.class_nbr = w.class_nbr
						and a.strm = w.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component = 'SEM' and enrl_status_reason ^= 'WDRW') as x
					on a.emplid = x.emplid
						and a.class_nbr = x.class_nbr
						and a.strm = x.strm
				left join (select distinct emplid, 
								class_nbr,
								unt_taken,
								strm
							from class_registration_&cohort_year.
							where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
								and ssr_component not in ('LAB','LEC','INT','STU','SEM') and enrl_status_reason ^= 'WDRW') as y
					on a.emplid = y.emplid
						and a.class_nbr = y.class_nbr
						and a.strm = y.strm
				group by a.emplid
            ;quit;

            proc sql;
				create table class_size_&cohort_year. as
				select distinct
					a.emplid
					,sum(b.total_enrl_hc) as fall_enrl_sum
					,round(avg(b.total_enrl_hc), .01) as fall_enrl_avg
					,sum(c.total_enrl_hc) as spring_enrl_sum
					,round(avg(c.total_enrl_hc), .01) as spring_enrl_avg
				from class_registration_&cohort_year. as a
				left join &dsn..class_vw as b
					on a.class_nbr = b.class_nbr
						and b.snapshot = 'census'
						and b.full_acad_year = "&cohort_year."
						and b.class_acad_career = 'UGRD'
						and substr(b.strm, 4, 1) = '7'
				left join &dsn..class_vw as c
					on a.class_nbr = c.class_nbr
						and c.snapshot = 'census'
						and c.full_acad_year = "&cohort_year."
						and c.class_acad_career = 'UGRD'
						and substr(c.strm, 4, 1) = '3'
				group by a.emplid
			;quit;
			
			proc sql;
				create table class_time_&cohort_year. as
				select distinct
					a.emplid
					,case when min(timepart(b.meeting_time_start)) < '10:00:00't then 1 else 0 end as fall_class_time_early
					,case when max(timepart(b.meeting_time_start)) > '16:00:00't then 1 else 0 end as fall_class_time_late
					,case when min(timepart(c.meeting_time_start)) < '10:00:00't then 1 else 0 end as spring_class_time_early
					,case when max(timepart(c.meeting_time_start)) > '16:00:00't then 1 else 0 end as spring_class_time_late
				from class_registration_&cohort_year. as a
				left join &dsn..class_mtg_pat_d_vw as b
					on a.class_nbr = b.class_nbr
						and b.snapshot = 'census'
						and b.full_acad_year = "&cohort_year."
						and b.class_acad_career = 'UGRD'
						and substr(b.strm, 4, 1) = '7'
				left join &dsn..class_mtg_pat_d_vw as c
					on a.class_nbr = c.class_nbr
						and c.snapshot = 'census'
						and c.full_acad_year = "&cohort_year."
						and c.class_acad_career = 'UGRD'
						and substr(c.strm, 4, 1) = '3'
				group by a.emplid
			;quit;

			proc sql;
				create table class_day_&cohort_year. as
				select distinct
					a.emplid
					,case when max(b.sun) = 'Y' then 1 else 0 end as fall_sun_class
					,case when max(b.mon) = 'Y' then 1 else 0 end as fall_mon_class
					,case when max(b.tues) = 'Y' then 1 else 0 end as fall_tues_class
					,case when max(b.wed) = 'Y' then 1 else 0 end as fall_wed_class
					,case when max(b.thurs) = 'Y' then 1 else 0 end as fall_thurs_class
					,case when max(b.fri) = 'Y' then 1 else 0 end as fall_fri_class
					,case when max(b.sat) = 'Y' then 1 else 0 end as fall_sat_class
					,case when max(c.sun) = 'Y' then 1 else 0 end as spring_sun_class
					,case when max(c.mon) = 'Y' then 1 else 0 end as spring_mon_class
					,case when max(c.tues) = 'Y' then 1 else 0 end as spring_tues_class
					,case when max(c.wed) = 'Y' then 1 else 0 end as spring_wed_class
					,case when max(c.thurs) = 'Y' then 1 else 0 end as spring_thurs_class
					,case when max(c.fri) = 'Y' then 1 else 0 end as spring_fri_class
					,case when max(c.sat) = 'Y' then 1 else 0 end as spring_sat_class
				from class_registration_&cohort_year. as a
				left join &dsn..class_mtg_pat_d_vw as b
					on a.class_nbr = b.class_nbr
						and b.snapshot = 'census'
						and b.full_acad_year = "&cohort_year."
						and b.class_acad_career = 'UGRD'
						and substr(b.strm, 4, 1) = '7'
				left join &dsn..class_mtg_pat_d_vw as c
					on a.class_nbr = c.class_nbr
						and c.snapshot = 'census'
						and c.full_acad_year = "&cohort_year."
						and c.class_acad_career = 'UGRD'
						and substr(c.strm, 4, 1) = '3'
				group by a.emplid
			;quit;
			
 			proc sql;           
				create table term_contact_hrs_&cohort_year. as
				select distinct
					a.emplid,
					sum(b.lec_contact_hrs) as fall_lec_contact_hrs,
					sum(c.lab_contact_hrs) as fall_lab_contact_hrs,
					sum(d.int_contact_hrs) as fall_int_contact_hrs,
					sum(e.stu_contact_hrs) as fall_stu_contact_hrs,
					sum(f.sem_contact_hrs) as fall_sem_contact_hrs,
					sum(g.oth_contact_hrs) as fall_oth_contact_hrs,
					coalesce(calculated fall_lec_contact_hrs, 0) + coalesce(calculated fall_lab_contact_hrs, 0) + coalesce(calculated fall_int_contact_hrs, 0) 
						+ coalesce(calculated fall_stu_contact_hrs, 0) + coalesce(calculated fall_sem_contact_hrs, 0) + coalesce(calculated fall_oth_contact_hrs, 0) as total_fall_contact_hrs,
					sum(h.lec_contact_hrs) as spring_lec_contact_hrs,
					sum(i.lab_contact_hrs) as spring_lab_contact_hrs,
					sum(j.int_contact_hrs) as spring_int_contact_hrs,
					sum(k.stu_contact_hrs) as spring_stu_contact_hrs,
					sum(l.sem_contact_hrs) as spring_sem_contact_hrs,
					sum(m.oth_contact_hrs) as spring_oth_contact_hrs,
					coalesce(calculated spring_lec_contact_hrs, 0) + coalesce(calculated spring_lab_contact_hrs, 0) + coalesce(calculated spring_int_contact_hrs, 0) 
						+ coalesce(calculated spring_stu_contact_hrs, 0) + coalesce(calculated spring_sem_contact_hrs, 0) + coalesce(calculated spring_oth_contact_hrs, 0) as total_spring_contact_hrs
				from class_registration_&cohort_year. as a
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as lec_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'LEC'
							group by subject_catalog_nbr) as b
					on a.subject_catalog_nbr = b.subject_catalog_nbr
						and a.ssr_component = b.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as lab_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'LAB'
							group by subject_catalog_nbr) as c
					on a.subject_catalog_nbr = c.subject_catalog_nbr
						and a.ssr_component = c.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as int_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'INT'
							group by subject_catalog_nbr) as d
					on a.subject_catalog_nbr = d.subject_catalog_nbr
						and a.ssr_component = d.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as stu_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'STU'
							group by subject_catalog_nbr) as e
					on a.subject_catalog_nbr = e.subject_catalog_nbr
						and a.ssr_component = e.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as sem_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component = 'SEM'
							group by subject_catalog_nbr) as f
					on a.subject_catalog_nbr = f.subject_catalog_nbr
						and a.ssr_component = f.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as oth_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '7' 
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')
							group by subject_catalog_nbr) as g
					on a.subject_catalog_nbr = g.subject_catalog_nbr
						and a.ssr_component = g.ssr_component
						and substr(a.strm,4,1) = '7'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as lec_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '3' 
								and ssr_component = 'LEC'
							group by subject_catalog_nbr) as h
					on a.subject_catalog_nbr = h.subject_catalog_nbr
						and a.ssr_component = h.ssr_component
						and substr(a.strm,4,1) = '3'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as lab_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '3' 
								and ssr_component = 'LAB'
							group by subject_catalog_nbr) as i
					on a.subject_catalog_nbr = i.subject_catalog_nbr
						and a.ssr_component = i.ssr_component
						and substr(a.strm,4,1) = '3'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as int_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '3' 
								and ssr_component = 'INT'
							group by subject_catalog_nbr) as j
					on a.subject_catalog_nbr = j.subject_catalog_nbr
						and a.ssr_component = j.ssr_component
						and substr(a.strm,4,1) = '3'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as stu_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '3' 
								and ssr_component = 'STU'
							group by subject_catalog_nbr) as k
					on a.subject_catalog_nbr = k.subject_catalog_nbr
						and a.ssr_component = k.ssr_component
						and substr(a.strm,4,1) = '3'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as sem_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '3' 
								and ssr_component = 'SEM'
							group by subject_catalog_nbr) as l
					on a.subject_catalog_nbr = l.subject_catalog_nbr
						and a.ssr_component = l.ssr_component
						and substr(a.strm,4,1) = '3'
				left join (select distinct
								subject_catalog_nbr,
								max(term_contact_hrs) as oth_contact_hrs,
								ssr_component
							from &dsn..class_vw
							where snapshot = "&snapshot."
								and full_acad_year = put(&cohort_year., 4.)
								and substr(strm,4,1) = '3' 
								and ssr_component not in ('LAB','LEC','INT','STU','SEM')
							group by subject_catalog_nbr) as m
					on a.subject_catalog_nbr = m.subject_catalog_nbr
						and a.ssr_component = m.ssr_component
						and substr(a.strm,4,1) = '3'
				group by a.emplid
			;quit;
			
			proc sql;
				create table fall_midterm_&cohort_year. as
				select distinct
					strm,
					emplid,
					class_nbr,
					crse_id,
					strip(subject) || ' ' || strip(catalog_nbr) as subject_catalog_nbr,
					case when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'A' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'A-' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'B+' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'B'	 and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'B-' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'C+' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'C'	 and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'C-' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'D+' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'D'	 and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'F'	 and enrl_status_reason ^= 'WDRW'	then unt_taken
																							else .
																							end as unt_taken,
					case when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'A' and enrl_status_reason ^= 'WDRW'	then 4.0
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'A-' and enrl_status_reason ^= 'WDRW'	then 3.7
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'B+' and enrl_status_reason ^= 'WDRW'	then 3.3
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'B'	 and enrl_status_reason ^= 'WDRW'	then 3.0
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'B-' and enrl_status_reason ^= 'WDRW'	then 2.7
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'C+' and enrl_status_reason ^= 'WDRW'	then 2.3
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'C'	 and enrl_status_reason ^= 'WDRW'	then 2.0
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'C-' and enrl_status_reason ^= 'WDRW'	then 1.7
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'D+' and enrl_status_reason ^= 'WDRW'	then 1.3
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'D'	 and enrl_status_reason ^= 'WDRW'	then 1.0
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'F'	 and enrl_status_reason ^= 'WDRW'	then 0.0
																							else .
																							end as fall_midterm_grade,
					case when calculated unt_taken is not null and enrl_status_reason ^= 'WDRW'		then 1
																									else 0
																									end as fall_midterm_grade_ind,
					case when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'S' and enrl_status_reason ^= 'WDRW'	then 1
																															else 0
																															end as fall_midterm_S_grade_ind,									
					case when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'X'	then 1
																							else 0
																							end as fall_midterm_X_grade_ind,
					case when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'Z'	then 1
																							else 0
																							end as fall_midterm_Z_grade_ind,
					case when enrl_status_reason = 'WDRW'	then 1
															else 0
															end as fall_midterm_W_grade_ind
				from acs.crse_grade_data
				where strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and stdnt_enrl_status = 'E'
			;quit;

			proc sql;
				create table spring_midterm_&cohort_year. as
				select distinct
					strm,
					emplid,
					class_nbr,
					crse_id,
					strip(subject) || ' ' || strip(catalog_nbr) as subject_catalog_nbr,
					case when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'A' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'A-' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'B+' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'B'	 and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'B-' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'C+' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'C'	 and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'C-' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'D+' and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'D'	 and enrl_status_reason ^= 'WDRW'	then unt_taken
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'F'	 and enrl_status_reason ^= 'WDRW'	then unt_taken
																							else .
																							end as unt_taken,
					case when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'A' and enrl_status_reason ^= 'WDRW'	then 4.0
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'A-' and enrl_status_reason ^= 'WDRW'	then 3.7
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'B+' and enrl_status_reason ^= 'WDRW'	then 3.3
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'B'	 and enrl_status_reason ^= 'WDRW'	then 3.0
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'B-' and enrl_status_reason ^= 'WDRW'	then 2.7
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'C+' and enrl_status_reason ^= 'WDRW'	then 2.3
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'C'	 and enrl_status_reason ^= 'WDRW'	then 2.0
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'C-' and enrl_status_reason ^= 'WDRW'	then 1.7
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'D+' and enrl_status_reason ^= 'WDRW'	then 1.3
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'D'	 and enrl_status_reason ^= 'WDRW'	then 1.0
						when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'F'	 and enrl_status_reason ^= 'WDRW'	then 0.0
																							else .
																							end as spring_midterm_grade,
					case when calculated unt_taken is not null and enrl_status_reason ^= 'WDRW'		then 1
																									else 0
																									end as spring_midterm_grade_ind,
					case when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'S' and enrl_status_reason ^= 'WDRW'	then 1
																															else 0
																															end as spring_midterm_S_grade_ind,									
					case when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'X'	then 1
																							else 0
																							end as spring_midterm_X_grade_ind,
					case when coalesce(crse_grade_input_mid, crse_grade_input_fin) = 'Z'	then 1
																							else 0
																							end as spring_midterm_Z_grade_ind,
					case when enrl_status_reason = 'WDRW'	then 1
															else 0
															end as spring_midterm_W_grade_ind
				from acs.crse_grade_data
				where strm = substr(put(&cohort_year., 4.), 1, 1) || substr(put(&cohort_year., 4.), 3, 2) || '3'
					and stdnt_enrl_status = 'E'
			;quit;
			
			proc sql;
				create table midterm_grades_&cohort_year. as
				select distinct
					a.emplid,
					b.fall_midterm_gpa_avg,
					c.fall_midterm_grade_count,
					d.fall_midterm_S_grade_count,
					e.fall_midterm_X_grade_count,
					f.fall_midterm_Z_grade_count,
					g.fall_midterm_W_grade_count,
					h.spring_midterm_gpa_avg,
					i.spring_midterm_grade_count,
					j.spring_midterm_S_grade_count,
					k.spring_midterm_X_grade_count,
					l.spring_midterm_Z_grade_count,
					m.spring_midterm_W_grade_count
				from cohort_&cohort_year. as a
				left join (select distinct emplid, round(sum(fall_midterm_grade * unt_taken) / sum(unt_taken), .01) as fall_midterm_gpa_avg from fall_midterm_&cohort_year. group by emplid) as b
					on a.emplid = b.emplid
				left join (select distinct emplid, sum(fall_midterm_grade_ind) as fall_midterm_grade_count from fall_midterm_&cohort_year. group by emplid) as c 
					on a.emplid = c.emplid
				left join (select distinct emplid, sum(fall_midterm_S_grade_ind) as fall_midterm_S_grade_count from fall_midterm_&cohort_year. group by emplid) as d
					on a.emplid = d.emplid
				left join (select distinct emplid, sum(fall_midterm_X_grade_ind) as fall_midterm_X_grade_count from fall_midterm_&cohort_year. group by emplid) as e
					on a.emplid = e.emplid
				left join (select distinct emplid, sum(fall_midterm_Z_grade_ind) as fall_midterm_Z_grade_count from fall_midterm_&cohort_year. group by emplid) as f
					on a.emplid = f.emplid
				left join (select distinct emplid, sum(fall_midterm_W_grade_ind) as fall_midterm_W_grade_count from fall_midterm_&cohort_year. group by emplid) as g
					on a.emplid = g.emplid
				left join (select distinct emplid, round(sum(spring_midterm_grade * unt_taken) / sum(unt_taken), .01) as spring_midterm_gpa_avg from spring_midterm_&cohort_year. group by emplid) as h
					on a.emplid = h.emplid
				left join (select distinct emplid, sum(spring_midterm_grade_ind) as spring_midterm_grade_count from spring_midterm_&cohort_year. group by emplid) as i
					on a.emplid = i.emplid
				left join (select distinct emplid, sum(spring_midterm_S_grade_ind) as spring_midterm_S_grade_count from spring_midterm_&cohort_year. group by emplid) as j
					on a.emplid = j.emplid
				left join (select distinct emplid, sum(spring_midterm_X_grade_ind) as spring_midterm_X_grade_count from spring_midterm_&cohort_year. group by emplid) as k
					on a.emplid = k.emplid
				left join (select distinct emplid, sum(spring_midterm_Z_grade_ind) as spring_midterm_Z_grade_count from spring_midterm_&cohort_year. group by emplid) as l
					on a.emplid = l.emplid
				left join (select distinct emplid, sum(spring_midterm_W_grade_ind) as spring_midterm_W_grade_count from spring_midterm_&cohort_year. group by emplid) as m
					on a.emplid = m.emplid
			;quit;
			
			proc sql;
				create table exams_detail_&cohort_year. as
				select distinct
					emplid,
					max(sat_sup_rwc) as sat_sup_rwc,
					max(sat_sup_ce) as sat_sup_ce,
					max(sat_sup_ha) as sat_sup_ha,
					max(sat_sup_psda) as sat_sup_psda,
					max(sat_sup_ei) as sat_sup_ei,
					max(sat_sup_pam) as sat_sup_pam,
					max(sat_sup_sec) as sat_sup_sec
				from &dsn..student_test_comp_sat_w
				where snapshot = 'census'
				group by emplid
			;quit;
			
			proc sql;
				create table housing_&cohort_year. as
				select distinct
					emplid,
					camp_addr_indicator,
					housing_reshall_indicator,
					housing_ssa_indicator,
					housing_family_indicator,
					afl_reshall_indicator,
					afl_ssa_indicator,
					afl_family_indicator,
					afl_greek_indicator,
					afl_greek_life_indicator
				from &dsn..new_student_enrolled_housing_vw
				where snapshot = 'census'
					and strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
					and acad_career = 'UGRD'
					and adj_admit_type_cat in ('FRSH')
			;quit;
			
			proc sql;
				create table housing_detail_&cohort_year. as
				select distinct
					emplid,
					'#' || put(building_id, z2.) as building_id
				from &dsn..student_housing
				where snapshot = 'census'
					and strm = substr(put(%eval(&cohort_year. - &lag_year.), 4.), 1, 1) || substr(put(%eval(&cohort_year. - &lag_year.), 4.), 3, 2) || '7'
			;quit;
			
			proc sql;
				create table dataset_&cohort_year. as
				select 
					a.*,
					b.pell_recipient_ind,
					coalesce(y.fall_term_gpa, x.fall_term_gpa) as fall_term_gpa,
					coalesce(y.fall_term_gpa_hours, x.fall_term_gpa_hours) as fall_term_gpa_hours,
					y.fall_term_D_grade_count,
					y.fall_term_F_grade_count,
					y.fall_term_W_grade_count,
					y.fall_term_I_grade_count,
					y.fall_term_X_grade_count,
					y.fall_term_U_grade_count,
					y.fall_term_S_grade_count,
					y.fall_term_P_grade_count,
					y.fall_term_Z_grade_count,
					y.fall_term_letter_count,
					y.fall_term_grade_count,
					z.spring_term_gpa,
					z.spring_term_gpa_hours,
					z.spring_term_D_grade_count,
					z.spring_term_F_grade_count,
					z.spring_term_W_grade_count,
					z.spring_term_I_grade_count,
					z.spring_term_X_grade_count,
					z.spring_term_U_grade_count,
					z.spring_term_S_grade_count,
					z.spring_term_P_grade_count,
					z.spring_term_Z_grade_count,
					z.spring_term_letter_count,
					z.spring_term_grade_count,
					aa.cum_gpa,
					aa.cum_gpa_hours,
					c.acad_plan,
					c.acad_plan_descr,
					c.plan_owner_org,
					c.plan_owner_org_descr,
					c.plan_owner_group_descrshort,
					c.business,
					c.cahnrs_anml,
					c.cahnrs_envr,
					c.cahnrs_econ,
					c.cahnrext,
					c.cas_chem,
					c.cas_crim,
					c.cas_math,
					c.cas_psyc,
					c.cas_biol,
					c.cas_engl,
					c.cas_phys,
					c.cas,
					c.comm,
					c.education,
					c.medicine,
					c.nursing,
					c.pharmacy,
					c.provost,
					c.vcea_bioe,
					c.vcea_cive,
					c.vcea_desn,
					c.vcea_eecs,
					c.vcea_mech,
					c.vcea,
					c.vet_med,
					c.lsamp_stem_flag,
					c.anywhere_stem_flag,
					v.dependent_snap,
					v.num_in_family,
					v.stdnt_have_dependents,
					v.stdnt_have_children_to_support,
					v.stdnt_agi,
					v.stdnt_agi_blank,
					d.fed_need,
					e.total_offer,
					e.total_accept,
					f.best,
					f.bestr,
					f.qvalue,
					f.act_engl,
					f.act_read,
					f.act_math,
					f.sat_erws,
					f.sat_mss,
					f.sat_comp,
					g.ad_dta,
					g.ad_ast,
					g.ad_hsdip,
					g.ad_ged,
					g.ad_ger,
					g.ad_gens,
					h.ap,
					h.rs,
					h.chs,
					h.ib,
					h.aice,
					largest(1, h.ib, h.aice) as IB_AICE,
					i.attendee_alive,
					i.attendee_campus_visit,
					i.attendee_cashe,
					i.attendee_destination,
					i.attendee_experience,
					i.attendee_fcd_pullman,
					i.attendee_fced,
					i.attendee_fcoc,
					i.attendee_fcod,
					i.attendee_group_visit,
					i.attendee_honors_visit,
					i.attendee_imagine_tomorrow,
					i.attendee_imagine_u,
					i.attendee_la_bienvenida,
					i.attendee_lvp_camp,
					i.attendee_oos_destination,
					i.attendee_oos_experience,
					i.attendee_preview,
					i.attendee_preview_jrs,
					i.attendee_shaping,
					i.attendee_top_scholars,
					i.attendee_transfer_day,
					i.attendee_vibes,
					i.attendee_welcome_center,
					i.attendee_any_visitation_ind,
					i.attendee_total_visits,
					j.athlete,
					k.remedial,
					l.min_week_from_term_begin_dt,
					l.max_week_from_term_begin_dt,
					l.count_week_from_term_begin_dt,
					(4.0 - m.fall_avg_difficulty) as fall_avg_difficulty,
					m.fall_avg_pct_withdrawn,
					m.fall_avg_pct_CDFW,
					m.fall_avg_pct_CDF,
					m.fall_avg_pct_DFW,
					m.fall_avg_pct_DF,
					(4.0 - m.spring_avg_difficulty) as spring_avg_difficulty,
					m.spring_avg_pct_withdrawn,
					m.spring_avg_pct_CDFW,
					m.spring_avg_pct_CDF,
					m.spring_avg_pct_DFW,
					m.spring_avg_pct_DF,
					r.fall_lec_count,
					r.fall_lab_count,
					r.fall_int_count,
					r.fall_stu_count,
					r.fall_sem_count,
					r.fall_oth_count,
					r.total_fall_units,
					r.spring_lec_count,
					r.spring_lab_count,
					r.spring_int_count,
					r.spring_stu_count,
					r.spring_sem_count,
					r.spring_oth_count,
					r.total_spring_units,
					w.fall_credit_hours,
					w.spring_credit_hours, 
					n.fall_lec_contact_hrs,
					n.fall_lab_contact_hrs,
					n.fall_int_contact_hrs,
					n.fall_stu_contact_hrs,
					n.fall_sem_contact_hrs,
					n.fall_oth_contact_hrs,
					n.total_fall_contact_hrs,
					n.spring_lec_contact_hrs,
					n.spring_lab_contact_hrs,
					n.spring_int_contact_hrs,
					n.spring_stu_contact_hrs,
					n.spring_sem_contact_hrs,
					n.spring_oth_contact_hrs,
					n.total_spring_contact_hrs,
					o.sat_sup_rwc,
					o.sat_sup_ce,
					o.sat_sup_ha,
					o.sat_sup_psda,
					o.sat_sup_ei,
					o.sat_sup_pam,
					o.sat_sup_sec,
					p.camp_addr_indicator,
					p.housing_reshall_indicator,
					p.housing_ssa_indicator,
					p.housing_family_indicator,
					p.afl_reshall_indicator,
					p.afl_ssa_indicator,
					p.afl_family_indicator,
					p.afl_greek_indicator,
					p.afl_greek_life_indicator,
					q.building_id,
					t.race_hispanic,
					t.race_american_indian,
					t.race_alaska,
					t.race_asian,
					t.race_black,
					t.race_native_hawaiian,
					t.race_white,
					u.fall_midterm_gpa_avg,
					u.fall_midterm_grade_count,
					u.fall_midterm_S_grade_count,
					u.fall_midterm_W_grade_count,
					u.spring_midterm_gpa_avg,
					u.spring_midterm_grade_count,
					u.spring_midterm_S_grade_count,
					u.spring_midterm_W_grade_count
				from cohort_&cohort_year. as a
				left join pell_&cohort_year. as b
					on a.emplid = b.emplid
				left join eot_term_gpa_&cohort_year. as x
					on a.emplid = x.emplid
				left join plan_&cohort_year. as c
					on a.emplid = c.emplid
				left join need_&cohort_year. as d
					on a.emplid = d.emplid
						and d.aid_year = "&cohort_year."
				left join aid_&cohort_year. as e
					on a.emplid = e.emplid
						and e.aid_year = "&cohort_year."
				left join exams_&cohort_year. as f
					on a.emplid = f.emplid
				left join degrees_&cohort_year. as g
					on a.emplid = g.emplid
				left join preparatory_&cohort_year. as h
					on a.emplid = h.emplid
				left join visitation_&cohort_year. as i
					on a.emplid = i.emplid
				left join athlete_&cohort_year. as j
					on a.emplid = j.emplid
				left join remedial_&cohort_year. as k
					on a.emplid = k.emplid
				left join date_&cohort_year. as l
					on a.emplid = l.emplid
				left join coursework_difficulty_&cohort_year. as m
					on a.emplid = m.emplid
				left join term_contact_hrs_&cohort_year. as n
					on a.emplid = n.emplid
				left join exams_detail_&cohort_year. as o
					on a.emplid = o.emplid
				left join housing_&cohort_year. as p
					on a.emplid = p.emplid
				left join housing_detail_&cohort_year. as q
					on a.emplid = q.emplid
				left join class_count_&cohort_year. as r
					on a.emplid = r.emplid
				left join race_detail_&cohort_year. as t
					on a.emplid = t.emplid
				left join midterm_grades_&cohort_year. as u
					on a.emplid = u.emplid
				left join dependent_&cohort_year. as v
					on a.emplid = v.emplid
				left join term_credit_hours_&cohort_year. as w
					on a.emplid = w.emplid
				left join eot_fall_term_grades_&cohort_year. as y
					on a.emplid = y.emplid
				left join eot_spring_term_grades_&cohort_year. as z
					on a.emplid = z.emplid
				left join eot_cum_grades_&cohort_year. as aa
					on a.emplid = aa.emplid
			;quit;
			
		%mend loop;
		""")

		print('Done\n')

		# Run SAS macro program to prepare data from census
		print('Run SAS macro program...')
		start = time.perf_counter()

		sas_log = sas.submit("""
		%loop;
		""")

		HTML(sas_log['LOG'])

		stop = time.perf_counter()
		print(f'Done in {(stop - start)/60:.1f} minutes\n')

		# Prepare data
		print('Prepare data...')

		sas.submit("""
		data training_set;
			set dataset_&start_cohort.-dataset_%eval(&end_cohort. - (3 * &lag_year.));
			if enrl_ind = . then enrl_ind = 0;
			if distance = . then acs_mi = 1; else acs_mi = 0;
			if distance = . then distance = 0;
			if pop_dens = . then pop_dens = 0;
			if educ_rate = . then educ_rate = 0;	
			if pct_blk = . then pct_blk = 0;	
			if pct_ai = . then pct_ai = 0;	
			if pct_asn = .	then pct_asn = 0;
			if pct_hawi = . then pct_hawi = 0;
			if pct_two = . then pct_two = 0;
			if pct_hisp = . then pct_hisp = 0;
			if pct_oth = . then pct_oth = 0;
			if pct_non = . then pct_non = 0;
			if median_inc = . then median_inc = 0;
			if median_value = . then median_value = 0;
			if gini_indx = . then gini_indx = 0;
			if pvrt_rate = . then pvrt_rate = 0;
			if educ_rate = . then educ_rate = 0;
			if city_large = . then city_large = 0;
			if city_mid = . then city_mid = 0;
			if city_small = . then city_small = 0;
			if suburb_large = . then suburb_large = 0;
			if suburb_mid = . then suburb_mid = 0;
			if suburb_small = . then suburb_small = 0;
			if town_fringe = . then town_fringe = 0;
			if town_distant = . then town_distant = 0;
			if town_remote = . then town_remote = 0;
			if rural_fringe = . then rural_fringe = 0;
			if rural_distant = . then rural_distant = 0;
			if rural_remote = . then rural_remote = 0;
			if ad_dta = . then ad_dta = 0;
			if ad_ast = . then ad_ast = 0;
			if ad_hsdip = . then ad_hsdip = 0;
			if ad_ged = . then ad_ged = 0;
			if ad_ger = . then ad_ger = 0;
			if ad_gens = . then ad_gens = 0;
			if ap = . then ap = 0;
			if rs = . then rs = 0;
			if chs = . then chs = 0;
			if ib = . then ib = 0;
			if aice = . then aice = 0;
			if ib_aice = . then ib_aice = 0;
			if athlete = . then athlete = 0;
			if remedial = . then remedial = 0;
			if sat_mss = . then sat_mss = 0;
			if sat_erws = . then sat_erws = 0;
			if high_school_gpa = . then high_school_gpa_mi = 1; else high_school_gpa_mi = 0;
			if high_school_gpa = . then high_school_gpa = 0;
			if transfer_gpa = . then transfer_gpa_mi = 1; else transfer_gpa_mi = 0;
			if transfer_gpa = . then transfer_gpa = 0;
			if last_sch_proprietorship = '' then last_sch_proprietorship = 'UNKN';
			if ipeds_ethnic_group_descrshort = '' then ipeds_ethnic_group_descrshort = 'NS';
			if fall_avg_pct_withdrawn = . then fall_avg_pct_withdrawn = 0;
			if fall_avg_pct_CDFW = . then fall_avg_pct_CDFW = 0;
			if fall_avg_pct_CDF = . then fall_avg_pct_CDF = 0;
			if fall_avg_pct_DFW = . then fall_avg_pct_DFW = 0;
			if fall_avg_pct_DF = . then fall_avg_pct_DF = 0;
			if fall_avg_difficulty = . then fall_crse_mi = 1; else fall_crse_mi = 0; 
			if fall_avg_difficulty = . then fall_avg_difficulty = 0;
			if fall_lec_contact_hrs = . then fall_lec_contact_hrs = 0;
			if fall_lab_contact_hrs = . then fall_lab_contact_hrs = 0;
			if fall_int_contact_hrs = . then fall_int_contact_hrs = 0;
			if fall_stu_contact_hrs = . then fall_stu_contact_hrs = 0;
			if fall_sem_contact_hrs = . then fall_sem_contact_hrs = 0;
			if fall_oth_contact_hrs = . then fall_oth_contact_hrs = 0;
			if total_fall_contact_hrs = . then total_fall_contact_hrs = 0;
			if fall_avg_pct_CDFW = . then fall_avg_pct_CDFW = 0;
			if fall_avg_pct_CDF = . then fall_avg_pct_CDF = 0;
			if fall_avg_pct_DFW = . then fall_avg_pct_DFW = 0;
			if fall_avg_pct_DF = . then fall_avg_pct_DF = 0;
			if fall_avg_difficulty = . then fall_crse_mi = 1; else fall_crse_mi = 0; 
			if fall_avg_difficulty = . then fall_avg_difficulty = 0;
			if spring_avg_pct_withdrawn = . then spring_avg_pct_withdrawn = 0;
			if spring_avg_pct_CDFW = . then spring_avg_pct_CDFW = 0;
			if spring_avg_pct_CDF = . then spring_avg_pct_CDF = 0;
			if spring_avg_pct_DFW = . then spring_avg_pct_DFW = 0;
			if spring_avg_pct_DF = . then spring_avg_pct_DF = 0;
			if spring_avg_difficulty = . then spring_crse_mi = 1; else spring_crse_mi = 0; 
			if spring_avg_difficulty = . then spring_avg_difficulty = 0;
			if spring_lec_count = . then spring_lec_count = 0;
			if spring_lab_count = . then spring_lab_count = 0;
			if spring_int_count = . then spring_int_count = 0;
			if spring_stu_count = . then spring_stu_count = 0;
			if spring_sem_count = . then spring_sem_count = 0;
			if spring_oth_count = . then spring_oth_count = 0;
			if spring_lec_contact_hrs = . then spring_lec_contact_hrs = 0;
			if spring_lab_contact_hrs = . then spring_lab_contact_hrs = 0;
			if spring_int_contact_hrs = . then spring_int_contact_hrs = 0;
			if spring_stu_contact_hrs = . then spring_stu_contact_hrs = 0;
			if spring_sem_contact_hrs = . then spring_sem_contact_hrs = 0;
			if spring_oth_contact_hrs = . then spring_oth_contact_hrs = 0;
			if total_spring_contact_hrs = . then total_spring_contact_hrs = 0;
			if fall_enrl_sum = . then fall_enrl_sum_mi = 1; else fall_enrl_sum_mi = 0;
			if fall_enrl_avg = . then fall_enrl_avg_mi = 1; else fall_enrl_avg_mi = 0;
			if spring_enrl_sum = . then spring_enrl_sum_mi = 1; else spring_enrl_sum_mi = 0;
			if spring_enrl_avg = . then spring_enrl_avg_mi = 1; else spring_enrl_avg_mi = 0;
			if fall_enrl_sum = . then fall_enrl_sum = 0;
			if fall_enrl_avg = . then fall_enrl_avg = 0;
			if spring_enrl_sum = . then spring_enrl_sum = 0;
			if spring_enrl_avg = . then spring_enrl_avg = 0;
			if fall_class_time_early = . then fall_class_time_early_mi = 1; else fall_class_time_early_mi = 0;
			if fall_class_time_late = . then fall_class_time_late_mi = 1; else fall_class_time_late_mi = 0;
			if spring_class_time_early = . then spring_class_time_early_mi = 1; else spring_class_time_early_mi = 0;
			if spring_class_time_late = . then spring_class_time_late_mi = 1; else spring_class_time_late_mi = 0;
			if fall_class_time_early = . then fall_class_time_early = 0;
			if fall_class_time_late = . then fall_class_time_late = 0;
			if spring_class_time_early = . then spring_class_time_early = 0;
			if spring_class_time_late = . then spring_class_time_late = 0;
			if fall_sun_class = . then fall_sun_class = 0;
			if fall_mon_class = . then fall_mon_class = 0;
			if fall_tues_class = . then fall_tues_class = 0;
			if fall_wed_class = . then fall_wed_class = 0;
			if fall_thurs_class = . then fall_thurs_class = 0;
			if fall_fri_class = . then fall_fri_class = 0;
			if fall_sat_class = . then fall_sat_class = 0;
			if spring_sun_class = . then spring_sun_class = 0;
			if spring_mon_class = . then spring_mon_class = 0;
			if spring_tues_class = . then spring_tues_class = 0;
			if spring_wed_class = . then spring_wed_class = 0;
			if spring_thurs_class = . then spring_thurs_class = 0;
			if spring_fri_class = . then spring_fri_class = 0;
			if spring_sat_class = . then spring_sat_class = 0;
			if total_fall_units = . then total_fall_units = 0;
			if total_spring_units = . then total_spring_units = 0;
			if fall_credit_hours = . then fall_credit_hours = 0;
			if spring_credit_hours = . then spring_credit_hours = 0;
			if fall_midterm_gpa_avg = . then fall_midterm_gpa_avg_mi = 1; else fall_midterm_gpa_avg_mi = 0;
			if fall_midterm_gpa_avg = . then fall_midterm_gpa_avg = 0;
			if fall_midterm_grade_count = . then fall_midterm_grade_count = 0;
			if fall_midterm_S_grade_count = . then fall_midterm_S_grade_count = 0;
			if fall_midterm_W_grade_count = . then fall_midterm_W_grade_count = 0;
			if spring_midterm_gpa_avg = . then spring_midterm_gpa_avg_mi = 1; else spring_midterm_gpa_avg_mi = 0;
			if spring_midterm_gpa_avg = . then spring_midterm_gpa_avg = 0;
			if spring_midterm_grade_count = . then spring_midterm_grade_count = 0;
			if spring_midterm_S_grade_count = . then spring_midterm_S_grade_count = 0;
			if spring_midterm_W_grade_count = . then spring_midterm_W_grade_count = 0;
			if fall_term_gpa = . then fall_term_gpa_mi = 1; else fall_term_gpa_mi = 0;
			if fall_term_gpa = . then fall_term_gpa = 0;
			if spring_term_gpa = . then spring_term_gpa_mi = 1; else spring_term_gpa_mi = 0;
			if spring_term_gpa = . then spring_term_gpa = 0;
			if fall_term_D_grade_count = . then fall_term_D_grade_count_mi = 1; else fall_term_D_grade_count_mi = 0;
			if fall_term_D_grade_count = . then fall_term_D_grade_count = 0;
			if fall_term_F_grade_count = . then fall_term_F_grade_count_mi = 1; else fall_term_F_grade_count_mi = 0;
			if fall_term_F_grade_count = . then fall_term_F_grade_count = 0;
			if fall_term_W_grade_count = . then fall_term_W_grade_count_mi = 1; else fall_term_W_grade_count_mi = 0;
			if fall_term_W_grade_count = . then fall_term_W_grade_count = 0;
			if fall_term_I_grade_count = . then fall_term_I_grade_count_mi = 1; else fall_term_I_grade_count_mi = 0;
			if fall_term_I_grade_count = . then fall_term_I_grade_count = 0;
			if fall_term_X_grade_count = . then fall_term_X_grade_count_mi = 1; else fall_term_X_grade_count_mi = 0;
			if fall_term_X_grade_count = . then fall_term_X_grade_count = 0;
			if fall_term_U_grade_count = . then fall_term_U_grade_count_mi = 1; else fall_term_U_grade_count_mi = 0;
			if fall_term_U_grade_count = . then fall_term_U_grade_count = 0;
			if fall_term_S_grade_count = . then fall_term_S_grade_count_mi = 1; else fall_term_S_grade_count_mi = 0;
			if fall_term_S_grade_count = . then fall_term_S_grade_count = 0;
			if fall_term_P_grade_count = . then fall_term_P_grade_count_mi = 1; else fall_term_P_grade_count_mi = 0;
			if fall_term_P_grade_count = . then fall_term_P_grade_count = 0;
			if fall_term_Z_grade_count = . then fall_term_Z_grade_count_mi = 1; else fall_term_Z_grade_count_mi = 0;
			if fall_term_Z_grade_count = . then fall_term_Z_grade_count = 0;
			if fall_term_letter_count = . then fall_term_letter_count_mi = 1; else fall_term_letter_count_mi = 0;
			if fall_term_letter_count = . then fall_term_letter_count = 0;
			if fall_term_grade_count = . then fall_term_grade_count_mi = 1; else fall_term_grade_count_mi = 0;
			if fall_term_grade_count = . then fall_term_grade_count = 0;
			fall_term_no_letter_count = fall_term_grade_count - fall_term_letter_count;
			if spring_term_D_grade_count = . then spring_term_D_grade_count_mi = 1; else spring_term_D_grade_count_mi = 0;
			if spring_term_D_grade_count = . then spring_term_D_grade_count = 0;
			if spring_term_F_grade_count = . then spring_term_F_grade_count_mi = 1; else spring_term_F_grade_count_mi = 0;
			if spring_term_F_grade_count = . then spring_term_F_grade_count = 0;
			if spring_term_W_grade_count = . then spring_term_W_grade_count_mi = 1; else spring_term_W_grade_count_mi = 0;
			if spring_term_W_grade_count = . then spring_term_W_grade_count = 0;
			if spring_term_I_grade_count = . then spring_term_I_grade_count_mi = 1; else spring_term_I_grade_count_mi = 0;
			if spring_term_I_grade_count = . then spring_term_I_grade_count = 0;
			if spring_term_X_grade_count = . then spring_term_X_grade_count_mi = 1; else spring_term_X_grade_count_mi = 0;
			if spring_term_X_grade_count = . then spring_term_X_grade_count = 0;
			if spring_term_U_grade_count = . then spring_term_U_grade_count_mi = 1; else spring_term_U_grade_count_mi = 0;
			if spring_term_U_grade_count = . then spring_term_U_grade_count = 0;
			if spring_term_S_grade_count = . then spring_term_S_grade_count_mi = 1; else spring_term_S_grade_count_mi = 0;
			if spring_term_S_grade_count = . then spring_term_S_grade_count = 0;
			if spring_term_P_grade_count = . then spring_term_P_grade_count_mi = 1; else spring_term_P_grade_count_mi = 0;
			if spring_term_P_grade_count = . then spring_term_P_grade_count = 0;
			if spring_term_Z_grade_count = . then spring_term_Z_grade_count_mi = 1; else spring_term_Z_grade_count_mi = 0;
			if spring_term_Z_grade_count = . then spring_term_Z_grade_count = 0;
			if spring_term_letter_count = . then spring_term_leter_count_mi = 1; else spring_term_leter_count_mi = 0;
			if spring_term_letter_count = . then spring_term_letter_count = 0;
			if spring_term_grade_count = . then spring_term_grade_count_mi = 1; else spring_term_grade_count_mi = 0;
			if spring_term_grade_count = . then spring_term_grade_count = 0;
			spring_term_no_letter_count = spring_term_grade_count - spring_term_letter_count;
			if first_gen_flag = '' then first_gen_flag_mi = 1; else first_gen_flag_mi = 0;
			if first_gen_flag = '' then first_gen_flag = 'N';
			if camp_addr_indicator ^= 'Y' then camp_addr_indicator = 'N';
			if housing_reshall_indicator ^= 'Y' then housing_reshall_indicator = 'N';
			if housing_ssa_indicator ^= 'Y' then housing_ssa_indicator = 'N';
			if housing_family_indicator ^= 'Y' then housing_family_indicator = 'N';
			if afl_reshall_indicator ^= 'Y' then afl_reshall_indicator = 'N';
			if afl_ssa_indicator ^= 'Y' then afl_ssa_indicator = 'N';
			if afl_family_indicator ^= 'Y' then afl_family_indicator = 'N';
			if afl_greek_indicator ^= 'Y' then afl_greek_indicator = 'N';
			if afl_greek_life_indicator ^= 'Y' then afl_greek_life_indicator = 'N';
			fall_withdrawn_hours = (total_fall_units - fall_credit_hours) * -1;
			if total_fall_units = 0 then fall_withdrawn_ind = 1; else fall_withdrawn_ind = 0;
			spring_withdrawn_hours = (total_spring_units - spring_credit_hours) * -1;
			if total_spring_units = 0 then spring_withdrawn = 1; else spring_withdrawn = 0;
			spring_midterm_gpa_change = spring_midterm_gpa_avg - fall_cum_gpa;
			unmet_need_disb = fed_need - total_disb;
			unmet_need_acpt = fed_need - total_accept;
			if unmet_need_acpt = . then unmet_need_acpt_mi = 1; else unmet_need_acpt_mi = 0;
			if unmet_need_acpt < 0 then unmet_need_acpt = 0;
			unmet_need_ofr = fed_need - total_offer;
			if unmet_need_ofr = . then unmet_need_ofr_mi = 1; else unmet_need_ofr_mi = 0;
			if unmet_need_ofr < 0 then unmet_need_ofr = 0;
			if fed_efc = . then fed_efc = 0;
			if fed_need = . then fed_need = 0;
			if total_disb = . then total_disb = 0;
			if total_offer = . then total_offer = 0;
			if total_accept = . then total_accept = 0;
		run;

		data validation_set;
			set dataset_%eval(&end_cohort. - (2 * &lag_year.))-dataset_%eval(&end_cohort. - &lag_year.);
			if enrl_ind = . then enrl_ind = 0;
			if distance = . then acs_mi = 1; else acs_mi = 0;
			if distance = . then distance = 0;
			if pop_dens = . then pop_dens = 0;
			if educ_rate = . then educ_rate = 0;	
			if pct_blk = . then pct_blk = 0;	
			if pct_ai = . then pct_ai = 0;	
			if pct_asn = .	then pct_asn = 0;
			if pct_hawi = . then pct_hawi = 0;
			if pct_two = . then pct_two = 0;
			if pct_hisp = . then pct_hisp = 0;
			if pct_oth = . then pct_oth = 0;
			if pct_non = . then pct_non = 0;
			if median_inc = . then median_inc = 0;
			if median_value = . then median_value = 0;
			if gini_indx = . then gini_indx = 0;
			if pvrt_rate = . then pvrt_rate = 0;
			if educ_rate = . then educ_rate = 0;
			if city_large = . then city_large = 0;
			if city_mid = . then city_mid = 0;
			if city_small = . then city_small = 0;
			if suburb_large = . then suburb_large = 0;
			if suburb_mid = . then suburb_mid = 0;
			if suburb_small = . then suburb_small = 0;
			if town_fringe = . then town_fringe = 0;
			if town_distant = . then town_distant = 0;
			if town_remote = . then town_remote = 0;
			if rural_fringe = . then rural_fringe = 0;
			if rural_distant = . then rural_distant = 0;
			if rural_remote = . then rural_remote = 0;
			if ad_dta = . then ad_dta = 0;
			if ad_ast = . then ad_ast = 0;
			if ad_hsdip = . then ad_hsdip = 0;
			if ad_ged = . then ad_ged = 0;
			if ad_ger = . then ad_ger = 0;
			if ad_gens = . then ad_gens = 0;
			if ap = . then ap = 0;
			if rs = . then rs = 0;
			if chs = . then chs = 0;
			if ib = . then ib = 0;
			if aice = . then aice = 0;
			if ib_aice = . then ib_aice = 0;
			if athlete = . then athlete = 0;
			if remedial = . then remedial = 0;
			if sat_mss = . then sat_mss = 0;
			if sat_erws = . then sat_erws = 0;
			if high_school_gpa = . then high_school_gpa_mi = 1; else high_school_gpa_mi = 0;
			if high_school_gpa = . then high_school_gpa = 0;
			if transfer_gpa = . then transfer_gpa_mi = 1; else transfer_gpa_mi = 0;
			if transfer_gpa = . then transfer_gpa = 0;
			if last_sch_proprietorship = '' then last_sch_proprietorship = 'UNKN';
			if ipeds_ethnic_group_descrshort = '' then ipeds_ethnic_group_descrshort = 'NS';
			if fall_avg_pct_withdrawn = . then fall_avg_pct_withdrawn = 0;
			if fall_avg_pct_CDFW = . then fall_avg_pct_CDFW = 0;
			if fall_avg_pct_CDF = . then fall_avg_pct_CDF = 0;
			if fall_avg_pct_DFW = . then fall_avg_pct_DFW = 0;
			if fall_avg_pct_DF = . then fall_avg_pct_DF = 0;
			if fall_avg_difficulty = . then fall_crse_mi = 1; else fall_crse_mi = 0; 
			if fall_avg_difficulty = . then fall_avg_difficulty = 0;
			if fall_lec_contact_hrs = . then fall_lec_contact_hrs = 0;
			if fall_lab_contact_hrs = . then fall_lab_contact_hrs = 0;
			if fall_int_contact_hrs = . then fall_int_contact_hrs = 0;
			if fall_stu_contact_hrs = . then fall_stu_contact_hrs = 0;
			if fall_sem_contact_hrs = . then fall_sem_contact_hrs = 0;
			if fall_oth_contact_hrs = . then fall_oth_contact_hrs = 0;
			if total_fall_contact_hrs = . then total_fall_contact_hrs = 0;
			if fall_avg_pct_CDFW = . then fall_avg_pct_CDFW = 0;
			if fall_avg_pct_CDF = . then fall_avg_pct_CDF = 0;
			if fall_avg_pct_DFW = . then fall_avg_pct_DFW = 0;
			if fall_avg_pct_DF = . then fall_avg_pct_DF = 0;
			if fall_avg_difficulty = . then fall_crse_mi = 1; else fall_crse_mi = 0; 
			if fall_avg_difficulty = . then fall_avg_difficulty = 0;
			if spring_avg_pct_withdrawn = . then spring_avg_pct_withdrawn = 0;
			if spring_avg_pct_CDFW = . then spring_avg_pct_CDFW = 0;
			if spring_avg_pct_CDF = . then spring_avg_pct_CDF = 0;
			if spring_avg_pct_DFW = . then spring_avg_pct_DFW = 0;
			if spring_avg_pct_DF = . then spring_avg_pct_DF = 0;
			if spring_avg_difficulty = . then spring_crse_mi = 1; else spring_crse_mi = 0; 
			if spring_avg_difficulty = . then spring_avg_difficulty = 0;
			if spring_lec_count = . then spring_lec_count = 0;
			if spring_lab_count = . then spring_lab_count = 0;
			if spring_int_count = . then spring_int_count = 0;
			if spring_stu_count = . then spring_stu_count = 0;
			if spring_sem_count = . then spring_sem_count = 0;
			if spring_oth_count = . then spring_oth_count = 0;
			if spring_lec_contact_hrs = . then spring_lec_contact_hrs = 0;
			if spring_lab_contact_hrs = . then spring_lab_contact_hrs = 0;
			if spring_int_contact_hrs = . then spring_int_contact_hrs = 0;
			if spring_stu_contact_hrs = . then spring_stu_contact_hrs = 0;
			if spring_sem_contact_hrs = . then spring_sem_contact_hrs = 0;
			if spring_oth_contact_hrs = . then spring_oth_contact_hrs = 0;
			if total_spring_contact_hrs = . then total_spring_contact_hrs = 0;
			if fall_enrl_sum = . then fall_enrl_sum_mi = 1; else fall_enrl_sum_mi = 0;
			if fall_enrl_avg = . then fall_enrl_avg_mi = 1; else fall_enrl_avg_mi = 0;
			if spring_enrl_sum = . then spring_enrl_sum_mi = 1; else spring_enrl_sum_mi = 0;
			if spring_enrl_avg = . then spring_enrl_avg_mi = 1; else spring_enrl_avg_mi = 0;
			if fall_enrl_sum = . then fall_enrl_sum = 0;
			if fall_enrl_avg = . then fall_enrl_avg = 0;
			if spring_enrl_sum = . then spring_enrl_sum = 0;
			if spring_enrl_avg = . then spring_enrl_avg = 0;
			if fall_class_time_early = . then fall_class_time_early_mi = 1; else fall_class_time_early_mi = 0;
			if fall_class_time_late = . then fall_class_time_late_mi = 1; else fall_class_time_late_mi = 0;
			if spring_class_time_early = . then spring_class_time_early_mi = 1; else spring_class_time_early_mi = 0;
			if spring_class_time_late = . then spring_class_time_late_mi = 1; else spring_class_time_late_mi = 0;
			if fall_class_time_early = . then fall_class_time_early = 0;
			if fall_class_time_late = . then fall_class_time_late = 0;
			if spring_class_time_early = . then spring_class_time_early = 0;
			if spring_class_time_late = . then spring_class_time_late = 0;
			if fall_sun_class = . then fall_sun_class = 0;
			if fall_mon_class = . then fall_mon_class = 0;
			if fall_tues_class = . then fall_tues_class = 0;
			if fall_wed_class = . then fall_wed_class = 0;
			if fall_thurs_class = . then fall_thurs_class = 0;
			if fall_fri_class = . then fall_fri_class = 0;
			if fall_sat_class = . then fall_sat_class = 0;
			if spring_sun_class = . then spring_sun_class = 0;
			if spring_mon_class = . then spring_mon_class = 0;
			if spring_tues_class = . then spring_tues_class = 0;
			if spring_wed_class = . then spring_wed_class = 0;
			if spring_thurs_class = . then spring_thurs_class = 0;
			if spring_fri_class = . then spring_fri_class = 0;
			if spring_sat_class = . then spring_sat_class = 0;
			if total_fall_units = . then total_fall_units = 0;
			if total_spring_units = . then total_spring_units = 0;
			if fall_credit_hours = . then fall_credit_hours = 0;
			if spring_credit_hours = . then spring_credit_hours = 0;
			if fall_midterm_gpa_avg = . then fall_midterm_gpa_avg_mi = 1; else fall_midterm_gpa_avg_mi = 0;
			if fall_midterm_gpa_avg = . then fall_midterm_gpa_avg = 0;
			if fall_midterm_grade_count = . then fall_midterm_grade_count = 0;
			if fall_midterm_S_grade_count = . then fall_midterm_S_grade_count = 0;
			if fall_midterm_W_grade_count = . then fall_midterm_W_grade_count = 0;
			if spring_midterm_gpa_avg = . then spring_midterm_gpa_avg_mi = 1; else spring_midterm_gpa_avg_mi = 0;
			if spring_midterm_gpa_avg = . then spring_midterm_gpa_avg = 0;
			if spring_midterm_grade_count = . then spring_midterm_grade_count = 0;
			if spring_midterm_S_grade_count = . then spring_midterm_S_grade_count = 0;
			if spring_midterm_W_grade_count = . then spring_midterm_W_grade_count = 0;
			if fall_term_gpa = . then fall_term_gpa_mi = 1; else fall_term_gpa_mi = 0;
			if fall_term_gpa = . then fall_term_gpa = 0;
			if spring_term_gpa = . then spring_term_gpa_mi = 1; else spring_term_gpa_mi = 0;
			if spring_term_gpa = . then spring_term_gpa = 0;
			if fall_term_D_grade_count = . then fall_term_D_grade_count_mi = 1; else fall_term_D_grade_count_mi = 0;
			if fall_term_D_grade_count = . then fall_term_D_grade_count = 0;
			if fall_term_F_grade_count = . then fall_term_F_grade_count_mi = 1; else fall_term_F_grade_count_mi = 0;
			if fall_term_F_grade_count = . then fall_term_F_grade_count = 0;
			if fall_term_W_grade_count = . then fall_term_W_grade_count_mi = 1; else fall_term_W_grade_count_mi = 0;
			if fall_term_W_grade_count = . then fall_term_W_grade_count = 0;
			if fall_term_I_grade_count = . then fall_term_I_grade_count_mi = 1; else fall_term_I_grade_count_mi = 0;
			if fall_term_I_grade_count = . then fall_term_I_grade_count = 0;
			if fall_term_X_grade_count = . then fall_term_X_grade_count_mi = 1; else fall_term_X_grade_count_mi = 0;
			if fall_term_X_grade_count = . then fall_term_X_grade_count = 0;
			if fall_term_U_grade_count = . then fall_term_U_grade_count_mi = 1; else fall_term_U_grade_count_mi = 0;
			if fall_term_U_grade_count = . then fall_term_U_grade_count = 0;
			if fall_term_S_grade_count = . then fall_term_S_grade_count_mi = 1; else fall_term_S_grade_count_mi = 0;
			if fall_term_S_grade_count = . then fall_term_S_grade_count = 0;
			if fall_term_P_grade_count = . then fall_term_P_grade_count_mi = 1; else fall_term_P_grade_count_mi = 0;
			if fall_term_P_grade_count = . then fall_term_P_grade_count = 0;
			if fall_term_Z_grade_count = . then fall_term_Z_grade_count_mi = 1; else fall_term_Z_grade_count_mi = 0;
			if fall_term_Z_grade_count = . then fall_term_Z_grade_count = 0;
			if fall_term_letter_count = . then fall_term_letter_count_mi = 1; else fall_term_letter_count_mi = 0;
			if fall_term_letter_count = . then fall_term_letter_count = 0;
			if fall_term_grade_count = . then fall_term_grade_count_mi = 1; else fall_term_grade_count_mi = 0;
			if fall_term_grade_count = . then fall_term_grade_count = 0;
			fall_term_no_letter_count = fall_term_grade_count - fall_term_letter_count;
			if spring_term_D_grade_count = . then spring_term_D_grade_count_mi = 1; else spring_term_D_grade_count_mi = 0;
			if spring_term_D_grade_count = . then spring_term_D_grade_count = 0;
			if spring_term_F_grade_count = . then spring_term_F_grade_count_mi = 1; else spring_term_F_grade_count_mi = 0;
			if spring_term_F_grade_count = . then spring_term_F_grade_count = 0;
			if spring_term_W_grade_count = . then spring_term_W_grade_count_mi = 1; else spring_term_W_grade_count_mi = 0;
			if spring_term_W_grade_count = . then spring_term_W_grade_count = 0;
			if spring_term_I_grade_count = . then spring_term_I_grade_count_mi = 1; else spring_term_I_grade_count_mi = 0;
			if spring_term_I_grade_count = . then spring_term_I_grade_count = 0;
			if spring_term_X_grade_count = . then spring_term_X_grade_count_mi = 1; else spring_term_X_grade_count_mi = 0;
			if spring_term_X_grade_count = . then spring_term_X_grade_count = 0;
			if spring_term_U_grade_count = . then spring_term_U_grade_count_mi = 1; else spring_term_U_grade_count_mi = 0;
			if spring_term_U_grade_count = . then spring_term_U_grade_count = 0;
			if spring_term_S_grade_count = . then spring_term_S_grade_count_mi = 1; else spring_term_S_grade_count_mi = 0;
			if spring_term_S_grade_count = . then spring_term_S_grade_count = 0;
			if spring_term_P_grade_count = . then spring_term_P_grade_count_mi = 1; else spring_term_P_grade_count_mi = 0;
			if spring_term_P_grade_count = . then spring_term_P_grade_count = 0;
			if spring_term_Z_grade_count = . then spring_term_Z_grade_count_mi = 1; else spring_term_Z_grade_count_mi = 0;
			if spring_term_Z_grade_count = . then spring_term_Z_grade_count = 0;
			if spring_term_letter_count = . then spring_term_leter_count_mi = 1; else spring_term_leter_count_mi = 0;
			if spring_term_letter_count = . then spring_term_letter_count = 0;
			if spring_term_grade_count = . then spring_term_grade_count_mi = 1; else spring_term_grade_count_mi = 0;
			if spring_term_grade_count = . then spring_term_grade_count = 0;
			spring_term_no_letter_count = spring_term_grade_count - spring_term_letter_count;
			if first_gen_flag = '' then first_gen_flag_mi = 1; else first_gen_flag_mi = 0;
			if first_gen_flag = '' then first_gen_flag = 'N';
			if camp_addr_indicator ^= 'Y' then camp_addr_indicator = 'N';
			if housing_reshall_indicator ^= 'Y' then housing_reshall_indicator = 'N';
			if housing_ssa_indicator ^= 'Y' then housing_ssa_indicator = 'N';
			if housing_family_indicator ^= 'Y' then housing_family_indicator = 'N';
			if afl_reshall_indicator ^= 'Y' then afl_reshall_indicator = 'N';
			if afl_ssa_indicator ^= 'Y' then afl_ssa_indicator = 'N';
			if afl_family_indicator ^= 'Y' then afl_family_indicator = 'N';
			if afl_greek_indicator ^= 'Y' then afl_greek_indicator = 'N';
			if afl_greek_life_indicator ^= 'Y' then afl_greek_life_indicator = 'N';
			fall_withdrawn_hours = (total_fall_units - fall_credit_hours) * -1;
			if total_fall_units = 0 then fall_withdrawn_ind = 1; else fall_withdrawn_ind = 0;
			spring_withdrawn_hours = (total_spring_units - spring_credit_hours) * -1;
			if total_spring_units = 0 then spring_withdrawn = 1; else spring_withdrawn = 0;
			spring_midterm_gpa_change = spring_midterm_gpa_avg - fall_cum_gpa;
			unmet_need_disb = fed_need - total_disb;
			unmet_need_acpt = fed_need - total_accept;
			if unmet_need_acpt = . then unmet_need_acpt_mi = 1; else unmet_need_acpt_mi = 0;
			if unmet_need_acpt < 0 then unmet_need_acpt = 0;
			unmet_need_ofr = fed_need - total_offer;
			if unmet_need_ofr = . then unmet_need_ofr_mi = 1; else unmet_need_ofr_mi = 0;
			if unmet_need_ofr < 0 then unmet_need_ofr = 0;
			if fed_efc = . then fed_efc = 0;
			if fed_need = . then fed_need = 0;
			if total_disb = . then total_disb = 0;
			if total_offer = . then total_offer = 0;
			if total_accept = . then total_accept = 0;
		run;

		data testing_set;
			set dataset_&end_cohort.;
			if enrl_ind = . then enrl_ind = 0;
			if distance = . then acs_mi = 1; else acs_mi = 0;
			if distance = . then distance = 0;
			if pop_dens = . then pop_dens = 0;
			if educ_rate = . then educ_rate = 0;	
			if pct_blk = . then pct_blk = 0;	
			if pct_ai = . then pct_ai = 0;
			if pct_asn = .	then pct_asn = 0;
			if pct_hawi = . then pct_hawi = 0;
			if pct_two = . then pct_two = 0;
			if pct_hisp = . then pct_hisp = 0;
			if pct_oth = . then pct_oth = 0;
			if pct_non = . then pct_non = 0;
			if median_inc = . then median_inc = 0;
			if median_value = . then median_value = 0;
			if gini_indx = . then gini_indx = 0;
			if pvrt_rate = . then pvrt_rate = 0;
			if educ_rate = . then educ_rate = 0;
			if city_large = . then city_large = 0;
			if city_mid = . then city_mid = 0;
			if city_small = . then city_small = 0;
			if suburb_large = . then suburb_large = 0;
			if suburb_mid = . then suburb_mid = 0;
			if suburb_small = . then suburb_small = 0;
			if town_fringe = . then town_fringe = 0;
			if town_distant = . then town_distant = 0;
			if town_remote = . then town_remote = 0;
			if rural_fringe = . then rural_fringe = 0;
			if rural_distant = . then rural_distant = 0;
			if rural_remote = . then rural_remote = 0;
			if ad_dta = . then ad_dta = 0;
			if ad_ast = . then ad_ast = 0;
			if ad_hsdip = . then ad_hsdip = 0;
			if ad_ged = . then ad_ged = 0;
			if ad_ger = . then ad_ger = 0;
			if ad_gens = . then ad_gens = 0;
			if ap = . then ap = 0;
			if rs = . then rs = 0;
			if chs = . then chs = 0;
			if ib = . then ib = 0;
			if aice = . then aice = 0;
			if ib_aice = . then ib_aice = 0;
			if athlete = . then athlete = 0;
			if remedial = . then remedial = 0;
			if sat_mss = . then sat_mss = 0;
			if sat_erws = . then sat_erws = 0;
			if high_school_gpa = . then high_school_gpa_mi = 1; else high_school_gpa_mi = 0;
			if high_school_gpa = . then high_school_gpa = 0;
			if transfer_gpa = . then transfer_gpa_mi = 1; else transfer_gpa_mi = 0;
			if transfer_gpa = . then transfer_gpa = 0;
			if last_sch_proprietorship = '' then last_sch_proprietorship = 'UNKN';
			if ipeds_ethnic_group_descrshort = '' then ipeds_ethnic_group_descrshort = 'NS';
			if fall_avg_pct_withdrawn = . then fall_avg_pct_withdrawn = 0;
			if fall_avg_pct_CDFW = . then fall_avg_pct_CDFW = 0;
			if fall_avg_pct_CDF = . then fall_avg_pct_CDF = 0;
			if fall_avg_pct_DFW = . then fall_avg_pct_DFW = 0;
			if fall_avg_pct_DF = . then fall_avg_pct_DF = 0;
			if fall_avg_difficulty = . then fall_crse_mi = 1; else fall_crse_mi = 0; 
			if fall_avg_difficulty = . then fall_avg_difficulty = 0;
			if fall_lec_contact_hrs = . then fall_lec_contact_hrs = 0;
			if fall_lab_contact_hrs = . then fall_lab_contact_hrs = 0;
			if fall_int_contact_hrs = . then fall_int_contact_hrs = 0;
			if fall_stu_contact_hrs = . then fall_stu_contact_hrs = 0;
			if fall_sem_contact_hrs = . then fall_sem_contact_hrs = 0;
			if fall_oth_contact_hrs = . then fall_oth_contact_hrs = 0;
			if total_fall_contact_hrs = . then total_fall_contact_hrs = 0;
			if fall_avg_pct_CDFW = . then fall_avg_pct_CDFW = 0;
			if fall_avg_pct_CDF = . then fall_avg_pct_CDF = 0;
			if fall_avg_pct_DFW = . then fall_avg_pct_DFW = 0;
			if fall_avg_pct_DF = . then fall_avg_pct_DF = 0;
			if fall_avg_difficulty = . then fall_crse_mi = 1; else fall_crse_mi = 0; 
			if fall_avg_difficulty = . then fall_avg_difficulty = 0;
			if spring_avg_pct_withdrawn = . then spring_avg_pct_withdrawn = 0;
			if spring_avg_pct_CDFW = . then spring_avg_pct_CDFW = 0;
			if spring_avg_pct_CDF = . then spring_avg_pct_CDF = 0;
			if spring_avg_pct_DFW = . then spring_avg_pct_DFW = 0;
			if spring_avg_pct_DF = . then spring_avg_pct_DF = 0;
			if spring_avg_difficulty = . then spring_crse_mi = 1; else spring_crse_mi = 0; 
			if spring_avg_difficulty = . then spring_avg_difficulty = 0;
			if spring_lec_count = . then spring_lec_count = 0;
			if spring_lab_count = . then spring_lab_count = 0;
			if spring_int_count = . then spring_int_count = 0;
			if spring_stu_count = . then spring_stu_count = 0;
			if spring_sem_count = . then spring_sem_count = 0;
			if spring_oth_count = . then spring_oth_count = 0;
			if spring_lec_contact_hrs = . then spring_lec_contact_hrs = 0;
			if spring_lab_contact_hrs = . then spring_lab_contact_hrs = 0;
			if spring_int_contact_hrs = . then spring_int_contact_hrs = 0;
			if spring_stu_contact_hrs = . then spring_stu_contact_hrs = 0;
			if spring_sem_contact_hrs = . then spring_sem_contact_hrs = 0;
			if spring_oth_contact_hrs = . then spring_oth_contact_hrs = 0;
			if total_spring_contact_hrs = . then total_spring_contact_hrs = 0;
			if fall_enrl_sum = . then fall_enrl_sum_mi = 1; else fall_enrl_sum_mi = 0;
			if fall_enrl_avg = . then fall_enrl_avg_mi = 1; else fall_enrl_avg_mi = 0;
			if spring_enrl_sum = . then spring_enrl_sum_mi = 1; else spring_enrl_sum_mi = 0;
			if spring_enrl_avg = . then spring_enrl_avg_mi = 1; else spring_enrl_avg_mi = 0;
			if fall_enrl_sum = . then fall_enrl_sum = 0;
			if fall_enrl_avg = . then fall_enrl_avg = 0;
			if spring_enrl_sum = . then spring_enrl_sum = 0;
			if spring_enrl_avg = . then spring_enrl_avg = 0;
			if fall_class_time_early = . then fall_class_time_early_mi = 1; else fall_class_time_early_mi = 0;
			if fall_class_time_late = . then fall_class_time_late_mi = 1; else fall_class_time_late_mi = 0;
			if spring_class_time_early = . then spring_class_time_early_mi = 1; else spring_class_time_early_mi = 0;
			if spring_class_time_late = . then spring_class_time_late_mi = 1; else spring_class_time_late_mi = 0;
			if fall_class_time_early = . then fall_class_time_early = 0;
			if fall_class_time_late = . then fall_class_time_late = 0;
			if spring_class_time_early = . then spring_class_time_early = 0;
			if spring_class_time_late = . then spring_class_time_late = 0;
			if fall_sun_class = . then fall_sun_class = 0;
			if fall_mon_class = . then fall_mon_class = 0;
			if fall_tues_class = . then fall_tues_class = 0;
			if fall_wed_class = . then fall_wed_class = 0;
			if fall_thurs_class = . then fall_thurs_class = 0;
			if fall_fri_class = . then fall_fri_class = 0;
			if fall_sat_class = . then fall_sat_class = 0;
			if spring_sun_class = . then spring_sun_class = 0;
			if spring_mon_class = . then spring_mon_class = 0;
			if spring_tues_class = . then spring_tues_class = 0;
			if spring_wed_class = . then spring_wed_class = 0;
			if spring_thurs_class = . then spring_thurs_class = 0;
			if spring_fri_class = . then spring_fri_class = 0;
			if spring_sat_class = . then spring_sat_class = 0;
			if total_fall_units = . then total_fall_units = 0;
			if total_spring_units = . then total_spring_units = 0;
			if fall_credit_hours = . then fall_credit_hours = 0;
			if spring_credit_hours = . then spring_credit_hours = 0;
			if fall_midterm_gpa_avg = . then fall_midterm_gpa_avg_mi = 1; else fall_midterm_gpa_avg_mi = 0;
			if fall_midterm_gpa_avg = . then fall_midterm_gpa_avg = 0;
			if fall_midterm_grade_count = . then fall_midterm_grade_count = 0;
			if fall_midterm_S_grade_count = . then fall_midterm_S_grade_count = 0;
			if fall_midterm_W_grade_count = . then fall_midterm_W_grade_count = 0;
			if spring_midterm_gpa_avg = . then spring_midterm_gpa_avg_mi = 1; else spring_midterm_gpa_avg_mi = 0;
			if spring_midterm_gpa_avg = . then spring_midterm_gpa_avg = 0;
			if spring_midterm_grade_count = . then spring_midterm_grade_count = 0;
			if spring_midterm_S_grade_count = . then spring_midterm_S_grade_count = 0;
			if spring_midterm_W_grade_count = . then spring_midterm_W_grade_count = 0;
			if fall_term_gpa = . then fall_term_gpa_mi = 1; else fall_term_gpa_mi = 0;
			if fall_term_gpa = . then fall_term_gpa = 0;
			if spring_term_gpa = . then spring_term_gpa_mi = 1; else spring_term_gpa_mi = 0;
			if spring_term_gpa = . then spring_term_gpa = 0;
			if fall_term_D_grade_count = . then fall_term_D_grade_count_mi = 1; else fall_term_D_grade_count_mi = 0;
			if fall_term_D_grade_count = . then fall_term_D_grade_count = 0;
			if fall_term_F_grade_count = . then fall_term_F_grade_count_mi = 1; else fall_term_F_grade_count_mi = 0;
			if fall_term_F_grade_count = . then fall_term_F_grade_count = 0;
			if fall_term_W_grade_count = . then fall_term_W_grade_count_mi = 1; else fall_term_W_grade_count_mi = 0;
			if fall_term_W_grade_count = . then fall_term_W_grade_count = 0;
			if fall_term_I_grade_count = . then fall_term_I_grade_count_mi = 1; else fall_term_I_grade_count_mi = 0;
			if fall_term_I_grade_count = . then fall_term_I_grade_count = 0;
			if fall_term_X_grade_count = . then fall_term_X_grade_count_mi = 1; else fall_term_X_grade_count_mi = 0;
			if fall_term_X_grade_count = . then fall_term_X_grade_count = 0;
			if fall_term_U_grade_count = . then fall_term_U_grade_count_mi = 1; else fall_term_U_grade_count_mi = 0;
			if fall_term_U_grade_count = . then fall_term_U_grade_count = 0;
			if fall_term_S_grade_count = . then fall_term_S_grade_count_mi = 1; else fall_term_S_grade_count_mi = 0;
			if fall_term_S_grade_count = . then fall_term_S_grade_count = 0;
			if fall_term_P_grade_count = . then fall_term_P_grade_count_mi = 1; else fall_term_P_grade_count_mi = 0;
			if fall_term_P_grade_count = . then fall_term_P_grade_count = 0;
			if fall_term_Z_grade_count = . then fall_term_Z_grade_count_mi = 1; else fall_term_Z_grade_count_mi = 0;
			if fall_term_Z_grade_count = . then fall_term_Z_grade_count = 0;
			if fall_term_letter_count = . then fall_term_letter_count_mi = 1; else fall_term_letter_count_mi = 0;
			if fall_term_letter_count = . then fall_term_letter_count = 0;
			if fall_term_grade_count = . then fall_term_grade_count_mi = 1; else fall_term_grade_count_mi = 0;
			if fall_term_grade_count = . then fall_term_grade_count = 0;
			fall_term_no_letter_count = fall_term_grade_count - fall_term_letter_count;
			if spring_term_D_grade_count = . then spring_term_D_grade_count_mi = 1; else spring_term_D_grade_count_mi = 0;
			if spring_term_D_grade_count = . then spring_term_D_grade_count = 0;
			if spring_term_F_grade_count = . then spring_term_F_grade_count_mi = 1; else spring_term_F_grade_count_mi = 0;
			if spring_term_F_grade_count = . then spring_term_F_grade_count = 0;
			if spring_term_W_grade_count = . then spring_term_W_grade_count_mi = 1; else spring_term_W_grade_count_mi = 0;
			if spring_term_W_grade_count = . then spring_term_W_grade_count = 0;
			if spring_term_I_grade_count = . then spring_term_I_grade_count_mi = 1; else spring_term_I_grade_count_mi = 0;
			if spring_term_I_grade_count = . then spring_term_I_grade_count = 0;
			if spring_term_X_grade_count = . then spring_term_X_grade_count_mi = 1; else spring_term_X_grade_count_mi = 0;
			if spring_term_X_grade_count = . then spring_term_X_grade_count = 0;
			if spring_term_U_grade_count = . then spring_term_U_grade_count_mi = 1; else spring_term_U_grade_count_mi = 0;
			if spring_term_U_grade_count = . then spring_term_U_grade_count = 0;
			if spring_term_S_grade_count = . then spring_term_S_grade_count_mi = 1; else spring_term_S_grade_count_mi = 0;
			if spring_term_S_grade_count = . then spring_term_S_grade_count = 0;
			if spring_term_P_grade_count = . then spring_term_P_grade_count_mi = 1; else spring_term_P_grade_count_mi = 0;
			if spring_term_P_grade_count = . then spring_term_P_grade_count = 0;
			if spring_term_Z_grade_count = . then spring_term_Z_grade_count_mi = 1; else spring_term_Z_grade_count_mi = 0;
			if spring_term_Z_grade_count = . then spring_term_Z_grade_count = 0;
			if spring_term_letter_count = . then spring_term_leter_count_mi = 1; else spring_term_leter_count_mi = 0;
			if spring_term_letter_count = . then spring_term_letter_count = 0;
			if spring_term_grade_count = . then spring_term_grade_count_mi = 1; else spring_term_grade_count_mi = 0;
			if spring_term_grade_count = . then spring_term_grade_count = 0;
			spring_term_no_letter_count = spring_term_grade_count - spring_term_letter_count;
			if first_gen_flag = '' then first_gen_flag_mi = 1; else first_gen_flag_mi = 0;
			if first_gen_flag = '' then first_gen_flag = 'N';
			if camp_addr_indicator ^= 'Y' then camp_addr_indicator = 'N';
			if housing_reshall_indicator ^= 'Y' then housing_reshall_indicator = 'N';
			if housing_ssa_indicator ^= 'Y' then housing_ssa_indicator = 'N';
			if housing_family_indicator ^= 'Y' then housing_family_indicator = 'N';
			if afl_reshall_indicator ^= 'Y' then afl_reshall_indicator = 'N';
			if afl_ssa_indicator ^= 'Y' then afl_ssa_indicator = 'N';
			if afl_family_indicator ^= 'Y' then afl_family_indicator = 'N';
			if afl_greek_indicator ^= 'Y' then afl_greek_indicator = 'N';
			if afl_greek_life_indicator ^= 'Y' then afl_greek_life_indicator = 'N';
			fall_withdrawn_hours = (total_fall_units - fall_credit_hours) * -1;
			if total_fall_units = 0 then fall_withdrawn_ind = 1; else fall_withdrawn_ind = 0;
			spring_withdrawn_hours = (total_spring_units - spring_credit_hours) * -1;
			if total_spring_units = 0 then spring_withdrawn = 1; else spring_withdrawn = 0;
			spring_midterm_gpa_change = spring_midterm_gpa_avg - fall_cum_gpa;
			unmet_need_disb = fed_need - total_disb;
			unmet_need_acpt = fed_need - total_accept;
			if unmet_need_acpt = . then unmet_need_acpt_mi = 1; else unmet_need_acpt_mi = 0;
			if unmet_need_acpt < 0 then unmet_need_acpt = 0;
			unmet_need_ofr = fed_need - total_offer;
			if unmet_need_ofr = . then unmet_need_ofr_mi = 1; else unmet_need_ofr_mi = 0;
			if unmet_need_ofr < 0 then unmet_need_ofr = 0;
			if fed_efc = . then fed_efc = 0;
			if fed_need = . then fed_need = 0;
			if total_disb = . then total_disb = 0;
			if total_offer = . then total_offer = 0;
			if total_accept = . then total_accept = 0;
		run;
		"""
		)

		print('Done\n')

		# Export data from SAS
		print('Export data from SAS...')

		sas_log = sas.submit("""
		libname valid \"Z:\\Nathan\\Models\\student_risk\\datasets\\\";

		%let valid_pass = 0;

		%if %sysfunc(exist(valid.ft_ft_1yr_validation_set)) 
			%then %do;
				data work.validation_set_compare;
					set valid.ft_ft_1yr_validation_set;
				run;
		
                proc compare data=validation_set compare=validation_set_compare method=absolute;
                run;                                                                        
			%end;
			
			%else %do;
				data work.validation_set_compare;
					set work.validation_set;
					stop;
				run;
				
				proc compare data=validation_set compare=validation_set_compare method=absolute;
				run;
			%end;

		%if &sysinfo ^= 0
			 
			%then %do;
				data valid.ft_ft_1yr_validation_set;
					set work.validation_set;
				run;
			%end;
			
			%else %do;
				%let valid_pass = 1;
			%end;

		libname training \"Z:\\Nathan\\Models\\student_risk\\datasets\\\";

		%let training_pass = 0;

		%if %sysfunc(exist(training.ft_ft_1yr_training_set)) 
			%then %do;
				data work.training_set_compare;
					set training.ft_ft_1yr_training_set;
				run;
                proc compare data=training_set compare=training_set_compare method=absolute;
                run;
			%end;
			
			%else %do;
				data work.training_set_compare;
					set work.training_set;
					stop;
				run;
				
				proc compare data=training_set compare=training_set_compare method=absolute;
				run;
			%end;

		%if &sysinfo ^= 0
			 
			%then %do;
				data training.ft_ft_1yr_training_set;
					set work.training_set;
				run;
			%end;
			
			%else %do;
				%let training_pass = 1;
			%end;
			
		libname testing \"Z:\\Nathan\\Models\\student_risk\\datasets\\\";

		%let testing_pass = 0;

		%if %sysfunc(exist(testing.ft_ft_1yr_testing_set)) 
			%then %do;
				data work.testing_set_compare;
					set testing.ft_ft_1yr_testing_set;
				run;

                proc compare data=testing_set compare=testing_set_compare method=absolute;
                run;  
			%end;
			
			%else %do;
				data work.testing_set_compare;
					set work.testing_set;
					stop;
				run;
				
				proc compare data=testing_set compare=testing_set_compare method=absolute;
				run;
			%end;
		
		%if &sysinfo ^= 0
			 
			%then %do;
				data testing.ft_ft_1yr_testing_set;
					set work.testing_set;
				run;
			%end;
			
			%else %do;
				%let testing_pass = 1;
			%end;
		""")

		HTML(sas_log['LOG'])

		print('Done\n')

		# End SAS session
		sas.endsas()
