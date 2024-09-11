#%%
import os
import runpy
import sys
import time
import traceback
from datetime import date, datetime
from glob import glob

import saspy

from student_risk import config

#%%
start = time.perf_counter()

#%%
sas = saspy.SASsession()

sas.submit("""
%let adm = adm;

libname &adm. odbc dsn=&adm. schema=dbo;
libname acs \"Z:\\Nathan\\Models\\student_risk\\supplemental_files\\\";

proc sort data=adm.xw_term out=acs.xw_term;
	by acad_career strm;
run;

data acs.xw_term;
	set asc.xw_term;
	by acad_career;
	if first.acad_career then idx = 1;
	else idx + 1;
	where acad_career = 'UGRD'
		and term_year <= year(today());
run;

proc sql;
	create table acs.adj_term as
	select
		base.acad_career,
		base.term_year,
        base.term_type,
        base.strm,
		base.full_acad_year,
		datepart(base.term_begin_dt) as term_begin_dt format=mmddyyd10.,
		coalesce(day(datepart(base.term_begin_dt)),99999) as begin_day,
		coalesce(week(datepart(base.term_begin_dt)),99999) as begin_week,
		coalesce(month(datepart(base.term_begin_dt)),99999) as begin_month,
		coalesce(year(datepart(base.term_begin_dt)),99999) as begin_year,
		datepart(base.term_census_dt) as term_census_dt format=mmddyyd10.,
        coalesce(day(datepart(base.term_census_dt)),99999) as census_day,
		coalesce(week(datepart(base.term_census_dt)),99999) as census_week,
		coalesce(month(datepart(base.term_census_dt)),99999) as census_month,
		coalesce(year(datepart(base.term_census_dt)),99999) as census_year,
		datepart(base.term_midterm_dt) as term_midterm_dt format=mmddyyd10.,
        coalesce(day(datepart(base.term_midterm_dt)),99999) as midterm_day,
        coalesce(week(datepart(base.term_midterm_dt)),99999) as midterm_week,
        coalesce(month(datepart(base.term_midterm_dt)),99999) as midterm_month,
        coalesce(year(datepart(base.term_midterm_dt)),99999) as midterm_year,
		datepart(base.term_end_dt) as term_eot_dt format=mmddyyd10.,
        day(datepart(base.term_end_dt)) as eot_day,
        week(datepart(base.term_end_dt)) as eot_week,
        month(datepart(base.term_end_dt)) as eot_month,
        year(datepart(base.term_end_dt)) as eot_year,
       	coalesce(datepart(intnx('dtday', next.term_begin_dt, -1)),99999) as term_end_dt format=mmddyyd10.,
		coalesce(day(datepart(intnx('dtday', next.term_begin_dt, -1))),99999) as end_day,
		coalesce(week(datepart(intnx('dtday', next.term_begin_dt, -1))),99999) as end_week,
		coalesce(month(datepart(intnx('dtday', next.term_begin_dt, -1))),99999) as end_month,
		coalesce(year(datepart(intnx('dtday', next.term_begin_dt, -1))),99999) as end_year
	from acs.xw_term as base
	left join acs.xw_term as next
		on base.acad_career = next.acad_career
			and base.idx = next.idx - 1
;quit;

filename adj_term \"Z:\\Nathan\\Models\\student_risk\\supplemental_files\\acad_calendar.csv\" encoding=\"utf-8\";

proc export data=acs.adj_term outfile=adj_term dbms=csv replace;

proc sql;
	select term_type into: term_type 
	from acs.adj_term 
	where term_begin_dt <= today()
		and term_end_dt >= today()
		and acad_career = 'UGRD'
;quit;

proc sql;
	select term_begin_dt into: term_begin_dt 
	from acs.adj_term 
	where term_begin_dt <= today()
		and term_end_dt >= today()
		and acad_career = 'UGRD'
;quit;
""")

term_type = sas.symget('term_type')
term_begin_dt = datetime.strptime(str(sas.symget('term_begin_dt')), '%m-%d-%Y').date()

sas.endsas()

#%%
class Logger(object):
	def __init__(self):
		self.terminal = sys.stdout
		self.log = open(f'Z:\\Nathan\\Models\\student_risk\\logs\\main\\log_{date.today()}.log', 'w')

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)  

	def flush(self):
		self.terminal.flush()
		self.log.flush()


#%%
if __name__ == '__main__':

	if date.today() == term_begin_dt or date.today().weekday() == 5:

		for filename in glob('Z:/Nathan/Models/student_risk/datasets/*.sas7bdat'):
			os.remove(filename)

	if term_type == 'SUM':
		
		sys.stdout = Logger()

		try:
			runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\sum\\ft_ft_1yr\\sr_prod_sum_ft_ft_1yr_eot.py')
			runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\sum\\ft_tr_1yr\\sr_prod_sum_ft_tr_1yr_eot.py')
			runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\sum\\ft_ft_2yr\\sr_prod_sum_ft_ft_2yr_eot.py')
			runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\sum\\ft_tr_2yr\\sr_prod_sum_ft_tr_2yr_eot.py')
		except config.EOTError as eot_error:
			print(eot_error)
		except KeyError as key_error:
			print(key_error)
		except:
			traceback.print_exc(file=sys.stdout)
		else:
			stop = time.perf_counter()
			print(f'Completed in {(stop - start)/60:.1f} minutes\n')

	elif term_type == 'SPR':

		sys.stdout = Logger()

		try:
			runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\spr\\ft_ft_1yr\\sr_prod_spr_ft_ft_1yr_mid.py')
			runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\spr\\ft_tr_1yr\\sr_prod_spr_ft_tr_1yr_mid.py')
			runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\spr\\ft_ft_2yr\\sr_prod_spr_ft_ft_2yr_mid.py')
			runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\spr\\ft_tr_2yr\\sr_prod_spr_ft_tr_2yr_mid.py')
		except config.MidError as mid_error:
			print(mid_error)
			
			try:
				runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\spr\\ft_ft_1yr\\sr_prod_spr_ft_ft_1yr_cen.py')
				runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\spr\\ft_tr_1yr\\sr_prod_spr_ft_tr_1yr_cen.py')
				runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\spr\\ft_ft_2yr\\sr_prod_spr_ft_ft_2yr_cen.py')
				runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\spr\\ft_tr_2yr\\sr_prod_spr_ft_tr_2yr_cen.py')
			except config.CenError as cen_error:
				print(cen_error)
			
				try:
					runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\spr\\ft_ft_1yr\\sr_prod_spr_ft_ft_1yr_eot.py')
					runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\spr\\ft_tr_1yr\\sr_prod_spr_ft_tr_1yr_eot.py')
					runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\spr\\ft_ft_2yr\\sr_prod_spr_ft_ft_2yr_eot.py')
					runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\spr\\ft_tr_2yr\\sr_prod_spr_ft_tr_2yr_eot.py')
				except config.EOTError as eot_error:
					print(eot_error)
				except KeyError as key_error:
					print(key_error)
				except:
					traceback.print_exc(file=sys.stdout)
				else:
					stop = time.perf_counter()
					print(f'Completed in {(stop - start)/60:.1f} minutes\n')

			except KeyError as key_error:
				print(key_error)
			except:
				traceback.print_exc(file=sys.stdout)
			else:
				stop = time.perf_counter()
				print(f'Completed in {(stop - start)/60:.1f} minutes\n')

		except KeyError as key_error:
			print(key_error)
		except:
			traceback.print_exc(file=sys.stdout)
		else:
			stop = time.perf_counter()
			print(f'Completed in {(stop - start)/60:.1f} minutes\n')

	elif term_type == 'FAL':
		
		sys.stdout = Logger()

		try:
			runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\fal\\ft_ft_1yr\\sr_prod_fal_ft_ft_1yr_mid.py')
			runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\fal\\ft_tr_1yr\\sr_prod_fal_ft_tr_1yr_mid.py')
			runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\fal\\ft_ft_2yr\\sr_prod_fal_ft_ft_2yr_mid.py')
			runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\fal\\ft_tr_2yr\\sr_prod_fal_ft_tr_2yr_mid.py')
		except config.MidError as mid_error:
			print(mid_error)

			try:
				runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\fal\\ft_ft_1yr\\sr_prod_fal_ft_ft_1yr_cen.py')
				runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\fal\\ft_tr_1yr\\sr_prod_fal_ft_tr_1yr_cen.py')
				runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\fal\\ft_ft_2yr\\sr_prod_fal_ft_ft_2yr_cen.py')
				runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\fal\\ft_tr_2yr\\sr_prod_fal_ft_tr_2yr_cen.py')
			except config.CenError as cen_error:
				print(cen_error)

				try:
					runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\fal\\ft_ft_1yr\\sr_prod_fal_ft_ft_1yr_adm.py')
					runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\fal\\ft_tr_1yr\\sr_prod_fal_ft_tr_1yr_adm.py')
					runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\fal\\ft_ft_2yr\\sr_prod_fal_ft_ft_2yr_adm.py')
					runpy.run_path(path_name='Z:\\Nathan\\Models\\student_risk\\student_risk\\prod\\fal\\ft_tr_2yr\\sr_prod_fal_ft_tr_2yr_adm.py')
				except config.AdmError as adm_error:
					print(adm_error)
				except KeyError as key_error:
					print(key_error)
				except:
					traceback.print_exc(file=sys.stdout)
				else:
					stop = time.perf_counter()
					print(f'Completed in {(stop - start)/60:.1f} minutes\n')

			except KeyError as key_error:
				print(key_error)
			except:
				traceback.print_exc(file=sys.stdout)
			else:
				stop = time.perf_counter()
				print(f'Completed in {(stop - start)/60:.1f} minutes\n')

		except KeyError as key_error:
			print(key_error)
		except:
			traceback.print_exc(file=sys.stdout)
		else:
			stop = time.perf_counter()
			print(f'Completed in {(stop - start)/60:.1f} minutes\n')

