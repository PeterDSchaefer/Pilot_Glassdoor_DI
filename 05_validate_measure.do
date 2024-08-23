********************************************************************************
*** ANALYZES CEO DISMISSALS IN A CEO-MONTH DATASET
*** Author: Peter Schaefer
*** This version: 2023/10/25



*===============================================================================
* DEFINITIONS AND SOURCES
{
**Basic settings
clear
set more off
set matsize 11000
version 17

** Folders 
global main_folder = 	"C:\Users\pesc713f\Desktop\01_Research\01_projects\14_Holistic_DI\01_pilot_di_from_gd_reviews"
global temp_folder = 	"02_temp_files"

** File directories
global gd_di_file = "04_proc_gd_reviews\02_dta\reviews_data.dta"
global our_di_file = "04_proc_gd_reviews\03_text_processing_steps\07_di_measures.csv"


cd $main_folder
}


** Import our DI File 
import delimited "$our_di_file", clear
save "$temp_folder\temp.dta", replace
label variable diversitymeasure "Diversity & Inclusion Scores from Machine Learning"

** Generate histogram
drop if diversitymeasure == 0
hist diversitymeasure
sum diversitymeasure, d
tab year



** Use Glassdoor DI File
use $gd_di_file, clear
rename company_name companyid
gen yearstr = substr(reviewDateTime, 1, 4)
destring yearstr, gen(year)
drop if ratingDiversityAndInclusion==0 
collapse (mean) ratingDiversityAndInclusion, by(companyid year)
merge 1:1 companyid year using "$temp_folder\temp.dta"
drop if _merge==2
drop _merge

label variable ratingDiversityAndInclusion "Glassdoor Diversity & Inclusion Rating"



** Generate scatter plot and calculate correlations
label variable diversitymeasure "Diversity & Inclusion Scores from Machine Learning"
drop if diversitymeasure == 0
corr ratingDiversityAndInclusion diversitymeasure
scatter ratingDiversityAndInclusion diversitymeasure  