### Data Information
* Archival data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Y28XIQJmmVjROrPddpf78rF9-5Zcrs6B?usp=sharing). 
* Experimental data are uploaded. 
   *  Future deprecation warnings come with the pacakges installed (```pyprocessmacro```,```pingouin```) and will be fixed.

### Archival Data Processing & Analysis
* Step 1: <b> [inequality_SE_prosper_data.py](https://github.com/jinyan0425/inequality_related/tree/main/Inequality_sharing_economy/archival/scripts/inequality_SE_prosper_data.py) </b> to prepare the prosper dataset (dataset needed: <i> prosper_loan_data.csv </i> and <i> state_abbv.csv </i> in [Google Drive](https://drive.google.com/drive/folders/1Y28XIQJmmVjROrPddpf78rF9-5Zcrs6B?usp=sharing). 

* Step 2: <b> [inequality_SE_master_data.py](https://github.com/jinyan0425/inequality_related/tree/main/Inequality_sharing_economy/archival/scripts/inequality_SE_master_data.py) </b> to create the master dataset (dataset needed: <i> propser_processed_data.csv </i> in [Google Drive](https://drive.google.com/drive/folders/1Y28XIQJmmVjROrPddpf78rF9-5Zcrs6B?usp=sharing). 

* Step 3: <b> [inequality_SE_archival_data_analysis.py](https://github.com/jinyan0425/inequality_related/tree/main/Inequality_sharing_economy/archival/scripts/inequality_SE_archival_data_analysis.py) </b> to analyze the archival data (dataset needed: <i> df_SE_master.csv</i> in [Google Drive](https://drive.google.com/drive/folders/1Y28XIQJmmVjROrPddpf78rF9-5Zcrs6B?usp=sharing).

### Experimental Data Processing & Analysis
* <b> inequality_SE_experiments_analysis.py </b> to analyze the experimental data (datasets needed: in [data folder](https://github.com/jinyan0425/inequality_related/tree/main/Inequality_sharing_economy/experiments/data); additional module needed: <i> [experiments_analysis_tools.py](https://github.com/jinyan0425/inequality_related/tree/main/Inequality_sharing_economy/experiments/scripts/experiments_analysis_tools.py) </i>)
* Outlier detection for experiment 4 can be accessed at [E4_outlier_analysis_mahalanobis_distance.ipynb](https://github.com/jinyan0425/inequality_related/tree/main/Inequality_sharing_economy/experiments/scripts/E4_outlier_analysis_mahalanobis_distance.ipynb).

### Libraries
* ```pandas```,```numpy```,
* ```matplotlib```,```seaborn```,
* ```sklearn```,```statsmodels```,```scipy```
* ```censusdata```,```pytrends```
