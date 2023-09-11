### Data Information
* Archival data can be downloaded on [Google drive](https://drive.google.com/drive/folders/1Y28XIQJmmVjROrPddpf78rF9-5Zcrs6B?usp=sharing). 
* Experimental data are uploaded. 
   *  Future deprecation warnings come with the pacakges installed (```pyprocessmacro```,```pingouin```) and will be fixed.

### Archival Data Processing & Analysis
* Step 1: <b> inequality_SE_prosper_data.py </b> to prepare the prosper dataset (dataset needed: <i> prosper_loan_data.csv </i> and <i> state_abbv.csv </i>

* Step 2: <b> inequality_SE_master_data.py </b> to create the master dataset (dataset needed: <i> propser_processed_data.csv </i>

* Step 3: <b> inequality_SE_archival_data_analysis.py </b> to analyze the archival data (dataset needed: <i> df_SE_master.csv </i>)

### Experimental Data Processing & Analysis
* <b> inequality_SE_experiments_analysis.py </b> to analyze the experimental data (dataset needed: all files in [data folder]([https://github.com/jinyan0425/inequality_related/tree/main/Inequality_sharing_economy/experiments/data](https://github.com/jinyan0425/inequality_related_projects/tree/sharing_economy/Inequality_sharing_economy/experiments/data)) needed; additional module needed: <i> experiments_analysis_tools.py </i>)

### Libraries
* ```pandas```,```numpy```,
* ```matplotlib```,```seaborn```,
* ```sklearn```,```statsmodels```,```pyprocessmacro```,```pingouin```,
* ```warnings```,```simple_colors```,```pickle```,```sys```,```os```
* [```census_data_collection```](https://github.com/jinyan0425/census_collection)
