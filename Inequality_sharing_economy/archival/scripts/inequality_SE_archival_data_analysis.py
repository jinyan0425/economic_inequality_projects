#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRPIT FOR VIRSUALIZE THE DISTRIBUTIONS OF VARIABLES RUNNING REGRESSION MODEL

@author: JINYAN XIANG
"""

import sys
import pickle

import pandas as pd 
import numpy as np
from simple_colors import *

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler ##import other methods as needed

from statsmodels.formula.api import rlm, ols, quantreg
from statsmodels.robust.norms import HuberT, AndrewWave ##import other methods as needed
from statsmodels.stats.api import het_breuschpagan
from statsmodels.compat import lzip

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


#LOAD DATA
def load_master_data(master_data_filepath):
    
    '''
    Parameters
    ----------
    master_data_filepath : STRING, url path.

    Returns
    -------
    DATAFRAME, the master dataframe.
    '''
    df = pd.read_csv(master_data_filepath)
    
    return df


#FEATURE SCALING
def scaling(df, scaler):
    '''
    Parameters
    ----------
    df: DATAFRAME, the dataframe to be transformed
    scaler: METHOD, method for scaling
    Returns
    -------
    DATAFRAME,the scaled dataframe
    ''' 
    df_for_scaling = df[['Gini_1y_est', 'Income_1y_est','Gini_5y_est', 'Income_5y_est',
                         'EstimatedReturn','ProsperScore',
                         'LoanDuration','Log_LoanDuration',
                         'LoanOriginalAmount','Investors']]
    df_no_scaling = df[['AveInvestment','Log_AveInvestment',
                        'ListingYear','ListingMonth','BorrowerState',
                        'AmountCategory_1k','AmountCategory_1_5k','AmountCategory_2k']]
    
    scaler_norm = scaler()
    scaled_data = scaler_norm.fit_transform(df_for_scaling)
    
    columns_for_scaling = df_for_scaling.columns
    
    df_scaled_temp = pd.DataFrame(scaled_data, columns=columns_for_scaling)
    df_scaled = df_scaled_temp.join(df_no_scaling)
    
    return df_scaled


#DRAW & SAVE DISTRIBUTION PLOTS
def get_distribution(var,var_name,kde_bool,discrete_bool,angle,df):
    '''
    Parameters
    ----------
    var : STRING, the countinous variable for drawing the histogram.
    var_name: STRING, the variable name
    kde_bool: BOOLIAN, whether draw the distribution curve or not
    angle: INTEGAR, the rotation angle of x-ticks
    df: DATAFRAME, the dataframe.

    Returns
    -------
    NONE, saving the distribution plots
    ''' 
    sns.displot(data = df,x = var, kde=kde_bool, discrete = discrete_bool, height=5, aspect=2)
    
    plt.title('Distribution of '+var_name, fontsize = 18)
    
    plt.xticks(rotation = angle)
    plt.xlabel(var_name, fontsize = 12)
    plt.ylabel('Counts', fontsize = 12)
    
    plt.tight_layout()
    
    #plt.show()
    plt.savefig('distribution of '+var_name.lower())
    
    
#DROP THE EXTREME OUTLIERS
def data_wo_extereme_outliers(var, percentile_low,percentile_high, df):
    '''
    Parameters
    ----------
    var : STRING, the variable of which the extreme outliers will be dropped.
    percentile_low: INTEGAR or FLOAT, the lower-bound criteria for identifying the extreme outliers
    percentile_high: INTEGAR or FLOAT, the lhigher-bound criteria for identifying the extreme outliers
    df: DATAFRAME, the dataframe.

    Returns
    -------
    DATAFRAME, the master dataframe without extereme outliers.
    '''
    percentile_low = np.percentile(df[var], percentile_low)
    percentile_high = np.percentile(df[var], percentile_high)
    
    df_wo_extreme_outliers = df[(df[var] > percentile_low)&(df[var] < percentile_high)]
    
    return df_wo_extreme_outliers


#RUN OLS LINEAR MODEL
def ols_reg(df, model, cov_type):
    '''
    Parameters
    ----------
    df: DATAFRAME, the dataframe for running the model
    model: STRING, the model to be estimated
    cov_type: STRING, covariance estimators, options: HC0, HC1, HC2, HC3, 'nonrobust' (default)
    
    Returns
    -------
    FITTED OLS MODEL
    '''
    ols_model = ols(model, data = df).fit(cov_type = cov_type) 
    return ols_model
   
 
#RUN ROBUST LINEAR MODEL
def rlm_reg(df, model, m_estimation, cov_type):
    '''
    Parameters
    ----------
    df: DATAFRAME, the dataframe for running the model
    model: STRING, the model to be estimated
    m_estimation: METHOD, the robust criterion function for downweighting outliers
                          options: LeastSquares, HuberT, RamsayE, AndrewWave, 
                                  TrimmedMean, Hampel, and TukeyBiweight
    cov_type: STRING, covariance estimators, options: H0, H1(default), H2, H3
    
    Returns
    -------
    FITTED RL MODEL
    '''
    rlm_model = rlm(model, data = df, M = m_estimation()).fit(cov = cov_type) 
    return rlm_model


#RUN QUANTILE REGRESSION
def quant_reg(df, model, quantile):
    '''
    Parameters
    ----------
    df: DATAFRAME, the dataframe for running the model
    model: STRING, the model to be estimated
    quantile: FLOAT, the conditional quantile of the outcome variable to be estimated
    
    Returns
    -------
    FITTED QUANTILE MODEL 
    '''
    quant_model = quantreg(model, data = df).fit(q = quantile) 
    return quant_model



#CHECK RESIDUAL DISTRIBUTION
def check_residuals(model, model_info):
    '''
    Parameters
    ----------
    model: MODEL, the regression model.
    model_info : STRING, the model information.

    Returns
    -------
    None.
    '''
    plt.figure(figsize=(20, 10))
    plt.scatter(x = model.model.endog, y = model.resid)
    
    plt.title('Fitted values vs. Residuals for ' + model_info, fontsize = 20)
    plt.xlabel('Fitted values', fontsize = 15)
    plt.ylabel('Residuals', fontsize = 15)
    
    #plt.show()
    plt.savefig('fitted values vs residuals plot of ' + model_info)
    
    BP_test = het_breuschpagan(model.resid, model.model.exog)
    BP_names = ['Lagrange multiplier statistic', 'p-value','f-value', 'f p-value']
    
    print('Breusch-Pagan Test Results:', lzip(BP_names, BP_test))



#SAVE THE MODEL
def save_model(model, model_filepath):
    '''
    Parameters
    ----------
    model: MODEL, the regression model.
    model_filepath : STRING, the pickle file path.

    Returns
    -------
    None.

    '''
    with open(model_filepath, 'wb') as file:
         pickle.dump(model.summary(), file)




def main():
    
    if len(sys.argv) == 3:
        
        master_data_filepath, model_filepath = sys.argv[1:]
        
        print(blue('...Loading data...\n    MASTER: {}'.format(master_data_filepath),['bold']))
        df = load_master_data(master_data_filepath)
        
        
        print(blue('\n...Scaling the data with min-max normalization... and saving the scaled dataset...\n',['bold']))
        df_scaled = scaling(df, MinMaxScaler)
        df_scaled.to_csv('df_SE_master_scaled.csv')
        
        
        print(blue('...Saving histogram of the average lending amount(raw and log transformed)...\n',['bold']))
        get_distribution('AveInvestment','Average Lending Amount',True, False,0,df)
        get_distribution('Log_AveInvestment','Average Lending Amount(log transformed)',True, False,0, df)
        
        print(blue('...Saving histogram of the focal predictor -- gini...\n',['bold']))
        get_distribution('Gini_1y_est','Gini index (1-year estimate)',True,False,0,df)
        
        print(blue('...Saving distribution of the focal moderator -- prosper score...\n',['bold']))
        get_distribution('ProsperScore','Prosper Score',False, True, 0,df)
        
        print(blue('...Saving distribution of other predictors -- estimated return & loan duration (raw and log transformed..\n',['bold']))
        get_distribution('EstimatedReturn','Estimated Return', True, False,0, df)
        get_distribution('LoanDuration','Loan Duration', True, False,0,df)
        get_distribution('Log_LoanDuration','Loan Duration (log transformed)', True, False,0,df)
        
        print(blue('...Saving distribution of the fixed effect factors...\n',['bold']))
        get_distribution('ListingYear', 'Listing Year',False, True, 0,df)
        get_distribution('BorrowerState', 'Borrower State',False, True,90,df)
        
        get_distribution('AmountCategory_1k','Amount Category_1k', False, True, 0, df)
        get_distribution('AmountCategory_1_5k','Amount Category_1_5k', False, True, 0, df)
        get_distribution('AmountCategory_2k','Amount Category_2k', False, True, 0, df)
        
        #using 1y-est Gini and income
        focal_model = 'Log_AveInvestment ~ Gini_1y_est*ProsperScore + Income_1y_est + Gini_1y_est*EstimatedReturn + Log_LoanDuration + C(ListingMonth) + C(BorrowerState) + Gini_1y_est*C(AmountCategory_1k)'
        #using 5y-est Gini and income for robustness check
        auxiliary_model = 'Log_AveInvestment ~ Gini_5y_est*ProsperScore + Income_5y_est + Gini_5y_est*EstimatedReturn + Log_LoanDuration + C(ListingMonth) + C(BorrowerState) + Gini_5y_est*C(AmountCategory_1k)'
        #using fixed year effect for robustness check
        auxiliary_model2 = 'Log_AveInvestment ~ Gini_1y_est*ProsperScore + Income_1y_est + Gini_1y_est*EstimatedReturn + Log_LoanDuration + C(ListingYear) + C(BorrowerState) + Gini_1y_est*C(AmountCategory_1k)'
        #using 1.5k as the cutoff point for 'Amount_Category'
        auxiliary_model3 = 'Log_AveInvestment ~ Gini_1y_est*ProsperScore + Income_1y_est + Gini_1y_est*EstimatedReturn + Log_LoanDuration + C(ListingMonth) + C(BorrowerState) + Gini_1y_est*C(AmountCategory_1_5k)'
        #using 2k as the cutoff point for 'Amount_Category'
        auxiliary_model4 = 'Log_AveInvestment ~ Gini_1y_est*ProsperScore + Income_1y_est + Gini_1y_est*EstimatedReturn + Log_LoanDuration + C(ListingMonth) + C(BorrowerState) + Gini_1y_est*C(AmountCategory_2k)'
        
        print(red('The focal model is {}...\n'.format(focal_model),['bold']))
        print(red('The first auxiliary model that uses 5 (vs. 1) year estimated Gini and Income is {}\n'.format(auxiliary_model),['bold']))
        print(red('The second auxiliary model that controls for fixed year (vs. month) effect is {}\n'.format(auxiliary_model2),['bold']))
        print(red('The third auxiliary model that uses 1.5k (vs. 1k) as the cutoff point for small-large lending amount category is {}\n'.format(auxiliary_model3),['bold']))
        print(red('The third auxiliary model that uses 2k (vs. 1k) as the cutoff point for small-large lending amount category is {}\n'.format(auxiliary_model4),['bold']))
        
      
        print(red('\n...Running the focal OLS model with fixed month & state effects (1y-est gini/income, HC3 heteroscedasticity robust covariance) and saving the model...\n',['bold']))
        print('for OLS models, use HC3 heteroscedasticity robust covariance to address heteroscedasticity \n')
        m_ols_focal = ols_reg(df_scaled, focal_model, 'HC3')
        #save_model(m_ols_focal,'ols_focal_month_state_full.pkl')
        print(m_ols_focal.summary())
        
        print(red('\n...Running the focal RLM model with fixed month & state effects (1y-est gini/income, HuberT weight function, H1 heteroscedasticity robust covariance) and saving the model...\n',['bold']))
        m_rlm_focal1 = rlm_reg(df_scaled, focal_model, HuberT, 'H1')
        #save_model(m_rlm_focal1,'rlm_focal_month_state_full(HuberT).pkl')
        print(m_rlm_focal1.summary())
        
        print(red('\n...Running the focal RLM model with fixed month & state effects (1y, AndrewWave, weight function, H1) and saving the model...\n',['bold']))
        m_rlm_focal2 = rlm_reg(df_scaled, focal_model, AndrewWave, 'H1')
        #save_model(m_rlm_focal2,'rlm_focal_month_state_full(AndrewWave).pkl')
        print(m_rlm_focal2.summary())
        
        print(red('\n...Running the focal Quantile Regression model with fixed month & state effects (1y, q = 0.5) and saving the model...\n',['bold']))
        m_quant_focal = quant_reg(df_scaled, focal_model, 0.5)
        #save_model(m_quant_focal,'quantile_focal_month_state_full(median).pkl')
        print(m_quant_focal.summary())
        
        print(red('\n...Running the auxiliary OLS model with fixed month & state effects (5y-est gini/income, HC3) and saving the model...\n',['bold']))
        m_ols_auxiliary = ols_reg(df_scaled, auxiliary_model, 'HC3')
        #save_model(m_ols_auxiliary,'ols_auxiliary_month_state_full (5y).pkl')
        print(m_ols_auxiliary.summary())
        
        print(red('\n...Running the auxiliary RLM model with fixed month & state effects (5y, HuberT, H1) and saving the model...\n',['bold']))
        m_rlm_auxiliary = rlm_reg(df_scaled, auxiliary_model, HuberT, 'H1')
        #save_model(m_rlm_auxiliary,'rlm_auxiliary_month_state_full(HuberT, 5y).pkl')
        print(m_rlm_auxiliary.summary())
        
        print(red('\n...Running the auxiliary Quantile Regression model with fixed month & state effects (5y, q = 0.5) and saving the model...\n',['bold']))
        m_quant_auxiliary = quant_reg(df_scaled, auxiliary_model, 0.5)
        #save_model(m_quant_auxiliary,'quantile_auxiliary_month_state_full (median, 5y).pkl')
        print(m_quant_auxiliary.summary())
        
        print(red('\n...Running the auxiliary RLM model with fixed year & state effects (1y, HuberT, H1) and saving the model...\n',['bold']))
        m_rlm_auxiliary2 = rlm_reg(df_scaled, auxiliary_model2, HuberT,'H1')
        #save_model(m_rlm_auxiliary2,'ols_auxiliary_year_state_full (HuberT).pkl')
        print(m_rlm_auxiliary2.summary())
        
        print(blue('...Dropping the extreme outliers (the top and bottom 1%)...\n',['bold']))
        df_scaled2 = data_wo_extereme_outliers('Log_AveInvestment', 1, 99, df_scaled)
        
        print(red('\n...Running the auxiliary OLS model with fixed month & state effects (1y, HC3, excluding outliers) and saving the model...\n',['bold']))
        m_ols_auxiliary2 = ols_reg(df_scaled2, focal_model, 'HC3')
        #save_model(m_ols_auxiliary2,'ols_auxiliary_month_state_wo_extreme_outliers.pkl')
        print(m_ols_auxiliary2.summary())
        
        print(red('\n...Running the auxiliary RLM model with fixed month & state effects (1y, HuberT, H1, 1.5k as the amount category cutoff point) and saving the model...\n',['bold']))
        m_rlm_auxiliary3 = rlm_reg(df_scaled, auxiliary_model3, HuberT, 'H1')
        #save_model(m_rlm_auxiliary3,'rlm_auxiliary_month_state_full(1.5k_cutoff).pkl')
        print(m_rlm_auxiliary3.summary())
        
        print(red('\n...Running the auxiliary RLM model with fixed month & state effects (1y, HuberT, H1, 2k) and saving the model...\n',['bold']))
        m_rlm_auxiliary4 = rlm_reg(df_scaled, auxiliary_model4, HuberT, 'H1')
        #save_model(m_rlm_auxiliary4,'rlm_auxiliary_month_state_full(1.5k_cutoff).pkl')
        print(m_rlm_auxiliary4.summary())
        
        print(red('\n...Running the auxiliary Quantile Regression model with fixed month & state effects (1y, q = 0.5, 1.5k) and saving the model...\n',['bold']))
        m_quant_auxiliary2 = quant_reg(df_scaled, auxiliary_model3, 0.5)
        #save_model(m_quant_auxiliary2,'quantile_auxiliary_month_state_full(1.5k_cutoff).pkl')
        print(m_quant_auxiliary2.summary())
        
        print(red('\n...Running the auxiliary Quantile Regression model with fixed month & state effects (1y, q = 0.5, 2k) and saving the model...\n',['bold']))
        m_quant_auxiliary3 = quant_reg(df_scaled, auxiliary_model4, 0.5)
        #save_model(m_quant_auxiliary3,'quantile_auxiliary_month_state_full(2k_cutoff).pkl')
        print(m_quant_auxiliary3.summary())
        
        
        print(blue('\n...Saving the focal RLM model: {}'.format(model_filepath),['bold']))
        save_model(m_rlm_focal1, model_filepath)
        
        print(blue('...Checking the residuals and homoscedasticity of the baseline model...\n',['bold']))
        check_residuals(m_rlm_focal1, 'RLM model with fixed month & state effect (Huber T Norm, H1 Cov)')
        
        
        print(red('Printing the focal RLM model \n'\
              'with fixed month and state effects\n'
              'using min_max normalization to address the different scale issue \n'\
              'and H1 covariance estimator to address the heteroscedasticity issue \n'\
              'and Huber T norm to address outlier issue\n',['bold']))
        
        print(m_rlm_focal1.summary())
        
        
    

        print(red('Focal RLM model with fixed month and state effects (Huber T Norm, H1 Cov) saved!',['bold']))

    
    else:
        print('Please provide the file path of the master dataframe'\
              'as the first argument and the file path of the pickle files to '\
              'save the model to as the second argument. \n\nExample: python3 '\
              'inequality_SE_archival_data_analysis.py df_SE_master.csv focal_rlm_fixed_month_state_model.pkl')


if __name__ == '__main__':
    main()