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
                        'ListingYear','ListingMonth','BorrowerState']]
    
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
   
 
#RUN ROBUST LINEAR MODEL 
## RLM was employed to address the potential issues related to heteroscedasticity and outliers stemming from the non-normal distribution of logarithmic average lending amount
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
## quantitle regression was employed as a robustness check as it is robust to non-normal distributions
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
     
        #define the model
        focal_model = 'Log_AveInvestment ~ Gini_1y_est + Income_1y_est + poverty_rate + ProsperScore + EstimatedReturn + Log_LoanDuration + Gini_1y_est*ProsperScore + Gini_1y_est*EstimatedReturn + C(ListingMonth) + C(BorrowerState)'
   
        
        print(red('The regression model is {}...\n'.format(focal_model),['bold']))
        
        
        print(red('\n...Running the fixed-effect RLM model with scaled data with non-scaled data for concrete interpretation (HuberT weight function, H1 heteroscedasticity robust covariance) and saving the model...\n',['bold']))
        m_rlm = rlm_reg(df, focal_model, HuberT, 'H1')
        #save_model(m_rlm,'rlm_non_scaled.pkl')
        print(m_rlm.summary())

        
        print(red('\n...Running the fixed-effect RLM model with scaled data (HuberT weight function, H1 heteroscedasticity robust covariance) and saving the model...\n',['bold']))
        m_rlm_scaled = rlm_reg(df_scaled, focal_model, HuberT, 'H1')
        #save_model(m_rlm_scaled,'rlm_scaled.pkl')
        print(m_rlm_scaled.summary())
        
        print(red('\n...Running the Quantile Regression with non-scaled data (q = 0.5) and saving the model...\n',['bold']))
        m_quant = quant_reg(df, focal_model, 0.5)
        #save_model(m_quant,'quantile_non_scaled.pkl')
        print(m_quant.summary())

        print(red('\n...Running the Quantile Regression with non-scaled data (q = 0.5) and saving the model...\n',['bold']))
        m_quant_scaled = quant_reg(df_scaled, focal_model, 0.5)
        #save_model(m_quant_scaled,'quantile_scaled.pkl')
        print(m_quant_scaled.summary())
        
        
        print(blue('...Dropping the extreme outliers (the top and bottom 1%)...\n',['bold']))
        df_scaled2 = data_wo_extereme_outliers('Log_AveInvestment', 1, 99, df_scaled)
        
        print(red('\n...Running the RLM model with extreme-outlier-excluded non-scaled data (1y, HuberT, H1) and saving the model...\n',['bold']))
        m_rlm_no_outlier = rlm_reg(df, focal_model, HuberT, 'H1')
        #save_model(m_rlm_no_outlier,'rlm_no_outlier.pkl')
        print(m_rlm_no_outlier.summary())
        
        print(red('\n...Running the  Quantile Regression model with extreme-outlier-excluded non-scaled data and saving the model...\n',['bold']))
        m_quant_no_outlier = quant_reg(m_quant_no_outlier, focal_model, 0.5)
        #save_model(m_quant_no_outlier,'quantile_no_outlier.pkl')
        print(m_quant_no_outlier.summary())


        print(blue('\n...Saving the focal RLM model: {}'.format(model_filepath),['bold']))
        save_model(m_rlm, model_filepath)
        
        print(blue('...Checking the residuals and homoscedasticity of the baseline model...\n',['bold']))
        check_residuals(m_rlm, 'RLM model with fixed month & state effect (Huber T Norm, H1 Cov)')
        
        print(red('Printing the focal RLM model \n'\
              'with fixed month and state effects\n'
              'using min_max normalization to address the different scale issue \n'\
              'and H1 covariance estimator to address the heteroscedasticity issue \n'\
              'and Huber T norm to address outlier issue\n',['bold']))
        
        print(m_rlm.summary())

        print(red('Focal RLM model with fixed month and state effects (Huber T Norm, H1 Cov) saved!',['bold']))

    
    else:
        print('Please provide the file path of the master dataframe'\
              'as the first argument and the file path of the pickle files to '\
              'save the model to as the second argument. \n\nExample: python3 '\
              'inequality_SE_archival_data_analysis.py df_SE_master.csv focal_rlm_model.pkl')


if __name__ == '__main__':
    main()
