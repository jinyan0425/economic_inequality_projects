#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT FOR PROCESSING PROSPER DATA
author: Jinyan Xiang

"""
import sys
import warnings
import os

import pandas as pd
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore")

#LOAD DATA
def load_data(prosper_filepath, state_abbv_filepath):
    '''
    Parameters
    ----------
    prosper_filepath : STRING, url path of the dataframe (the original propser data)
    state_abbv_filepath : STRING, url path of the dataframe (state -abbv. information)

    Returns
    -------
    DATAFRAME, the prosper dataframe.
    '''
    df_prosper_og = pd.read_csv(prosper_filepath)
    df_state_abbv = pd.read_csv(state_abbv_filepath)
    
    #add the full name of states based on the abbrivations for future merge
    df_prosper_og = df_prosper_og.merge(df_state_abbv, how = 'inner', 
                                        left_on ='BorrowerState', 
                                        right_on = 'Abbv')
    
    df_prosper_og.drop('Abbv', axis = 1, inplace = True)
    
    
    return df_prosper_og
  

  
#CLEAN DATA
def clean_data(df_prosper_og):
    '''
    Parameters
    ----------
    df_prosper_og : DATAFRAME,the dataframe of the original propser data
    df_state_abbv : DATAFRAME,the dataframe for imputing state information

    Returns
    -------
    DATAFRAME, the cleaned dataframe.
    '''
    #copy the original dataset for data processing
    df_prosper = df_prosper_og.copy()
    
    #drop duplicated listings based on 'ListingNumber'
    df_prosper.drop_duplicates(subset = ['ListingNumber'],inplace = True)
    
    
    #drop listings with missing 'BorrowerState'(the proxy of the focal predictor) 
    # and 'ProsperScore' (the potential moderator, per data description, ProsperScore was not available for listings before July 2009) 
    df_prosper.dropna(subset = ['BorrowerState'],inplace = True)
    df_prosper.dropna(subset = ['ProsperScore'],inplace = True)
    
    
    #drop non-fully-funded listings (3 listings were funded over 100%)
    df_prosper = df_prosper.query('PercentFunded >= 1.0')
    
    
    #create time variables: 'ListingYear','ListingMonth','LoanDuration' 
    #based on ListingCreationDate'and 'ListingCreationDate'
    for i in ['ListingCreationDate','LoanOriginationDate']:
        df_prosper[i] = df_prosper[i].astype('datetime64') ## correct the data type
    
    df_prosper['ListingYear'] = df_prosper['ListingCreationDate'].dt.to_period('Y')
    df_prosper['ListingMonth'] = df_prosper['ListingCreationDate'].dt.to_period('M')
    
    ##'LoanDuration' refers to how long a listing was listed on Prosper (in Days)
    loan_duration = df_prosper['LoanOriginationDate'] - df_prosper['ListingCreationDate']
    df_prosper['LoanDuration'] = loan_duration/np.timedelta64(1, 'D')
    
    ##create 'Log_LoanDuration' because the raw data of 'LoanDuration' is right skewed
    df_prosper['Log_LoanDuration'] = np.log10(df_prosper['LoanDuration'])
    

    #create "AveInvestment" and "Log_AveInvestment" as the focal outcome variables
    df_prosper['AveInvestment'] = df_prosper['LoanOriginalAmount'] / df_prosper['Investors']
    df_prosper['Log_AveInvestment'] = np.log10(df_prosper['AveInvestment']) ## log_transformed becasue the raw data of 'AveInvestement' is right skewed
    
    ##draw the distribuiton of 'Log_AveInvestment'
    sns.displot(df_prosper['Log_AveInvestment'],height=5, aspect=2)
    

    #create a categorical variable to specify two groups in terms of the size of average investment 
    #because 'Log_AveInvestment' follows a iregular distribution
    #the focal categorization criteria is whether Log_AveInvestement <= 3 (or AveInvestment = 1000)
    #because this cutooff point is close the valley of the bi-modal distribution and easy for intepretation
    #63.10% of the data are below this cutoff point
    #for robustness check, aslo set cutoff points at 
    #AveInvestment = 1500 (64.41% below) and 2000 (66.35% below)
    df_prosper['AmountCategory_1k'] = np.where(df_prosper['AveInvestment'] <= 1000, 'small_amount', 'large_amount')
    df_prosper['AmountCategory_1_5k'] = np.where(df_prosper['AveInvestment'] <= 1500, 'small_amount', 'large_amount')
    df_prosper['AmountCategory_2k'] = np.where(df_prosper['AveInvestment'] <= 2000, 'small_amount', 'large_amount')

    return df_prosper



#GET INO
def data_info(df_prosper):
    
    print(df_prosper.head())
    
    print(df_prosper.info())
    
    
    
#SAVE DATA
def save_data(df_prosper):
    
    '''
    Parameters
    ----------
    df : DATAFRAME, the cleaned dataframe.

    Returns
    -------
    None.
    '''
    
    #save the clean dataset into the work directory.
    df_prosper.to_csv('propser_processed_data.csv',index=False)
    
    

def main():
    
    cwd = os.getcwd()
    print('The working directory is\n', cwd)
    
    print('\n')
    
    if len(sys.argv) == 3:

        prosper_filepath,state_abbv_filepath = sys.argv[1:]
        

        print('Loading data...\n  PROSPER: {}\n  STATE_ABBV: {}'.format(prosper_filepath,state_abbv_filepath))
        df = load_data(prosper_filepath, state_abbv_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n')
        save_data(df)
        
        print('Cleaned data saved to the work directory!\n\n')
        
        print('HERE IS THE SUMMARY of PROSPER DATA\n')
        data_info(df)
    
    
    else:
        print('Please provide the file paths of Prosper and state-abbv dataset'\
              'as the second and third arguments'\
              '\n\nExample: python3 inequality_SE_prosper_data.py prosper_loan_data.csv state_abbv.csv')


if __name__ == '__main__':
    main()

 