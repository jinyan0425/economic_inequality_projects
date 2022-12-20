"""
SCRPIT FOR CREATING SHARING ECONOMY MATER DATA
@author: JINYAN XIANG
"""
import os
import sys

import pandas as pd 
import numpy as np

from census_data_collection import ACS_data #pacakge information can be at found https://github.com/jinyan0425/census_collection

    

def load_prosper_data(prosper_processed_filepath):
    '''
    Parameters
    ----------
    prosper_filepath : STRING, url path.
    Returns
    -------
    DATAFRAME, the prosper dataframe.
    '''
    df_prosper = pd.read_csv(prosper_processed_filepath)
    
    return df_prosper



def collect_ACS_data(year_range):
    '''
    Parameters
    ----------
    year_range : LIST of INTEGARS, years of ACS data to be collected.
    Returns
    -------
    DATAFRAME, the ACS dataframe with information collected.
    '''
    SE_geo = 'state: *' #please note that any space between state and ":" could cause errors
    SE_var_dict = {'B19083_001E':'Gini','B19013_001E':'Income'}
    
    api_key = "2ce6be4a8f71a52336616ae611a7979c33880c8b"
    
    SE_ACS_1y = ACS_data(1, SE_var_dict, SE_geo, api_key)
    SE_ACS_5y = ACS_data(5, SE_var_dict, SE_geo, api_key)
    
    df_SE_ACS_1y = SE_ACS_1y.collect_data_multi_years(year_range)
    df_SE_ACS_5y = SE_ACS_5y.collect_data_multi_years(year_range)
    
    df_SE_ACS_combined = df_SE_ACS_1y.merge(df_SE_ACS_5y, 
                                            on = ['GEO_ID','GEO_NAME','state','year'])
    
    
    return df_SE_ACS_combined
    
    
def create_mater_data(df_prosper, df_SE_ACS_combined):
    '''
    Parameters
    ----------
    DATAFRAMES : two dataframes to be merged.
    Returns
    -------
    DATAFRAME, merged dataframe.
    '''
    df_SE_master = df_prosper.merge(df_SE_ACS_combined, how = 'inner', 
                                    left_on =['state','ListingYear'],
                                    right_on = ['state','year'])
    return df_SE_master


#GET INO
def data_info(df_SE_master):
    
    print(df_SE_master.head())
    print(df_SE_master.info())
    

#SAVE DATA
def save_data(df_SE_master):
    
    '''
    Parameters
    ----------
    df : DATAFRAME, the cleaned dataframe.
    Returns
    -------
    None.
    '''
    
    df_SE_master.to_csv('df_SE_master.csv',index=False)


def main():
    
    cwd = os.getcwd()
    print('The working directory is\n', cwd)
    
    print('\n')
    
    if len(sys.argv) == 2:

        prosper_processed_filepath = sys.argv[1]
        

        print('Loading data...\n  PROSPER: {}\n'.format(prosper_processed_filepath))
        df_prosper = load_prosper_data(prosper_processed_filepath)
        
        print('Collecting data from ACS...\n')
        year_range = np.arange(2010, 2015,1)
        df_SE_ACS_combined = collect_ACS_data(year_range)

        print('Creating the master data...\n')
        df_master = create_mater_data(df_prosper, df_SE_ACS_combined)
        
        print('Saving data...\n')
        save_data(df_master)
        
        print('The master data saved to the work directory!\n\n')
        
        print('HERE IS THE SUMMARY of MASTER DATA\n')
        data_info(df_master)
    
    
    else:
        print('Please provide the file paths of Prosper and state-abbv dataset'\
              'as the second arguments'\
              '\n\nExample: python3 inequality_SE_master_data.py prosper_processed_data.csv')


if __name__ == '__main__':
    main()
