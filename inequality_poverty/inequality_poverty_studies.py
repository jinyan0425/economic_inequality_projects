# -*- coding: utf-8 -*-
"""
SCRIPT FOR DATA PROCESSING & ANALYSIS (PROJECT: INEQUALITY & POVERTY)
author: Jinyan Xiang

"""

import sys
import os

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
from pingouin import mixed_anova
import scipy.stats 

from IPython.display import display
import os
import sys
from simple_colors import *



#LOAD DATA
##create dictionaries to recode demographic variables
dict_ideology_cat = {1:'conservative', 2:'Conservative', 3:'Conservative', 
                     4:'Neutral', 5:'Liberal', 6:'Liberal', 7:'Liberal', np.NaN:'Not reported'}
dict_affiliation = {1:'Republican', 2: 'Democratic', 
                        3: 'Indpendent', 4: 'Others', np.NaN: 'Not Reported'}
    
dict_edu = {1:'Less than high school', 2:'High school graduate', 
                3:'Some college', 4:'Two-year college degree', 5:'Four-year college degree',
                6:'Professional degree', 7:'Doctorate'}
    
dict_income = {1:'Less than 10,000', 2:'10,000 - 19,999', 3:'20,000 - 29,999',
                   4:'30,000 - 39,999', 5:'40,000 - 49,999', 6:'50,000 - 59,999',
                   7:'60,000 - 69,999', 8:'70,000 - 79,999', 9:'80,000 - 89,999',
                   10:'90,000 - 99,999', 11:'100,000 - 149,999', 12:'More than 150,000'}
dict_social_class = {1:'Working class', 2:'Lower-middle class', 3:'Middle class', 
                     4:'Upper-middle class', 5:'Upper class'}

dict_religion = {1:'Protestant', 2:'Catholic', 3:'Jewish', 4:'Mormon', 
                     5:'Muslim', 6:'Atheist', 7:'No religion',8:'Other religion',
                     9:'Agnostic'}

dict_gender = {1:'Male', 2:'Female'}

dict_hispanic = {1:'Yes', 2:'No'}
dict_ethnicity = {1:'White', 2:'Black or African American', 3:'American Indian or Alaska Native',
                      4:'Asian', 5:'Native Hawaiian or Pacific Islander', 6:'Other', np.NaN: 'Hispanic'}


def load_data(s1_filepath, s2_filepath, s3_filepath):
    '''
    Parameters
    ----------
    s1_filepath : STRING, filepath of study1 (survey study).
    s2_filepath : STRING, filepath of study2 (experiment study).
    s3_filepath : STRING, filepath of study3 (IAT study).

    Returns
    -------
    df_survey : DATAFRAME, the full dataframe of study 1.
    df_survey_country : DATAFRAME, the subset of the dataframe study 1 (across-country condition).
    df_survey_time : DATAFRAME, the subset of the dataframe study 1 (over-time condition).
    df_survey_demo : DATAFRAME, the subset of the dataframe study 1 (demographic information).
    df_exp : DATAFRAME, the full dataframe of study 2.
    df_iat : DATAFRAME, the full dataframe of study 3.

    '''
    df_survey = pd.read_csv(s1_filepath)
    ##recode the demographic information in study 1
    df_survey = df_survey.replace({'ideology':dict_ideology_cat, 'education':dict_edu,
                                   'income': dict_income, 'religion':dict_religion,
                                   'gender': dict_gender,
                                   'hispanic':dict_hispanic, 'ethnicity':dict_ethnicity})
    
    df_survey['id'] = df_survey.index
    
    df_survey_country = df_survey.query('condition == "country"').dropna(axis = 1).reset_index(drop=True)
    df_survey_time = df_survey.query('condition == "time"').dropna(axis = 1).reset_index(drop=True)
    df_survey_demo = df_survey[['id','ideology', 'education', 'income', 'religion', 'gender', 'age','hispanic', 'ethnicity']]
    
    
    df_exp = pd.read_csv(s2_filepath)
    ##recode the demographic information in study 2
    df_exp = df_exp.replace({'ideology':dict_ideology_cat,'income': dict_income,'gender': dict_gender})
    
    ##recode the demographic information in study 3    
    df_iat = pd.read_csv(s3_filepath)
    df_iat = df_iat.replace({'ideology':dict_ideology_cat, 'affiliation':dict_affiliation, 
                                'social_class':dict_social_class, 'income': dict_income,
                                'gender': dict_gender,
                                'hispanic':dict_hispanic, 'ethnicity':dict_ethnicity})
    
    return df_survey, df_survey_country, df_survey_time, df_survey_demo, df_exp, df_iat


#GET DEMOGRAPHIC INFORMATION
def get_demo(df, demo_var):
    '''
    Parameters
    ----------
    df : DATAFRAME, the dataframe of which demographic information will be returned
    demo_var : STRING, the demographic variable

    Returns
    -------
    demo_tb : DATAFRAME, the demographic information for the demographic variable
                         (if cateogrical/descrete demographic variable, return counts and percents;
                          if countinous demographic variable, return mean, sd and quantiles)
    '''
    print('demographic information for {}:'.format(demo_var))
    
    if df[demo_var].dtypes == 'object':
        demo_tb = df[demo_var].value_counts().reset_index(name = 'count')
        demo_tb['percent'] = round(demo_tb['count']/demo_tb['count'].sum(axis = 0), 3)
        demo_tb.rename(columns = {'index':demo_var}, inplace = True)
        demo_tb.sort_values(by = 'count')
    
    elif df[demo_var].dtypes in ['int64','float64']:
        demo_tb= df[demo_var].describe().reset_index(name = 'stat')
        demo_tb.rename(columns = {'index':demo_var}, inplace = True)

    else: 
        print('abnormal data type; recheck needed')
        demo_tb = pd.DataFrame()
    
    return demo_tb 



#STUDY 1 _ DATA TRANSFORMATION
def transform_survey_data(df_survey): 
    ##transform the wide format to long format that fits multi-level mixed effect analysis
    '''
    Parameters
    ----------
    df : DATAFRAME, the loaded and recoded dataframe of study 1
    Returns
    -------
    df_survey_transformed: DATAFRAME, the transformed dataframe  
   '''
    df_survey.reset_index(inplace = True)
    
    for col_name in df_survey.columns:
        if col_name.startswith('GINI'):
            if col_name.split('_')[2] in ['1','1.1']:
                new_col_name = 'rich_'+ str(int(float(col_name.split('_')[1])*10)) #to avoid confusion, multiply the original gini floats by 10 (e.g., 0.2 -> 2)
                df_survey.rename(columns = {col_name : new_col_name}, inplace = True)
                
            if col_name.split('_')[2] in ['2','2.1']:
                new_col_name = 'poor_'+ str(int(float(col_name.split('_')[1])*10))
                df_survey.rename(columns = {col_name : new_col_name}, inplace = True)
                
            if col_name.split('_')[2] in ['3','3.1']:
                new_col_name = 'others_'+ str(int(float(col_name.split('_')[1])*10))
                df_survey.rename(columns = {col_name : new_col_name}, inplace = True)
    
    df_temp1 = pd.wide_to_long(df_survey, stubnames= ['rich','poor','others'], i='index', j='gini',sep='_')
    df_temp1.reset_index(inplace = True)
    
    df_survey_transformed = df_temp1.melt(id_vars=['id','gini'], value_vars=['rich', 'poor','others'], 
                             var_name='income_group', value_name='perception')
    df_survey_transformed['condition'] = df_survey['condition'].values[0]
    
    return df_survey_transformed


def create_survey_master_data(df_sub1, df_sub2, df_demo):
    '''
    Parameters
    ----------
    df_sub1 : DATAFRAME, the cleaned subset of the dataframe study 1 (cross-country condition).
    df_sub2 : DATAFRAME, the cleaned subset of the dataframe study 1 (over-time condition).
    df_demo : DATAFRAME, the cleaned subset of the dataframe study 1 (demographic information).

    Returns
    -------
    df_survey_master : DATAFRAME, the cleaned master dataframe study 1
    df_survey_master_country : DATAFRAME, the cleaned master subset dataframe of the dataframe study 1 (ross-country condition).
    df_survey_master_time : DATAFRAME, the cleaned master subset dataframe of the dataframe study 1 (over-time condition).

    '''
    df_survey_master_temp = transform_survey_data(df_sub1).append(transform_survey_data(df_sub2))
    
    df_survey_master = df_survey_master_temp.merge(df_demo, how = 'inner', 
                                                   left_on = ['id'], right_on = ['id'])
    
    df_survey_master = df_survey_master.sort_values(by = 'id')
    
    df_survey_master['gini'] = df_survey_master['gini'].values/10
    
    df_survey_master['income_group_code'] = np.where(df_survey_master['income_group'] == 'rich',1,
                                                 (np.where(df_survey_master['income_group'] == 'poor',2,3)))
    
    df_survey_master_country = df_survey_master.query('condition == "country"')
    df_survey_master_time = df_survey_master.query('condition == "time"')
    
    return df_survey_master, df_survey_master_country, df_survey_master_time


#STUDY 1 _ DATA ANALYSIS
def get_survey_results_mixedLM(df): 
    ##generalized mixed linear model
    '''
    Parameters
    ----------
    df : DATAFRAME, the dataframe of data analysis

    Returns
    -------
    DATAFRAMES : the first table displays the descpritive statistics
                 the second table displays the generalized linear mixed effect results 
    '''
    model = smf.mixedlm("perception ~ C(income_group_code) * gini", 
                        df, groups = 'id').fit()
    
    des_stat = df.query('income_group_code!=3').groupby(['income_group','gini'])['perception'].describe()
    
    print('Coding information of income group: 1 = rich, 2 = poor \n')
    
    return display(des_stat, model.summary())

#STUDY 1 _ VIRSUALIZE RESULTS (LINE CHART)
def virsualize_survey_results(df, fig_num, condition):
    '''
    Parameters
    ----------
    df : DATAFRAME, the dataframe
    fig_num : INTEGAR, the figure number
    condition : STRING, the condition 

    Returns
    -------
    NONE :  saving the figure
    '''
    plt.figure(figsize = (18,12))
    
    title = 'Figure {}: Perceived Share of Popoulation in Each Income Tier \n at Different Levels of Economic Inequality (the $\it{}$ condition)'.format(fig_num, condition)
    
    plt.title(title,fontsize = 18, ha = 'center')
               
    plt.ylim([20,45])
               
    plt.ylabel("Perceived Share of Population", fontsize = 15)           
    plt.xlabel("Economic Inequality (measured in Gini)",fontsize = 15)
    
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 12)
               
    sns.lineplot(data = df.query('income_group != "others"'), 
                 x = "gini", y="perception", 
                 hue = 'income_group', style ='income_group',
                 markers=True, markersize=20, dashes=[(1, 1), (3, 3)],linewidth=5,
                 err_style='band', errorbar=('se',1))   
    
    plt.legend(loc=2, title = 'Income Group', title_fontsize = 'x-large', 
           labels = ['rich', 'poor'], fontsize = 'large')
    
    plt.savefig('Figure {}'.format(fig_num))



#STUDY 2 _ DATA TRANSFORMATION
def transform_exp_data(df_exp):
    '''
    Parameters
    ----------
    df_exp : DATAFRAME, the dataframeo of study 2
    
    Returns
    -------
    DATAFRAME : the transformed dataframe of study 2
    '''
    df_exp['id'] = df_exp.index
    df_exp_transformed = pd.wide_to_long(df_exp, 
                                         stubnames= 'perception',
                                         i = ['id', 'inequality'],
                                         j = 'income_group', sep = '_').reset_index()
    df_exp_transformed['income_group'] = np.where(df_exp_transformed['income_group'] == 1, 
                                                  'rich', 'poor')
    
    return df_exp_transformed


#STUDY 2 _ DATA ANALYSIS (using mixed ANOVA for 2*3 within-between subject design)
def get_exp_results_mixedANOVA(df):
    '''
    Parameters
    ----------
    df_exp : DATAFRAME, the transformed dataframe of study 2
    
    Returns
    -------
    DATAFRAMES: the first table displays the descpritive statistics
                the second table displays the ANOVA results (
                #pairwise comparisions on 6-pair interactions only were conducted in SPSS)
    '''
    des_stat = df.groupby(['inequality','income_group'])['perception'].describe()
    
    aov = mixed_anova(dv='perception', between='inequality',
                  within='income_group', subject='id', effsize="ng2", data=df)
    
    return display(des_stat, aov.round(3))


#STUDY 2 _ VIRSUALIZE RESULTS (BAR CHART)
def virsualize_exp_results(df, fig_num):
    '''
    Parameters
    ----------
    df : DATAFRAME, the dataframe
    fig_num : INTEGAR, the figure number

    Returns
    -------
    NONE : saving the figure
    '''
    df['inequality_w_gini'] = np.where(df['inequality'] == 'low', 'low: gini = 0.2',(np.where(df['inequality'] == 'medium', 'medium: gini = 0.4', 'high: gini = 0.6')))
    
    plt.figure(figsize = (18,12))
    
    title = 'Figure {}: Perceived Share of Popoulation in Each Income Tier at \n Different Levels of Economic Inequality (mixed-design experiment)'.format(fig_num)
    plt.title(title, fontsize = 20, ha = 'center')
    
    sns.barplot(data=df, x="income_group", y="perception", 
                hue="inequality_w_gini", hue_order= ['low: gini = 0.2','medium: gini = 0.4','high: gini = 0.6'])
    
    plt.ylabel("Perceived Share of Population", fontsize = 18)           
    plt.xlabel("Income Tiers",fontsize = 18)
    
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    
    plt.legend(loc=2, title = 'Levels of Economic Inequality', title_fontsize = 'x-large', fontsize = 'x-large')
    
    plt.savefig('Figure {}'.format(fig_num))


#STUDY 3 _ CREATE SUB-SAMPLE DATA FOR GROUP DIFFERENCE ANALYSIS
def create_sub_samples(df, group_condition, group):
    '''
    Parameters
    ----------
    df : DATAFRAME, the dataframe of study 3
    group_condition : STRING, the subgroup category (e.g., gender)
    group : STRING, the subgroup (e.g., females for gender)
 
    Returns
    -------
    TUPLE : the shape information of the subsample
    '''
    df_sub = df[df[group_condition] == group]
    df_sub.to_csv('df_iat_sub_'+group.lower()+'.csv')
    
    return df_sub.shape
 
#STUDY 3 _ DATA ANALYSIS
##main data analysis will be conductd using the IAT open-source tool http://iatgen.org/
##main pre-registration describes the exlusion criteria for data analysis https://osf.io/vtgux

##post-hoc analysis on group difference 
##(gender: males vs female, political affiliation - Republican vs. Democratics)
##means (m1, m2), sds (s1, s2) and sample size(n1, n2) are results from the iat analysis using the sub-samples
def get_t_test_results(m1, m2, s1, s2, n1, n2, des, test_num):
    '''
    Parameters
    ----------
    m1, m2, s1, s2: FLOAT, means of group 1 & 2, standard deviations of group 1 & 2
    n1, n2: INTEGAR, sample size of group 1 & 2
    des : STRING, the description information of the test
    test_num : INTEGAR or STRING, the order or name of the test
    Returns
    -------
    None, priting out the results
    '''
    df = n1+n2-2
    
    s_p_sq = ((n1-1)*(s1**2)+(n2-1)*(s2**2))/df
    s_p = np.sqrt(s_p_sq)
    
    m_diff = round(m1-m2, 3)
    
    t_score = (m1-m2)/(s_p*(np.sqrt(1/n1+1/n2)))
    
    p_value = scipy.stats.t.sf(abs(t_score), df = df)*2 ##two-sided test
    
    if p_value < 0.001:
        p_value_new = '< 0.001'
    else:
        p_value_new = '= {}'.format(round(p_value, 3))
        
    if p_value <= 0.05:
        conclusion = 'statistically significant'
    else:
        conclusion = 'NOT statistically significant'
    
    print('TEST {}: the difference between {} is {} (t({}) = {}, p {}, {})'.
          format(test_num, des, m_diff, df, round(t_score, 3), p_value_new, conclusion))
    

#SAVE PROCESSED DATASETS
def save_data(df, saved_filepath):
    
    '''
    Parameters
    ----------
    df : DATAFRAME, the cleaned dataframe (in each study)

    Returns
    -------
    None: saving the data
    '''
    
    #save the clean datasets into the work directory.
    df.to_csv(saved_filepath+'.csv', index=False)



def main():
    
    
    cwd = os.getcwd()
    print('The working directory is\n', cwd)
    
    print('\n')
    
    print(yellow('\n:::STUDY DESIGNS & DATA CAN BE FOUND AT https://osf.io/pyuek/?view_only=1ffd9c21463d403dbef6fa9fbb74a2c8:::\n',['bold','underlined']))
    
    print('\n')
    
    if len(sys.argv) == 4:

        s1_filepath, s2_filepath, s3_filepath = sys.argv[1:]
        

        print(blue('...Loading data...\n  STUDY1 - MULTI-LEVEL DESIGN SURVEY: {}\n  STUDY2 - MIXED DESIGN EXPERIMENT: {} \n  STUDY3 - IAT :{}\n'.format(s1_filepath, s2_filepath, s3_filepath),['bold']))
        df_survey, df_survey_country, df_survey_time, df_survey_demo, df_exp, df_iat = load_data(s1_filepath, s2_filepath, s3_filepath)


        print(red('\n:::STUDY 1: DATA PROCESSING AND ANALYSIS:::\n',['bold']))
        
        print(red('\n...transforming and processing the data of study 1...\n'))
        df_survey_master, df_survey_master_country, df_survey_master_time = create_survey_master_data(df_survey_country, df_survey_time, df_survey_demo)
        
        print(red('\n...running mixed-effects generalized liner model for the cross-country condition...\n'\
                  'fixed effects of income group (i.e., rich, poor, vs. others), the Gini index (i.e., 0.2 to 0.7) and the interaction between income group and Gini index \n'\
                  'random intercepts for participants to account for the clustering\n'))
        get_survey_results_mixedLM(df_survey_master_country.query('income_group_code != 3'))
        
        print(red('\n...virsualizing the results for the cross-country condition...\n'))
        virsualize_survey_results(df_survey_master_country, '1a', 'across-country')
        
        print(red('\n...running mixed-effects generalized liner model for the over-time condition & saving the figure...\n'))
        get_survey_results_mixedLM(df_survey_master_time.query('income_group_code != 3'))
        
        print(red('\n...virsualizing the results for the over-time condition & saving the figure...\n'))
        virsualize_survey_results(df_survey_master_country, '1b', 'over-time')
        
        print(red('\n...getting the demographic information of Study 1...\n'))
        for demo_var in df_survey_demo.columns[1:]:
            display(get_demo(df_survey_demo, demo_var))
        
        print(blue('\n...saving df_survey_master (full) for S1...\n'))
        save_data(df_survey_master,'df_survey_master')
        
        print(blue('\n...saving df_survey_master_time (over time) for S1...\n'))
        save_data(df_survey_master_time,'df_survey_master_time')
        
        print(blue('\n...saving df_survey_master_country (across country) for S1...\n'))
        save_data(df_survey_master_country,'df_survey_master_country')
        
        print(blue('\n...saving df_survey_demo (demo information) for S1...\n'))
        save_data(df_survey_demo,'df_survey_demo')
        
        
        print(red('\n:::STUDY 2: DATA PROCESSING AND ANALYSIS:::\n',['bold']))
        
        print(red('\n...transforming and processing the data of study 2...\n'))
        df_exp_transformed = transform_exp_data(df_exp)
        
        print(red('\n...running mixed-design ANOVA...\n'\
                  'within factor: income group (2 level), between-subject factor: economic inequality (3 level)\n'))
        get_exp_results_mixedANOVA(df_exp_transformed)
        
        print(red('\n...virsualizing the results for the mixed-design experiment..\n'))
        virsualize_exp_results(df_exp_transformed, 2)
        
        print(red('\n...getting the demographic information of Study 2...\n'))
        for demo_var in ['ideology', 'income', 'gender', 'age']:
            display(get_demo(df_exp, demo_var))
        
        print(blue('\n...saving df_exp_transformed (long format)for S2...\n'))
        
        
    
        print(red('\n:::STUDY 3: DATA PROCESSING AND POST-HOC ANALYSIS::\n',['bold']))
        
        print(red('\n...getting the demographic information of Study 3...\n'))
        for demo_var in ['gender', 'age', 'hispanic', 'ethinicity',
                 'social class', 'income',
                 'ideology', 'affiliation']:
            display(get_demo(df_iat, demo_var))
            
        print(red('\n...creating sub samples & saving the datasets: gender (male vs. female), political affiliation (republican vs. democratic)...\n'))
        create_sub_samples(df_iat, 'gender', 'Male')
        create_sub_samples(df_iat, 'gender', 'Female')
        create_sub_samples(df_iat, 'affiliation', 'Republican')
        create_sub_samples(df_iat, 'affiliation', 'Democratic')
        
        print(red('\n...main data analysis is conductd using the IAT open-source tool http://iatgen.org/' + '...\n'))
    
        print(red('\n...running the post-host t-test to detect group difference in D-score...\n'))
        
        des_gender_full = 'males and females (full sample)' 
        get_t_test_results(.21, .14, .33, .39, 69, 31, des_gender_full, 1)
        print('\n')
        
        des_gender_sub = 'males and females (sub-sample with participants who passed the exclusion criteria)' 
        get_t_test_results(.40, .40, .27, .40, 30, 13, des_gender_sub, 2)
        print('\n')
        
        des_aff_full = 'Republican and Democratic (full sample)' 
        get_t_test_results(.44, .09, .37, .30, 71, 18, des_aff_full, 3)
        print('\n')
        
        des_aff_sub = 'Republican and Democratic (sub-sample with participants who passed the exclusion criteria)'
        get_t_test_results(.48, .29, .28, .31, 20, 14, des_aff_sub, 4)
        print('\n')
        
        print(yellow(':::THIS IS THE END!:::\n\n',['bold','underlined']))
        
    else:
        print('Please provide the file paths of S1, S2, S3'\
              ' as the second, third and fourth arguments'\
              '\n\nExample: python3 inequality_poverty_studies.py S1_inequality_poverty_survey.csv S2_inequality_poverty_experiment.csv S3_inequality_poverty_IAT.csv')
    
            
if __name__ == '__main__':
    main()