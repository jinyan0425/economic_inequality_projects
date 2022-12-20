#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPTS FOR ANALYZING EXPERIMENT DATA

@author: JINYANXIANG
"""

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display

from scipy.stats import chi2_contingency, fisher_exact
from pyprocessmacro import Process
import pingouin as pg


class experiment:
    #get the metadata of the experiment to retrive design information
    meta_data = pd.read_csv('meta_data.csv', index_col = 0).fillna('NA')
    
    def __init__ (self, filepath, exp_title):
        
        """
        
        Attributes
        ----------
        filepath : STRING, the filepath of the experiment result data
        
        
        exp_title: STRINGS, 
        the title of the experiment

        """

        
        #read the dataset of the experiment
        self.filepath = filepath
        self.data = pd.read_csv(self.filepath)
        
        #retrieve the variables from the dataset
        manipulation_var = []
        mediator_var = []
        dv_var = []
        
        for var in self.data.columns:
            if var.startswith('check'):
                manipulation_var.append(var)
            if var.startswith('med'):
                mediator_var.append(var)
            if var.startswith('dv'):
                dv_var.append(var)
        
        #assign the instance attributes
        self.manipulation_var = manipulation_var
        self.mediator_var = mediator_var
        self.dv_var = dv_var
        
        self.focal_condition = experiment.meta_data.loc[exp_title,'focal_con']
        self.moderation_condition = experiment.meta_data.loc[exp_title,'mod_con']
        self.exp_title = exp_title
        self.exp_des = experiment.meta_data.loc[exp_title,'exp_info']
    
   
#GET DESIGN, SAMPLE AND VARIABLE INFORMATION OF A GIVEN EXPERIMENT    
    def get_design_info(self):
        print('Experiment Information is: ' + self.exp_des)
        
    
    def get_sample_info(self):
        print('Sample Size = {}'.format(self.data.shape[0]))
        
        if self.moderation_condition == 'NA':
            print(self.data.groupby([self.focal_condition]).size())
        else:
            print(self.data.groupby([self.focal_condition, self.moderation_condition]).size())   
    
    def get_condition_level(self, condition):
        df = self.data
        condition_dict = {}
        
        for i in range(df[condition].unique().shape[0]):
            condition_dict['level{} for condition-{}'.format(i+1, condition)]=(df[condition+'_des'].unique()[i]+' '+condition, df[condition].unique()[i])
        
        return condition_dict
        
    def get_var_info(self):
        print("Variables are:\n",
              "focal condition: {} \n".format(self.focal_condition),
              "moderation condition: {}\n".format(self.moderation_condition),
              "manipulation measure(s): {}\n".format(self.manipulation_var),
              "mediator measure(s): {}\n".format(self.mediator_var),
              "dv measure(s): {}\n".format(self.dv_var))

        
#CHECK DV DISTRIBUTION
    def get_var_distribution(self, var, condition, bin_num):

        sns.displot(self.data, x = var, bins = bin_num, hue = condition)
        
        return plt.savefig('distribution of {} in {}'.format(var, self.exp_title))
        

        
 #CHECK RELIABILITY OF MEAURES   
    def reliability_check (self, var_list, var_des):
        '''
        Parameters
        ----------
        var_list : LIST
            item of a measure (N of items >=2)
        var_des : STRING
            the measure name

        Returns
        -------
        print the reliability information of the measures
        correlation for two-item measure
        cronbach's alpha for the  three-or-more-item measures

        '''
        
        df = self.data
        
        if len(var_list) == 1:
            print('Single item & reliability check is not applicable')
        
        elif len(var_list) == 2:
            correlation = pg.corr(df[var_list[0]], df[var_list[1]]).round(3)
            print('Pearson Correlation = {} (2 items, {})'.format(correlation['r'].values[0], var_des))
    
        else:
            reliability = round(pg.cronbach_alpha(data=df[var_list])[0],3)
            print('Crohnbach Alpha = {} ({} items, {})'.format(reliability, len(var_list), var_des))
            
            
#CALCULATED MEASURE MEANS (WHEN MULTIPLE ITEMS ARE USED)    
    def compute_measure_mean(self, var_list, var_name):
        '''
        Parameters
        ----------
        var_list : LIST
            list of measures, mean of which to be calculated.
        var_name : STRING
            the variable name for the calculated measure

        Returns
        -------
        DATAFRAME
            the updated dataframe with the calcualted measure mean.

        '''    
        df = self.data
        
        df[var_name] = np.nanmean(df[var_list], axis = 1)
        
        self.data = df

        return self.data

    
#Get CORRELATION MATRIX
    def get_correlation_matrix(self, var_list, method):
        
        df = self.data
        
        matrix = df[var_list].corr(method = method)
        
        return matrix
        

#TWO-SAMPLE T TEST FOR MANIPULATION CHECK  
    def two_sample_t_test(self, var, condition, description): ##t-test
        '''
        Parameters
        ----------
        var: STRING
            the depdendent variable in the two-sample t-test
        condition : STRING
            the group information for the two sample t-test
        description: STRING
            description of the variable(s)(e.g., manipulation check)

        Returns
        -------
        DATAFRAMES
            the first table shows the descriptive statistics
            the second table shows the t-test statistics
       '''
        df = self.data
        
        levels = list(self.get_condition_level(condition).values())
        
        des = df.groupby([condition])[var].describe()

        sample1 = df[df[condition] == levels[0][1]][var]
        sample2 = df[df[condition] == levels[1][1]][var]
        
        t_test_results = pg.ttest(sample1, sample2,  paired = False, alternative = 'two-sided').round(3)
    
        print (description + ' Results (independent t-test, two-sided) \n')
    
        return display(des, 
                       t_test_results)  
    
    
#ONE-WAY ANOVA FOR TESTING THE GROUP DIFFERENCE BY CONDTIONS IN MANIPULATION, MEDIATOR(S) AND DV(S) 
    def one_way_anova(self, var_list, condition, description): #one-way avova
        '''
        Parameters
        ----------
        var_list : LIST
            the list of variables (mediator(s) or dv(s))
        condition : STRING
            the group for the one-way anova
        description: STRING
            description of the dependent variable(s)(e.g., mediator, dv1, dv2)

        Returns
        -------
        DATAFRAMES
            the first table shows the descriptive statistics
            the second table shows the F-test statistics

        '''  
        df = self.data
    
        df['var_mean'] = np.nanmean(df[var_list], axis = 1)
        
        des = df.groupby([condition])['var_mean'].describe()
        
        
        one_way_anova_results= pg.anova(data = df, dv = 'var_mean', 
                                 between = condition, effsize = 'n2', detailed = True).round(3)
    
        print (description + ' Results (one-way ANOVA) \n')
        
        
        return display(des, one_way_anova_results)


#N-WAY ANOVA FOR TESTING THE GROUP DIFFERENCE BY CONDTIONS IN MANIPULATION, MEDIATOR(S) AND DV(S) 
    def n_way_anova(self, var_list, condition_list, description): #one_way_anova
        '''
        Parameters
        ----------
        var_list : LIST
            the list of variables (mediator(s) or dv(s))
        condition_list : List
            the list of groups for the n-way anova
        description : STRING
            description of the dependent variable(s)(e.g., mediator, dv1, dv2)

        Returns
        -------
        DATAFRAMES
            the first table shows the descriptive statistics
            the second table shows the F-test statistics

        '''  
        df = self.data
    
        df['var_mean'] = np.nanmean(df[var_list], axis = 1)
        
        des = df.groupby(condition_list)['var_mean'].describe()
        
        n_way_anova_results = pg.anova(data = df, dv = 'var_mean', 
                                 between = condition_list, effsize = 'n2',
                                 ss_type = 3, detailed = True).round(3)
    
        print (description + ' Results ({}-way ANOVA) \n'.format(len(condition_list)))
        
        return display(des, n_way_anova_results)

    def one_way_ancova (self, var_list, condition, cov_list, description):
        df = self.data
        '''
        Parameters
        ----------
        var_list : LIST
            the list of variables (mediator(s) or dv(s))
        condition: STRING
            the group for the one-way anova
        cov_list : LIST
            the list of covariates
        description : STRING
            description of the dependent variable(s)(e.g., mediator, dv1, dv2) #not adjusted by covariates

        Returns
        -------
        DATAFRAMES
            the first table shows the descriptive statistics
            the second table shows the F-test statistics

        '''
        df['var_mean'] = np.nanmean(df[var_list], axis = 1)
        
        des = df.groupby(condition)['var_mean'].describe()
        
        one_way_ancova_results = pg.ancova(data = df, dv ='var_mean', covar = cov_list, between = condition)
        
        print (description + ' Results (one-way ANCOVA)')
        
        return display(des, one_way_ancova_results)
        

        
#TWO SAMPLE MEDIAN TEST
    def median_test(self, var, condition, description):
        '''
        Parameters
        ----------
        var : STRING
            the dependent variable
        condition : STRING
            the group for the median test (two group only)
        description: STRING
            description of the dependent variable(s)(e.g., mediator, dv1, dv2)

        Returns
        -------
        STRINGS
            the contigency table, chi-square tests (with or without Yates correction), nonparamatric Fisher exact test
        '''
        df = self.data
        
        median_cutoff = np.nanmedian(df[var])
        
        levels = list(self.get_condition_level(condition).values())
        
        sample1 = df[df[condition] == levels[0][1]][var]
        sample2 = df[df[condition] == levels[1][1]][var]
    
        df['median_coded'] = np.where(df[var] <=  median_cutoff, 0, 1)
        self.data = df
        
        def count_partition(sample, median_cutoff):
            
            count_equal_below = 0
            count_above = 0
            
            for value in sample:
                if value <= median_cutoff:
                    count_equal_below += 1
                else:
                    count_above +=1
            
            return count_equal_below, count_above
        
        below_l1, above_l1 = count_partition(sample1, median_cutoff)
        below_l2, above_l2 = count_partition(sample2, median_cutoff)
        
        contigency_table = np.array([[below_l1, below_l2], [above_l1, above_l2]])
        
        chi2, p, dof, ex = chi2_contingency(contigency_table, correction=False)
        chi2_Yates, p_Yates, dof_Yates, ex_Yates = chi2_contingency(contigency_table, correction=True)
        oddsratio_fisher, p_fisher = fisher_exact(contigency_table, alternative='two-sided')
        
        print(description + ' Results (median test) \n')
        print('the contigency table is \n{}\n'.format(pd.DataFrame(contigency_table, 
                                                                   columns = [levels[0][0],levels[1][0]], index = ['equal_below_median', 'above_median'])))
        print('Chi-square(df = {}) = {:.3f}, p = {:.3f}'.format(dof, chi2, p))
        print('Chi-square (with Yates Correction)(df = {}) = {:.3f}, p = {:.3f}'.format(dof_Yates, chi2_Yates, p_Yates))
        print('Fisher exact test p = {:.3f}'.format(p_fisher))
        
        
#CHI-SQUARED TEST FOR TESTING GROUP DIFFERENCE BY CONDITONS IN MANIPULATION (BINARY OUTCOME VARIABLE)
    def chi_squared_test(self, var, condition, description):
        '''
        Parameters
        ----------
        var : STRING
            the binary dependent variable
        condition : STRING
            the group for the chi-sqaured test
        description: STRING
            description of the dependent variable(s)(e.g., mediator, dv1, dv2)

        Returns
        -------
        DATAFRAMES
            the first table shows the descriptive statistics
            the second table shows the Chi-squared-test statistics
        '''
        observed, expected, stats = pg.chi2_independence(self.data,  x= condition, y = Y)
        
        print (description + ' independent Chi-squared test')
        
        return display(expected, stats)    
    
#VIRSUALIZE RESULTS (BAR PLOT for continous dv and COUNT PLOT for categocial dv)
    def get_bar_plot(self, dv, focal_condition, moderation_condition, legend_labels):
        '''
        Parameters
        ----------
        dv : STRING
            the dependent variable
        focal_condition, moderation_condition : STRING
            the groups, if there is no moderation condition, input None
        legend_labels: LIST
            a list of levels of the moderation condition, if if there is no moderation condition, put [] (an empty list)

        Returns
        -------
        Saved figure.
        '''
     
        ax = sns.barplot(data = self.data, x = focal_condition, y = dv, 
                         hue = moderation_condition, errorbar=('se'))
    
        dv_name = " ".join(dv.split("_")[1:]).capitalize()
    
        if moderation_condition == None:
            title_name = '{}: The Impacts of {} on {}'.format(self.exp_title,
                                                             focal_condition.capitalize(),dv_name)
        else:
            title_name = '{}: The Interaction between {} and {} on {}'.format(self.exp_title,
                                                                              focal_condition.capitalize(),
                                                                              moderation_condition.capitalize(),
                                                                              dv_name)
            og_legend_labels, _ = ax.get_legend_handles_labels()
            plt.legend(og_legend_labels, legend_labels, title = moderation_condition.capitalize())
        
        plt.title(title_name)
        
        plt.xlabel(focal_condition.capitalize())
        plt.xticks([0,1],['low', 'high'])
    
        plt.ylabel(dv_name)
    

     
        return plt.show()


#MODERATION & MEDIATION ANALYSIS, PROCESS BASED
##source: https://github.com/QuentinAndre/pyprocessmacro
    def moderation_mediation_analysis(self, X, M_list, Y, W, cov_list, model_num, logit_bool):
        '''
        Parameters
        ----------
        X: STRING
            the focal predictor (continous, if binary, pls coded as 0, 1, multicategorical is not available)
        M_list : LIST
            the list of mediator(s), countinous
        Y: STRING
            the focal outcome variable (of continous, logit_bool == True, if binary, logit_bool = False, multicategorical is not available)
        W: STRING
           the focal moderator (continous, if binary, pls coded as 0, 1, multicategorical is not available), if none, put 'NA'
        cov_list : LIST
            the list of covariates, if none, put [] (an empty list)
        model_num: INTEGAR
             the model number, see the model information at file:///Users/jinyanxiang/Downloads/andrewhayes.pdf
             note that this function only works for model 1,4, 7, 8, 14, 15, 59 that will be used by the current project
             for other models, please modify the code
        logit_bool: BOOLIAN
             indicate whether the outcome variable is binary or not

        Returns
        -------
        STRING
             the summary of results

        '''  
        if model_num == 1:
            process_results = Process(data= self.data, model = model_num, 
                                      x = X, y = Y, 
                                      m = W, 
                                      controls = cov_list,
                                      logit = logit_bool)
            
        if model_num == 4:
            process_results = Process(data= self.data, model = model_num, 
                                      x = X, y = Y, 
                                      m = M_list, 
                                      controls = cov_list,
                                      logit = logit_bool)
        
        if model_num in [7, 8, 14, 15, 59]:
            process_results = Process(data= self.data, model = model_num, 
                                      x = X, y = Y, 
                                      m = M_list, w = W, 
                                      controls = cov_list,
                                      logit = logit_bool)
        
        return process_results.summary()
