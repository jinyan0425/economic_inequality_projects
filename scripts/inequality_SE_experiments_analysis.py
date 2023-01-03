#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:37:11 2022

@author: jinyanxiang
"""


import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


from simple_colors import *
from experiments_analysis_tools import experiment



#EXPRIEMENT 1
##manipulation check
exp1_manipulation_pretest = experiment('E1_manipulation_pretest.csv','E1_pretest')

#the main effect study (manipulation: mock google search page, context: lodgesharing)
exp1 = experiment('E1_inequality_SE_lodge_main.csv', 'E1')

#EXPRIEMENT 2a/b
##the mediation studies (manipulation: income distribution, context: p2p lending)
exp2 = experiment('E2_inequality_SE_p2p_mediation.csv', 'E2')

#EXPRIEMENT 3a/b 
##the mediation studies (manipulation: income distribution (3a) & street image (3b), context: ride-sharing
exp3a = experiment('E3a_inequality_SE_ride_mediation(explict).csv', 'E3a')
exp3b = experiment('E3b_inequality_SE_ride_mediation(implict).csv', 'E3b')

#EXPRIEMENT 4
exp4 = experiment('E4_inequality_SE_lodge_moderation.csv', 'E4')


def main():
    print(yellow('\n:::ALL METHODS:::\n',['bold', 'underlined']))
    method_list = [method for method in dir(experiment) if method.startswith('__') is False]
    print(method_list)
    
    print('...\n...\n')
    
    print(red(':::ANALYSIS FOR EXPRIMENT 1::: \n', ['bold', 'underlined']))
    
    print(blue('...Getting design information of experiment 1 - manipulation pretest...\n',['bold']))
    exp1_manipulation_pretest.get_design_info()
    print(blue('\n...Getting sample information of experiment 1 - manipulation pretest...\n',['bold']))
    exp1_manipulation_pretest.get_sample_info()
    print(blue('\n...Getting variable information of experiment 1 - manipulation pretest...\n',['bold']))
    exp1_manipulation_pretest.get_var_info()
    
    print(blue('\n...Checking the reliability of the manipulation check measures...\n',['bold']))
    exp1_manipulation_pretest.reliability_check (exp1_manipulation_pretest.manipulation_var, 'manipulation check pretest for E1')
    print(blue('\n...Checking the manipulation in E1 pretest...\n',['bold']))
    exp1_manipulation_pretest.compute_measure_mean(exp1_manipulation_pretest.manipulation_var, 'manipulation_check')
    exp1_manipulation_pretest.two_sample_t_test('manipulation_check', exp1_manipulation_pretest.focal_condition, 'Manipulation Check')
    
    print('...\n...\n')
    
    
    print(blue('...Getting design information of experiment 1...\n',['bold']))
    exp1.get_design_info()
    print(blue('\n...Getting sample information of experiment 1...\n',['bold']))
    exp1.get_sample_info()
    print(blue('\n...Getting variable information of experiment 1...\n',['bold']))
    exp1.get_var_info()
    print(blue('\n...Checking the group difference by condition on DV (willingness) in experiment 1 (ANOVA)...\n',['bold']))
    exp1.one_way_anova(exp1.dv_var, exp1.focal_condition, 'Willingness to choose the lodge-sharing service (vs. a comparable hotel) (DV)')
    
    print('...\n...\n')
    
    print(red(':::ANALYSIS FOR EXPRIMENT 2::: \n', ['bold', 'underlined']))
    print(blue('...Getting design information of experiment 2...\n',['bold']))
    exp2.get_design_info()
    print(blue('\n...Getting sample information of experiment 2...\n',['bold']))
    exp2.get_sample_info()
    print(blue('\n...Getting variable information of experiment 2...\n',['bold']))
    exp2.get_var_info()
    
    print(blue('\n...Checking the reliability of the manipulation check in experiment 2...\n',['bold']))
    exp2.reliability_check(exp2.manipulation_var, 'manipulation check: perceived inequality')
    print(blue('\n...Checking the manipulation in experiment 2...\n',['bold']))
    exp2.compute_measure_mean(exp2.manipulation_var, 'manipulation_check') 
    exp2.two_sample_t_test('manipulation_check',exp2.focal_condition, 'Manipulation Check')
    
    print(blue('\n...Checking the reliability of the mediator in experiment 2...\n',['bold']))
    exp2.reliability_check(exp2.mediator_var, 'mediatior_interpersonal_trust')
    print(blue('\n...Checking the group difference by condition on mediator (interpersonal trust) in experiment 2... \n',['bold']))
    exp2.compute_measure_mean(exp2.mediator_var, 'mediatior_interpersonal_trust') 
    exp2.one_way_anova(['mediatior_interpersonal_trust'],exp2.focal_condition, 'Mediator')
    
    print(blue('\n...Plotting the distribution of the first DV lending amount...\n'))
    exp2.get_var_distribution(exp2.dv_var[0],exp2.focal_condition+'_des',30)
    print(blue('\n...Checking the group difference by condition on DV (lending amount) in experiment 2 (Median Test due to irregular distribution of the lending amount)... \n',['bold']))
    exp2.median_test(exp2.dv_var[0], exp2.focal_condition, 'Lending Amount in USD (DV1)')
    print(blue('\n...Checking the group difference by condition on DV (willingness) in experiment 2 (ANOVA)... \n',['bold']))
    exp2.one_way_anova([exp2.dv_var[1]], exp2.focal_condition, 'Willingness to lend (DV2)')
    
    print(blue('\n...Mediation Analysis experiment 2...\n',['bold']))
    exp2.moderation_mediation_analysis(exp2.focal_condition,['mediatior_interpersonal_trust'],'median_coded',
                                       'NA',[], 4, False)
    print('\n...')
    exp2.moderation_mediation_analysis(exp2.focal_condition, ['mediatior_interpersonal_trust'], exp2.dv_var[1], 
                                       'NA', [], 4, False)
    print('\n...')
    exp2.moderation_mediation_analysis(exp2.focal_condition, ['mediatior_interpersonal_trust'], exp2.dv_var[1], 
                                       'NA', [], 4, False)
    
    print('...\n...\n')

    
    print(red(':::ANALYSIS FOR EXPRIMENT 3a::: \n', ['bold', 'underlined']))
    print(blue('...Getting design information of experiment 3a...\n',['bold']))
    exp3a.get_design_info()
    print(blue('\n...Getting sample information of experiment 3a...\n',['bold']))
    exp3a.get_sample_info()
    print(blue('\n...Getting variable information of experiment 3a...\n',['bold']))
    exp3a.get_var_info()
    
    print(blue('\n...Checking the reliability of the manipulation check in experiment 3a...\n',['bold']))
    exp3a.reliability_check(exp3a.manipulation_var, 'manipulation check: perceived inequality')
    print(blue('\n...Checking the manipulation in experiment 3a...\n',['bold']))
    exp3a.compute_measure_mean(exp3a.manipulation_var, 'manipulation_check')
    exp3a.two_sample_t_test('manipulation_check',exp3a.focal_condition, 'Manipulation Check')
    
    print(blue('\n...Checking the reliability of the mediator in experiment 3a...\n',['bold']))
    exp3a.reliability_check(exp3a.mediator_var[0:1], 'mediatior_interpersonal_trust')
    print(blue('\n...Checking the group difference by condition on mediator (interpersonal trust) in experiment 3a... \n',['bold']))
    exp3a.compute_measure_mean(exp3a.mediator_var[0:1], 'mediatior_interpersonal_trust')
    exp3a.one_way_anova(['mediatior_interpersonal_trust'],exp3a.focal_condition, 'Mediator')
    
    print(blue('\n...Checking the group difference by condition on DV (willingness) in experiment 3a... \n',['bold']))
    exp3a.one_way_anova(exp3a.dv_var, exp3a.focal_condition, 'Willingness to choose the ride-sharing service over other available transportation (DV)')

    print(blue('\n...Mediation Analysis for experiment 3a (interpersonal trust only)...\n',['bold']))
    exp3a.moderation_mediation_analysis(exp3a.focal_condition, ['mediatior_interpersonal_trust'],
                                        exp3a.dv_var[0], 'NA',[], 4, False)
    print(blue('\n...Mediation Analysis for experiment 3a (test the alternative mechanism -- perceived safety)...\n',['bold']))
    exp3a.moderation_mediation_analysis(exp3a.focal_condition, ['mediatior_interpersonal_trust','med_safe'],
                                        exp3a.dv_var[0], 'NA', [], 4, False)
    
    print('...\n...\n')
    
    print(red(':::ANALYSIS FOR EXPRIMENT 3b::: \n', ['bold', 'underlined']))
    print(blue('...Getting design information of experiment 3b...\n',['bold']))
    exp3b.get_design_info()
    print(blue('\n...Getting sample information of experiment 3b...\n',['bold']))
    exp3b.get_sample_info()
    print(blue('\n...Getting variable information of experiment 3b...\n',['bold']))
    exp3b.get_var_info()
    
    print(blue('\n...Checking the reliability of the manipulation check in experiment 3b...\n',['bold']))
    exp3b.reliability_check(exp3b.manipulation_var, 'manipulation check: perceived inequality')
    print(blue('\n...Checking the manipulation in experiment 3b...\n',['bold']))
    exp3b.compute_measure_mean(exp3b.manipulation_var, 'manipulation_check')
    exp3b.two_sample_t_test('manipulation_check',exp3b.focal_condition, 'Manipulation Check')
    
    print(blue('\n...Checking the reliability of the mediator in experiment 3b...\n',['bold']))
    exp3b.reliability_check(exp3b.mediator_var, 'mediatior_interpersonal_trust')
    print(blue('\n...Checking the group difference by condition on mediator (interpersonal trust) in experiment 3b... \n',['bold']))
    exp3b.compute_measure_mean(exp3b.mediator_var, 'mediatior_interpersonal_trust')
    exp3b.one_way_anova(['mediatior_interpersonal_trust'],exp3b.focal_condition, 'Mediator')
    
    print(blue('\n...Checking the group difference by condition on DV (willingness) in experiment 3b',['bold']))
    print(blue('(controlling for the econ knowledge, which siginificantly varies across condition due a possible failure in randomization - replication study will be conducted to resolve the issues)...\n',['bold']))
               
    exp3b.compute_measure_mean(['econ','econ_inequality'], 'econ_knowledge')
    exp3b.one_way_ancova(exp3b.dv_var, exp3b.focal_condition, 'econ_knowledge', 'Willingness to choose the ride-sharing service over other available transportation (DV)')
     
    print(blue('\n...Mediation Analysis for experiment 3b (re-test the alternative mechanism -- perceived safety and control fro the econ_knowledge accoridngly)...\n',['bold']))
    exp3b.moderation_mediation_analysis(exp3b.focal_condition, ['mediatior_interpersonal_trust','med_safe'],
                                        exp3b.dv_var[0], 
                                        'NA',['econ_knowledge'], 4, False)
    
    print(red('\n ::: 3a and 3b suggest full mediation from economic inequality on willingness to use ridesharing service (consumers engagement in the sharinge economy), and also evidence that interpersonal trust is the mechanism after controlling for perceived safety, which is an alternative explanation ::: \n',['bold']))

    print('...\n...\n')
    
    print(red(':::ANALYSIS FOR EXPRIMENT 4::: \n', ['bold', 'underlined']))
    print(blue('...Getting design information of experiment 4...\n',['bold']))
    exp4.get_design_info()
    print(blue('\n...Getting sample information of experiment 4...\n',['bold']))
    exp4.get_sample_info()
    print(blue('\n...Getting variable information of experiment 4...\n',['bold']))
    exp4.get_var_info()
    
    print(blue('\n...Checking the manipulation in experiment 4...\n',['bold']))
    exp4.reliability_check([exp4.manipulation_var[0]], 'manipulation check: inequality')
    exp4.reliability_check([exp4.manipulation_var[1]], 'manipulation check: familiarity')
    
    exp4.two_sample_t_test(exp4.manipulation_var[0],exp4.focal_condition, 'Manipulation Check (inequality)')
    exp4.two_sample_t_test(exp4.manipulation_var[1],exp4.moderation_condition, 
                           'Manipulation Check (familiarity)')
    
    
    print(blue('\n...Checking the reliability of the mediator in experiment 4...\n',['bold']))
    exp4.reliability_check(exp4.mediator_var, 'Meidator')
    
    print(blue('\n...Checking the group difference by condition on mediator (interpersonal trust) in experiment 4... \n',['bold']))
    exp4.compute_measure_mean(exp4.mediator_var, 'mediatior_interpersonal_trust')
    exp4.n_way_anova(['mediatior_interpersonal_trust'],
                     [exp4.focal_condition,exp4.moderation_condition], 'Mediator')
    
    print(blue('\n...Checking the reliability of the DV (willingness) in experiment 4...\n',['bold']))
    exp4.reliability_check(exp4.dv_var, 'Willingness to serve to host (DV)')
    
    print(blue('\n...Checking the group difference by condition on DV (willingness) in experiment 4...\n', ['bold']))
    exp4.compute_measure_mean(exp4.dv_var, 'dv_willingness_to_serve_the_host')
    exp4.n_way_anova(exp4.dv_var,[exp4.focal_condition,exp4.moderation_condition], 'Willingness to serve the guest (DV)')
    
    exp4.moderation_mediation_analysis(exp4.focal_condition, 
                                   ['mediatior_interpersonal_trust'], 
                                   exp4.dv_var[0],
                                   exp4.moderation_condition, 
                                   [], 7, False)
    
    print(yellow(':::THIS IS THE END!:::\n\n',['bold','underlined']))
    


if __name__ == '__main__':
    main()
