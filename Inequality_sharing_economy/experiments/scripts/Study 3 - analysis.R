#import libraries
library(psych)
library(dplyr)
library(pwr)
library(pscl)
library(car)
library(effects)
library(effectsize)


#load data
setwd('/Users/jinyanxiang/Desktop/revision_1/studies/') ##NEED TO REMOVE
data <- read.csv("S3_main_tool_sharing/S3_tool_sharing.csv")
View(data)

#recode categorical vars with values
data$gender_coded <- recode_factor(data$gender, 
                                   '1' = 'male', 
                                   '2' = 'female',
                                   '5' = 'prefer_not_to_say')

data$inequality_coded <- recode_factor(data$inequality,
                                       '0' = 'low',
                                       '1' = 'high')

data$exp_specific_coded <- recode_factor(data$exp_specific,
                                    '1' = 'experienced',
                                    '2' = 'inexperienced')

data$exp_general_coded <- recode_factor(data$exp_general,
                                         '1' = 'experienced',
                                         '2' = 'inexperienced')

data$exp_any_coded <- recode_factor(data$exp_any,
                                    '1' = 'experienced',
                                    '2' = 'inexperienced')

data$exp_any_coded <- recode_factor(data$exp_any,
                                       '1' = 'experienced',
                                       '2' = 'inexperienced')


#get the sample information
##condition
inequality_count<-table(data$inequality_coded) #low = 111 vs. high = 91
inequality_count

##gender
gender_count<-table(data$gender_coded) #male = 91 vs. female = 109
gender_count

##age
summary(data$age) #41.02
sd(data$age) #12.72

##exp
table(data$exp_specific) #w/exp = 11 vs. w/o exp = 191
table(data$exp_general) #w/exp = 22 vs. w/o exp = 180
table(data$exp_any_coded) #w/exp = 25 vs. w/o exp = 177

#manipulation check
##calculate the manipulation check of inequality 
data <- data %>%
  mutate(manipulation_check = (check_1 + check_2)/2)

##reliability 
inequality_alpha <- data[, c('check_1', 'check_2')]
cronbach.alpha(inequality_alpha) #.93
cor(inequality_alpha) #.88

##check
leveneTest(manipulation_check ~ inequality_coded, data = data) #p = .07
t.test(manipulation_check ~ inequality_coded, data = data, 
       alternative = "two.sided", var.equal = TRUE) #t = -10.50, df = 200, p < .001

aggregate(manipulation_check ~ inequality_coded, data, mean) #low = 2.95 vs. high = 5.08
aggregate(manipulation_check ~ inequality_coded, data, sd) #low = 1.34 vs. high = 1.54


#main effect
##descriptive
aggregate(willingness ~ inequality_coded, data, mean) #low = 5.63 vs. high = 5.00
aggregate(willingness ~ inequality_coded, data, sd) #low = 1.43 vs. high = 1.53

##type 2 ANOVA
willingness <- lm(willingness ~ inequality_coded, data = data)
Anova(willingness, type = 2) #F(1,200) = 9.13, p = .003

##effect size & achieved power
eta_squared(willingness, partial = TRUE) #eta-squared = .044, f = .214
pwr.anova.test(k = 2, n = 101, f = eta2_to_f(.044), sig.level = 0.05) #85.86% power


#interaction between experience and inequality
##descriptive
aggregate(willingness ~ inequality_coded * exp_any_coded, data, mean) 
aggregate(willingness ~ inequality_coded * exp_any_coded, data, sd) 
#high: w/exp = 5.08(1.08) vs. w/o exp = 4.99(1.59)
#low: w/exp = 5.46(1.61) vs. w/o exp = 5.63(1.41)

##type 2 ANOVA as interaction does not work
willingness_int <- lm(willingness ~ inequality_coded * exp_any_coded, data = data)
Anova(willingness_int, type = 2)
