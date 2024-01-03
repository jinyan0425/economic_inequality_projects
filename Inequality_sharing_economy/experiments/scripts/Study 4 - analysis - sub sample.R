#import libraries
library(psych)
library(dplyr)
library(pwr)
library(pscl)
library(car)
library(effects)
library(effectsize)
library(ltm)
library(WebPower)

#load data
setwd('/Users/jinyanxiang/Desktop/revision_1/studies/') ##NEED TO REMOVE
data <- read.csv("S4_med_lending/Study 4 - data.csv")
View(data)

#recode categorical vars with values
data$gender_coded <- recode_factor(data$gender, 
                                   '1' = 'male', 
                                   '2' = 'female',
                                   '3' = 'non_binary_third_gender', 
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

#exclude participants who failed to correctly answered all recap questions at their second attempt (per pre-reg)
data_sub <- data %>% filter(recap_code == 1)
View(data_sub)
#ANAYSIS IS CONDUCTED WITH THE SUB SAMPLE#

#get the sample information
##condition
inequality_count_sub <- table(data_sub$inequality_coded) #low = 112 vs. high = 101
inequality_count_sub

##gender
gender_count_sub<-table(data_sub$gender_coded) #male = 114 vs. female = 94 vs. 4 not to say & 1 missing
gender_count_sub

##age
summary(data_sub$age) #43.55
sd(data_sub$age) #12.47

##exp
table(data_sub$exp_specific) #w/exp = 12 vs. w/o exp = 201
table(data_sub$exp_general) #w/exp = 7 vs. w/o exp = 206
table(data_sub$exp_any_coded) #w/exp = 18 vs. w/o exp = 195


#manipulation check
##calculate the manipulation check of inequality 
data_sub <- data_sub %>%
  mutate(manipulation_check = (check_1 + check_2)/2)


##reliability 
inequality_alpha_sub <- data_sub[, c('check_1', 'check_2')]
cor(inequality_alpha_sub) #.91

##check
leveneTest(manipulation_check ~ inequality_coded, data = data_sub) #p = .61
t.test(manipulation_check ~ inequality_coded, data = data_sub, 
       alternative = "two.sided", var.equal = TRUE) #t = -16.06, df = 211, p < .001

aggregate(manipulation_check ~ inequality_coded, data_sub, mean) #low = 3.33 vs. high = 5.81
aggregate(manipulation_check ~ inequality_coded, data_sub, sd) #low = 1.10 vs. high = 1.14


#main effect
##willingness
###descriptive
aggregate(willingness ~ inequality_coded, data_sub, mean) #low = 3.45 vs. high = 2.90
aggregate(willingness ~ inequality_coded, data_sub, sd) #low = 1.59 vs. high = 1.41

###type 2 ANOVA
willingness_sub <- lm(willingness ~ inequality_coded, data = data_sub)
Anova(willingness_sub, type = 2) #F(1,211) = 6.94, p = .009

###effect size & achieved power
eta_squared(willingness_sub, partial = TRUE) #eta-squared = .032, f = .182
pwr.anova.test(k = 2, n = 106, f = eta2_to_f(.032), sig.level = 0.05) #75.02% power

##lending_amount
###descriptive
aggregate(lending_amount ~ inequality_coded, data_sub, mean) #low = 1057.82 vs. high = 679.22
aggregate(lending_amount ~ inequality_coded, data_sub, sd) #low = 972.88 vs. high = 749.36

###distribution
hist(data_sub$lending_amount) #not normally distributed, therefore, median-test will be performed (per pre-reg)

###median test
mood.medtest(lending_amount ~ inequality_coded, 
             data = data_sub, exact = FALSE) #chi-square = 8.73, p = .003
###dichotomy lending amount and get descriptive
summary(data_sub$lending_amount) #median = 506
data_sub$lending_amount_coded <- ifelse(data_sub$lending_amount <= 506, 0, 1)
table(data_sub$lending_amount_coded, data_sub$inequality_coded) 
#<= median: low = 45 vs. high = 62
#> median: low = 67 vs. high = 39
#> 
###effect size & achieved power for Chi-square Test
w = sqrt(8.73/(211*1))
w #w = 0.203407
pwr.chisq.test(w=w, N = 211, df=1, sig.level=0.05) #84.01% power


###auxiliary ANOVA
lending_amount_sub <- lm(lending_amount ~ inequality_coded, data = data_sub)
Anova(lending_amount_sub, type = 2) #F(1,211) = 9.96, p = .002
eta_squared(lending_amount_sub, partial = TRUE) #eta-squared = .045, f = .217
pwr.anova.test(k = 2, n = 106, f = eta2_to_f(.045), sig.level = 0.05) #88.22% power


#mediation
##trustworthiness
###calculate 
data_sub <- data_sub %>%
  mutate(trustworthiness = (trustworthy + honest)/2)

###reliability
trustworthiness_alpha_sub <- data_sub[, c('trustworthy', 'honest')]
cronbach.alpha(trustworthiness_alpha_sub) #.91
cor(trustworthiness_alpha_sub) #.84

###descriptive
aggregate(trustworthiness ~ inequality_coded, data_sub, mean) #low = 4.17 vs. high = 3.61
aggregate(trustworthiness ~ inequality_coded, data_sub, sd) #low = 1.23 vs. high = 1.13

###type 2 ANOVA
trustworthiness_sub <- lm(trustworthiness ~ inequality_coded, data = data_sub)
Anova(trustworthiness_sub, type = 2) #F(1,211) = 12.13, p < .001

###effect size & achieved power
eta_squared(trustworthiness_sub, partial = TRUE) #eta-squared = .054, f = .240
pwr.anova.test(k = 2, n = 106, f = eta2_to_f(.054), sig.level = 0.05) #93.36% power

##perceived risk
data_sub <- data_sub %>%
  mutate(risk = (risk_1 + risk_2)/2)

###reliability
risk_alpha_sub <- data_sub[, c('risk_1', 'risk_2')]
cronbach.alpha(risk_alpha_sub) #.97
cor(risk_alpha_sub) #.93

###descriptive
aggregate(risk ~ inequality_coded, data_sub, mean) #low = 4.69 vs. high = 5.33
aggregate(risk ~ inequality_coded, data_sub, sd) #low = 1.41 vs. high = 1.20

###type 2 ANOVA
risk_sub <- lm(risk ~ inequality_coded, data = data_sub)
Anova(risk_sub, type = 2) #F(1,211) = 12.86, p < .001

###effect size & achieved power
eta_squared(risk_sub, partial = TRUE) #eta-squared = .057, f = .0.246
pwr.anova.test(k = 2, n = 106, f = eta2_to_f(.057), sig.level = 0.05) #94.56% power


#Serial Mediation Analysis
process (data = data_sub, 
         y = 'willingness', 
         x = 'inequality', 
         #m = c('trustworthiness','risk'),
         m = c('risk','trustworthiness'),
         model = 6,
         effsize = 1,
         total = 1,
         stand = 1,
         boot = 10000,
         modelbt = 1,
         seed = 1)

process (data = data_sub, 
         y = 'lending_amount_coded', 
         x = 'inequality', 
         #m = c('trustworthiness','risk'),
         m = c('risk','trustworthiness'),
         model = 6,
         effsize = 1,
         total = 1,
         stand = 1,
         boot = 10000,
         modelbt = 1,
         seed = 1)

process (data = data_sub, 
         y = 'lending_amount', 
         x = 'inequality', 
         #m = c('trustworthiness','risk'),
         m = c('risk','trustworthiness'),
         model = 6,
         effsize = 1,
         total = 1,
         stand = 1,
         boot = 10000,
         modelbt = 1,
         seed = 1)
