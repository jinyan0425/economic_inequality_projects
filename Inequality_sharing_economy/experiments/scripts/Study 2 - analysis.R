#import libraries
library(psych)
library(dplyr)
library(effectsize)
library(pscl)
library(pwr)
library(standardize)
library(Hmisc)


#load data
data <- read.csv("S2_main_accomodation/Study 2 - data.csv")
data <- read.csv('/Users/jinyanxiang/Desktop/revision_1/studies/S2_main_accomodation/Study 2 - data.csv')
View(data)


#impute missing value of experience
##two participants input wrong information about their experience
##I impute their values with median as the distribution of experience is not normal
hist(data$exp_length)
data$exp_length[is.na(data$exp_length)] <- median(data$exp_length, na.rm = TRUE)

#recode categorical vars with values
data$gender_coded <- recode_factor(data$gender, 
                               '1' = 'male', 
                               '2' = 'female', 
                               '3' = 'non_binary_third_gender', 
                               '4' = 'other')

data$bad_response_coded <- recode_factor(data$bad_response, 
                                         '0' = 'passed', 
                                         '1' = 'failed')

#get the sample information
##gender
gender_count<-table(data$gender_coded) #71 males vs. 65 females vs. 2 binary/third-gender vs. 1 not to say
gender_count

##age
summary(data$age) #37.12
sd(data$age) #9.55

##bad respondents
bad_response_count<-table(data$bad_response_coded) #108 passed vs. 31 failed
bad_response_count


#calculate inequality as the focal IV
data <- data %>%
  mutate(inequality_mean = (inequality_1 + inequality_2 + inequality_3)/3)

#reliability test of inequality
inequality_alpha <- data[, c('inequality_1', 'inequality_2', 'inequality_3')]
cronbach.alpha(inequality_alpha) #alpha = .77

#get the descriptive information of continuous focal IVs and DV
focal_cont_vars <- data[, c('willingness', 
                        'inequality_mean', 'inequality_1', 'inequality_2', 'inequality_3',
                        'exp_length','income')]

for (col_name in colnames(focal_cont_vars)) {
  col_data <- focal_cont_vars[[col_name]]
  col_mean <- mean(col_data, na.rm = TRUE)  
  col_sd <- sd(col_data, na.rm = TRUE)
  

  cat("Variable:", col_name, "\n")
  cat("Mean (M):", col_mean, "\n")
  cat("Standard Deviation (SD):", col_sd, "\n\n")
}
###Variable: willingness  Mean (M): 5.669065  Standard Deviation (SD): 1.293173 
###Variable: inequality_mean Mean (M): 5.067146  Standard Deviation (SD): 1.074653 


#get the correlation
cor_test_result <- cor.test(data$inequality_mean, data$willingness)
cor_test_result$estimate #r = -.186
cor_test_result$conf.int #95CI = [-0.34 -0.020]
cor_test_result$p.value #p = .029


## calculate achieved power
pwr.r.test(r = -.18551402, n = 139, sig.level = .05, alternative = 'two.sided') ###59.55% of power
pwr.r.test(r = -.18551402, n = 139, sig.level = .05, alternative = 'less') ###70.96% of power

#get the regression
model_cov <- lm(willingness ~ inequality_mean + exp_length + income + gender + race_White, data = data)
summary(model_cov)
standardize_parameters(model_cov)
confint(model_cov)


#repeat the analysis for participants passed the check
selected_data <- data %>% filter(bad_response == 0)
View(selected_data)

focal_cont_vars_selected <- selected_data[, c('willingness', 
                            'inequality_mean', 'inequality_1', 'inequality_2', 'inequality_3',
                            'exp_length','income')]

inequality_alpha_selected <- selected_data[, c('inequality_1', 'inequality_2', 'inequality_3')]
cronbach.alpha(inequality_alpha_selected) #alpha = .79

for (col_name in colnames(focal_cont_vars_selected)) {
  col_data <- focal_cont_vars_selected[[col_name]]
  col_mean <- mean(col_data, na.rm = TRUE)  
  col_sd <- sd(col_data, na.rm = TRUE)
  
  
  cat("Variable:", col_name, "\n")
  cat("Mean (M):", col_mean, "\n")
  cat("Standard Deviation (SD):", col_sd, "\n\n")
}
###Variable: willingness Mean (M): 5.685185 Standard Deviation (SD): 1.357899 
###Variable: inequality_mean  Mean (M): 5.055556  Standard Deviation (SD): 1.082764 

cor_test_result_selected <- cor.test(selected_data$inequality_mean, selected_data$willingness)
cor_test_result_selected$estimate #r = -.217
cor_test_result_selected$conf.int #95CI = [-0.39 -0.029]
cor_test_result_selected$p.value #p = .024

pwr.r.test(r = -.21682566, n = 108, sig.level = .05, alternative = 'two.sided') ###62.05% of power
pwr.r.test(r = -.21682566, n = 108, sig.level = .05, alternative = 'less') ###73.35% of power


model_cov_selected <- lm(willingness ~ inequality_mean + exp_length + income + gender + race_White, data = selected_data)
summary(model_cov_selected)
standardize_parameters(model_cov_selected)
confint(model_cov_selected)
