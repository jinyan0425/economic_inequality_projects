#Import required libraries
library(sensemakr)
library(lmtest)
library(psych)
library(sandwich)
library(parameters)
library(sandwich)
library(clubSandwich)
library(miceadds)
library(mice)
library(nlme)
library(Metrics)



#Load data
## define function -- load_data
load_data <- function(file_path) {
  data <- read.csv(file_path)
  return(data)
}

## get data paths
df_attain_1y_path = '/Users/jinyanxiang/Desktop/Dissertation/Education/Archival Data Analysis/Github/data/df_attain_master_1y.csv'
df_enroll_1y_path = '/Users/jinyanxiang/Desktop/Dissertation/Education/Archival Data Analysis/Github/data/df_enroll_master_1y.csv'

df_attain_5y_path = '/Users/jinyanxiang/Desktop/Dissertation/Education/Archival Data Analysis/Github/data/df_attain_master_5y.csv'
df_enroll_5y_path = '/Users/jinyanxiang/Desktop/Dissertation/Education/Archival Data Analysis/Github/data/df_enroll_master_5y.csv'

## call data-loading function and load data
df_attain_1y = load_data(df_attain_1y_path)
df_attain_5y = load_data(df_attain_5y_path)
df_enroll_1y = load_data(df_enroll_1y_path)
df_enroll_5y = load_data(df_enroll_5y_path)



#Fit a GLM model to address the bounded nature of the proportion dvs
## define the function
m.glm <- function(data, response_variable) {
  formula <- as.formula(paste(response_variable, "~ gini + log10_income + population + unemployment + factor(Year) + State"))
  model <- glm(formula, data = data, family = quasibinomial(link = 'logit'))
  model_summary <- summary(model)
  return(list(model = model, summary = model_summary))
}



#Fit a GLM model with clustered errors to address the potential violation of equal variance
m.glm.cluster <- function(data, response_variable) {
  formula <- as.formula(paste(response_variable, "~ gini + log10_income + population + unemployment + factor(Year) + State"))
  model <- glm.cluster(formula, data = data, cluster = 'GEO_ID', family = quasibinomial(link = 'logit'))
  model_summary <- summary(model)
  return(list(model = model, summary = model_summary))
}



#Use HC5 (MacKinnon and White Method) to get estimation robust to heteroscedasticity, serial correlation, and clustering.
##It assumes a conditional heteroscedasticity structure and is suitable for panel data models where there may be heteroscedasticity and serial correlation.
##It is an improvement over HC0, HC1, and HC2 in terms of robustness to serial correlation.
##HC5 is less biased in the presence of serial correlation compared to some other methods.
m.glm.robust <- function(model){
  results <- coeftest(model, vcov = vcovHC(model, method = 'HC5', cluser = 'GEO_ID'))
  return(results)
}



#Auxiliary analysis: fit a Generalized Least Square model
##this will ignore the bounded nature of the DV
m.gls <- function(data, response_variable) {
  formula <- as.formula(paste(response_variable, "~ gini + log10_income + population + unemployment + factor(Year) + State"))
  model <- gls(formula, correlation = corSymm(form = ~ 1 | GEO_ID), na.action = na.omit,
               data = data) #general correlation structure is used
  model_summary <- summary(model)

  return(list(model = model, summary = model_summary))
}



#Conduct sensitivity analysis using point estimate
sensitivity <- function(gini_beta, gini_se, gini_dof, description) {
  point.estimate <- sensemakr(gini_beta, gini_se, gini_dof)
  message <- paste("Data, Model, Variable:", description)
  cat(message, "\n", "\n")
  sensitivity_summary <- summary(point.estimate)
  return(sensitivity_summary)
}



#Compare predicted value vs. actual value & model fit
compare.predicted.values <- function(data, response_variable, 
                                     model.glm, model.glm.cluster, model.gls){
  column_names <- c("Mean", "SD", "Range","RMSE")
  
  results_df <- data.frame(Row = character(0), Mean = numeric(0), SD = numeric(0), Range = character(0), stringsAsFactors = FALSE)
  
  values_to_calculate <- list(
    Actual = data[[response_variable]],  # Use double brackets to access the column by name
    GLM.Predicted = predict(model.glm, type = 'response'),
    GLM.Cluster.Predicted = model.glm.cluster$glm_res$fitted.values,
    GLS.Predicted = predict(model.gls, type = 'response')
  )
  
  for (i in 1:length(values_to_calculate)) {
    name <- names(values_to_calculate)[i]
    value <- values_to_calculate[[i]]
    row_name <- name
    mean_value <- mean(value, na.rm = TRUE)
    sd_value <- sd(value, na.rm = TRUE)
    range_value <- paste(range(value, na.rm = TRUE), collapse = " - ")
    actual_value <- data[[response_variable]]
    RMSE_value <- rmse(actual_value, value)
      
    results_df <- rbind(results_df, data.frame(Row = row_name, Mean = mean_value, SD = sd_value, Range = range_value, 
                                               RMSE = RMSE_value))
  }
  
  message <- paste("Actual vs. Predicted Values for", response_variable)
  cat(message, "\n", "\n")
  
  return(results_df)
}



# Call the function with your data and response variable
##Attainment: ba & higher, 18-24y
###Fit the model
####GLM model
glm.attain_ba_higher_18_24 <- m.glm(df_attain_1y, "ba_higher_18_24D")
####GLM model with robust error
glm.robust.attain_ba_higher_18_24 <-m.glm.robust(glm.attain_ba_higher_18_24$model)
####GML model with clustered sd
glm.cluster.attain_ba_higher_18_24 <- m.glm.cluster(df_attain_1y, "ba_higher_18_24D")
####GLS model
gls.attain_ba_higher_18_24 <- m.gls(df_attain_1y, "ba_higher_18_24D")

###Get model summary
glm.attain_ba_higher_18_24$summary
glm.robust.attain_ba_higher_18_24
glm.cluster.attain_ba_higher_18_24$summary
gls.attain_ba_higher_18_24$summary

###Get sensitivity analysis
sens.glm.attain_ba_higher_18_24 <- sensitivity(7.447, 0.3709, 8122,
                                               "One-year Estimated Education Attainment, GLM, Bachelor's or higher for population aged between 18 and 24")
sens.glm.robust.attain_ba_higher_18_24 <- sensitivity(7.447, 0.1987, 8122,
                                                      "One-year Estimated Education Attainment, GLM-robust std, Bachelor's or higher for population aged between 18 and 24")
sens.glm.cluster.attain_ba_higher_18_24 <- sensitivity(7.447, 0.4832, 8122,
                                                       "One-year Estimated Education Attainment, GLM-clustered std, Bachelor's or higher for population aged between 18 and 24")
sens.gls.attain_ba_higher_18_24 <- sensitivity(0.587, 0.0310, 8122,
                                               "One-year Estimated Education Attainment, GLS std, Bachelor's or higher for population aged between 18 and 24")

##Compare actual vs. predicted value
compare.predicted.values(df_attain_1y, "ba_higher_18_24D",
                         glm.attain_ba_higher_18_24$model, glm.cluster.attain_ba_higher_18_24$model, gls.attain_ba_higher_18_24$model)


##Attainment: ba & higher, 25-34y
###Fit the model
####GLM model
glm.attain_ba_higher_25_34 <- m.glm(df_attain_1y, "ba_higher_25_34D")
####GLM model with robust error
glm.robust.attain_ba_higher_25_34 <-m.glm.robust(glm.attain_ba_higher_25_34$model)
####GML model with clustered sd
glm.cluster.attain_ba_higher_25_34 <- m.glm.cluster(df_attain_1y, "ba_higher_25_34D")
####GLS model
gls.attain_ba_higher_25_34 <- m.gls(df_attain_1y, "ba_higher_25_34D")

###Get model summary
glm.attain_ba_higher_25_34$summary
glm.robust.attain_ba_higher_25_34
glm.cluster.attain_ba_higher_25_34$summary
gls.attain_ba_higher_25_34$summary

###Get sensitivity analysis
sens.glm.attain_ba_higher_25_34 <- sensitivity(9.202, 0.3137, 8122,
                                               "One-year Estimated Education Attainment, GLM, Bachelor's or higher for population aged between 25 and 34")
sens.glm.robust.attain_ba_higher_25_34 <- sensitivity(9.202, 0.1987, 8122,
                                                      "One-year Estimated Education Attainment, GLM-robust std, Bachelor's or higher for population aged between 25 and 34")
sens.glm.cluster.attain_ba_higher_25_34 <- sensitivity(9.202, 0.5580, 8122,
                                                       "One-year Estimated Education Attainment, GLM-clustered std, Bachelor's or higher for population aged between 25 and 34")
sens.gls.attain_ba_higher_25_34 <- sensitivity(0.886, 0.0609, 8122,
                                               "One-year Estimated Education Attainment, GLS std, Bachelor's or higher for population aged between 25 and 34")

##Compare actual vs. predicted value
compare.predicted.values(df_attain_1y, "ba_higher_25_34D",
                         glm.attain_ba_higher_25_34$model, glm.cluster.attain_ba_higher_25_34$model, gls.attain_ba_higher_25_34$model)


##Attainment: ba, 25plus
###Fit the model
####GLM model
glm.attain_ba_25plus <- m.glm(df_attain_1y, "ba_higher_25_plusD")
####GLM model with robust error
glm.robust.attain_ba_25plus <-m.glm.robust(glm.attain_ba_25plus$model)
####GML model with clustered sd
glm.cluster.attain_ba_25plus <- m.glm.cluster(df_attain_1y, "ba_higher_25_plusD")
####GLS model
gls.attain_ba_25plus <- m.gls(df_attain_1y, "ba_higher_25_plusD")

###Get model summary
glm.attain_ba_25plus$summary
glm.robust.attain_ba_25plus
glm.cluster.attain_ba_25plus$summary
gls.attain_ba_25plus$summary

###Get sensitivity analysis
sens.glm.attain_ba_25plus <- sensitivity(8.612, .0990, 8122,
                                               "One-year Estimated Education Attainment, GLM, Bachelor's for population aged over 25")
sens.glm.robust.attain_ba_25plus <- sensitivity(8.612, 0.1335, 8122,
                                                      "One-year Estimated Education Attainment, GLM-robust std, Bachelor's for population aged over 25")
sens.glm.cluster.attain_ba_25plus <- sensitivity(8.612, 0.3992, 8122,
                                                       "One-year Estimated Education Attainment, GLM-clustered std, Bachelor's for population aged over 25")
sens.gls.attain_ba_25plus <- sensitivity(0.341,  0.0309, 8122,
                                               "One-year Estimated Education Attainment, GLS std, Bachelor's for population aged over 25")

##Compare actual vs. predicted value
compare.predicted.values(df_attain_1y, "ba_higher_25_34D",
                         glm.attain_ba_25plus$model, glm.cluster.attain_ba_25plus$model, gls.attain_ba_25plus$model)


##Attainment: graduate or professional, 25plus
###Fit the model
####GLM model
glm.attain_grad_prof_25plus <- m.glm(df_attain_1y, "grad_prof_25_plusD")
####GLM model with robust error
glm.robust.attain_grad_prof_25plus <-m.glm.robust(glm.attain_grad_prof_25plus$model)
####GML model with clustered sd
glm.cluster.attain_grad_prof_25plus <- m.glm.cluster(df_attain_1y, "grad_prof_25_plusD")
####GLS model
gls.attain_grad_prof_25plus <- m.gls(df_attain_1y, "grad_prof_25_plusD")

###Get model summary
glm.attain_grad_prof_25plus$summary
glm.robust.attain_grad_prof_25plus
glm.cluster.attain_grad_prof_25plus$summary
gls.attain_grad_prof_25plus$summary

###Get sensitivity analysis
sens.glm.attain_ba_25plus <- sensitivity(8.717, 0.0106, 8122,
                                         "One-year Estimated Education Attainment, GLM, Graduate or Professional for population aged over 25")
sens.glm.robust.attain_ba_25plus <- sensitivity(8.717, 0.1775, 8122,
                                                "One-year Estimated Education Attainment, GLM-robust std, Graduate or Professional for population aged over 25")
sens.glm.cluster.attain_ba_25plus <- sensitivity(8.612, 0.5268, 8122,
                                                 "One-year Estimated Education Attainment, GLM-clustered std, Graduate or Professional for population aged over 25")
sens.gls.attain_ba_25plus <- sensitivity(0.249,  0.0193, 8122,
                                         "One-year Estimated Education Attainment, GLS std, Graduate or Professional for population aged over 25")

##Compare actual vs. predicted value
compare.predicted.values(df_attain_1y, "ba_higher_25_34D",
                         glm.attain_grad_prof_25plus$model, glm.cluster.attain_grad_prof_25plus$model, gls.attain_grad_prof_25plus$model)

##Enrollment: college & graduate, 18-24y
###Fit the model
####GLM model
glm.enroll_college_grad_18_24 <- m.glm(df_enroll_1y, "college_grad_18_24D")
####GLM model with robust error
glm.robust.enroll_college_grad_18_24 <-m.glm.robust(glm.enroll_college_grad_18_24$model)
####GML model with clustered sd
glm.cluster.enroll_college_grad_18_24 <- m.glm.cluster(df_enroll_1y, "college_grad_18_24D")
####GLS model
gls.enroll_college_grad_18_24 <- m.gls(df_enroll_1y, "college_grad_18_24D")


###Get model summary
glm.enroll_college_grad_18_24$summary
glm.robust.enroll_college_grad_18_24
glm.cluster.enroll_college_grad_18_24$summary
gls.enroll_college_grad_18_24$summary

###Get sensitivity analysis
sens.glm.enroll_college_grad_18_24 <- sensitivity(7.970, 0.2142, 7776,
                                               "One-year Estimated School Enrollment, GLM, College or graduate school enrollment for population aged between 18 and 24")
sens.glm.robust.enroll_college_grad_18_24 <- sensitivity(7.970, 0.2603, 7776,
                                                      "One-year Estimated School Enrollment, GLM-robust std, College or graduate school enrollment for population aged between 18 and 24")
sens.glm.cluster.enroll_college_grad_18_24 <- sensitivity(7.970, 0.7361, 7776,
                                                       "One-year Estimated School Enrollment, GLM-clustered std, College or graduate school enrollment for population aged between 18 and 24")
sens.gls.enroll_college_grad_18_24 <- sensitivity(0.4847, 0.0856, 7776,
                                               "One-year Estimated School Enrollment, GLS std, College or graduate school enrollment for population for population aged between 18 and 24")

##Compare actual vs. predicted value
compare.predicted.values(df_enroll_1y, "college_grad_18_24D",
                         glm.enroll_college_grad_18_24$model, glm.cluster.enroll_college_grad_18_24$model, gls.enroll_college_grad_18_24$model)

df_enroll_1y[["college_grad_18_24D"]]

##Enrollment: college, 3years plus
###Fit the model
####GLM model
glm.enroll_college_3plus <- m.glm(df_enroll_1y, "college_3_plusD")
####GLM model with robust error
glm.robust.enroll_college_3plus<-m.glm.robust(glm.enroll_college_3plus$model)
####GML model with clustered sd
glm.cluster.enroll_college_3plus <- m.glm.cluster(df_enroll_1y, "college_3_plusD")
####GLS model
gls.enroll_college_3plus <- m.gls(df_enroll_1y, "college_3_plusD")

###Get model summary
glm.enroll_college_3plus$summary
glm.enroll_college_3plus$summary
glm.robust.enroll_college_3plus
gls.enroll_college_3plus$summary

###Get sensitivity analysis
sens.glm.enroll_college_3plus <- sensitivity(5.678, 0.1726, 7776,
                                                  "One-year Estimated School Enrollment, GLM, College enrollment for population over 3 years")
sens.glm.robust.enroll_college_3plus <- sensitivity(7.970, 0.2047, 7776,
                                                         "One-year Estimated School Enrollment, GLM-robust std, College enrollment for population over 3 years")
sens.glm.cluster.enroll_college_3plus <- sensitivity(7.970, 0.5873, 7776,
                                                          "One-year Estimated School Enrollment, GLM-clustered std, College enrollment for population over 3 years")
sens.gls.enroll_college_3plus <- sensitivity(0.1387, 0.0433, 7776,
                                                  "One-year Estimated School Enrollment, GLS std, College enrollment for population over 3 years")

##Compare actual vs. predicted value
compare.predicted.values(df_enroll_1y, "college_3_plusD",
                         glm.enroll_college_3plus$model, glm.cluster.enroll_college_3plus$model, gls.enroll_college_3plus$model)


##Enrollment: graduate or professional school, 3years plus
###Fit the model
####GLM model
glm.enroll_grad_prof_3plus <- m.glm(df_enroll_1y, "grad_prof_3_plusD")
####GLM model with robust error
glm.robust.enroll_grad_prof_3plus<-m.glm.robust(glm.enroll_grad_prof_3plus$model)
####GML model with clustered sd
glm.cluster.enroll_grad_prof_3plus <- m.glm.cluster(df_enroll_1y, "grad_prof_3_plusD")
####GLS model
gls.enroll_grad_prof_3plus <- m.gls(df_enroll_1y, "grad_prof_3_plusD")

###Get model summary
glm.enroll_grad_prof_3plus$summary
glm.robust.enroll_grad_prof_3plus
glm.cluster.enroll_grad_prof_3plus$summary
gls.enroll_grad_prof_3plus$summary

###Get sensitivity analysis
sens.glm.enroll_grad_prof_3plus <- sensitivity(8.640, 0.1686, 7776,
                                             "One-year Estimated School Enrollment, GLM, Graduate or professional school enrollment for population over 3 years")
sens.glm.robust.enroll_grad_prof_3plus <- sensitivity(8.640, 0.2088, 7776,
                                                    "One-year Estimated School Enrollment, GLM-robust std, Graduate or professional school enrollment for population over 3 years")
sens.glm.cluster.enroll_grad_prof_3plus <- sensitivity(8.640, 0.5651, 7776,
                                                     "One-year Estimated School Enrollment, GLM-clustered std, Graduate or professional school enrollment for population over 3 years")
sens.gls.enroll_grad_prof_3plus <- sensitivity(0.164, 0.0170, 7776,
                                             "One-year Estimated School Enrollment, GLS std, Graduate or professional school enrollment for population over 3 years")

##Compare actual vs. predicted value
compare.predicted.values(df_enroll_1y, "grad_prof_3_plusD",
                         glm.enroll_college_3plus$model, glm.cluster.enroll_college_3plus$model, gls.enroll_college_3plus$model)
