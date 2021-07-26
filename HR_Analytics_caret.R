#---------------------------------------------------------------
### HR Analytics - Predict Employee Churn in R ###
#   Author: Hannah Roos, 23.06.2021
#---------------------------------------------------------------


rm(list = ls())

setwd("C:/Users/hanna/Dropbox/Methods_2019_2021/R-Scripts/Analysis")
current_date <- Sys.Date()
path_plot <- ("C:/Users/hanna/Dropbox/Medium")


library(data.table)
library(skimr)


#https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
ibm_dat <- fread('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# explore data
str(ibm_dat)
skim(ibm_dat)


ibm_dat[ , `:=`(median_compensation = median(MonthlyIncome)),by = .(JobLevel) ]
ibm_dat[ , `:=`(CompensationRatio = (MonthlyIncome/median_compensation)), by =. (JobLevel)]
ibm_dat[ , `:=`(CompensationLevel = factor(fcase(
                                             CompensationRatio %between% list(0.75,1.25), "Average",
                                             CompensationRatio %between% list(0, 0.74), "Below",
                                             CompensationRatio %between% list(1.26,2),  "Above"),
                                             levels = c("Below","Average","Above"))),
                                                                          by = .(JobLevel) ]



# how much turnover in dataset?
turnover_rate <- ibm_dat[, list(count = .N, rate = (.N/nrow(ibm_dat))), by = Attrition]
turnover_rate

# sample small to medium-sized data
set.seed(123)
ibm_126 <- ibm_dat[sample(.N, 126)]

#---------------------------------------------------------------------
# Test some preassumptions visually #
#---------------------------------------------------------------------
library(vcd)
library(vcdExtra)
mosaic(~ Attrition + JobSatisfaction, data = ibm_dat,
       main = "Job Satisfaction against Turnover", shade = TRUE, legend = TRUE)

#highly satisfied and extremely unsatisfied employees are overrepresented among leavers
#what makes them leave if they are actually happy with the job?


# specify colors
library(RColorBrewer)
myCol <- rbind(brewer.pal(8, "Blues")[c(5,7,8)],
               brewer.pal(8, "Reds")[c(4,6,8)],
               brewer.pal(8, "RdPu")[c(4,6,8)],
               brewer.pal(8, "BuPu")[c(5,7,8)])

# Correlation: The higher employee's job satisfaction,the lower the turnover rate.
# Boxplot
plot_jobsatisfaction <- ggplot(ibm_126, aes(x = Attrition, y = JobSatisfaction,fill = Attrition))+
                          geom_boxplot(width=0.1)+
                          scale_fill_manual(values = myCol)+
                          ylab("Job Satisfaction")+
                          xlab("Employee Attrition")+
                          theme(legend.box.background = element_rect(),
                                strip.background = element_rect(colour = "black", fill = "lightgrey"), 
                                axis.title.x = element_blank(), axis.ticks.x = element_blank(),
                                axis.text.x = element_blank())+
                          ggtitle("Are leavers less satisfied than stayers?")
ggsave(plot_jobsatisfaction, file=paste0(current_date,"_","Distribution_Jobsatisfaction.png"), path=path_plot,width = 15, height = 10, units = "cm")



# Job satisfaction amplifies the effect of monthly income on attrition. 
# Hypothesis: Someone who is satisfied with the job tolerates a low to moderate income and stays

# Monthly income vs. job satisfaction -> turnover?
plot_income <- ggplot(ibm_126, aes(x = as.factor(JobSatisfaction), y = MonthlyIncome,fill = Attrition))+
                geom_boxplot()+
                scale_fill_manual(values = myCol)+
                ylab("Monthly Income")+
                xlab("Job Satisfaction")+
                theme(legend.box.background = element_rect(),
                      strip.background = element_rect(colour = "black", fill = "lightgrey"))+
                ggtitle("How are monthly income and job satisfaction related to employee attrition?")
ggsave(plot_income, file=paste0(current_date,"_","Interaction_Income.png"), path=path_plot,width = 20, height = 10, units = "cm")

# Pay competetiveness vs. job satisfaction -> turnover?
plot_CompRatio <- ggplot(ibm_126, aes(x = as.factor(JobSatisfaction), y = CompensationRatio,fill = Attrition))+
                    geom_boxplot()+
                    scale_fill_manual(values = myCol)+
                    ylab("Competensation Ratio")+
                    xlab("Job Satisfaction")+
                    theme(legend.box.background = element_rect(),
                          strip.background = element_rect(colour = "black", fill = "lightgrey"))+
                    ggtitle("How are pay competetiveness and job satisfaction affect employee turnover?")
                  # supported if we plot compa_ratio on the y axis
                  # People who leave even if they are highly satisfied do not seem to go for monetary reasons
ggsave(plot_CompRatio, file=paste0(current_date,"_","Interaction_CompRatio.png"), path=path_plot,width = 20, height = 10, units = "cm")

sanity_check <- ibm_126[, list(count = .N, rate = (.N/nrow(ibm_126))), by = JobSatisfaction]
sanity_check



# What about personal reasons?
library(AppliedPredictiveModeling)
transparentTheme(trans = .4)
library(caret)

featurePlot(x = ibm_126[, c(7,9,18,20)], 
                    y = as.factor(ibm_126$Attrition),
                    plot = "density", 
                    ## Pas in options to xyplot() to 
                    ## make it prettier
                    scales = list(x = list(relation="free"), 
                                  y = list(relation="free")), 
                    adjust = 1.5, 
                    pch = "|", 
                    layout = c(4, 1), 
                    auto.key = list(columns = 2))

library(vcd)
library(vcdExtra)
mosaic(~ Attrition + EnvironmentSatisfaction, data = ibm_126,
       main = "Environment Satisfaction against Turnover", shade = TRUE, legend = TRUE)
mosaic(~ Attrition + JobInvolvement, data = ibm_126,
       main = "Job Involvement against Turnover", shade = TRUE, legend = TRUE)
mosaic(~ Attrition + WorkLifeBalance, data = ibm_126,
       main = "Work-Life-Balance against Turnover", shade = TRUE, legend = TRUE)
mosaic(~ Attrition + RelationshipSatisfaction, data = ibm_126,
       main = "Relationship Satisfaction against Turnover", shade = TRUE, legend = TRUE)



#-------------------------------------------------------------------------------------------------
# FEATURE PREPROCESSInG
#-------------------------------------------------------------------------------------------------

# look at data to find variables that probably do not have any predictive power
colnames(ibm_126)


# clean up data
ibm_126 <- ibm_126[,-c("DailyRate","EducationField", "EmployeeCount", "EmployeeNumber",
                               "MonthlyRate","StandardHours","TotalWorkingYears","StockOptionLevel","Gender", 
                               "Over18", "OverTime", "median_compensation")]
ibm_reduced <- as.data.frame(unclass(ibm_126),stringsAsFactors=TRUE)

#---------------------------------------------------------
# FIND HIGHLY CORRELATED VARS
#----------------------------------------------------------
# load the library
#library(mlbench)
library(caret)

# find numeric values
nums <- unlist(lapply(ibm_reduced, is.numeric))
# save numeric variables for later
ibm_nums <- ibm_reduced[,nums]
# show numeric variables
head(ibm_nums)

# calculate correlation matrix
correlationMatrix <- cor(ibm_nums)
# summarize the correlation matrix
correlationMatrix

# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print colnames of highly correlated attributes
colnames(ibm_nums[,highlyCorrelated])
correlationMatrix[,highlyCorrelated]



# remove highly correlated variables to overcome multicollinearity
colnames(ibm_nums)
highlyCorrelated <- c(1,7,11,16,17,19)
ibm_nums <- ibm_nums[,-highlyCorrelated]
#----------------------------------------------------------
# CREATE DUMMY VARIABLES
#----------------------------------------------------------

# select factor variables to convert, but leave Attrition out
vars_to_dummy <- ibm_reduced[,sapply(ibm_reduced, is.factor) & colnames(ibm_reduced) != "Attrition"]
head(vars_to_dummy)

# Create dummy variables with caret
dummies <- dummyVars( ~ ., data = vars_to_dummy)
ibm_dummy <- predict(dummies, newdata = vars_to_dummy)
# New dataframe to work with later
ibm_sample <- data.frame(ibm_dummy, ibm_nums, Attrition = ibm_reduced$Attrition)
View(ibm_sample)

#--------------------------------------------------------------------------------
# REMOVE NON INFORMATIVE PREDICTORS


# remove near zero variables (except for attr)
remove_cols <- nearZeroVar(ibm_sample, names = TRUE)
remove_cols

# Get all column names 
all_cols <- names(ibm_sample)

# Remove from data
ibm_final<- ibm_sample[ , setdiff(all_cols, remove_cols)]

# make sure that factor levels of attrition are correctly ordered
levels(ibm_final$Attrition)
ibm_final$Attrition <- factor(ibm_final$Attrition, levels = c("Yes", "No"))
# double check
levels(ibm_final$Attrition)

#-------------------------------------------------------------------------
# MODELLING PART 
#-------------------------------------------------------------------------
library(naivebayes)
library(ranger)



# create data folds for cross validation
myFolds <- createFolds(ibm_final$Attrition, k = 5)

f1 <- function (data, lev = NULL, model = NULL) {
  precision <- posPredValue(data$pred, data$obs, positive = "Yes")
  recall  <- sensitivity(data$pred, data$obs, postive = "Yes")
  f1_val <- (2 * precision * recall) / (precision + recall)
  names(f1_val) <- c("F1")
  f1_val
} 



# Create reusable trainControl object: myControl
myControl <- trainControl(
  method = "repeatedcv", 
  repeats = 5, 
  summaryFunction = f1,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE,
  savePredictions = "final",
  index = myFolds
)

model_baseline <- caret::train(
  Attrition ~ MonthlyIncome + JobSatisfaction + MonthlyIncome*JobSatisfaction,
  data = ibm_final,
  metric = "F1",
  method = "glm",
  family = "binomial",
  trControl = myControl
)

# Fit glmnet model
model_glmnet <- caret::train(
  Attrition ~ .,
  data = ibm_final,
  metric = "F1",
  method = "glmnet",
  trControl = myControl
)

# Fit naive bayes: model_naivebayes
model_naivebayes <- caret::train(
  Attrition ~ .,
  data = ibm_final, 
  method = "naive_bayes",
  metric = "F1",
  trControl = myControl
)


# Fit random forest: model_rf
model_rf <- caret::train(
  Attrition ~ .,
  data = ibm_final, 
  method = "ranger",
  metric = "F1",
  trControl = myControl
)




# Fit xgboost model
model_xgboost <- caret::train(
  Attrition ~ .,
  data = ibm_final,
  metric = "F1",
  method = "xgbTree",
  trControl = myControl
)

#------------------------------------------------------------------------------------------
# ASSESS BASELINE MODEL
#------------------------------------------------------------------------------------------
model_baseline
summary(model_baseline)

base_Pred <- predict.train(model_baseline, ibm_final, type = "raw")
confusionMatrix(base_Pred, ibm_final$Attrition, mode = "prec_recall") 

#------------------------------------------------------------------------------------------
# COMPARE MODELS
#------------------------------------------------------------------------------------------

# Create model_list
model_list <- list(baseline = model_baseline, naive_bayes = model_naivebayes,  glmnet = model_glmnet, random_forest = model_rf, xgboost= model_xgboost)

# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)

# Summarize the results
summary(resamples)

bwplot(resamples, metric = "F1")


#------------------------------------------------------------------------------------------
# CHECK MODEL PERFORMANCE - CONFUSION MATRIX
#------------------------------------------------------------------------------------------

xgb_Pred <- predict.train(model_xgboost, ibm_final, type = "raw")
confusionMatrix(xgb_Pred, ibm_final$Attrition, mode = "prec_recall") 


#---------------------------------------------------------------------------------------
# PREDICTION ON ACTIVE EMPLOYEES
#---------------------------------------------------------------------------------------

# get indices of employees that still work for the company
still_active <- setDT(ibm_final)[Attrition == "No", which = TRUE]
probs <- predict.train(model_xgboost, ibm_final, type = "prob")

# save predicted probs for still active employees
risks <- probs[still_active,]
emp_active <- ibm_126[still_active,]

# save row index
risks$index <- 1:nrow(risks)
#find employees who may be at risk leaving the company
emp_risk <-setDT(risks)[order(-Yes)]
emp_risk

# show employee data who could leave soon
emp_indices <- emp_risk$index
top_5 <- head(emp_active[emp_indices,],5)
View(top_5)


#---------------------------------------------------------------------------------------------------------
# BACKUP
#---------------------------------------------------------------------------------------------------------


#----------------------------------------------------------
# TRAIN - TEST - SPLIT
#----------------------------------------------------------


trainIndex <- createDataPartition(ibm_sample$Attrition, p = .6, list = F)
train <- ibm_sample[ trainIndex,]
valid <- ibm_sample[-trainIndex,]



# Compare class distribution between training and validation set
setDT(train)[, list(count = .N, rate = (.N/nrow(train))), by = Attrition]
setDT(valid)[, list(count = .N, rate = (.N/nrow(train))), by = Attrition]




# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(diabetes~., data=PimaIndiansDiabetes, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)

roc_imp <- filterVarImp(x = train[, -ncol(train)], y = train$Turnover)
roc_imp



# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(ibm_126_reduced_2[,1:21], as.matrix(ibm_126_reduced_2[,22]), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))



# estimate variable importance - glmnet
importance <- varImp(model_glmnet, scale=FALSE)
# summarize importance
print(importance_glmnet)
# plot importance
plot(importance_glmnet)

# estimate variable importance - random forest
importance <- varImp(model_rf, scale=FALSE)
# summarize importance
print(importance_rf)
# plot importance
plot(importance_rf)

# estimate variable importance - xgboost
importance_xgboost <- varImp(model_xgboost, scale=FALSE)
# summarize importance
print(importance_xgboost)
# plot importance
plot(importance_xgboost)
