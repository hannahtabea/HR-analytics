#-------------------------------------------------------------------------------
### HR Analytics - Predict Employee Churn in R ###
#   Author: Hannah Roos
#-------------------------------------------------------------------------------

# use here package to set up the right paths while making it less hard-wired for other users
library(here)
current_date <- Sys.Date()
path_dat <- here('HR analytics','WA_Fn-UseC_-HR-Employee-Attrition.csv')

# get data
library(data.table)
ibm_dat <- fread(path_dat, stringsAsFactors = T)

# explore data
library(skimr)
str(ibm_dat)
skim(ibm_dat)

# speed up processing if possible
plan(multicore) 

#-------------------------------------------------------------------------------
# DATA WRANGLING
#-------------------------------------------------------------------------------

ibm_dat[ , `:=`(median_compensation = median(MonthlyIncome)),by = .(JobLevel) ]
ibm_dat[ , `:=`(CompensationRatio = (MonthlyIncome/median_compensation)), by =. (JobLevel)]
ibm_dat[ , `:=`(CompensationLevel = factor(fcase(
                                             CompensationRatio %between% list(0.75,1.25), "Average",
                                             CompensationRatio %between% list(0, 0.75), "Below",
                                             CompensationRatio %between% list(1.25,2),  "Above"),
                                             levels = c("Below","Average","Above"))),
                                                                          by = .(JobLevel) ]

# how much turnover in dataset?
turnover_rate <- ibm_dat[, list(count = .N, rate = (.N/nrow(ibm_dat))), by = Attrition]
turnover_rate

# # sample small to medium-sized data
# set.seed(123)
# ibm_147 <- ibm_dat[sample(.N, 147)]
 
# look at data to find variables that probably do not have any predictive power
colnames(ibm_147)

# clean up data
ibm_reduced <- ibm_dat[,-c("DailyRate","EducationField", "EmployeeCount", 
                           "MonthlyRate","StandardHours","TotalWorkingYears","StockOptionLevel",
                           "Gender", "Over18", "OverTime", "median_compensation")]


# make sure that factor levels of attrition are correctly ordered
levels(ibm_reduced$Attrition)
ibm_reduced$Attrition <- factor(ibm_reduced$Attrition, levels = c("Yes", "No"))
# double check
levels(ibm_reduced$Attrition)

#-------------------------------------------------------------------------------
# MODELLING PART 
#-------------------------------------------------------------------------------

library(caret)

# create data folds for cross validation
trainIndex <- createDataPartition(ibm_reduced$Attrition, p = .7,
                                  list = FALSE,
                                  times = 1)
train <- ibm_reduced[ trainIndex,]
test  <- ibm_reduced[-trainIndex,]

## deal with class imbalance - upsampling
library(ROSE)
library(dplyr)
set.seed(9560)
train <- ROSE(Attrition ~ ., data  = train)$data %>% 
  mutate(Attrition = factor(Attrition, levels = c("Yes", "No"))) # ROSE has reversed factor levels, therefore order them again...
# check if it worked
table(train$Attrition)

f1 <- function (data, lev = NULL, model = NULL) {
  precision <- posPredValue(data$pred, data$obs, positive = "Yes")
  recall  <- sensitivity(data$pred, data$obs, postive = "Yes")
  f1_val <- (2 * precision * recall) / (precision + recall)
  names(f1_val) <- c("F1")
  f1_val
}

# create k folds - not needed if borrowed from tidymodels 
myFolds <- createFolds(train$Attrition, k = 10)

# Create reusable trainControl object
myControl <- trainControl(
  method = "repeatedcv", 
  repeats = 3, 
  summaryFunction = f1,
  classProbs = TRUE, 
  verboseIter = TRUE,
  savePredictions = "final",
  returnResamp = "final",
  preProcOptions = list(cutoff = 0.75),
  index = myFolds
  # index = train_cv_caret$index, # use when split is borrowed from tidymodels
  # indexOut = train_cv_caret$indexOut # see above
)

# Create reusable train function
methods <- c("glmnet", "xgbTree")

train_model <- function(x) {
  model <- caret::train(
                    Attrition ~ .,
                    data = train,
                    metric = "F1",
                    method = x,
                    preProcess = c("scale", "nzv","corr"),
                    trControl = myControl,
                    tuneLength = 4
                  )
  return(assign(paste0("model_", x),model, envir = .GlobalEnv))
}

lapply(methods, train_model)

#-------------------------------------------------------------------------------
# MODEL COMPARISON
#-------------------------------------------------------------------------------

# Create model_list
model_list <- list(glmnet = model_glmnet, xgboost = model_xgbTree)

# Pass model_list to resamples
resamples <- resamples(model_list)

# Summarize the results
summary(resamples)
bwplot(resamples, metric = "F1")

# get ROC results
#install.packages("MLeval")
library(MLeval)
evaluation <- evalm(model_list,gnames=c('glmnet','xgboost'))

#-------------------------------------------------------------------------------
# MODEL PERFORMANCE - CONFUSION MATRIX & VISuALIZATION
#-------------------------------------------------------------------------------
# visualize tuning parameters against performance
plot(model_xgbTree, type=c("g", "o"))

# cross-validated training performance
# assess winner model on test data
xgb_pred_train <- predict.train(model_xgbTree, train, type = "raw")
bal_accuracy(train, truth = train$Attrition, estimate = xgb_pred_train)
confusionMatrix(xgb_pred_train, train$Attrition, mode = "prec_recall")

xgb_pred_test <- predict.train(model_xgbTree, test, type = "raw")
bal_accuracy(test, truth = test$Attrition, estimate = xgb_pred_test)
confusionMatrix(xgb_pred_test, test$Attrition, mode = "prec_recall")

#-------------------------------------------------------------------------------
# PREDICTION ON ACTIVE EMPLOYEES
#-------------------------------------------------------------------------------

# save employees who did not churn
no_churn <- subset(test,Attrition == "No")
# get class probabilities from employees who are still active
probs <- predict.train(model_xgbTree, no_churn, type = "prob")
# merge with employee data
still_active <- data.table(
  subset(no_churn,select = c("EmployeeNumber","Attrition")),
  probs)

# show employees who are most at risk to leave soon
head(still_active[order(-Yes)],5)

#-------------------------------------------------------------------------------
# BONUS: Feature importance
#-------------------------------------------------------------------------------

# estimate variable importance
importance <- varImp(model_xgbTree, scale=FALSE)
# summarize importance
importance
# plot importance
plot(importance)
