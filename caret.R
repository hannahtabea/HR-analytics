#-------------------------------------------------------------------------------
### HR Analytics - Predict Employee Churn in R ###
#   Author: Hannah Roos
#-------------------------------------------------------------------------------

# use here package to set up the right paths while making it less hard-wired for other users
library(here)
current_date <- Sys.Date()
path_dat <- here('WA_Fn-UseC_-HR-Employee-Attrition.csv')
path_plot <- here('Plots')

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

# sample small to medium-sized data
set.seed(123)
ibm_147 <- ibm_dat[sample(.N, 147)]


# look at data to find variables that probably do not have any predictive power
colnames(ibm_147)


# clean up data
ibm_reduced <- ibm_147[,-c("DailyRate","EducationField", "EmployeeCount", "EmployeeNumber",
                           "MonthlyRate","StandardHours","TotalWorkingYears","StockOptionLevel",
                           "Gender", "Over18", "OverTime", "median_compensation")]


# make sure that factor levels of attrition are correctly ordered
levels(ibm_reduced$Attrition)
ibm_reduced$Attrition <- factor(ibm_reduced$Attrition, levels = c("Yes", "No"))
# double check
levels(ibm_reduced$Attrition)

# deal with class imbalance
library(ROSE)
set.seed(9560)
ibm_balanced <- ROSE(Attrition ~ ., data  = ibm_reduced)$data
# check if it worked
table(ibm_balanced$Attrition) 

#-------------------------------------------------------------------------------
# MODELLING PART 
#-------------------------------------------------------------------------------

library(caret)

# create data folds for cross validation
# myFolds <- createFolds(ibm_reduced$Attrition, k = 5)

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
  returnResamp = "final",
  preProcOptions = list(cutoff = 0.75),
  index = train_cv_caret$index,
  indexOut = train_cv_caret$indexOut
)


# Create reusable train function
methods <- c("glmnet", "xgbTree")

train_model <- function(x) {
  model <- caret::train(
                    Attrition ~ .,
                    data = ibm_balanced,
                    metric = "F1",
                    method = x,
                    preProcess = c("scale", "nzv","corr"),
                    trControl = myControl
                  )
  return(assign(paste0("model_", x),model, envir = .GlobalEnv))
}


lapply(methods, train_model)

#-------------------------------------------------------------------------------
# MODEL COMPARISON
#-------------------------------------------------------------------------------

# Create model_list
model_list <- list(baseline = model_glmnet, xgboost = model_xgbTree)

# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)

# Summarize the results
summary(resamples)

bwplot(resamples, metric = "F1")


#-------------------------------------------------------------------------------
# MODEL PERFORMANCE - CONFUSION MATRIX
#-------------------------------------------------------------------------------

# assess baseline model
model_glm
summary(model_glm)

base_Pred <- predict.train(model_glm, ibm_reduced, type = "raw")
confusionMatrix(base_Pred, ibm_reduced$Attrition, mode = "prec_recall") 

# assess winner model
xgb_Pred <- predict.train(model_xgbTree, ibm_reduced, type = "raw")
confusionMatrix(xgb_Pred, ibm_reduced$Attrition, mode = "prec_recall")


#-------------------------------------------------------------------------------
# PREDICTION ON ACTIVE EMPLOYEES
#-------------------------------------------------------------------------------

# save employee index to keep track
ibm_reduced$ID <- ibm_147$EmployeeNumber
# save employees who did not churn
no_churn <- subset(ibm_reduced,Attrition == "No")
# get class probabilities from employees who are still active
probs <- predict.train(model_xgbTree, no_churn, type = "prob")
# merge with employee data
still_active <- data.table(no_churn,probs)

# show employees who are most at risk to leave soon
head(still_active[order(-Yes)],5)

