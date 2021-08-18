#-------------------------------------------------------------------------------
### HR Analytics - Predict Employee Churn in R ###
#   Author: Hannah Roos
#-------------------------------------------------------------------------------

# use here package to set up the right paths 
library(here)
current_date <- Sys.Date()
path_dat <- here('WA_Fn-UseC_-HR-Employee-Attrition.csv')
path_plot <- here('Plots')

# get data
library(readr)
ibm_dat <- read_csv(path_dat)

# explore data
library(skimr)
str(ibm_dat)
skim(ibm_dat)

# speed up processing if possible
library(furrr)
plan(multicore) 
#-------------------------------------------------------------------------------
# DATA WRANGLING
#-------------------------------------------------------------------------------

library(tidyverse)
ibm_dat <- ibm_dat %>% 
           # create CompensationRatio by joblevel
            group_by(JobLevel) %>%
            mutate(median_compensation = median(MonthlyIncome),
                   CompensationRatio = (MonthlyIncome/median(MonthlyIncome)),
                   CompensationLevel = case_when(
              between(CompensationRatio, 0.75,1.25) ~ "Average",
              between(CompensationRatio, 0, 0.75) ~ "Below",
              between(CompensationRatio, 1.25, 2) ~ "Above"
            )) %>%
            ungroup() %>%
            # convert all characters to factors
            mutate_if(is.character, as.factor)
            

# how much turnover in dataset?
turnover_rate <- ibm_dat %>% 
                 group_by(Attrition) %>%
                  summarise(n = n()) %>%
                  mutate(rate = n / sum(n))
turnover_rate


# sample small to medium-sized data
set.seed(123)
ibm_147 <- sample_n(ibm_dat,147,replace = F)


# look at data to find variables that probably do not have any predictive power
colnames(ibm_147)


# clean up data
ibm_reduced <- ibm_147 %>%
               select(-c("DailyRate","EducationField", "EmployeeCount", "EmployeeNumber",
                           "MonthlyRate","StandardHours","TotalWorkingYears","StockOptionLevel",
                           "Gender", "Over18", "OverTime", "median_compensation"))

#-------------------------------------------------------------------------------
# DATA PREPARATION
#-------------------------------------------------------------------------------

library(rsample)
library(tidymodels)

#create 90/10 split
ibm_split <- initial_split(ibm_reduced, prop = 9/10, strata = Attrition)

# Create the training data
train <- ibm_split %>%
  training()

test <- ibm_split %>%
  testing()

# Check class imbalance
train %>%
  group_by(Attrition) %>%
  summarise(
    n = n(),
    perc = n/nrow(.)
  )

# create data folds for cross validation
myFolds <- vfold_cv(train,
                    v = 5,
                    repeats = 5,
                    strata = Attrition)
# save resampling for caret to make fair comparisons later
train_cv_caret <- rsample2caret(myFolds)

#-------------------------------------------------------------------------------
# FEATURE PREPROCESSING
#-------------------------------------------------------------------------------

library(themis)

set.seed(9560)
# create reusable recipe for all models
ibm_rec_balanced <- train %>%
  recipe(Attrition ~ .) %>%
  # normalize all numeric predictors
  step_normalize(all_numeric()) %>%
  # create dummy variables 
  step_dummy(all_nominal(), - all_outcomes()) %>%
  # remove zero variance predictors
  step_nzv(all_predictors(), - all_outcomes()) %>%
  # remove highly correlated vars
  step_corr(all_numeric(), threshold = 0.75) %>%
  # deal with class imbalance
  step_rose(Attrition)


# create model-specific recipes
log_spec <- 
  logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") 

xgb_spec <- 
  boost_tree( trees = tune(), tree_depth = tune()) %>%
  set_mode("classification")



# create a workflow SET
library(workflowsets)

my_models <- 
  workflow_set(
    preproc = list(ibm_rec_balanced),
    models = list(glmnet = log_spec,  xgbTree = xgb_spec),
    cross = TRUE
  )
my_models


# actual training
model_race <- 
  my_models %>% 
  workflow_map("tune_grid", resamples = myFolds, initial = 5,verbose = TRUE,
               metrics = metric_set(bal_accuracy, yardstick::precision, yardstick::recall, f_meas))




#-------------------------------------------------------------------------------
# MODEL COMPARISON
#-------------------------------------------------------------------------------

# show metrices for the models
model_race %>% collect_metrics() %>%
  group_by(wflow_id)

# show performance
autoplot(model_race)

# select best workflow for the model that worked well
best_results <- 
  model_race %>% 
  extract_workflow_set_result("recipe_glmnet") %>% 
  select_best(metric = "f_meas")
best_results

# show final metrics for the training set
glmnet_test_results <- model_race %>% 
                          extract_workflow("recipe_glmnet") %>% 
                          finalize_workflow(best_results) %>% 
                          last_fit(split = ibm_split, metrics = metric_set(bal_accuracy, yardstick::precision, yardstick::recall, f_meas)) %>%
                          collect_metrics()
glmnet_test_results


#-------------------------------------------------------------------------------
# TO DO's
#-------------------------------------------------------------------------------

# train test split and model performance on test data for caret
# make tuning grids comparable to each other (caret's default tuning parameters)
# create confusion matrix for tidymodels
# make predictions with the winning model to find employees at risk to leave soon
# track times for model tuning with profvis

# done:
# ----
# more a bit more data: 147 instead of 126
# included minority sampling to deal with class imbalance
# reduced model comparison to two competing models only
# included furrrs multicore processing
# finalized workflow 

