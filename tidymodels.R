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

# make sure that factor levels of attrition are correctly ordered
levels(ibm_reduced$Attrition)
ibm_reduced$Attrition <- factor(ibm_reduced$Attrition, levels = c("Yes", "No"))
# double check
levels(ibm_reduced$Attrition)

#-------------------------------------------------------------------------------
# DATA PREPARATION
#-------------------------------------------------------------------------------

library(rsample)
library(tidymodels)

# #create 90/10 split
ibm_split <- initial_split(ibm_reduced, prop = 4/5, strata = Attrition)

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

# create reusable recipe for all models
ibm_rec <- train %>%
  recipe(Attrition ~ .) %>%
  # normalize all numeric predictors
  step_normalize(all_numeric()) %>%
  # create dummy variables 
  step_dummy(all_nominal(), - all_outcomes()) %>%
  # remove zero variance predictors
  step_nzv(all_predictors(), - all_outcomes()) %>%
  # remove highly correlated vars
  step_corr(all_numeric(), threshold = 0.75) 

  # deal with class imbalance
  # step_rose(Attrition)


# create model-specific recipes
log_spec <- 
  logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") 

xgb_spec <- 
  boost_tree(tree_depth = tune(),trees = tune(), learn_rate = tune(), 
             loss_reduction = tune(), min_n = tune(), sample_size = tune()) %>%
  set_mode("classification")


# create a custom grid and control object

#len <- ncol(train)
# glmnet_grid <- expand.grid(mixture = seq(0, 1, length = len),# alpha
#                            penalty = c(0, 10 ^ seq(-1, -4, length = len - 1)))# lambda
glmnet_grid <- grid_random(parameters(log_spec), size = 16)
# xgbTree_grid <- expand.grid(tree_depth = seq(1, len), # max_depth
#                             trees = floor((1:len) * 50), # n_rounds
#                             learn_rate = c(.3, .4), # eta
#                             loss_reduction = 0, # gamma
#                             # mtry = runif(len, min = .3, max = .7), # colsample_bytree -> only specific engines
#                             min_n = sample(0:20, size = len, replace = TRUE), # min_child_weight
#                             sample_size = seq(.5, 1, length = len)) # subsample
xgbTree_grid <- grid_random(parameters(xgb_spec), size = 256)

grid_ctrl <-
  control_grid(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE,
    # show progress
    verbose = TRUE
  )

# create a workflow SET
library(workflowsets)
my_models <- 
  workflow_set(
    preproc = list(ibm_rec),
    models = list(glmnet = log_spec,  xgbTree = xgb_spec),
    cross = TRUE
  ) %>%
  # add custom grid
  option_add(grid = glmnet_grid, id = "recipe_glmnet") %>%
  option_add(grid = xgbTree_grid, id = "recipe_xgbTree")
my_models


# actual training
model_race <- 
  my_models %>% 
  workflow_map("tune_grid", resamples = myFolds, verbose = TRUE,
               control = grid_ctrl,
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
  extract_workflow_set_result("recipe_xgbTree") %>% 
  select_best(metric = "f_meas")
best_results

# show final metrics for the training set
xgbTree_wkfl <- model_race %>% 
                          extract_workflow("recipe_xgbTree") %>% 
                          finalize_workflow(best_results) %>% 
                          last_fit(split = ibm_split, metrics = f_meas) 

xgbTree_test_results <- xgbTree_wkfl %>%
                        collect_metrics()
xgbTree_test_results

# create confusion matrix
conf_mat(xgbTree_wkfl$.predictions[[1]],
         truth = Attrition,
         estimate = .pred_class)

#-------------------------------------------------------------------------------
# PREDICTION ON ACTIVE EMPLOYEES
#-------------------------------------------------------------------------------

# find employee IDs based on split object indices

# save employee index to keep track
ibm_reduced$ID <- ibm_147$EmployeeNumber 

# subset on employees that have not left

# make predictions on these employees (change test to no_churn)
predict(xgbTree_wkfl, test, type = "prob") %>%
  mutate(Pred_attr = ifelse(.pred_Yes > 0.5, "Yes", "No")) %>%
  bind_cols(test %>% select(Attrition))

# order by predicted probability to leave and show the first 5 suggestions


# done:
# ----
# made ROSE optional (a bit artificial and we want to make predictions with real data)
# created custom grids with the similar to caret to ensure more comparability
