#-------------------------------------------------------------------------------
### HR Analytics - Predict Employee Churn in R ###
#   Author: Hannah Roos
#-------------------------------------------------------------------------------

# use here package to set up the right paths 
library(here)
current_date <- Sys.Date()
path_dat <- here('HR analytics','WA_Fn-UseC_-HR-Employee-Attrition.csv')

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
ibm_split <- initial_split(ibm_reduced, prop = 7/10, strata = Attrition)

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

test %>%
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
  step_corr(all_numeric(), threshold = 0.75) %>%
  # deal with class imbalance
  step_rose(Attrition)


# create model-specific recipes
log_spec <- 
  logistic_reg(penalty = tune(), # lambda
               mixture = tune()) # alpha
               %>% 
  set_engine("glmnet") 

xgb_spec <- 
  boost_tree(tree_depth = tune(), # max_depth
             trees = tune(), # n_rounds 
             learn_rate = tune(), # eta
             loss_reduction = tune(), # gamma
             min_n = tune(), # min_child_weight 
             sample_size = tune()) %>% # subsample
  set_mode("classification")


# create a custom grid and control object
glmnet_grid <- grid_random(parameters(log_spec), size = 16)
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
model_race <- my_models %>% 
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

# combine parameter combinations with metrics and predictions
results <- model_race %>% 
  extract_workflow_set_result("recipe_xgbTree")

# select best workflow
best_results <- results %>%
  select_best(metric = "f_meas")


# finalize workflow
xgbTree_wkfl <- model_race %>%
  extract_workflow("recipe_xgbTree") %>%
  finalize_workflow(best_results)

# train on training data and test on test data
xgbTree_final <- xgbTree_wkfl %>%
  last_fit(split = ibm_split, metrics = metric_set(bal_accuracy, yardstick::precision, yardstick::recall, f_meas)) 

# assess model performance across different folds of test data
xgbTree_test_results <- xgbTree_wkfl %>%
  fit_resamples(resamples = myFolds,
                metrics = metric_set(bal_accuracy, yardstick::precision, yardstick::recall, f_meas))

# create confusion matrix
conf_mat(xgbTree_final$.predictions[[1]],
         truth = Attrition,
         estimate = .pred_class)

# Cross-validated training performance - balanced accuracy
percent(show_best(results, n = 1)$mean)

# Test performance - balanced accuracy
percent(xgbTree_final$.metrics[[1]]$.estimate[[1]])


#-------------------------------------------------------------------------------
# PREDICTION ON ACTIVE EMPLOYEES
#-------------------------------------------------------------------------------

# create fit object based on finalized workflow 
whole_fit <-  xgbTree_final$.workflow[[1]] %>%
  fit(ibm_147)

# set employee data without outcome
employees <- ibm_147 %>%
  dplyr::select(-Attrition)

# make predictions on these employees based on finalized model
employee_pred <- predict(whole_fit, employees, type = "prob") %>%
  mutate(Pred_attr = ifelse(.pred_Yes > 0.5, "Yes", "No")) %>%
  bind_cols(ibm_147 %>% select(Attrition, EmployeeNumber))

# order by predicted probability to leave and show the first 5 suggestions on employees who have not left yet
employee_pred %>% 
  filter(Attrition == "Yes") %>% 
  arrange(desc(.pred_Yes)) %>% 
  head()
