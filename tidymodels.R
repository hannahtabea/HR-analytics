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



# make sure that factor levels of attrition are correctly ordered
levels(ibm_dat$Attrition)
ibm_dat$Attrition <- factor(ibm_dat$Attrition, levels = c("Yes", "No"))
# double check
levels(ibm_dat$Attrition)


# look at data to find variables that probably do not have any predictive power
colnames(ibm_dat)


# clean up data
ibm_reduced <- ibm_dat %>%
  select(-c("DailyRate","EducationField", "EmployeeCount", 
            "MonthlyRate","StandardHours","TotalWorkingYears","StockOptionLevel",
            "Gender", "Over18", "OverTime", "median_compensation"))

#-------------------------------------------------------------------------------
# DATA PREPARATION
#-------------------------------------------------------------------------------

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

# create data folds for cross validation - 10 folds
myFolds <- vfold_cv(train, repeats = 3,
                    strata = Attrition)

# save resampling for caret to make fair comparisons later
train_cv_caret <- rsample2caret(myFolds)

#-------------------------------------------------------------------------------
# FEATURE PREPROCESSING
#-------------------------------------------------------------------------------
#devtools::install_github("stevenpawley/recipeselectors")
library(recipeselectors)
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

# Prepare for parallel processing
all_cores <- parallel::detectCores(logical = TRUE)
doParallel::registerDoParallel(cores = all_cores)

#-------------------------------------------------------------------------------
# MODEL FITTING
#-------------------------------------------------------------------------------

# create model-specific recipes
log_spec <- 
  logistic_reg(penalty = tune(), # lambda
               mixture = tune()) %>% # alpha 
  set_engine("glmnet") 

xgb_spec <- 
  parsnip::boost_tree(mtry = tune(), # colsample_bytree
                      sample_size = tune(), # subsample
                      tree_depth = tune(), # max_depth
                      trees = 2000, # n_rounds 
                      learn_rate = tune(), # eta
                      loss_reduction = tune(), # gamma
                      min_n = tune()) %>% # min_child_weight
  set_mode("classification")%>%
  set_engine("xgboost")


# Create params object for glmnet
glmnet_params <- 
  dials::parameters(list(
    penalty(), 
    mixture()
  ))


# Create params object for XGB
xgb_params <- 
  dials::parameters(list(
    min_n(),
    tree_depth(),
    learn_rate(),
    loss_reduction(),
    sample_size = sample_prop(),
    finalize(mtry(), train)
  ))


# Generate irregular grids
glmnet_grid <- grid_latin_hypercube(glmnet_params,
                           size = 16 # like caret
                            )
xgbTree_grid <- grid_latin_hypercube(xgb_params, 
                            size = 256 #like caret
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
  option_add(grid = xgbTree_grid, id = "recipe_xgbTree") %>%
  option_add(grid = glmnet_grid, id = "recipe_glmnet") 

my_models


# create custom metrics
ibm_metrics <- metric_set(bal_accuracy, roc_auc, yardstick::sensitivity, yardstick::specificity, yardstick::precision, f_meas)

# actual training
model_race <- my_models %>% 
#  option_add(param_info = rfe_param) %>%
  workflow_map("tune_grid", resamples = myFolds, verbose = TRUE,
               control = tune::control_grid(verbose = TRUE, extract = identity),
               metrics = ibm_metrics)


#-------------------------------------------------------------------------------
# MODEL COMPARISON
#-------------------------------------------------------------------------------

# show metrices for the models
model_race %>% collect_metrics(metrics = ibm_metrics) %>%
  group_by(wflow_id)

# show performance of competing models
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

# assess model performance across different folds of train data
xgbTree_res_results <- xgbTree_wkfl %>%
  fit_resamples(resamples = myFolds,
                metrics = ibm_metrics)
# get metrices
collect_metrics(xgbTree_res_results)

# train on training data and test on test data
xgbTree_final <- xgbTree_wkfl %>%
  last_fit(split = ibm_split, metrics = ibm_metrics) 

# create confusion matrix on test data
conf_mat(xgbTree_final$.predictions[[1]],
         truth = Attrition,
         estimate = .pred_class)

# Cross-validated training performance - ROC_AUC
percent(show_best(results, n = 1, metric = "roc_auc")$mean)

# Test performance - ROC_AUC
percent(xgbTree_final$.metrics[[1]]$.estimate[[6]])

#-------------------------------------------------------------------------------
# PREDICTION ON ACTIVE EMPLOYEES
#-------------------------------------------------------------------------------

# create fit object based on finalized workflow 
employee_fit <-  xgbTree_final$.workflow[[1]] %>%
  fit(train)

# set employee test data without outcome
employees <- test %>%
  dplyr::select(-Attrition)

# make predictions on these employees based on finalized model
employee_pred <- predict(employee_fit, employees, type = "prob") %>%
  mutate(Pred_attr = ifelse(.pred_Yes > 0.5, "Yes", "No")) %>%
  bind_cols(test %>% select(Attrition, EmployeeNumber))

# order by predicted probability to leave and show the first 5 suggestions on employees who have not left yet
employee_pred %>% 
  filter(Attrition == "Yes") %>% 
  arrange(desc(.pred_Yes)) %>% 
  head()

# plot predictions
employee_pred %>%
  ggplot() +
  geom_density(aes(x = .pred_Yes, fill = Attrition),
               alpha = 0.5)+
  geom_vline(xintercept = 0.5,linetype = "dashed")+
  ggtitle("Predicted probability distributions vs. actual Attrition outcomes")+
  theme_bw()

# show roc curve
employee_pred %>% 
  roc_curve(truth = Attrition, .pred_Yes) %>% 
  autoplot()


#-------------------------------------------------------------------------------
# BONUS: FEATURE IMPORTANCE 
#-------------------------------------------------------------------------------

# basic model specification
basic_spec <- boost_tree(mode = "classification") %>%
  set_engine("xgboost")

# prepare data based on recipe (Again)
prepped <- prep(ibm_rec)

# fit again (pull_importances not applicable to resamples object)
basic_fitted <- basic_spec %>%
  fit(Attrition ~ ., juice(prepped))

# Get importances
pull_importances(basic_fitted)
