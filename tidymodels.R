#-------------------------------------------------------------------------------
### HR Analytics - Predict Employee Churn in R ###
#   Author: Hannah Roos
#-------------------------------------------------------------------------------

# use here package to set up the right paths 
current_date <- Sys.Date()
path_dat <- here::here('HR analytics','WA_Fn-UseC_-HR-Employee-Attrition.csv')

# get data
ibm_dat <- readr::read_csv(path_dat)

# explore data
str(ibm_dat)
skimr::skim(ibm_dat)

#-------------------------------------------------------------------------------
# DATA WRANGLING
#-------------------------------------------------------------------------------

library(tidymodels)

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


# create 90/10 split
set.seed(693)
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
  themis::step_rose(Attrition)

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
                           size = 9 # like caret
                            )
xgbTree_grid <- grid_latin_hypercube(xgb_params, 
                            size = 108 #like caret
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


#system.time ({
# actual tuning
model_race <- my_models %>% 
  workflow_map("tune_grid", resamples = myFolds, verbose = TRUE,
               control = tune::control_grid(verbose = TRUE),
               metrics = ibm_metrics)
#})

#-------------------------------------------------------------------------------
# MODEL COMPARISON
#-------------------------------------------------------------------------------

# show metrices for the models
model_race %>% collect_metrics(metrics = ibm_metrics) %>%
  group_by(wflow_id)

# show performance of competing models
autoplot(model_race)

#-------------------------------------------------------------------------------
# MODEL FINALIZATION
#-------------------------------------------------------------------------------

# combine parameter combinations with metrics and predictions
results <- model_race %>% 
  extract_workflow_set_result("recipe_glmnet")

# select best workflow
best_results <- results %>%
  select_best(metric = "f_meas")

# finalize workflow
glmnet_wkfl <- model_race %>%
  extract_workflow("recipe_glmnet") %>%
  finalize_workflow(best_results)

# assess model performance across different folds of train data
glmnet_res_results <- glmnet_wkfl %>%
  fit_resamples(resamples = myFolds,
                metrics = ibm_metrics,
                control = control_resamples(save_pred = TRUE))

# get metrices of training folds
collect_metrics(glmnet_res_results)

# train on training data and test on test data
glmnet_final <- glmnet_wkfl %>%
  last_fit(split = ibm_split, metrics = ibm_metrics) 


#-------------------------------------------------------------------------------
# PREDICTIONS & CONFUSION MATRIX
#-------------------------------------------------------------------------------

# get performance metrics on test data in last_fit object
glmnet_final$.metrics

# create fit object on training data
glmnet_fit <- fit(glmnet_wkfl, train)
# predict
glmnet_train_pred <- predict(glmnet_fit, train) %>%
                      bind_cols(train[2]) 
# create confusion matrix on train data
conf_mat(glmnet_train_pred,
          truth = Attrition,
          estimate = .pred_class)

# create confusion matrix on test data
conf_mat(glmnet_final$.predictions[[1]],
         truth = Attrition,
         estimate = .pred_class)

#-------------------------------------------------------------------------------
# PERFORMANCE VISUALIZATIONS
#-------------------------------------------------------------------------------

# plot predictions
data.frame(glmnet_final$.predictions) %>%
  ggplot() +
  geom_density(aes(x = .pred_Yes, fill = Attrition),
               alpha = 0.5)+
  geom_vline(xintercept = 0.5,linetype = "dashed")+
  ggtitle("Predicted class probabilities coloured by attrition")+
  theme_bw()


# show roc curve
data.frame(glmnet_final$.predictions) %>% 
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
