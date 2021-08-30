# Data wrangling and load data
url <- "https://raw.githubusercontent.com/hannahtabea/HR-analytics/8c7abc5ef610c1f7ecc4596cf0ce6f55a2ffccf1/WA_Fn-UseC_-HR-Employee-Attrition.csv"
ibm_dat <- fread(url) %>%
  # make sure that factor levels are correctly ordered to ensure correct performance metrics!!!
  # the first level should be your level of interest (e.g., YES)
  mutate(Attrition = factor(Attrition, levels = c("Yes", "No")))

ibm_dat[ , `:=`(MedianCompensation = median(MonthlyIncome)),by = .(JobLevel) ]
ibm_dat[ , `:=`(CompensationRatio = (MonthlyIncome/MedianCompensation)), by =. (JobLevel)]
ibm_dat[ , `:=`(CompensationLevel = factor(fcase(
  CompensationRatio %between% list(0.75,1.25), "Average",
  CompensationRatio %between% list(0, 0.75), "Below",
  CompensationRatio %between% list(1.25,2),  "Above"),
  levels = c("Below","Average","Above"))),
  by = .(JobLevel) ][, c("EmployeeCount","StandardHours","Over18") := NULL]


library(rsample)
set.seed(69)
ibm_split <- initial_split(ibm_dat, strata = Attrition)

# Create the training data
train <- ibm_split %>%
  training()

# Create the test data
test <- ibm_split %>%
  testing()


# Check class imbalance
train %>%
  group_by(Attrition) %>%
  summarise(
    n = n(),
    perc = n/nrow(.)
    )

# Modelling without adressing class imbalance

ibm_rec_imbalance <- recipe(Attrition ~ ., data = train) %>%
  # normalize all numeric predictors
  step_normalize(all_numeric()) %>%
  # create dummy variables 
  step_dummy(all_nominal(), - all_outcomes()) %>%
  # remove zero variance predictors
  step_nzv(all_predictors(), - all_outcomes()) %>%
  # remove highly correlated vars
  step_corr(all_numeric(), threshold = 0.75) 

# set model
ibm_log_mod <- logistic_reg() %>%
  set_engine("glm")

# create workflow
ibm_wflow_imbal <- workflow() %>%
  add_model(ibm_log_mod) %>%
  add_recipe(ibm_rec_imbalance)

# fit model to training data 
ibm_fit_imbal <- ibm_wflow_imbal %>%
  fit(data = train)

# extract model fit from parsnip and tidy output
ibm_imbal_res <- ibm_fit_imbal %>%
  extract_fit_parsnip() %>%
  tidy()

# make predictions on test data
ibm_preds_imbal <- predict(ibm_fit_imbal, test, type = "prob") %>%
  mutate(Pred_attr = factor(ifelse(.pred_Yes > 0.5, "Yes", "No"), levels = c("Yes", "No"))) %>%
  bind_cols(test %>% select(Attrition))
ibm_preds_imbal

# show roc curve
ibm_preds_imbal %>% 
  roc_curve(truth = Attrition, .pred_Yes) %>% 
  autoplot()

# confusion matrix
conf_mat(ibm_preds_imbal,
         truth = Attrition,
         estimate = Pred_attr)

# set metrics 
multi_met <- metric_set(accuracy, yardstick::precision, yardstick::recall, spec)

# show metrics
ibm_metrics_imbal <- ibm_preds_imbal %>% 
  multi_met(truth = Attrition, estimate = Pred_attr)
ibm_metrics_imbal
