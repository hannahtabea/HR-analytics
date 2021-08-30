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


# Check class balance
train %>%
  group_by(Attrition) %>%
  summarise(
    n = n(),
    perc = n/nrow(.)
  )

# Modelling without adressing class balance
ibm_rec_balance <- recipe(Attrition ~ ., data = train) %>%
  # normalize all numeric predictors
  step_normalize(all_numeric()) %>%
  # create dummy variables 
  step_dummy(all_nominal(), - all_outcomes()) %>%
  # remove zero variance predictors
  step_nzv(all_predictors(), - all_outcomes()) %>%
  # remove highly correlated vars
  step_corr(all_numeric(), threshold = 0.75) %>%
  step_rose(Attrition)

# set model
ibm_log_mod <- logistic_reg() %>%
  set_engine("glm")

# create workflow
ibm_wflow_bal <- workflow() %>%
  add_model(ibm_log_mod) %>%
  add_recipe(ibm_rec_balance)

# fit model to training data 
ibm_fit_bal <- ibm_wflow_bal %>%
  fit(data = train)

# extract model fit from parsnip and tidy output
ibm_bal_res <- ibm_fit_bal %>%
  extract_fit_parsnip() %>%
  tidy()

# make predictions on test data
ibm_preds_bal <- predict(ibm_fit_bal, test, type = "prob") %>%
  mutate(Pred_attr = factor(ifelse(.pred_Yes > 0.5, "Yes", "No"), levels = c("Yes", "No"))) %>%
  bind_cols(test %>% select(Attrition))
ibm_preds_bal

# show roc curve
ibm_preds_bal %>% 
  roc_curve(truth = Attrition, .pred_Yes) %>% 
  autoplot()

# confusion matrix
conf_mat(ibm_preds_bal,
         truth = Attrition,
         estimate = Pred_attr)

# set metrics 
multi_met <- metric_set(accuracy, yardstick::precision, yardstick::recall, spec)

# show metrics
ibm_metrics_bal <- ibm_preds_bal %>% 
  multi_met(truth = Attrition, estimate = Pred_attr)
ibm_metrics_bal
