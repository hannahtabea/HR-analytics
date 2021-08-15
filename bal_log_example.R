# Data wrangling and load data
url <- "https://raw.githubusercontent.com/hannahtabea/HR-analytics/8c7abc5ef610c1f7ecc4596cf0ce6f55a2ffccf1/WA_Fn-UseC_-HR-Employee-Attrition.csv"
ibm_dat <- fread(url) %>%
  mutate(Attrition = factor(Attrition))

ibm_dat[ , `:=`(MedianCompensation = median(MonthlyIncome)),by = .(JobLevel) ]
ibm_dat[ , `:=`(CompensationRatio = (MonthlyIncome/MedianCompensation)), by =. (JobLevel)]
ibm_dat <- ibm_dat %>%
  mutate(CompensationLevel =  case_when(
    CompensationRatio > 0.75 & CompensationRatio <= 1.25 ~ "Average",
    CompensationRatio >= 0 & CompensationRatio <= 0.75 ~ "Below",
    CompensationRatio >1.25  ~ "Above",
    TRUE ~ "Other"))
ibm_dat$CompensationLevel <- as.factor(ibm_dat$CompensationLevel)
ibm_dat$EmployeeCount <- NULL
ibm_dat$StandardHours <- NULL
ibm_dat$Over18 <- NULL

library(rsample)
ibm_split <- initial_split(ibm_dat, strata = Attrition)

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

# Modelling with adressing class imbalance

ibm_rec_balanced <- recipe(Attrition ~ ., data = train) %>%
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


ibm_log_mod <- logistic_reg() %>%
  set_engine("glm")

ibm_wflow_bal <- workflow() %>%
  add_model(ibm_log_mod) %>%
  add_recipe(ibm_rec_balanced)

ibm_fit_bal <- ibm_wflow_bal %>%
  fit(data = train)

ibm_bal_res <- ibm_fit_bal %>%
  pull_workflow_fit() %>%
  tidy()

ibm_preds_bal <- predict(ibm_fit_bal, test, type = "prob") %>%
  mutate(Pred_attr = ifelse(.pred_Yes > 0.5, "Yes", "No")) %>%
  bind_cols(test %>% select(Attrition))

ibm_preds_bal$Pred_attr <- as.factor(ibm_preds_bal$Pred_attr)

ibm_preds_bal %>% 
  roc_curve(truth = Attrition, .pred_No) %>% 
  autoplot()

# set metrics 
multi_met <- metric_set(accuracy, precision, recall, spec)

ibm_metrics_bal <- ibm_preds_bal %>% 
  multi_met(truth = Attrition, estimate = Pred_attr)
