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


ibm_log_mod <- logistic_reg() %>%
  set_engine("glm")

ibm_wflow_imbal <- workflow() %>%
  add_model(ibm_log_mod) %>%
  add_recipe(ibm_rec_imbalance)

ibm_fit_imbal <- ibm_wflow_imbal %>%
  fit(data = train)

ibm_imbal_res <- ibm_fit_imbal %>%
  pull_workflow_fit() %>%
  tidy()

ibm_preds_imbal <- predict(ibm_fit_imbal, test, type = "prob") %>%
  mutate(Pred_attr = ifelse(.pred_Yes > 0.5, "Yes", "No")) %>%
  bind_cols(test %>% select(Attrition))

ibm_preds_imbal$Pred_attr <- as.factor(ibm_preds_imbal$Pred_attr)

ibm_preds_imbal %>% 
  roc_curve(truth = Attrition, .pred_No) %>% 
  autoplot()

# set metrics 
multi_met <- metric_set(accuracy, precision, recall, spec)

ibm_metrics_imbal <- ibm_preds_imbal %>% 
  multi_met(truth = Attrition, estimate = Pred_attr)
