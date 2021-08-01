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
ibm_126 <- sample_n(ibm_dat,126,replace = F)


#-------------------------------------------------------------------------------
# VERY BASIC FEATURE PREPROCESSING
#-------------------------------------------------------------------------------

# look at data to find variables that probably do not have any predictive power
colnames(ibm_126)


# clean up data
ibm_reduced <- ibm_126 %>%
               select(-c("DailyRate","EducationField", "EmployeeCount", "EmployeeNumber",
                           "MonthlyRate","StandardHours","TotalWorkingYears","StockOptionLevel",
                           "Gender", "Over18", "OverTime", "median_compensation"))


# make sure that factor levels of attrition are correctly ordered
levels(ibm_reduced$Attrition)
ibm_reduced$Attrition <- factor(ibm_reduced$Attrition, levels = c("Yes", "No"))
# double check
levels(ibm_reduced$Attrition)

#-------------------------------------------------------------------------------
# MODELLING PART 
#-------------------------------------------------------------------------------

library(tidymodels)
# for naive bayes
library(discrim)
library(klaR)

# create data folds for cross validation
myFolds <- vfold_cv(ibm_reduced,
                        v = 5,
                        #repeats = 5,
                        strata = Attrition)


# create a reusable recipe for feature preprocessing and model formula 
base_recipe <- 
  recipe(Attrition ~ ., data = ibm_reduced) %>% 
  # normalize all numeric predictors
  step_normalize(all_numeric()) %>%
  # create dummy variables 
  step_dummy(all_nominal(), - all_outcomes()) %>%
  # remove zero variance predictors
  step_nzv(all_predictors(), - all_outcomes()) 

filter_recipe <- base_recipe %>%
  # remove highly correlated vars
  step_corr(all_numeric(), threshold = 0.75) 
  

# create model-specific recipes
log_spec <- 
  logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glm") 

glmnet_spec <- 
  logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") 

rf_spec <- 
  rand_forest( trees = tune(), min_n = tune()) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

xgb_spec <- 
  boost_tree( trees = tune(), tree_depth = tune()) %>%
  set_mode("classification")

nb_spec <- 
  discrim::naive_Bayes()%>%
  set_mode("classification")


# create a workflow SET
library(workflowsets)

my_models <- 
  workflow_set(
    preproc = list(simple = base_recipe, filter = filter_recipe),
    models = list(glm = log_spec, glmnet = glmnet_spec, naive_bayes = nb_spec, 
                  ranger = rf_spec, xgbTree = xgb_spec),
    cross = TRUE
  )
my_models

# use no filter on glmnet or glm
my_models <- 
  my_models %>% 
  anti_join(tibble(wflow_id = c("filter_glm", "filter_glmnet")), 
            by = "wflow_id")

# actual training
model_res <- 
  my_models %>% 
  workflow_map("tune_bayes", resamples = myFolds, initial = 5,
               iter = 20,verbose = TRUE,
               metrics = metric_set(bal_accuracy))




#-------------------------------------------------------------------------------
# MODEL COMPARISON
#-------------------------------------------------------------------------------

#glm did not work
try <- model_res %>% 
       filter(wflow_id != "simple_glm")

# show metrices for all other models
try %>% collect_metrics() %>%
  group_by(wflow_id) %>%
  summarise(performance = mean(mean))

#autoplot(model_res)
autoplot(try)

#-------------------------------------------------------------------------------
# TO DO's
#-------------------------------------------------------------------------------

# find out why the classic glm won't work
# improve model performance - why are the algorithms not able to detect positive cases? (precision fail)
# set f1 to standard metric again
# make the tuning process comparable to caret workflow

# finalize the last fit for the baseline and winning model
# retrieve several performance metrics: confusion matrix + precision, recall, f1 and balanced accuracy for each of the models?
# use predictions from winning model to find employees at risk to leave soon


