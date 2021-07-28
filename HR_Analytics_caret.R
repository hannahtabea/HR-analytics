#-------------------------------------------------------------------------------
### HR Analytics - Predict Employee Churn in R ###
#   Author: Hannah Roos, 23.06.2021, 27.07.2021
#-------------------------------------------------------------------------------

# use here package to set up the right paths while making it less hard-wired for other users
library(here)
current_date <- Sys.Date()
path_dat <- here('WA_Fn-UseC_-HR-Employee-Attrition.csv')
path_plot <- here('Plots')

# get data
library(data.table)
ibm_dat <- fread(path_dat, stringsAsFactors = T)

# explore data
library(skimr)
str(ibm_dat)
skim(ibm_dat)


#-------------------------------------------------------------------------------
# DATA WRANGLING
#-------------------------------------------------------------------------------

ibm_dat[ , `:=`(median_compensation = median(MonthlyIncome)),by = .(JobLevel) ]
ibm_dat[ , `:=`(CompensationRatio = (MonthlyIncome/median_compensation)), by =. (JobLevel)]
ibm_dat[ , `:=`(CompensationLevel = factor(fcase(
                                             CompensationRatio %between% list(0.75,1.25), "Average",
                                             CompensationRatio %between% list(0, 0.75), "Below",
                                             CompensationRatio %between% list(1.25,2),  "Above"),
                                             levels = c("Below","Average","Above"))),
                                                                          by = .(JobLevel) ]



# how much turnover in dataset?
turnover_rate <- ibm_dat[, list(count = .N, rate = (.N/nrow(ibm_dat))), by = Attrition]
turnover_rate

# sample small to medium-sized data
set.seed(123)
ibm_126 <- ibm_dat[sample(.N, 126)]


#-------------------------------------------------------------------------------
# VERY BASIC FEATURE PREPROCESSING
#-------------------------------------------------------------------------------

# look at data to find variables that probably do not have any predictive power
colnames(ibm_126)


# clean up data
ibm_reduced <- ibm_126[,-c("DailyRate","EducationField", "EmployeeCount", "EmployeeNumber",
                           "MonthlyRate","StandardHours","TotalWorkingYears","StockOptionLevel",
                           "Gender", "Over18", "OverTime", "median_compensation")]


# make sure that factor levels of attrition are correctly ordered
levels(ibm_reduced$Attrition)
ibm_reduced$Attrition <- factor(ibm_reduced$Attrition, levels = c("Yes", "No"))
# double check
levels(ibm_reduced$Attrition)

#-------------------------------------------------------------------------------
# MODELLING PART 
#-------------------------------------------------------------------------------

library(caret)
library(naivebayes)
library(ranger)


# create data folds for cross validation
myFolds <- createFolds(ibm_reduced$Attrition, k = 5)

f1 <- function (data, lev = NULL, model = NULL) {
  precision <- posPredValue(data$pred, data$obs, positive = "Yes")
  recall  <- sensitivity(data$pred, data$obs, postive = "Yes")
  f1_val <- (2 * precision * recall) / (precision + recall)
  names(f1_val) <- c("F1")
  f1_val
} 


# Create reusable trainControl object: myControl
myControl <- trainControl(
  method = "repeatedcv", 
  repeats = 5, 
  summaryFunction = f1,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE,
  savePredictions = "final",
  # remove vars those intercorrelations exceed 0.75
  preProcOptions = list(cutoff = 0.75),
  index = myFolds
)

# Create reusable train function
methods <- c("glm","glmnet","naive_bayes", "ranger", "xgbTree")

train_model <- function(x) {
  model <- caret::train(
                    Attrition ~ .,
                    data = ibm_reduced,
                    metric = "F1",
                    method = x,
                    preProcess = c("scale", "nzv","corr"),
                    trControl = myControl
                  )
  return(assign(paste0("model_", x),model, envir = .GlobalEnv))
}

lapply(methods, train_model)

#-------------------------------------------------------------------------------
# MODEL COMPARISON
#-------------------------------------------------------------------------------

# Create model_list
model_list <- list(baseline = model_glm, naive_bayes = model_naive_bayes,  
                   glmnet = model_glmnet,random_forest = model_ranger, xgboost = model_xgbTree)

# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)

# Summarize the results
summary(resamples)

bwplot(resamples, metric = "F1")


#-------------------------------------------------------------------------------
# MODEL PERFORMANCE - CONFUSION MATRIX
#-------------------------------------------------------------------------------

# assess baseline model
model_glm
summary(model_glm)

base_Pred <- predict.train(model_glm, ibm_reduced, type = "raw")
confusionMatrix(base_Pred, ibm_reduced$Attrition, mode = "prec_recall") 

# assess winner model
xgb_Pred <- predict.train(model_xgbTree, ibm_reduced, type = "raw")
confusionMatrix(xgb_Pred, ibm_reduced$Attrition, mode = "prec_recall")


#-------------------------------------------------------------------------------
# PREDICTION ON ACTIVE EMPLOYEES
#-------------------------------------------------------------------------------

# save employee index to keep track
ibm_reduced$ID <- ibm_126$EmployeeNumber
# save employees who did not churn
no_churn <- subset(ibm_reduced,Attrition == "No")
# get class probabilities from employees who are still active
probs <- predict.train(model_xgbTree, no_churn, type = "prob")
# merge with employee data
still_active <- data.table(no_churn,probs)

# show employees who are most at risk to leave soon
head(still_active[order(-Yes)],5)


#-------------------------------------------------------------------------------
# VISUALIZATIONS 
#-------------------------------------------------------------------------------

library(ggplot2)

# specify custom theme
theme_custom <- function(){ 
  
  theme(
    
    # background
    strip.background = element_rect(colour = "black", fill = "lightgrey"),
    
    # x-axis already represented by legend
    axis.title.x = element_blank(),               
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank(),                
    
    # legend box
    legend.box.background = element_rect())
  
}

# specify colors
library(RColorBrewer)
myCol <- rbind(brewer.pal(8, "Blues")[c(5,7,8)],
               brewer.pal(8, "Reds")[c(4,6,8)])



# Correlation: The higher employee's job satisfaction,the lower the turnover rate.
# Boxplot
plot_jobsatisfaction <- ggplot(ibm_126, aes(x = Attrition, y = JobSatisfaction,fill = Attrition))+
  geom_boxplot(width=0.1)+
  scale_fill_manual(values = myCol)+
  ylab("Job Satisfaction")+
  xlab("Employee Attrition")+
  theme_custom()+
  ggtitle("Are leavers less satisfied than stayers?")
ggsave(plot_jobsatisfaction, file=paste0(current_date,"_","Distribution_Jobsatisfaction.png"), path=path_plot,width = 15, height = 10, units = "cm")



# Job satisfaction amplifies the effect of monthly income on attrition. 
# Hypothesis: Someone who is satisfied with the job tolerates a low to moderate income and stays

# Monthly income vs. job satisfaction -> turnover?
plot_income <- ggplot(ibm_126, aes(x = as.factor(JobSatisfaction), y = MonthlyIncome,fill = Attrition))+
  geom_boxplot()+
  scale_fill_manual(values = myCol)+
  ylab("Monthly Income")+
  xlab("Job Satisfaction")+
  theme_custom()+
  ggtitle("How are monthly income and job satisfaction related to employee attrition?")
ggsave(plot_income, file=paste0(current_date,"_","Interaction_Income.png"), path=path_plot,width = 20, height = 10, units = "cm")

# Pay competetiveness vs. job satisfaction -> turnover?
plot_CompRatio <- ggplot(ibm_126, aes(x = as.factor(JobSatisfaction), y = CompensationRatio,fill = Attrition))+
  geom_boxplot()+
  scale_fill_manual(values = myCol)+
  ylab("Competensation Ratio")+
  xlab("Job Satisfaction")+
  theme_custom()+
  ggtitle("How are pay competetiveness and job satisfaction affect employee turnover?")
# supported if we plot compa_ratio on the y axis
# People who leave even if they are highly satisfied do not seem to go for monetary reasons
ggsave(plot_CompRatio, file=paste0(current_date,"_","Interaction_CompRatio.png"), path=path_plot,width = 20, height = 10, units = "cm")


# check distribution 
sanity_check <- ibm_126[, list(count = .N, rate = (.N/nrow(ibm_126))), by = JobSatisfaction]
sanity_check


# What about personal reasons?
library(AppliedPredictiveModeling)
transparentTheme(trans = .4)
library(caret)

featurePlot(x = ibm_126[, c(7,9,18,20)], 
            y = as.factor(ibm_126$Attrition),
            plot = "density", 
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(4, 1), 
            auto.key = list(columns = 2))

# get mosaic plots
library(vcd)
library(vcdExtra)

mosaic(~ Attrition + EnvironmentSatisfaction, data = ibm_126, shade = TRUE, legend = TRUE)
mosaic(~ Attrition + JobInvolvement, data = ibm_126, shade = TRUE, legend = TRUE)
mosaic(~ Attrition + WorkLifeBalance, data = ibm_126, shade = TRUE, legend = TRUE)
mosaic(~ Attrition + RelationshipSatisfaction, data = ibm_126, shade = TRUE, legend = TRUE)

