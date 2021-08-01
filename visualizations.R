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
