# Cost Sensitive Learning

# read the csv file 
datFlight <- read.csv(file.choose(), stringsAsFactors = TRUE)
datFlight[,8:10] <- lapply(datFlight[,8:10],factor)

str(datFlight)
summary(datFlight)

# c. 
# install.packages("caret")
library(caret)
datTrain<-list()
datTest<-list()
n <- 3
Sum_Test_flights <- 0

for (i in 1:n)
{ 
  seed_Number<-100+400*(i-1) # seed will be 100, 500, 900 i=1, 2 and 3
  set.seed(seed_Number)
  inTrain <- createDataPartition(datFlight$delay, p=0.67, list=FALSE)
  datTrain[[i]] <- datFlight[inTrain,]
  datTest[[i]] <- datFlight[-inTrain,]
  Sum_Test_flights <-Sum_Test_flights + nrow(datTest[[i]]) 
  print(paste("the proportion table of traindata",i,sep=""))
  print(prop.table(table(datTrain[[i]]$delay)),sep="")
  print(paste("the proportion table of testdata",i,sep=""))
  print(prop.table(table(datTest[[i]]$delay)))
}
# the average number of test instances over all three test sets
Avg_Test_flights<-Sum_Test_flights/n
Avg_Test_flights #726

## Build Models and Evaluations

library(e1071)
library(rminer)
NB_sum_TP <- 0
NB_sum_TN <- 0
NB_sum_FP <- 0
NB_sum_FN <- 0

NB_total_result<-array()

for(i in 1:n)
{

  NBmodel <- naiveBayes(delay ~ ., data=datTrain[[i]])
  print(NBmodel)
  NBpredictions <- predict(NBmodel, datTest[[i]])
  NB_confusionMatrix<-confusionMatrix(NBpredictions,datTest[[i]]$delay,  positive = "ontime",  dnn = c("Prediction","True"))
  print(NB_confusionMatrix) 
  NB_sum_TP <- NB_sum_TP+NB_confusionMatrix$table[2,2]
  NB_sum_FP <- NB_sum_FP+NB_confusionMatrix$table[2,1]
  NB_sum_TN <- NB_sum_TN+NB_confusionMatrix$table[1,1]
  NB_sum_FN <- NB_sum_FN+NB_confusionMatrix$table[1,2]
  NB_result<-mmetric(datTest[[i]]$delay, NBpredictions,c("ACC","PRECISION","TPR","F1"))
  NB_total_result<-cbind(NB_total_result,NB_result)
}

NB_total_result
rowMeans(NB_total_result[1:7,-1])

# ACC       PRECISION1 PRECISION2     TPR1    TPR2        F11        F12 
# 79.70615   47.17876   85.46703   36.40662   90.14245   41.07549   87.74101 

NB_mean_TP <- NB_sum_TP/n
NB_mean_TN <- NB_sum_TN/n
NB_mean_FP <- NB_sum_FP/n
NB_mean_FN <- NB_sum_FN/n


#  average net-benefit per customer over all three testing results
cost <- NB_mean_FP*(1000) + NB_mean_FN*(50)
Benefit <- NB_mean_TN * 500
Net_Benefit <- (Benefit-cost)/Avg_Test_flights
Net_Benefit
# -92.1258

# cost matrix to specify the cost of misclassifying a delay flight as a on time flight to be 10 times the cost of misclassifying a on time to delay.  

matrix_dim <- list(c("Predicted_delayed", "Predicted_ontime"), c("delayed", "ontime"))
costMatrix <- matrix(c(0,10,1,0), nrow = 2, dimnames = matrix_dim)
print(costMatrix)

# In a For loop, build, predict and evaluate C50 classifiers using this cost matrix with three pairs of train and test sets 
cost_sum_TP <- 0
cost_sum_TN <- 0
cost_sum_FP <- 0
cost_sum_FN <- 0

library(e1071)
library(rminer)

library(C50)
cost_total_result <- array()
for(i in 1:n)
{
  
  cost_model <- C5.0(delay~., data = datTrain[[i]], cost= costMatrix)
  cost_predictions <- predict(cost_model, datTest[[i]])
  cost_confusionMatrix<-confusionMatrix(cost_predictions,datTest[[i]]$delay,  positive = "ontime",  dnn = c("Prediction","True"))
  print(cost_confusionMatrix)
  cost_sum_TP <- cost_sum_TP+cost_confusionMatrix$table[2,2]
  cost_sum_FP <- cost_sum_FP+cost_confusionMatrix$table[2,1]
  cost_sum_TN <- cost_sum_TN+cost_confusionMatrix$table[1,1]
  cost_sum_FN <- cost_sum_FN+cost_confusionMatrix$table[1,2]
  cost_result<-mmetric(datTest[[i]]$delay, cost_predictions,c("ACC","PRECISION","TPR","F1"))
  cost_total_result<-cbind(cost_total_result,cost_result)
  }
rowMeans(cost_total_result[1:7,-1])

# ACC PRECISION1 PRECISION2       TPR1       TPR2        F11        F12 
# 52.43343   26.76842   91.56841   82.74232   45.12821   40.40169   60.27705 


# the average net-benefit per customer over all three testing results.
cost <- cost_sum_FP/n*(1000) + cost_sum_FN/n*(50)
Benefit <- cost_sum_TN/n * 500
Net_Benefit <- (Benefit-cost)/Avg_Test_flights
Net_Benefit # 24.72452

##########################
# Linear Regression

# Import and explore data using read.csv, str, summary.  
houseprice <- read.csv(file.choose())
str(houseprice)
summary(houseprice)

# Split the data frame. Set seed equals to 500. Use a random 50% sample of data for 	training. 
# Randomly divide the rest of the data into two even-sized (25%) test sets for 	evaluation.
set.seed(500)
inTrain <- createDataPartition(y= houseprice$Price, p = 0.5, list = FALSE)
datTrain <- houseprice[inTrain,]
testdata <- houseprice[-inTrain,]
inTest <- createDataPartition(testdata$Price, p = 0.5, list = FALSE)
testdata1 <- testdata[inTest,]
testdata2 <- testdata[-inTest,]

#  Train a base linear regression model with all predictors (all the attributes besides Price) 	on the training sample and show the summary of the trained model.  

##Building the model based on the training sample
predict_model <- lm(Price ~., data = datTrain)
summary(predict_model)


# The model is a good fit because almost 86% of response variable variations is explained by the model.
# Also, adjusted R2 is the preferred measure as it adjusts for the number of variables considered.It is always lower than R-squared. It is 0.8365 in this case.The model is still a good fit.
# F statistic is 41.28 which means there is a good relationship between predictor and response variables. The value of F statistic should be further from 1.
# Also, the overall p value is less than the significance level. 

# SqFt, Offers, BrickYes and NeighbourhoodWest. Also, Bathrooms has significance relationship with Price but it is less significant than the ones mentioned before.

# the base linear regression model’s predictions for Price in the two test sets. 

##Predict the expenses for the examples in the testing sample 1
insur_prediction1 <- predict(predict_model, testdata1)
insur_prediction1

insur_prediction2 <- predict(predict_model, testdata2)
insur_prediction2

# Use mmetric to generate and show eight metrics (shown in tutorials) to evaluate 	prediction accuracy and explanatory power of the model. 

library(rminer)
mmetric(testdata1$Price, insur_prediction1, c("MAE", "RMSE", "MAPE", "RMSPE", 
                                                 "RRSE","RAE", "R2", "COR"))

mmetric(testdata2$Price, insur_prediction2, c("MAE", "RMSE", "MAPE", "RMSPE", 
                                             "RRSE","RAE", "R2", "COR"))
