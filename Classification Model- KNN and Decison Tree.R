# Classification Model: K Nearest Neighbors

# Import and explore data using read.csv, str and summary.
donation <- read.csv(file.choose())
str(donation)
summary(donation)

# Transform the target variable B to a factor variable.
donation$B <- factor(donation$B)
str(donation)

# Partition the data to training and testing based on the row index order. Use the first 50% of rows as training sample; and the following 25% rows as testing data 1, 
# and the last 25% rows as testing data 2. How many rows of observations/instances are in training sample? How many rows in testing sample 1? How many rows in testing sample 2?
library(caret)
dtrain <- donation[1:374,]
dtest1 <- donation[375:561,]
dtest2 <- donation[562:748,]
nrow(dtrain)
# there are 374 rows of observations in the training sample.
nrow(dtest1)
# there are 187 rows in testing sample 1.
nrow(dtest2)
# there are 187 rows in testing sample 2.

# Load RWeka package. Build the KNN classification model using training sample. Given maximum K = 20, without any weighting strategies, how many K has been selected for optimal performance?
library(RWeka)
KNN_model <- IBk(B~., data = dtrain, control= Weka_control(K = 20, X = TRUE))
KNN_model
#Ans: 16 K has been selected for optimal performance

# Build another KNN classification model using training sample, with weights adding to nearest neighbors (weight = the inverse of their distance).Given maximum K = 20, how many K has been selected for optimal performance?
KNN_Model2 <- IBk(B~., data = dtrain, control=Weka_control(K = 20, X = TRUE, I=TRUE))
KNN_Model2
#Ans: 12 K has been selected for optimal performance.

# Build a third KNN classification model using training sample, with weights adding to nearest neighbors (weight = 1-distance).
# Given maximum K = 20, how many K has been selected for optimal performance?
KNN_Model3 <- IBk(B~., data = dtrain, control=Weka_control(K = 20, X =TRUE, F=TRUE))
KNN_Model3
#Ans: 16 K has been selected for optimal performaces.

# Generate predictions for both testing sample 1 and testing sample 2 using the first KNN model built in question 4.
KNN_prediction1 <- predict(KNN_model, dtest1)
KNN_prediction2 <- predict(KNN_model, dtest2)

# Load the rminer package. Evaluate the prediction performance for testing sample 1 using mmetric function (seven evaluation metrics: accuracy, prediction for both classes, recall for both classes and F measure for both classes).
library(rminer)
mmetric(dtest1$B, KNN_prediction1, c("ACC", "PRECISION", "TPR", "F1"))
	
# Determine which class is class 1 and which class is class 2 in the output.
str(KNN_prediction1)
# Ans: class 1 is "0" and class 2 is "1".

# Based on the evaluation metrics, if we intend to predict only class “0” accurately, do we have a good model? 
# Ans: yes, we have a good model to accurately predict class "0".

# If we intend to predict only class “1” accurately, do we still have a good model?
# Ans: We do not have a good model to accurately predict class "1".
	
# Evaluate the prediction performance for testing sample 2 using mmeitrc function (seven evaluation metrics). 
mmetric(dtest2$B, KNN_prediction2, c("ACC", "PRECISION", "TPR", "F1"))
# ACC       PRECISION1 PRECISION2    TPR1       TPR2        F11        F12 
# 87.16578   87.16578    0.00000  100.00000    0.00000   93.14286    0.00000 

# Do we have significantly huge differences between the performance on the two testing samples?
# Ans: yes, we can see significantly huge differences between the performance of the two testing samples.

###########################################
# Decision tree
# Import and explore data

datFlight <- read.csv(file.choose(), stringsAsFactors = TRUE)
datFlight[, 8:10] <- lapply(datFlight[,8:10], factor)

# Use the str() and summary commands to provide a listing of the imported columns and their basic statistics. 
str(datFlight)
summary(datFlight)

# Using a seed of 100, randomly select 60% of the rows into training (e.g. called traindata).
library(caret)
library(C50)
library(rminer)
set.seed(100)
InTrain <- createDataPartition(y = datFlight$delay, p = 0.6, list = FALSE)
head(InTrain,10)
traindata <- datFlight[InTrain,]
testdata <- datFlight[-InTrain,]
# Divide the other 40% of the rows evenly into two holdout test/validation sets (e.g., called testdata1 and testdata2). 
Intest <- createDataPartition(testdata$delay, p = 0.5, list = FALSE)
testdata1 <- testdata[Intest,]
testdata2 <- testdata[-Intest,]

# Inspect (show) the distributions of the target variable in the subsets. They should preserve the distribution of the target variable in the whole data set. 
nrow(datFlight) # 2201
nrow(traindata) # 1321
nrow(testdata1) # 441
nrow(testdata2) # 439
prop.table(table(traindata$delay))
# delayed    ontime 
# 0.1945496 0.8054504 
prop.table(table(testdata1$delay))
# delayed    ontime 
# 0.1950113 0.8049887 
prop.table(table(testdata2$delay))
# delayed    ontime 
# 0.1936219 0.8063781

# C5.0 decision tree classifiers
str(traindata)
# a.	Build/train a tree model
# i.	Build the tree using the C50 function with default settings 
tree_model1 <- C5.0(traindata[-11], traindata$delay)
tree_model1
# ii.	Show the (textual) model/tree. 
summary(tree_model1)
plot(tree_model1)
# Apply and evaluate the trained model:
# Generate predictions (i.e. estimations) of the values of the target variable for the testing instances.
prediction1 <- predict(tree_model1, testdata1, type = "class")
summary(prediction1)
# delayed  ontime 
# 45     396 
prediction2 <- predict(tree_model1, testdata2, type = "class")
summary(prediction2)
# delayed  ontime 
# 35     404


# Generate a confusion matrix that shows the counts of true-positive, true-negative, false-positive and false-negative predictions for both testdata1 and testdata2. Consider Ontime as positive class.
confusionMatrix(prediction1,testdata1$delay, positive = "ontime", dnn= c("Prediction", "True"))
#           True
# Prediction delayed ontime
# delayed      26     19
# ontime       60    336
confusionMatrix(prediction2,testdata2$delay, positive = "ontime", dnn =c("Prediction", "True"))
#         True
# Prediction delayed ontime
# delayed      26      9
# ontime       59    345


# Generate seven performance metrics - Accuracy (percent of all correctly classified testing instances),
library(rminer)
mmetric(testdata1$delay, prediction1, c("ACC", "PRECISION", "TPR", "F1"))
#       ACC PRECISION1 PRECISION2       TPR1       TPR2        F11        F12 
#   82.08617   57.77778   84.84848   30.23256   94.64789   39.69466   89.48069 
mmetric(testdata2$delay, prediction2, c("ACC", "PRECISION", "TPR", "F1"))
#        ACC PRECISION1 PRECISION2       TPR1       TPR2        F11        F12 
#  84.51025   74.28571   85.39604   30.58824   97.45763   43.33333   91.02902 
# 4)	 C50 pruning
# a.	Build another C50 tree using the train set by changing the confidence factor to 0.05 (i.e. CF=0.05 in C50 function’s control).
tree_model2 <-  C5.0(traindata[-11], traindata$delay, control = C5.0Control(CF = 0.05))
tree_model2
summary(tree_model2)
plot(tree_model2)
# the tree size is 10

# c.	Generate predictions, confusion matrixes and performance metrics using two test sets.
prediction21 <- predict(tree_model2, testdata1, type = "class")
summary(prediction21)
# delayed  ontime 
# 40     401 
prediction22 <- predict(tree_model2, testdata2, type = "class")
summary(prediction21)
# delayed  ontime 
# 40     401 

# confusion matrix
confusionMatrix(prediction21,testdata1$delay, positive = "ontime", dnn= c("Prediction", "True"))
#          True
# Prediction delayed ontime
# delayed      21     19
# ontime       65    336
confusionMatrix(prediction22,testdata2$delay, positive = "ontime", dnn =c("Prediction", "True"))
#          True
# Prediction delayed ontime
# delayed      19      8
# ontime       66    346

# metric
mmetric(testdata1$delay, prediction21, c("ACC", "PRECISION", "TPR", "F1"))
#        ACC PRECISION1 PRECISION2    TPR1       TPR2        F11        F12 
# 80.95238   52.50000   83.79052   24.41860   94.64789   33.33333   88.88889 

mmetric(testdata2$delay, prediction22, c("ACC", "PRECISION", "TPR", "F1"))
#      ACC  PRECISION1 PRECISION2   TPR1       TPR2        F11        F12 
# 83.14351   70.37037   83.98058   22.35294   97.74011   33.92857   90.33943

#  Returning to the default pruning setting, build another C50 tree with only two predictors of your choice. 
# Build a tree using the predictors of your choice in the train set.
tree_model3 <-  C5.0(delay~ deptime + schedtime, data = traindata)
summary(tree_model3)

# Tree size is20

# Generate predictions, confusion matrices and performance metrics using two test sets.
prediction31 <- predict(tree_model3, testdata1, type = "class")
summary(prediction31)
# delayed  ontime 
# 39     402 
prediction32 <- predict(tree_model3, testdata2, type = "class")
summary(prediction32)
# delayed  ontime 
# 39     400

# confusion matrix
confusionMatrix(prediction31,testdata1$delay, positive = "ontime", dnn= c("Prediction", "True"))
#           True
# Prediction delayed ontime
# delayed      38      1
# ontime       48    354

confusionMatrix(prediction32,testdata2$delay, positive = "ontime", dnn =c("Prediction", "True"))
#           True
# Prediction delayed ontime
# delayed      36      3
# ontime       49    351

# metric

library(rminer)
mmetric(testdata1$delay, prediction31, c("ACC", "PRECISION", "TPR", "F1"))
# ACC PRECISION1 PRECISION2       TPR1       TPR2        F11        F12 
# 88.88889   97.43590   88.05970   44.18605   99.71831   60.80000   93.52708 
mmetric(testdata2$delay, prediction32, c("ACC", "PRECISION", "TPR", "F1"))
# ACC PRECISION1 PRECISION2       TPR1       TPR2        F11        F12 
# 88.15490   92.30769   87.75000   42.35294   99.15254   58.06452   93.10345 

  
  
  