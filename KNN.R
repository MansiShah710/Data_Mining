# Part 2: Classification Model: K Nearest Neighbors
# A.
# 1.	Import and explore data using read.csv, str and summary.
donation <- read.csv(file.choose())
str(donation)
summary(donation)

# 2.	Transform the target variable B to a factor variable.
donation$B <- factor(donation$B)
str(donation)

# 3.	Partition the data to training and testing based on the row index order. Use the first 50% of rows as training sample; and the following 25% rows as testing data 1, 
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

# 4.	Load RWeka package. Build the KNN classification model using training sample. Given maximum K = 20, without any weighting strategies, how many K has been selected for optimal performance?
library(RWeka)
KNN_model <- IBk(B~., data = dtrain, control= Weka_control(K = 20, X = TRUE))
KNN_model
#Ans: 16 K has been selected for optimal performance

# 5.	Build another KNN classification model using training sample, with weights adding to nearest neighbors (weight = the inverse of their distance).Given maximum K = 20, how many K has been selected for optimal performance?
KNN_Model2 <- IBk(B~., data = dtrain, control=Weka_control(K = 20, X = TRUE, I=TRUE))
KNN_Model2
#Ans: 12 K has been selected for optimal performance.

# 6.	Build a third KNN classification model using training sample, with weights adding to nearest neighbors (weight = 1-distance).
# Given maximum K = 20, how many K has been selected for optimal performance?
KNN_Model3 <- IBk(B~., data = dtrain, control=Weka_control(K = 20, X =TRUE, F=TRUE))
KNN_Model3
#Ans: 16 K has been selected for optimal performaces.

# 7.	Generate predictions for both testing sample 1 and testing sample 2 using the first KNN model built in question 4.
KNN_prediction1 <- predict(KNN_model, dtest1)
KNN_prediction2 <- predict(KNN_model, dtest2)

# 8.	Load the rminer package. Evaluate the prediction performance for testing sample 1 using mmetric function (seven evaluation metrics: accuracy, prediction for both classes, recall for both classes and F measure for both classes).
library(rminer)
mmetric(dtest1$B, KNN_prediction1, c("ACC", "PRECISION", "TPR", "F1"))

# 9.	
# Determine which class is class 1 and which class is class 2 in the output.
str(KNN_prediction1)
# Ans: class 1 is "0" and class 2 is "1".

# What are the values for the seven evaluation metrics respectively? 
#        ACC PRECISION1 PRECISION2       TPR1       TPR2        F11        F12 
#   79.67914   81.54762   63.15789   95.13889   27.90698   87.82051   38.70968 

# Based on the evaluation metrics, if we intend to predict only class “0” accurately, do we have a good model? 
# Ans: yes, we have a good model to accurately predict class "0".

# If we intend to predict only class “1” accurately, do we still have a good model?
# Ans: We do not have a good model to accurately predict class "1".

# 10.	
# Evaluate the prediction performance for testing sample 2 using mmeitrc function (seven evaluation metrics). 
mmetric(dtest2$B, KNN_prediction2, c("ACC", "PRECISION", "TPR", "F1"))
# ACC       PRECISION1 PRECISION2    TPR1       TPR2        F11        F12 
# 87.16578   87.16578    0.00000  100.00000    0.00000   93.14286    0.00000 

# Do we have significantly huge differences between the performance on the two testing samples?
# Ans: yes, we can see significantly huge differences between the performance of the two testing samples.

###########################################

# Part 4
# 1)	Import and explore data
# a.	Open FlightDelay.csv and store the results into a data frame, e.g., called datFlight.All of the character values should be imported as factors. Transform specific numeric values such as weather condition, day of week and day of month as factors.
datFlight <- read.csv(file.choose(), stringsAsFactors = TRUE)
datFlight[, 8:10] <- lapply(datFlight[,8:10], factor)

# b.	Use the str() and summary commands to provide a listing of the imported columns and their basic statistics. 
str(datFlight)
summary(datFlight)

# 2)	Prepare data for classification
# a.	Using a seed of 100, randomly select 60% of the rows into training (e.g. called traindata).
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

# b.	Inspect (show) the distributions of the target variable in the subsets. They should preserve the distribution of the target variable in the whole data set. 
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

# 3)	C5.0 decision tree classifiers
str(traindata)
# a.	Build/train a tree model
# i.	Build the tree using the C50 function with default settings 
tree_model1 <- C5.0(traindata[-11], traindata$delay)
tree_model1
# ii.	Show the (textual) model/tree. 
summary(tree_model1)
plot(tree_model1)
# iii.	How many leaves are in the tree?
#Ans: there are 16 leaves
# iv.	What is the predictor that first splits the tree? 
#Ans: Weather is the predictor that splits the table first.

# b.	Find rules (paths) in the tree 
# i.	Find one path in the tree to a leaf node that is classified to ontime. Starting with the condition on the first (or top) branch of the path, write down the conditions on the tree branches belonging to this path. 
#Ans: if (weather = 0) and (daymonth in {1,2,3,6,7,8,9,10,11,12,14,17,19,20,21,22,23,24,31}), and (schedtime > 1525), and (deptime <= 2145), then ontime.

# ii.	How many conditions and how many unique predictors are in your selected rule?
#Ans: There are four conditions and unique predictors are: weather, daymonth, schedtime and deptime.

# iii.	What is this rule’s misclassification error rate (e.g., 20/50 misclassified)?
#Ans:  the misclassification error rate is 38/316 (misclassified).

# iv.	Similarly, describe a rule that classifies an instance to delay.
#Ans: If (weather = 1) then delayed

# v.	What is this rule’s misclassification error?
#Ans: there is no misclassification error.

# vi.	Find a shorter or longer rule with fewer or more conditions for ontine than previous rules. Repeat this for Delay. Show these two rules and their misclassification errors.

# ontime rule:
# if (weather = 0), and (daymonth in {4,5,13,15,16,18,25,26,27,28,29,30})
# and (flightnumber <= 2186), and (dest in {EWR,LGA}), and (deptime <= 1857), then ontime 
#Ans: misclassification errors: (21/186)

# delay rule:
#Ans: if (weather = 0), and (daymonth in {1,2,3,6,7,8,9,10,11,12,14,17,19,20,21,22,23,24,31}),
      # and (schedtime > 1525), and (deptime > 2145), then delayed
#Ans: there is no misclassification error.

# vii.	What are the reasons that long rules are included in a decision tree model?
#Ans: longer rules results in better accuracy and less errors. Also, an important pattern is not missed.

# viii.	What is the disadvantage of a long rule?
#Ans: Longer rules increases the complexity of decision tree. There is also a risk of overfitting the model.

# c.	Apply and evaluate the trained model:
# i.	Generate predictions (i.e. estimations) of the values of the target variable for the testing instances.
prediction1 <- predict(tree_model1, testdata1, type = "class")
summary(prediction1)
# delayed  ontime 
# 45     396 
prediction2 <- predict(tree_model1, testdata2, type = "class")
summary(prediction2)
# delayed  ontime 
# 35     404


# ii.	Generate a confusion matrix that shows the counts of true-positive, true-negative, false-positive and false-negative predictions for both testdata1 and testdata2. Consider Ontime as positive class.
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


# iii.	Generate seven performance metrics - Accuracy (percent of all correctly classified testing instances),
library(rminer)
mmetric(testdata1$delay, prediction1, c("ACC", "PRECISION", "TPR", "F1"))
#       ACC PRECISION1 PRECISION2       TPR1       TPR2        F11        F12 
#   82.08617   57.77778   84.84848   30.23256   94.64789   39.69466   89.48069 
mmetric(testdata2$delay, prediction2, c("ACC", "PRECISION", "TPR", "F1"))
#        ACC PRECISION1 PRECISION2       TPR1       TPR2        F11        F12 
#  84.51025   74.28571   85.39604   30.58824   97.45763   43.33333   91.02902 

# iv.	Report all performance differences in the same performance metric between the two data sets that are more than 10%.
#Ans: PRECISION1: 28.5%

# Does this tree generalize well over these two testing sets? Explain the reason for your answer.
#Ans: For class "1", this tree generalizes well over the two testing sets but for class "0", precision1 has more than 10% differences thatswhy, class '0' do not generalize well.

# 4)	 C50 pruning
# a.	Build another C50 tree using the train set by changing the confidence factor to 0.05 (i.e. CF=0.05 in C50 function’s control).
tree_model2 <-  C5.0(traindata[-11], traindata$delay, control = C5.0Control(CF = 0.05))
tree_model2
summary(tree_model2)
plot(tree_model2)
# b.	Describe the size of the tree built. 
#Ans: the tree size is 10

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


# d.	Report all performance differences in the same performance metric between the two data sets that are more than 10%. 
#Ans: PRECISION1 has more than 10% difference: 34.04%

# Does this tree generalize well over these two testing sets? Explain the reason for your answer.
#Ans: For class "1", this tree generalizes well over the two testing sets but for class "0", precision1 has more than 10% differences thatswhy class '0' do not generalize well.

# e.	Would you adopt this pruning setting? Why or why not?
#Ans: yes, I would adopt this pruning setting because it resulted in less complex tree. All the metrics except PRECISION1 have less than 10% differences.

# 5)	 Returning to the default pruning setting, build another C50 tree with only two predictors of your choice. 
# a.	Build a tree using the predictors of your choice in the train set.
tree_model3 <-  C5.0(delay~ deptime + schedtime, data = traindata)
summary(tree_model3)

# b.	Describe the size of the tree built. 
#Ans: 20

# c.	Generate predictions, confusion matrices and performance metrics using two test sets.
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

# d.	Report all performance differences in the same performance metric between the two data sets that are more than 10%. # Does this tree generalize well over these two testing sets? 

#Ans: no metric with more than 10% differences.
#Ans: yes, this tree generalize well over the two testing samples.
  
  
  