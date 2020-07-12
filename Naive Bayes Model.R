
# Assignment 5: Naïve Bayes Model

# e1071 naiveBayes classifiers

datFlight <- read.csv(file.choose())
str(datFlight)
datFlight[, 8:10] <- lapply(datFlight[,8:10], factor)

# Using the str() and summary commands to provide a listing of the imported columns and their basic statistics
str(datFlight)
summary(datFlight)

# Loading the caret package. Using a seed of 100, 500 and 900, randomly select 67% of a file three times into three training sets and save other 33% in three testing sets respectively. 

install.packages("caret")
library(caret)
datTrain <- list()
datTest <- list()
n <- 3
sumtest <-  0

for(i in 1:n)
{
  ##define the seed number for iteration
  seedNumber <- 100+400*(i-1) ##seed number will be 100, 500, 900
  set.seed(seedNumber)
  InTrain<-createDataPartition(datFlight$delay, p= 0.67, list = FALSE)
  
  ##save the training dataset as the element at index i of the list
  datTrain[[i]]<-datFlight[InTrain,]
  datTest[[i]] <- datFlight[-InTrain,]
  
  sumtest <- nrow(datTest[[i]])+ sumtest
}

sumtest
# the average number of examples in testing sets.
sumtest/n
  
library(e1071)
library(rminer)

NB_total_result <- array()

for(j in 1:n)
{
  # Build a Naïve Bayesian models using the naiveBayes function in e1071 with each train data.
  NBmodel[[j]] <-naiveBayes(delay~., data = datTrain[[j]])
  
  ##  the values of A-priori probabilities - P(Delay) for the delay class and P(Ontime) for the ontime class for each model
  print(NBmodel[[j]])  # model1: P(delay) is 0.1945763 and P(Ontime) is 0.8054237
                       # model2: P(delay) is 0.1945763 and P(Ontime) is 0.8054237
                       # model3: P(delay) is 0.1945763 and P(Ontime) is 0.8054237
  
  ## Generate predictions (i.e. estimations) of the values of the target variable for instances in each test data. 
  NB_prediction <- predict(NBmodel[[j]], datTest[[j]])
  
    ##Generate the confusion matrix
    NB_matrix <- confusionMatrix(NB_prediction, datTest[[j]]$delay, positive = "delayed",
                               dnn = c("Prediction", "True"))
    print(NB_matrix)
  
    #             True
    # Prediction  delayed ontime
    #  delayed      51     51
    #  ontime       90    534
  
    ##Generate the 7 evaluation metrics using mmetric
   NB_result <- mmetric(datTest[[j]]$delay, NB_prediction,c("ACC", "PRECISION", "TPR", "F1"))
  
   ##combine new result to the array
    NB_total_result <- rbind(NB_total_result, NB_result)
  }

    ##final result for mmetric function
    NB_total_result

    ##check the structure of the predictions
    str(NB_prediction)
    ## delayed is class 1, ontime is class 2







