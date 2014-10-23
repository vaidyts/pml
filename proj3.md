---
title: "Modeling and Prediction using Weight Lifting dataset"
output: html_document
---

### Executive Summary

Under the category of Human Activity Recognition (HAR) is a dataset called Weight Lifting Exercise (WLE). This dataset contains data generated during weight lifting performed by 6 participants, who were asked to lift the weights using different techniques with various sensors attached to their body, the sensor data being the predictors and the weight lifting technique being the response variable. We are given a training and test dataset. The objective of this project is to develop a model based on the provided training data and give the output of the test samples.

The main steps in the above exercise involved reading the data, looking at the summary statistics, selecting the variables for modeling, partitioning the given samples into train and test subsets, model development using the training data, testing the model on the test subset, error estimation, and finally prediction of the response for the sample set provided. Different models were tried out and in the end, the best model turned out to be a random Forest model which outperformed by far the other techniques. The model was able to obtain a near perfect classification of the training and test data, and a perfect classification of the assignment data. Details are given below:

### Data input, exploratory analysis, and variable selection for modeling
The given data contains 19622 rows (observations) and 160 columns (variables). This is split into a 75/25 partition for training and testing purposes respectively. On the training subset, summary statistics of each of the variables is collected. It is seen that 96 of the 160 variables have close to 98% (19216/19620) "NA" or "blank" entries. It is decided to remove these variables from the training. 4 other variables related to serial number and timestamp are also removed from the dataset. This pruning results in a reduced dataset containing 56 variables. 


Reading the data

```r
train_given <- read.csv("pml-training.csv")
test_given <- read.csv("pml-testing.csv")
indexTrain <- createDataPartition(y=train_given$classe,p=0.75,list=FALSE)
train <- train_given[indexTrain,]
test <- train_given[-indexTrain,]
```
Calculation of summary statistics of the variables

```r
fn <- "summary.txt"
raw <- train_given
for (i in 1:ncol(raw)) { 
  x <- names(raw)[i]
  write(x, file=fn, append=TRUE, sep="\t",ncolumns = 7)
  x <- names(summary(raw[,i], maxsum=7))
  write(x,file=fn, append=TRUE, sep="\t", ncolumns = 7)
  x <- summary(raw[,i], maxsum=7)
  write(x,file=fn, append=TRUE, sep="\t", ncolumns = 7)
  write("\n", file=fn, append=TRUE)
}
```
Variable selection for modeling

```r
varindex <- c(2, 6:11, 37:49, 60:68, 84:86, 102, 113:124, 140, 151:159)
train_x <- train[,varindex]
train_y <- train[,160]
train_xy <- data.frame(train_x,classe=train_y)
test_x <- test[,varindex]
test_y <- test[,160]
validate <- test_given[,varindex]
```
### Model development 
In this section, different models are attempted

### Model=rpart
The first model tried is rpart which stands for recursive partitioning. The model is run with default options. The resulting accuracy is ~ 0.6 (60%) on the training subset. Attempts are made to improve this further with the help of other options such as center, scale, pca, cv, etc. Unfortunately no significant improvement is seen and the overall accuracy obtained is ~ 60%

```r
fit_rpart_1 <- train(classe~., method="rpart", data=train_xy)
fit_rpart_1
```

```
## CART 
## 
## 14718 samples
##    55 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp    Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.03  0.6       0.50   0.03         0.04    
##   0.04  0.5       0.38   0.05         0.08    
##   0.12  0.3       0.04   0.04         0.06    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.0282.
```


```r
fit_rpart_2 <- train(classe~., method="rpart", data=train_xy,
                     preProcess=c("center","scale"))
fit_rpart_3 <- train(classe~., method="rpart", data=train_xy,
                     preProcess=c("center","scale","pca"))
fit_rpart_4 <- train(classe~., method="rpart", data=train_xy,
                     preProcess=c("center","scale","pca"),trControl=trainControl(method="cv"))
```

### Model=tree
In this section, tree based algorithms are used. With the default options we get a misclassification 
rate of 0.22 (22%) which implies an accuracy rate of 0.78 (78%). This is definitely an improvement over the previous section. Changing the control parameters does not result in a significant improvement.

```r
fit_tree_1 <- tree(classe~.,data=train_xy)
summary(fit_tree_1)
```

```
## 
## Classification tree:
## tree(formula = classe ~ ., data = train_xy)
## Variables actually used in tree construction:
##  [1] "roll_belt"         "pitch_forearm"     "num_window"       
##  [4] "roll_forearm"      "magnet_dumbbell_z" "magnet_dumbbell_y"
##  [7] "yaw_belt"          "user_name"         "accel_dumbbell_z" 
## [10] "pitch_belt"        "accel_dumbbell_y" 
## Number of terminal nodes:  25 
## Residual mean deviance:  1.18 = 17400 / 14700 
## Misclassification error rate: 0.22 = 3236 / 14718
```

### Model=randomForest
Here the random forest approach is attempted. With default parameters, the misclassification rate is only 0.26%. It is attempted to improve the models further by changing the control parameters such as ```importance```,```proximity```, ```mtry```, etc. Best results are obtained with ```mtry``` of 11

```r
fit_rf_1 <- randomForest(x=train_x,y=train_y,xtest=test_x,ytest=test_y)
fit_rf_2 <- randomForest(x=train_x,y=train_y,xtest=test_x,ytest=test_y,importance=TRUE)
fit_rf_3 <- randomForest(x=train_x,y=train_y,xtest=test_x,ytest=test_y,proximity=TRUE)
fit_rf_4 <- randomForest(x=train_x,y=train_y,xtest=test_x,ytest=test_y,mtry=10)
fit_rf_5 <- randomForest(x=train_x,y=train_y,xtest=test_x,ytest=test_y,mtry=11)
fit_rf_6 <- randomForest(x=train_x,y=train_y,xtest=test_x,ytest=test_y,mtry=12)
```
Below are the details of the best achieved fit using Random Forest technique and the confusion matrices on the train, test and validation data subsets. We can see from the confusion matrices that the random forest algorithm does extremely well on all of them. Finally we also obtain the outputs predicted for the validation dataset. This concludes the modeling exercise. 

```
## 
## Call:
##  randomForest(x = train_x, y = train_y, xtest = test_x, ytest = test_y,      mtry = 11) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 11
## 
##         OOB estimate of  error rate: 0.2%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4184    0    0    0    1   0.0002389
## B    7 2838    3    0    0   0.0035112
## C    0    5 2562    0    0   0.0019478
## D    0    0    8 2403    1   0.0037313
## E    0    0    0    4 2702   0.0014782
##                 Test set error rate: 0.12%
## Confusion matrix:
##      A   B   C   D   E class.error
## A 1395   0   0   0   0    0.000000
## B    0 949   0   0   0    0.000000
## C    0   2 853   0   0    0.002339
## D    0   0   1 802   1    0.002488
## E    0   0   0   2 899    0.002220
```

```
## 
## Call:
##  randomForest(x = train_x, y = train_y, xtest = validate, mtry = 11) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 11
## 
##         OOB estimate of  error rate: 0.2%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4184    0    0    0    1   0.0002389
## B    7 2840    1    0    0   0.0028090
## C    0    6 2561    0    0   0.0023374
## D    0    0   10 2402    0   0.0041459
## E    0    0    0    4 2702   0.0014782
```

```
## $predicted
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

### Importance measures
It is also important to get an idea of the relative importance of the variables. From the Gini measure we can get a good idea of the variables that are important. The following table and plots provide this information.

```
##                   MeanDecreaseGini MeanDecreaseGini.1 cumratio
## num_window                  1314.2            0.11295   0.1130
## roll_belt                   1008.6            0.08669   0.1996
## pitch_forearm                641.4            0.05513   0.2548
## yaw_belt                     638.8            0.05490   0.3097
## magnet_dumbbell_z            557.9            0.04795   0.3576
## pitch_belt                   527.2            0.04531   0.4029
## magnet_dumbbell_y            523.8            0.04502   0.4480
## roll_forearm                 410.9            0.03531   0.4833
## magnet_dumbbell_x            327.2            0.02813   0.5114
## roll_dumbbell                283.1            0.02433   0.5357
```

![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-101.png) ![plot of chunk unnamed-chunk-10](figure/unnamed-chunk-102.png) 

### Summary / Conclusion
In summary, we have developed a model of the WLE dataset by looking at different models and selecting the random Forest model, trained and tested it, and successfully used it for prediction of new observations with excellent accuracy.
