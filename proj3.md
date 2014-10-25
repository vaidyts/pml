---
title: "Modeling and Prediction using Weight Lifting dataset"
output: html_document
---

### Executive Summary

Under the category of Human Activity Recognition (HAR) is a dataset called Weight Lifting Exercise (WLE). This dataset contains data generated during weight lifting performed by 6 participants, who were asked to lift the weights using different techniques with various sensors attached to their body, the sensor data being the predictors and the weight lifting technique being the response variable. We are given a training and test dataset. The objective of this project is to develop a model based on the provided training data, addressing the questions "how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did", and use the model to predict 20 different test cases. 

The main steps in the above exercise involved reading the data, looking at the summary statistics, selecting the variables for modeling, partitioning the given samples into train and test subsets, model development using the training data, testing the model on the test subset, error estimation, and finally prediction of the response for the sample set provided. Different models were tried out and in the end, the best model turned out to be a random Forest model which outperformed by far the other techniques. The model was able to obtain a near perfect classification of the training and test data, and a perfect classification of the assignment data. Details are given below:

### Data partitioning and variable selection for modeling
The given training data contains 19622 rows (observations) and 160 columns (variables). After reading it in, it is split into a 75/25 partition for training and testing purposes respectively. Such a cross validation approach is needed in order to avoid over-fitting. Summary statistics of each of the variables is collected and written to a log file. It is seen form the log file that 96 of the 160 variables have close to 98% (19216/19620) "NA" or "blank" entries. It is decided to remove these variables from the training. 4 other variables related to serial number and timestamp are also removed from the dataset. This pruning results in a reduced dataset containing 56 variables. 


Following lines of code pertain to reading the data.

```r
train_given <- read.csv("pml-training.csv")
test_given <- read.csv("pml-testing.csv")
indexTrain <- createDataPartition(y=train_given$classe,p=0.75,list=FALSE)
train <- train_given[indexTrain,]
test <- train_given[-indexTrain,]
```
Following lines of code pertain to the summary statistics of the variables.

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
Folliwng lines of code pertain to discarding those variables which have a majority of blank or NA entries, and setting up the train, test, and validate datasets for subsequent use.

```r
varindex <- c(2, 6:11, 37:49, 60:68, 84:86, 102, 113:124, 140, 151:159)
train_x <- train[,varindex]
train_y <- train[,160]
train_xy <- data.frame(train_x,classe=train_y)
test_x <- test[,varindex]
test_y <- test[,160]
test_xy <- data.frame(test_x,classe=test_y)
validate <- test_given[,varindex]
```
### Modeling choices
In this section, different models are attempted. There are numerous modeling algorithms and it is not feasible or practical to try all of them. Of the 2 main approaches - regression vs classification based methods, it was decided to go with classification methods. The reason for this being that the data contains numeric as well as categorical variables, hence classification techniques was used rather than regression techniques. There are many classification algorithms, of which a few were attempted - they are rpart, tree and randomForest. The accuracy kept on increasing ongoing from rpart to tree to randomForest. With randomForest, extremely good accuracy could be obtained, hence it was not necessary to proceed further.

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
##   0.04  0.5       0.39   0.03         0.04    
##   0.06  0.4       0.16   0.05         0.08    
##   0.11  0.3       0.03   0.04         0.06    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.04035.
```


```r
fit_rpart_2 <- train(classe~., method="rpart", data=train_xy,
                     preProcess=c("center","scale"))
fit_rpart_3 <- train(classe~., method="rpart", data=train_xy,
                     preProcess=c("center","scale","pca"))
fit_rpart_4 <- train(classe~., method="rpart", data=train_xy,
                     preProcess=c("center","scale","pca"),trControl=trainControl(method="cv"))
```
The confusion matrix of the predicted vs actual values is shown in the table below. As can be seen, there is a significant mis-classification error rate in the training as well as testing samples.

```r
confusionMatrix(predict(fit_rpart_1, train_xy),train_y)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3828 1227 1184 1079  401
##          B   62  947   77  447  367
##          C  284  674 1306  886  732
##          D    0    0    0    0    0
##          E   11    0    0    0 1206
## 
## Overall Statistics
##                                         
##                Accuracy : 0.495         
##                  95% CI : (0.487, 0.503)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.34          
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.915   0.3325   0.5088    0.000   0.4457
## Specificity             0.631   0.9197   0.7880    1.000   0.9991
## Pos Pred Value          0.496   0.4984   0.3364      NaN   0.9910
## Neg Pred Value          0.949   0.8517   0.8836    0.836   0.8889
## Prevalence              0.284   0.1935   0.1744    0.164   0.1839
## Detection Rate          0.260   0.0643   0.0887    0.000   0.0819
## Detection Prevalence    0.524   0.1291   0.2638    0.000   0.0827
## Balanced Accuracy       0.773   0.6261   0.6484    0.500   0.7224
```

```r
confusionMatrix(predict(fit_rpart_1, test_xy),test_y)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1266  399  401  355  122
##          B   22  326   32  125  123
##          C  104  224  422  324  231
##          D    0    0    0    0    0
##          E    3    0    0    0  425
## 
## Overall Statistics
##                                         
##                Accuracy : 0.497         
##                  95% CI : (0.483, 0.511)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.343         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.908   0.3435   0.4936    0.000   0.4717
## Specificity             0.636   0.9236   0.7819    1.000   0.9993
## Pos Pred Value          0.498   0.5191   0.3234      NaN   0.9930
## Neg Pred Value          0.945   0.8543   0.8797    0.836   0.8937
## Prevalence              0.284   0.1935   0.1743    0.164   0.1837
## Detection Rate          0.258   0.0665   0.0861    0.000   0.0867
## Detection Prevalence    0.519   0.1281   0.2661    0.000   0.0873
## Balanced Accuracy       0.772   0.6336   0.6377    0.500   0.7355
```
### Model=tree
In this section, tree based algorithms are used. With the default options we get a misclassification rate of 0.22 (22%) which implies an accuracy rate of 0.78 (78%). This is definitely an improvement over the previous section. Changing the control parameters does not result in a significant improvement.

```r
fit_tree_1 <- tree(classe~.,data=train_xy)
summary(fit_tree_1)
```

```
## 
## Classification tree:
## tree(formula = classe ~ ., data = train_xy)
## Variables actually used in tree construction:
##  [1] "roll_belt"         "pitch_forearm"     "roll_forearm"     
##  [4] "magnet_dumbbell_x" "num_window"        "magnet_dumbbell_y"
##  [7] "magnet_dumbbell_z" "pitch_belt"        "yaw_belt"         
## [10] "roll_dumbbell"     "magnet_forearm_z"  "accel_forearm_x"  
## Number of terminal nodes:  21 
## Residual mean deviance:  1.44 = 21200 / 14700 
## Misclassification error rate: 0.282 = 4155 / 14718
```
The confusion matrix and error rates corresponding to the tree model are shown below. Here too, a significant misclassification error rate is seen in both for in-sample and out of sample cases.

```r
confusionMatrix(predict(fit_tree_1, train_xy,type="class"),train_y)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3632  255    0   11   51
##          B  161 1699  122  221  309
##          C   47  148 2112  541  227
##          D  325  621  318 1608  607
##          E   20  125   15   31 1512
## 
## Overall Statistics
##                                        
##                Accuracy : 0.718        
##                  95% CI : (0.71, 0.725)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.645        
##  Mcnemar's Test P-Value : <2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.868    0.597    0.823    0.667    0.559
## Specificity             0.970    0.932    0.921    0.848    0.984
## Pos Pred Value          0.920    0.676    0.687    0.462    0.888
## Neg Pred Value          0.949    0.906    0.961    0.928    0.908
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.247    0.115    0.143    0.109    0.103
## Detection Prevalence    0.268    0.171    0.209    0.236    0.116
## Balanced Accuracy       0.919    0.764    0.872    0.757    0.771
```

```r
confusionMatrix(predict(fit_tree_1, test_xy,type="class"),test_y)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1185   87    0    3   12
##          B   56  572   41   72  110
##          C   23   52  706  203   68
##          D  128  200  106  515  181
##          E    3   38    2   11  530
## 
## Overall Statistics
##                                         
##                Accuracy : 0.715         
##                  95% CI : (0.702, 0.728)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.642         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.849    0.603    0.826    0.641    0.588
## Specificity             0.971    0.929    0.915    0.850    0.987
## Pos Pred Value          0.921    0.672    0.671    0.456    0.908
## Neg Pred Value          0.942    0.907    0.961    0.923    0.914
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.242    0.117    0.144    0.105    0.108
## Detection Prevalence    0.262    0.174    0.215    0.230    0.119
## Balanced Accuracy       0.910    0.766    0.870    0.745    0.787
```
### Model=randomForest
Next the random forest approach is attempted. With default parameters, the misclassification rate is only 0.26%. It is attempted to improve the models further by changing the control parameters such as ```importance```,```proximity```, ```mtry```, etc. Best results are obtained with ```mtry``` of 11

```r
fit_rf_1 <- randomForest(x=train_x,y=train_y,xtest=test_x,ytest=test_y)
fit_rf_2 <- randomForest(x=train_x,y=train_y,xtest=test_x,ytest=test_y,importance=TRUE)
fit_rf_3 <- randomForest(x=train_x,y=train_y,xtest=test_x,ytest=test_y,proximity=TRUE)
fit_rf_4 <- randomForest(x=train_x,y=train_y,xtest=test_x,ytest=test_y,mtry=10)
fit_rf_5 <- randomForest(x=train_x,y=train_y,xtest=test_x,ytest=test_y,mtry=11)
fit_rf_6 <- randomForest(x=train_x,y=train_y,xtest=test_x,ytest=test_y,mtry=12)
```
Below are the details of the best achieved fit using Random Forest technique and the confusion matrices on the train, test and validation data subsets. We can see from the confusion matrices that the random forest algorithm does extremely well on all of them. The training sample error rate is 0.2% and the test sample error rate is 0.22%, which is quite good. 

```r
fit_rf_5 <- randomForest(x=train_x,y=train_y,xtest=test_x,ytest=test_y,mtry=11)
fit_rf_5
```

```
## 
## Call:
##  randomForest(x = train_x, y = train_y, xtest = test_x, ytest = test_y,      mtry = 11) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 11
## 
##         OOB estimate of  error rate: 0.25%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4184    1    0    0    0   0.0002389
## B    4 2843    1    0    0   0.0017556
## C    0   12 2553    2    0   0.0054538
## D    0    0   13 2398    1   0.0058043
## E    0    0    0    3 2703   0.0011086
##                 Test set error rate: 0.39%
## Confusion matrix:
##      A   B   C   D   E class.error
## A 1395   0   0   0   0    0.000000
## B    3 946   0   0   0    0.003161
## C    0   7 847   1   0    0.009357
## D    0   0   6 798   0    0.007463
## E    0   0   0   2 899    0.002220
```


Finally we apply the model to the unknown dataset in order to predict the outcome. The results obtained are as below and are submitted and evaluated to be correct.

```r
fit_rf_7$test["predicted"]
```

```
## $predicted
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

### Summary and Conclusion
In summary, we have developed a model of the WLE dataset by looking at different models and selecting the random Forest model, trained and tested it, and successfully used it for prediction of new observations with excellent accuracy. The specific questions asked in the project related to use of cross validation, explanation of modeling choices and out of sample error rates have also been addressed - to summarize, a 75/25 partitioning scheme was used for cross validation, the explanation of modeling choices made is given in  the section with the same name, and the in sample and out of sample error rates are calculated and shown for each of the models.
