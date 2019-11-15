rm(list=ls())
# setting the working directory.
setwd("D:/R and PYTHON files/data set/project 2")
getwd()

# loading the train and test data to the environment.
train=read.csv("train.csv",header=TRUE)
test=read.csv("test.csv",header=TRUE)

# installing the necessary libraries.
#install.packages (c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
                    #"MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees'))

#  explorin the datasets
dim(train)# 200000 observation and 202 features
dim(test) # 200000 observations and 201 features
str(train) # to know the features class type
str(test) # to know the features class type

# checcking for missing values.
sum(is.na(train))# no missing values in train dataset
sum(is.na(test)) # no misssing values in test dataset

# target feature analysis.
unique(train$target)# unique values are 0 and 1.
table(train$target)# 0=179902, 1=20098
hist(train$target)# the train data has one dependeant variable called "target" whose values are compromised of imbalance class values.



# visualization
library(ggplot2)
require(gridExtra)
# barplot
plot=ggplot(train,aes(target))+theme_bw()+geom_bar(stat='count',fill='brown')
grid.arrange(plot)



# deleting the features which are not wanting.
train= subset(train,select=-c(ID_code))
test= subset(test,select=-c(ID_code))


# preparing the data for model developement 
library(caret)
train_index=createDataPartition(train$target, p = .80, list = FALSE)
train1=train[train_index,]
test2=train[-train_index,]


# for logistic regression target variable must be numeric.
train1$target=as.numeric(train1$target)
# logistic regression modell
logit_model=glm(target~.,data=train1)
summary(logit_model)
logit_predictions=predict(logit_model,newdata=test2[,-1],type="response")
# predicting for test data
test_predictions=predict(logit_model,newdata=test,type="response")
#write(capture.output(test_predictions),"logistic model predictions new R.txt")

# decision tree classifier.
library(C50)
# tree model requires a factor target variable
train1$target=as.factor(train1$target)
tree_model= C5.0(target ~., train1, trials = 3, rules = TRUE)
tree_predictions=predict(tree_model,test2[,-1],type="class")
# prediction of test case dependent feature.
testcase_pred=predict(tree_model,test,type="class")
# model evaluation
ConfMatrix_tree_mat = table(test2$target,tree_predictions )
confusionMatrix(ConfMatrix_tree_mat)
# Accuracy=89.38 %
# FNR=  9.50 %
# Recall= 90.4 %
# precision= 98.4 %
# F score= 94.1 %


#write(capture.output(confusionMatrix(ConfMatrix_bayes_mat)),"bayes confusion_matrix R.txt")


# naive bayes
library(e1071)
nb_model=naiveBayes(target~.,data=train1)
#(summary(nb_model))
nb_predicion=predict(nb_model,newdata=test2[,-1],type="class")
# prediction of test cases
nb_test_pred=predict(nb_model,newdata=test,type="class")
# model evaluation
ConfMatrix_bayes_mat = table(test2$target,nb_predicion)
confusionMatrix(ConfMatrix_bayes_mat)
# Accuracy= 92.1 %
# FNR= 6.8 %
# Recall= 93.1 %
# precision=  98.4 %
# F score= 95.6 %






# random forest classifier
library(randomForest)
rf_model=randomForest(target~.,train1,importance=TRUE,ntree=50)
#Extract rules fromn random forest
#transform rf /object to an inTrees' format
library(inTrees)
treeList = RF2List(rf_model)  
#Extract rules
rules= extractRules(treeList, train1[,-1])
#Visualize some rules
rules[1:2,]
#Make rules more readable:
readrules = presentRules(rules, colnames(train1))
readrules[1:2,]
#Predict test data using random forest model
rf_Predictions = predict(rf_model, test2[,-1])
# prediction of test cases.
rf_Pred_test = predict(rf_model, test)
# model evaluation
ConfMatrix_random_mat = table(test2$target,rf_Predictions )
confusionMatrix(ConfMatrix_random_mat)
#write(capture.output(rf_model),"rf_model R.txt")

# Accuracy=89.76 %
# FNR= 10.2 %
# Recall= 89.7 %
# Precision= 99.9 %
# F score= 94.5 %

# note:-
# train1 is the training dataset.
# test2 is the testing dataset.
# train1 and test2 are created for the sake of calculating confusion matrix.
# test is the actual datset which must be predicted.




