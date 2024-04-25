setwd("~/Documents/R_Projects/Obesity_Classification")
library(tidyverse)
library(xgboost)
library(dplyr)
train = read.csv('./00_Data/train.csv',
                 na.strings = c(""), stringsAsFactors=TRUE)

test = read.csv('./00_Data/test.csv',
                na.strings = c(""), stringsAsFactors=TRUE)

##########################################
########## Data Pre-processing ###########
##########################################

str(train)

#Convert columns into factors if they are factors
# All factor variables are correctly imported as factors

# Before dropping any NA data take a backup
train_backup <- train
train[!complete.cases(train),]

# If necessary adjust the factor levels here.
# label <- dplyr::recode(label, '1'=1, '2'=0, .default = NA_real_)

# train data had less factors than test data for the train$CALC variable
# So we made train data same as test for CALC's factor lev
table(train$CALC, useNA = 'always')
table(test$CALC, useNA = 'always')
train$CALC <- factor(train$CALC, levels = levels(test$CALC))
table(train$CALC, useNA = 'always')

str(train)

train$NObeyesdad <- as.numeric(train$NObeyesdad)
# For XGBoost label must start from 0, so we must reorder the factors.
train$NObeyesdad <- dplyr::recode(train$NObeyesdad, '1'=0, '2'=1, '3'=2, '4'=3, '5'=4, '6'=5, '7'=6, .default = NA_real_)

# One-Hot Encode here? if so changing to all numeric not necessary
# install.packages('mltools')
library(mltools)
library(data.table)
# install.packages('data.table')
train <- one_hot(data.table(train))
str(train)

# XGBoost does not work with factors, so we must change all factors into numeric type
# train$Gender <- as.numeric(train$Gender)
# train$family_history_with_overweight <- as.numeric(train$family_history_with_overweight)
# train$FAVC <- as.numeric(train$FAVC)
# train$CAEC <- as.numeric(train$CAEC)
# train$SMOKE <- as.numeric(train$SMOKE)
# train$SCC <- as.numeric(train$SCC)
# train$CALC <- as.numeric(train$CALC)
# train$MTRANS <- as.numeric(train$MTRANS)
# train$NObeyesdad <- as.numeric(train$NObeyesdad)



# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(train$NObeyesdad, SplitRatio = 0.75)
training_set = subset(train, split == TRUE)
test_set = subset(train, split == FALSE)


#########################################
## Fitting XGBoost to the Training set ##
#########################################
# data[!complete.cases(data),]
# write_csv(as.data.frame(data), file = './03_Results/data.csv')
# write_csv(as.data.frame(training_set), file = './03_Results/data2.csv')
# Data Prep
# Remove the column we are trying to predict
# data <- as.matrix(training_set[-18])
#if using one hot
data <- as.matrix(training_set[,-33])
# Specify the column we are trying to predict
label <- training_set$NObeyesdad
# # For XGBoost label must start from 0, so we must reorder the factors.
# label <- dplyr::recode(label, '1'=0, '2'=1, '3'=2, '4'=3, '5'=4, '6'=5, '7'=6, .default = NA_real_)
# label <- as.factor(train$Survived)
table(label)
#XGBoost cannot have character variables so we must convert the data to a number.
mode(data) <- 'double'

# set.seed(123)
# Fit the xgboost model to the training_set.
classifier = xgboost(data = data, label = label, nrounds = 20, num_class = 7, objective = "multi:softmax")

# Predicting the Test set results
newdata = as.matrix(test_set[,-33])
mode(newdata) <- 'double'
y_pred = predict(classifier, newdata = newdata)
y_pred

cm2 = table(as.matrix(test_set[, 1]), y_pred)
cm2

# Create a confustion matrix and test accuracy
library(caret)
# test_set$NObeyesdad <- as.numeric(test_set$NObeyesdad)
# test_set$NObeyesdad <- dplyr::recode(test_set$NObeyesdad , '1'=0, '2'=1, '3'=2, '4'=3, '5'=4, '6'=5, '7'=6, .default = NA_real_)
test_set$NObeyesdad <- as.factor(test_set$NObeyesdad)
y_pred <- as.factor(y_pred)
confusionMatrix(y_pred, test_set$NObeyesdad)


imp <- xgb.importance(model = classifier)
xgb.plot.importance(imp)

# Save model so it does not need to be retrained.
xgb.save(classifier, 'xgb.model')
# classifier <- xgb.load('xgb.model')
##########################################
#### Applying k-Fold Cross Validation ####
##########################################
# table(training_set$NObeyesdad, useNA = 'always')
# table(test_set$NObeyesdad, useNA = 'always')

folds = createFolds(training_set$NObeyesdad, k = 10)
#lapply (list apply) apply the function to our list (folds) x is each fold
cv = lapply(folds, function(x) {
  training_fold = as.matrix(training_set[-x, ])
  test_fold = as.matrix(test_set[x, ])
  # mode(test_fold) <- 'double'
  classifier = xgboost(data = data, label = label, nrounds = 20, num_class = 7, objective = "multi:softmax")
  y_pred = predict(classifier, newdata = as.matrix(test_fold[,-33]))
  # y_pred = (y_pred >= 0.5) Only needed for binomial classification
  cm = table(test_fold[, 33], y_pred)
  # Need to add the rest of the accuracy calculation cm[3,3] cm[4,4] etc
  accuracy = (cm[1,1] + cm[2,2] + cm[3,3] + cm[4,4] + cm[5,5] + cm[6,6]) / (cm[1,1] + cm[1,2] + cm[1,3] + cm[1,4]+ cm[1,5] + cm[1,6] + 
                                                                              cm[2,1] + cm[2,2] + cm[2,3] + cm[2,4]+ cm[2,5] + cm[2,6] + 
                                                                              cm[3,1] + cm[3,2] + cm[3,3] + cm[3,4]+ cm[3,5] + cm[3,6] + 
                                                                              cm[4,1] + cm[4,2] + cm[4,3] + cm[4,4]+ cm[4,5] + cm[4,6] + 
                                                                              cm[5,1] + cm[5,2] + cm[5,3] + cm[5,4]+ cm[5,5] + cm[5,6] + 
                                                                              cm[6,1] + cm[6,2] + cm[6,3] + cm[6,4]+ cm[6,5] + cm[6,6])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))




# 
# test_set$NObeyesdad <- as.factor(test_set$NObeyesdad)
# test_set$NObeyesdad <- dplyr::recode(test_set$NObeyesdad , '1'=0, '2'=1, '3'=2, '4'=3, '5'=4, '6'=5, '7'=6, .default = NA_real_)
# pm <- test_set$NObeyesdad
# cv1 <- as.factor(cv$Fold01)
# confusionMatrix(cv1, test_set$NObeyesdad)
# test_set$NObeyesdad <- factor(test_set$NObeyesdad, levels = levels(training_set$NObeyesdad))

# test_set$NObeyesdad <- as.factor(test_set$NObeyesdad)
# test_set$NObeyesdad <- dplyr::recode(test_set$NObeyesdad , '1'=0, '2'=1, '3'=2, '4'=3, '5'=4, '6'=5, '7'=6, .default = NA_real_)
# test_set$NObeyesdad <- as.numeric(test_set$NObeyesdad)

