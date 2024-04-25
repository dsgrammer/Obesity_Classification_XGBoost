library(tidyverse)
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

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(train$NObeyesdad, SplitRatio = 0.75)
training_set = subset(train, split == TRUE)
test_set = subset(train, split == FALSE)

##########################################
################ Modeling ################
##########################################
library(randomForest)
library(caret)
# install.packages("randomForest")

# Fit the classifier
classifier <- randomForest(NObeyesdad ~ ., data=training_set, proximity=TRUE)
print(classifier)

# Predict the results on train (not really necessary)
p1 <- predict(classifier, training_set)
p1

#Make the confusion matrix for train set
confusionMatrix(p1, training_set$NObeyesdad)


# Predicting the Test set results
y_pred = predict(classifier, newdata=test_set)
y_pred

# Making the Confusion Matrix for test set
confusionMatrix(y_pred, test_set$NObeyesdad)

