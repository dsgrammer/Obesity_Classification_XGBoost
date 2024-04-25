library(tidyverse)
train = read.csv('/home/derek/Documents/R_Projects/Obesity_Classification/Data/train.csv',
                 na.strings = c(""), stringsAsFactors=TRUE)

test = read.csv('/home/derek/Documents/R_Projects/Obesity_Classification/Data/test.csv',
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

str(test)

#Convert columns into factors if they are factors
# All factor variables are correctly imported as factors

# Before dropping any NA data take a backup
test_backup <- test
test[!complete.cases(test),]

# If necessary adjust the factor levels here.
# label <- dplyr::recode(label, '1'=1, '2'=0, .default = NA_real_)

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(train$NObeyesdad, SplitRatio = 0.75)
# training_set = subset(train, split == TRUE)
# test_set = subset(train, split == FALSE)

#########################################
########## Logistic regression ##########
#########################################
# summary(glm(formula = NObeyesdad ~ .,
#             family = binomial,
#             data = train))

# library(nnet)
# summary(multinom(formula = NObeyesdad ~ .,
#                  data = train))


table(train$CALC, useNA = 'always')
table(test$CALC, useNA = 'always')
# train data had less factors than test data for the train$CALC variable
# So we made train data same as test for CALC's factor lev
train$CALC <- factor(train$CALC, levels = levels(test$CALC))

library(randomForest)
library(caret)
# install.packages("randomForest")
classifier <- randomForest(NObeyesdad ~ ., data=train, proximity=TRUE)
print(classifier)
p1 <- predict(classifier, train)
confusionMatrix(p1, train$NObeyesdad)

# Predicting the Test set results

y_pred = predict(classifier, newdata=test)
y_pred
write.csv(y_pred, file = 'obese.csv')

# Making the Confusion Matrix
cm = table(test[, 1], y_pred)
cm
# confusionMatrix(y_pred, test$NObeyesdad)
# write.table(cm, file = "obese_or_not.csv")
