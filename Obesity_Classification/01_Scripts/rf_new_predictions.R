library(tidyverse)
test = read.csv('./00_Data/test.csv',
                na.strings = c(""), stringsAsFactors=TRUE)

##########################################
########## Data Pre-processing ###########
##########################################
str(test)

#Convert columns into factors if they are factors
# All factor variables are correctly imported as factors

# Before dropping any NA data take a backup
test_backup <- test
test[!complete.cases(test),]

# If necessary adjust the factor levels here.
# label <- dplyr::recode(label, '1'=1, '2'=0, .default = NA_real_)

#########################################
################ Predict ################
#########################################
library(randomForest)
library(caret)
y_pred = predict(classifier, newdata=test)
y_pred
table(y_pred)
df <- as.data.frame(y_pred)
final_df <- cbind(test, df)
# write.csv(y_pred, file = 'rf_new_predictions.csv', col.names = c("ID", "test$NObeyesdad"))
write_csv(final_df, file = './03_Results/rf_new_predictions_fulldf2.csv')
