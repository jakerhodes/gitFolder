library(randomForest)
library(caret)

#----------------------------------------------------#
#                Load in the data
#----------------------------------------------------#
set.seed(0)
data <- read.csv('combinedImputed.csv')

train <- data[1:1460, 2:80]
test <- data[1461:2919, 2:80]
prices <- data[1:1460, 81]

#---------------------------------------------------#
#               One-hot data
#---------------------------------------------------#

dmy <- dummyVars(" ~ .", data = data)
oneHot <- data.frame(predict(dmy, newdata = data))

oneHotTrain <- oneHot[1:1460, 2:300]
oneHotTest <- oneHot[1461:2919, 2:300]

oneHotTraindf <- as.data.frame(oneHotTrain)
oneHotTestdf <- as.data.frame(oneHotTest)
#----------------------------------------------------#
#               Random Forests on Data
#----------------------------------------------------#

rf <- randomForest(train, prices, ntree = 1000, importance = TRUE)
yhat <- predict(rf, test)
write.csv(yhat, 'rfSubmission.csv')

#----------------------------------------------------#
#               Random Forests on One-Hot
#----------------------------------------------------#

rfOneHot <- randomForest(oneHotTrain, prices)
yhatOneHot <- predict(rfOneHot, oneHotTest)
write.csv(yhatOneHot, 'rfOneHotSubmission.csv')
