library(randomForest)
library(mice)

set.seed(0)
train <- read.csv('train.csv')
test <- read.csv('test.csv')

X <- train[, c(2:6, 8:72, 76:80)]
y <- train[, 81]

Xtest <- test[, c(2:6, 8:72, 76:80)]

combinedTrainTest <- rbind(X, Xtest)

#Creating imputed training set
XImpute <- rfImpute(X, y, iter = 10, ntree = 500)
XImpute <- XImpute[-1]

trainClass <- lapply(XImpute, class)
XImpute[which(trainClass == 'integer')] <- sapply(
  XImpute[which(trainClass == 'integer')], as.numeric)

#Creating imputed test set
XtestImputeTemp <- mice(Xtest, nnet.MaxNWts = 5000)
XtestImpute <- complete(XtestImputeTemp, 1)

testClass <- lapply(XtestImpute, class)
XtestImpute[which(testClass == 'integer')] <- sapply(
  XtestImpute[which(testClass == 'integer')], as.numeric)


combinedImputed <- rbind(XImpute, XtestImpute)


rf <- randomForest(combinedImputed[1:1460,], y)

yhat <- predict(rf, combinedImputed[1461:2919,])

submission <- cbind(1461:2919, yhat)

write.table(submission,
          'G:/My Drive/AAA USU Phd ALL/Spring 2020/CS 6665/Project/rcode/rfSubmission.csv',
          row.names = FALSE,
          col.names = FALSE,
          sep = ',')
