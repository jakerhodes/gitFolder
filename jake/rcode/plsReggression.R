library(pls)

oneHotTraindf <- as.data.frame(oneHotTrain)
oneHotTestdf <- as.data.frame(oneHotTest)

pls <- plsr(SalePrice ~., data = oneHotTraindf, validation = 'none')

summary(pls)

explvar(pls)

yhat <- predict(pls, oneHotTestdf)

col.names(yhat) <- c('ID', 'SalePrice')

write.csv(yhat, 'plsAllComps.csv')

#-----------------------------------------------------------#
#                   Random Forest on PLS
#-----------------------------------------------------------#
