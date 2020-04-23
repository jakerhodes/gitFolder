library(randomForest)

rfSelection <- randomForest(oneHotTrain, prices, ntree = 5000, importance = TRUE,
                           proximity = TRUE)


importancePlot <- varImpPlot(rfSelection) 

importantVars <- as.data.frame(rfSelection$importance)
importantVars <- topVars[order(-topVars$`%IncMSE`),]

topVars <- row.names(importantVars[1:65,])

topVarsTrain <- oneHotTrain[, topVars]

topVarsTest <- oneHotTest[, topVars]

rfTop <- randomForest(topVarsTrain, prices, ntree = 1000)

yhatTop <- predict(rfTop, topVarsTest)
yhatTop <- as.data.frame(cbind(1461:2919, yhatTop))
colnames(yhatTop) <- c('ID', 'SalePrice')

write.table(yhatTop, 'yhatTop65.csv', sep = ',', row.names = FALSE)
