library(EZtune)


set.seed(0)

train_gbm <- eztune(oneHotTraindf, prices, method = 'gbm', fast = TRUE)
eztune_cv(oneHotTraindf, prices, train_gbm)


yhat <- predict(train_gbm$model, oneHotTestdf, n.tree = train_gbm$n.trees)
sub <- as.data.frame(cbind(1461:2919, yhat))
colnames(sub) <- c('ID', 'SalePrice')

write.csv(sub, 'gbmSubmission.csv', row.names = F)

#------------------------------------------------------#


train_gbm_log <- eztune(oneHotTraindf, log(prices), method = 'gbm', fast = TRUE)


yhat <- predict(train_gbm_log$model, oneHotTestdf, n.tree = train_gbm_log$n.trees)

exp_yhat <- exp(yhat)
sub_log <- as.data.frame(cbind(1461:2919, exp_yhat))
colnames(sub_log) <- c('ID', 'SalePrice')

write.csv(sub_log, 'gbmSubmissionLogPrices.csv', row.names = F)
