library(glmnet)
library(Metrics)
set.seed(0)
glmOneHot <- glmnet(as.matrix(oneHotTrain), prices)

coef.glmnet(glmOneHot)

plot(glmOneHot)

print(glmOneHot)

cv.fit <- cv.glmnet(as.matrix(oneHotTrain), prices)

plot(cv.fit)
coef(cv.fit)
#Prediciton with lambda.1se
glm.predict <- predict(cv.fit, newx = as.matrix(oneHotTest), s = 'lambda.1se')
yhat <- as.data.frame(cbind(1461:2919, glm.predict))
colnames(yhat) <- c('ID', 'SalePrice')

write.table(yhat, 'glmpredict1se.csv', sep = ',', row.names = FALSE)


#Prediction with lambda.min
glm.predict <- predict(cv.fit, newx = as.matrix(oneHotTest), s = 'lambda.min')
yhat <- as.data.frame(cbind(1461:2919, glm.predict))
colnames(yhat) <- c('ID', 'SalePrice')

write.table(yhat, 'glmpredictmin.csv', sep = ',', row.names = FALSE)

resub <- predict(cv.fit, newx = as.matrix(oneHotTrain), s = 'lambda.min')

rmse(log(prices), log(resub))
