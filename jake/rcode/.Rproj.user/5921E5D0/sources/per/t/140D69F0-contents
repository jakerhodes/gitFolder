#---------------------------------------------------------------#
#                        Libraries                
#---------------------------------------------------------------#
library(glmnet)
library(EZtune)
library(randomForest)
library(elasticnet)
library(olsrr)
library(gbm)
library(ggplot2)
library(xgboost)
library(plotly)
#---------------------------------------------------------------#
#                    Read in the data                
#---------------------------------------------------------------#
set.seed(0)
data <- read.csv('transformed_data.csv')

data_train <- data[1:1460, -1]

x_train <- data[1:1460, 2:301]
y_train <- data[1:1460, 302]

x_test <- data[1461:2919, 2:301]

x_train <- x_train[-c(31, 89, 108, 333, 411, 524, 589, 826, 1001), ]
y_train <- y_train[-c(31, 89, 108, 333, 411, 524, 589, 826, 1001)]
#---------------------------------------------------------------#
#                     Linear Regression                
#---------------------------------------------------------------#
set.seed(0)
ls <- lm(log.SalePrice ~., data = data_train)
ls.predict <- predict(ls, x_test)

cooks.distance(ls)
ols_plot_cooksd_chart(ls) +
  theme(plot.title = element_blank(),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14)) +
  xlab('Observation Index') +
  scale_y_continuous(limits = c(0, 0.35))



ols <- lm(log.SalePrice ~., data = data_train)
ols.predict <- predict(ols, x_test)
ols.submission <- data.frame('ID' = 1461:2919, 'SalePrice' = exp(ols.predict))
write.table(ols.submission, './submissions/ols.submission.csv', row.names = FALSE, sep = ',')

#---------------------------------------------------------------#
#                     Ridge Regression                
#---------------------------------------------------------------#

#Ridge with lambda.min
set.seed(0)
ridge.cv <- cv.glmnet(as.matrix(x_train), as.matrix(y_train), alpha = 0)
ridge.min.predict <- predict(ridge.cv, as.matrix(x_test), s = 'lambda.min')
ridge.min.submission <- data.frame('ID' = 1461:2919, 'SalePrice' = as.numeric(exp(ridge.min.predict)))
write.table(ridge.min.submission, 'ridge.min.submission.csv', row.names = FALSE, sep = ',')

#Ridge with lambda.1se
set.seed(0)
ridge.1se.predict <- predict(ridge.cv, as.matrix(x_test), s = 'lambda.1se')
ridge.1se.submission <- data.frame('ID' = 1461:2919, 'SalePrice' = as.numeric(exp(ridge.1se.predict)))
write.table(ridge.1se.submission, 'ridge.1se.submission.csv', row.names = FALSE, sep = ',')
#---------------------------------------------------------------#
#                     Lasso Regression                
#---------------------------------------------------------------#
#lasso with lambda.min
set.seed(0)
lasso.cv <- cv.glmnet(as.matrix(x_train), as.matrix(y_train), alpha = 1)
plot(lasso.cv)
lasso.min.predict <- predict(lasso.cv, as.matrix(x_test), s = 'lambda.min')
lasso.min.submission <- data.frame('ID' = 1461:2919, 'SalePrice' = as.numeric(exp(lasso.min.predict)))
write.table(lasso.min.submission, 'lasso.min.submission.csv', row.names = FALSE, sep = ',')

#lasso with lambda.1se
set.seed(0) 
lasso.1se.predict <- predict(lasso.cv, as.matrix(x_test), s = 'lambda.1se')
lasso.1se.submission <- data.frame('ID' = 1461:2919, 'SalePrice' = as.numeric(exp(lasso.1se.predict)))
write.table(lasso.1se.submission, 'lasso.1se.submission.csv', row.names = FALSE, sep = ',')
#---------------------------------------------------------------#
#                     Random Forest Regression                
#---------------------------------------------------------------#
set.seed(0)
rf <- randomForest(x_train, y_train, importance = TRUE, keep.forest = TRUE)
varImpPlot(rf, type = 1)
rf.predict <- predict(rf, x_test)
rf.submission <- data.frame('ID' = 1461:2919, 'SalePrice' = as.numeric(exp(rf.predict)))
write.table(rf.submission, 'rf.submission.csv', row.names = FALSE, sep = ',')
#---------------------------------------------------------------#
#                     Boosting Machine                
#---------------------------------------------------------------#
set.seed(0)
gbm_train <- eztune(x_train, y_train, method = 'gbm', fast = TRUE)
gbm.predict <- predict(gbm_train$model, newdata = x_test, n.tree = gbm_train$n.trees)

gbm.submission <- data.frame('ID' = 1461:2919, 'SalePrice' = as.numeric(exp(gbm.predict)))
write.table(gbm.submission, 'gbm.submission.csv', row.names = FALSE, sep = ',')

#GBM fast = FALSE
set.seed(0)
gbm_train_slow <- eztune(x_train, y_train, method = 'gbm', fast = FALSE)
gbm.predict_slow <- predict(gbm_train_slow$model, x_test, n.tree = gbm_train_slow$n.trees)

gbm.submission_slow <- data.frame('ID' = 1461:2919, 'SalePrice' = as.numeric(exp(gbm.predict_slow)))
write.table(gbm.submission, 'gbm.submission_slow.csv', row.names = FALSE, sep = ',')


#---------------------------------------------------------------#
#                     ElasticNet Regression                
#---------------------------------------------------------------#
#enet with lambda.min
set.seed(0)
enet.cv <- cv.glmnet(as.matrix(x_train), as.matrix(y_train), alpha = .8)
enet.min.predict <- predict(enet.cv, as.matrix(x_test), s = 'lambda.min')
enet.min.submission <- data.frame('ID' = 1461:2919, 'SalePrice' = as.numeric(exp(enet.min.predict)))
write.table(enet.min.submission, 'enet.min.submission.csv', row.names = FALSE, sep = ',')

#enet with lambda.1se
set.seed(0) 
enet.1se.predict <- predict(enet.cv, as.matrix(x_test), s = 'lambda.1se')
enet.1se.submission <- data.frame('ID' = 1461:2919, 'SalePrice' = as.numeric(exp(enet.1se.predict)))
write.table(enet.1se.submission, 'enet.1se.submission.csv', row.names = FALSE, sep = ',')

#---------------------------------------------------------------#
#                     xgboost               
#---------------------------------------------------------------#
set.seed(0)

params <- list(colsample_bytree = 0.4603, gamma = 0.0468,
               learning_rate = 0.05, max_depth = 3,
               min_child_weight = 1.7817, n_estimators = 2200,
               alpha = 0.4640, lambda = 0.8571,
               subsample = 0.5213)


xgb <- xgboost(data = as.matrix(x_train),
               label = y_train,
               params = params,
               nrounds = 2200)
xgb.predict <- predict(xgb, as.matrix(x_test))

xgb.submission <- data.frame('ID' = 1461:2919, 'SalePrice' = as.numeric(exp(xgb.predict)))
write.table(xgb.submission, 'xgb.submission.csv', row.names = FALSE, sep = ',')

