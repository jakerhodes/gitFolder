#---------------------------------------------------------------#
#                        Libraries                
#---------------------------------------------------------------#
library(glmnet)
library(EZtune)
library(randomForest)
library(elasticnet)
library(olsrr)
library(gbm)
library(Metrics)
library(xgboost)

#---------------------------------------------------------------#
#                    Read in the data                
#---------------------------------------------------------------#
data <- read.csv('transformed_data.csv')
data <- data[1:1460, ]
data <- data[-c(31, 89, 108, 333, 411, 524, 589, 826, 1001), ]

x <- data[, 2:(length(data[1,]) - 1)]
y <- data$log.SalePrice
#---------------------------------------------------------------#
#                    Train/Test Split               
#---------------------------------------------------------------#
set.seed(0)
samp <- sample(length(y), size = floor(.7 * length(y)))

x_train <- x[samp, ]
y_train <- y[samp]

x_test <- x[-samp, ]
y_test <- y[-samp]

#---------------------------------------------------------------#
#                     Linear Regression                
#---------------------------------------------------------------#
set.seed(0)
ols <- lm(log.SalePrice ~., data[samp, ])
ols.predict <- predict(ols, data[-samp, ])

ols.rmse <- rmse(y_test, ols.predict)
ols.rmse

#---------------------------------------------------------------#
#                     Ridge Regression                
#---------------------------------------------------------------#

#Ridge with lambda.min
set.seed(0)
ridge.cv <- cv.glmnet(as.matrix(x_train), as.matrix(y_train), alpha = 0)
ridge.min.predict <- predict(ridge.cv, as.matrix(x_test), s = 'lambda.min')
ridge.min.rmse <- rmse(y_test, ridge.min.predict)
ridge.min.rmse 

#Ridge with lambda.1se
set.seed(0)
ridge.1se.predict <- predict(ridge.cv, as.matrix(x_test), s = 'lambda.1se')
ridge.1se.rmse <- rmse(y_test, ridge.1se.predict)
ridge.1se.rmse

#---------------------------------------------------------------#
#                     Lasso Regression                
#---------------------------------------------------------------#
#lasso with lambda.min
set.seed(0)
lasso.cv <- cv.glmnet(as.matrix(x_train), as.matrix(y_train), alpha = 1)
lasso.min.predict <- predict(lasso.cv, as.matrix(x_test), s = 'lambda.min')
lasso.min.rmse <- rmse(y_test, lasso.min.predict)
lasso.min.rmse

#lasso with lambda.1se
set.seed(0) 
lasso.1se.predict <- predict(lasso.cv, as.matrix(x_test), s = 'lambda.1se')
lasso.1se.rmse <- rmse(y_test, lasso.1se.predict)
lasso.1se.rmse

#---------------------------------------------------------------#
#                     Random Forest Regression                
#---------------------------------------------------------------#
set.seed(0)
rf <- randomForest(x_train, y_train)
rf.predict <- predict(rf, x_test)
rf.rmse <- rmse(y_test, rf.predict)
rf.rmse

rf.tune <- tuneRF(x_train, y_train, stepFactor=1.5, improve=1e-5, ntree=500)

#---------------------------------------------------------------#
#                     ElasticNet Regression                
#---------------------------------------------------------------#
#enet with lambda.min
set.seed(0)
enet.cv <- cv.glmnet(as.matrix(x_train), as.matrix(y_train), alpha = .5)
enet.min.predict <- predict(enet.cv, as.matrix(x_test), s = 'lambda.min')
enet.min.rmse <- rmse(y_test, enet.min.predict)
lasso.min.rmse

#enet with lambda.1se
set.seed(0) 
enet.1se.predict <- predict(enet.cv, as.matrix(x_test), s = 'lambda.1se')
enet.1se.rmse <- rmse(y_test, enet.1se.predict)
enet.1se.rmse

#---------------------------------------------------------------#
#                     Boosting Machine                
#---------------------------------------------------------------#
set.seed(0)
gbm_train <- eztune(x_train, y_train, method = 'gbm', fast = TRUE)
gbm.predict <- predict(gbm_train$model, newdata = x_test, n.tree = gbm_train$n.trees)

gbm.rmse <- rmse(y_test, gbm.predict)
gbm.rmse

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

xgb.rmse <- rmse(y_test, xgb.predict)
xgb.rmse

