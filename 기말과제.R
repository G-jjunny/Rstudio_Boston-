rmse <- function(yi, yhat_i){
  
  sqrt(mean((yi - yhat_i)^2))
  
}



binomial_deviance <- function(y_obs, yhat){
  
  epsilon = 0.0001
  
  yhat = ifelse(yhat < epsilon, epsilon, yhat)
  
  yhat = ifelse(yhat > 1-epsilon, 1-epsilon, yhat)
  
  a = ifelse(y_obs==0, 0, y_obs * log(y_obs/yhat))
  
  b = ifelse(y_obs==1, 0, (1-y_obs) * log((1-y_obs)/(1-yhat)))
  
  return(2*sum(a + b))
  
}
panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...){
  
  usr <- par("usr"); on.exit(par(usr))
  
  par(usr = c(0, 1, 0, 1))
  
  r <- abs(cor(x, y))
  
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  
  txt <- paste0(prefix, txt)
  
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  
  text(0.5, 0.5, txt, cex = cex.cor * r)
  
}
#install.packages("glmnet")
#install.packages("randomForest")
#install.packages("gbm")
#install.packages("ROCR")
library(tidyverse)
library(gridExtra)
library(MASS)
library(glmnet)
library(randomForest)
library(gbm)
library(rpart)
library(boot)
library(data.table)
library(ROCR)
library(ggplot2)
library(dplyr)
library(gridExtra)

data <- Boston

glimpse(data)

summary(data) 

set.seed(1810)

pairs(data %>% dplyr::sample_n(min(1000, nrow(data))),
      
      lower.panel=function(x,y){ points(x,y); abline(0, 1, col='red')},
      
      upper.panel = panel.cor) 

set.seed(1810)
n <- nrow(data)
idx <- 1:n
training_idx <- sample(idx, n * .60)
idx <- setdiff(idx, training_idx)
validate_idx <- sample(idx, n * .20)
test_idx <- setdiff(idx, validate_idx)
training <- data[training_idx,]
validation <- data[validate_idx,]
test <- data[test_idx,]  

data_lm_full <- lm(medv ~ ., data=training)
summary(data_lm_full)

predict(data_lm_full, newdata = data[1:5,])

data_lm_full2 = lm(medv ~ .^2, data = training)
summary(data_lm_full2)

length(coef(data_lm_full2))

library(MASS)
data_step = stepAIC(data_lm_full, scope = list(upper = ~ .^2, lower = ~1)) 

summary(data_step)

length(coef(data_step))

y_obs <- validation$medv

yhat_lm <- predict(data_lm_full, newdata=validation)

yhat_lm_2 <- predict(data_lm_full2, newdata=validation)

yhat_step <- predict(data_step, newdata=validation)

rmse(y_obs, yhat_lm)

rmse(y_obs, yhat_lm_2)

rmse(y_obs, yhat_step)

xx <- model.matrix(medv ~ .^2-1, data)

x <- xx[training_idx, ]

y <- training$medv

glimpse(x)

data_cvfit <- cv.glmnet(x, y)

plot(data_cvfit) 

coef(data_cvfit, s = c("lambda.1se"))

coef(data_cvfit, s = c("lambda.min"))

predict(data_cvfit, s="lambda.min", newx = x[1:5, ])

y_obs <- validation$medv
yhat_glmnet <- predict(data_cvfit, s="lambda.min", newx=xx[validate_idx,])
yhat_glmnet <- yhat_glmnet[,1] # change to a vector from [n*1] matrix
rmse(y_obs, yhat_glmnet)

data_tr <- rpart(medv ~ ., data = training)
data_tr
opar <- par(mfrow = c(1,1), xpd = NA)
plot(data_tr)
text(data_tr, use.n = TRUE)

par(opar)

yhat_tr <- predict(data_tr, validation)
rmse(y_obs, yhat_tr)

set.seed(1810)

data_rf <- randomForest(medv ~ ., training)
data_rf

opar <- par(mfrow=c(1,2))

plot(data_rf)

varImpPlot(data_rf)

par(opar) 

yhat_rf <- predict(data_rf, newdata=validation)
rmse(y_obs, yhat_rf)

set.seed(1810)

data_gbm <- gbm(medv ~ ., data=training,
                
                n.trees=30000, cv.folds=3, verbose = TRUE)
(best_iter = gbm.perf(data_gbm, method="cv"))

yhat_gbm <- predict(data_gbm, n.trees=best_iter, newdata=validation)
rmse(y_obs, yhat_gbm)

data.frame(lm = rmse(y_obs, yhat_step),
           glmnet = rmse(y_obs, yhat_glmnet),
           rf = rmse(y_obs, yhat_rf),
           gbm = rmse(y_obs, yhat_gbm)) %>%
  reshape2::melt(value.name = 'rmse', variable.name = 'method')

rmse(test$medv, predict(data_rf, newdata = test))

boxplot(list(lm = y_obs-yhat_step,
             
             glmnet = y_obs-yhat_glmnet,
             
             rf = y_obs-yhat_rf,
             
             gbm = y_obs-yhat_gbm), ylab="Error in Validation Set")

abline(h=0, lty=2, col='red') 

pairs(data.frame(y_obs=y_obs,
                 
                 yhat_lm=yhat_step,
                 
                 yhat_glmnet=c(yhat_glmnet),
                 
                 yhat_rf=yhat_rf,
                 
                 yhat_gbm=yhat_gbm),
      
      lower.panel=function(x,y){ points(x,y); abline(0, 1, col='red')},
      
      upper.panel = panel.cor) 
