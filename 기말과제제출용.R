rmse <- function(yi, yhat_i){
  sqrt(mean((yi - yhat_i)^2))
}

library(tidyverse)
library(MASS) #로버스트 선형회귀를 위한 패키지
library(glmnet) #라쏘, 능형, 일래스틱넷 모형을 위한 패키지
library(randomForest) #탠덤 포레스트 모형을 위한 패키지
library(gbm) #부스팅을 위한 패키지
library(rpart) #나무모형을 위한 패키지
library(boot) #유의미한 변수선택을 위한 패키지
library(data.table) #데이터 테이블 형을 위한 패키지
library(ROCR) #ROC 곡선을 위한 패키지
library(ggplot2) #데이터 시각화를 위한 패키지
library(dplyr) #데이터 가공 문법을 위한 패키지
library(gridExtra) #그래프를 격자 형태로 보기 위한 패키지

housing <- read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", strip.white = T)
data <- tbl_df(housing)
names(data) <- c('crim', 'zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat','medv')
glimpse(data) #반응변수는 medv: 주택가격

panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}

pairs(data %>% dplyr::sample_n(min(1000, nrow(data))),
      lower.panel=function(x,y){ points(x,y); abline(0, 1, col='red')},
      upper.panel = panel.cor) 

set.seed(1606)
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

data_lm_full2 = lm(medv ~ .^2, data = training)
summary(data_lm_full2)

length(coef(data_lm_full2))

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
data_cvfit <- cv.glmnet(x, y)
plot(data_cvfit) 

coef(data_cvfit, s=c("lambda.1se"))
log(data_cvfit$lambda.1se)
length(which(coef(data_cvfit, s=c("lambda.1se")) !=0))-1


coef(data_cvfit, s = c("lambda.min"))
log(data_cvfit$lambda.min)
length(which(coef(data_cvfit, s=c("lambda.min")) !=0))-1

predict(data_cvfit, s="lambda.min", newx = x[1:5, ])

y_obs <- validation$medv
yhat_glmnet <- predict(data_cvfit, s="lambda.min", newx=xx[validate_idx,])
#yhat_glmnet <- yhat_glmnet[,1] # change to a vector from [n*1] matrix
rmse(y_obs, yhat_glmnet)

# 트리
data_tr <- rpart(medv ~ ., data = training)
par(mfrow = c(1,1), xpd = NA)
plot(data_tr)
text(data_tr, use.n = TRUE)

predict(data_tr,newdata = data[1:5, ])

y_obs <- validation$medv
yhat_tr <- predict(data_tr, validation)
rmse(y_obs, yhat_tr)

set.seed(1607)
data_rf <- randomForest(medv ~ ., training)

data_rf

plot(data_rf)

varImpPlot(data_rf)

predict(data_rf, newdata=data[1:5,])

y_obs <- validation$medv
yhat_rf <- predict(data_rf, newdata=validation)
rmse(y_obs, yhat_rf)

set.seed(1607)

data_gbm <- gbm(medv ~ ., data=training,
                n.trees=30000, cv.folds=3, verbose = TRUE)
(best_iter = gbm.perf(data_gbm, method="cv"))

y_obs<-validation$medv
yhat_gbm <- predict(data_gbm, n.trees=best_iter, newdata=validation)
rmse(y_obs, yhat_gbm)

data.frame(
  method=c('lm','glmnet','tr','rf','gbm'),
  rmse=c(rmse(y_obs, yhat_step),
           rmse(y_obs, yhat_glmnet),
           rmse(y_obs, yhat_tr),
           rmse(y_obs, yhat_rf),
           rmse(y_obs, yhat_gbm)))

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
