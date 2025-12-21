## ----LoadData-------------------------------------------------------------------------------------------
library(faraway)
data("esdcomp")
##
head(esdcomp)

colnames(esdcomp)[4] <- "sex"

## ----fitComplexComplaints-------------------------------------------------------------------------------
glm.complaints.1 <- glm(complaints ~ (poly(visits, degree = 3) +
                                        residency + 
                                        poly(revenue, degree = 3) +
                                        poly(hours, degree = 3)) * sex,
                        family = "poisson",
                        data = esdcomp)


## ----summaryComplexModel--------------------------------------------------------------------------------
summary(glm.complaints.1)


## ----NtoPratio------------------------------------------------------------------------------------------
( N <- nrow(esdcomp) )
( p <- length(coef(glm.complaints.1)))


## ----getTrees-------------------------------------------------------------------------------------------
d.trees <- read.csv2("../../Datasets/TreesChamagne2017_Lab.csv",
                     stringsAsFactors = TRUE)
str(d.trees)


## ----fitTwomodelsToCompare------------------------------------------------------------------------------
## 1) the "simple" model
lm.trees.1 <- lm(growth.rate ~ species + poly(density.site, degree = 2),
                 data = d.trees)
##
## 2) the "complex" model
lm.trees.2 <- lm(growth.rate ~ species * (poly(age, degree = 3) + 
                                            poly(diversity.site, degree = 3) +
                                            poly(density.site, degree = 3)),
                 data = d.trees)


## ----Rsquared-------------------------------------------------------------------------------------------
summary(lm.trees.1)$r.squared
##
summary(lm.trees.2)$r.squared


## ----adjRsquared----------------------------------------------------------------------------------------
summary(lm.trees.1)$adj.r.squared
##
summary(lm.trees.2)$adj.r.squared


## ----ComputeMSEINSAMPLE---------------------------------------------------------------------------------
mean((d.trees$growth.rate - predict(lm.trees.1))^2)
##
mean((d.trees$growth.rate - predict(lm.trees.2))^2)



## ----OrderAndCut----------------------------------------------------------------------------------------
## Some new commands we will use:
( folds <- cut(1:25, breaks = 6, labels = FALSE) )
##
which(folds == 2)



## ----SIzeTwoSamples-------------------------------------------------------------------------------------
nrow(d.trees)


## ----splitData------------------------------------------------------------------------------------------

( folds <- cut(1:nrow(d.trees), breaks = 2, labels = FALSE) )
## look at the number of observations in each fold
table(folds)



## ----trainTest------------------------------------------------------------------------------------------
## 1) prepare data
( ind.test <- which(folds == 1) )
d.trees.test <- d.trees[ind.test, ]
d.trees.train <- d.trees[- ind.test, ]
##
nrow(d.trees.test)
nrow(d.trees.train)
##
## 2) fit the model with "train" data
lm.trees.1.train <- lm(formula = formula(lm.trees.1),
                       data = d.trees.train)
## 
## 3) make prediction on the test data
predicted.trees.1.test <- predict(lm.trees.1.train,
                                newdata = d.trees.test)
##
## 4) compute MSE on the test data
mean((d.trees.test$growth.rate - predicted.trees.1.test)^2)




## ----trainTestCOmplex-----------------------------------------------------------------------------------
## 2) fit the model with "train" data
lm.trees.2.train <- lm(formula = formula(lm.trees.2),
                       data = d.trees.train)
## 
## 3) make prediction on the test data
predicted.trees.2.test <- predict(lm.trees.2.train,
                                newdata = d.trees.test)
##
## 4) compute MSE on the test data
mean((d.trees.test$growth.rate - predicted.trees.2.test)^2)


## ----RandomlypermuteRows--------------------------------------------------------------------------------
 ## randomly permute the row-numbers:
set.seed(1)
n <- nrow(d.trees)
## permuted row-numbers (= indices)
( inds.permuted <- sample(1:n, replace = FALSE) ) 
## we permute the data according to the permuted row-numbers
d.trees.permuted <- d.trees[inds.permuted, ]
dim(d.trees.permuted)
dim(d.trees)
## we create the 2 folds of (roughly) equal size as before:
( folds <- cut(1:n, breaks = 2, labels = FALSE) ) 
##
( ind.test <- which(folds == 1) )
d.trees.test <- d.trees.permuted[ind.test, ]
d.trees.train <- d.trees.permuted[- ind.test, ]



## ----fitftyFifty200Times--------------------------------------------------------------------------------
set.seed(22)
##
mse.simple <- c()
mse.complex <- c()
##
for(i in 1:200){
  ## 1) prepare data
  ## randomly permute the rows:
  n <- nrow(d.trees)
  inds.permuted <- sample(1:n, replace = FALSE)
  d.trees.permuted <- d.trees[inds.permuted, ]

  ## create 2 (roughly) equally sized folds:
  folds <- cut(1:n, breaks = 2, labels = FALSE)
  ##
  ind.test <- which(folds == 1)
  d.trees.test <- d.trees.permuted[ind.test, ]
  d.trees.train <- d.trees.permuted[- ind.test, ]
  ##
  ## simple model ##
  ##
  ## 2) fit the model with "train" data
  lm.trees.1.train <- lm(formula = formula(lm.trees.1),
                       data = d.trees.train)
  ## 
  ## 3) make prediction on the test data
  predicted.trees.1.test <- predict(lm.trees.1.train,
                                newdata = d.trees.test)
  ##
  ## 4) compute MSE
  mse.simple[i] <- mean((d.trees.test$growth.rate - predicted.trees.1.test)^2)

  ##
  ## complex model ##
  ##
  ## 2) fit the model with "train" data
  lm.trees.2.train <- lm(formula = formula(lm.trees.2),
                       data = d.trees.train)
  ## 
  ## 3) make prediction on the test data
  predicted.trees.2.test <- predict(lm.trees.2.train,
                                newdata = d.trees.test)
  ##
  ## 4) compute MSE
  mse.complex[i] <- mean((d.trees.test$growth.rate - predicted.trees.2.test)^2)
  }


## ----meanRsquared---------------------------------------------------------------------------------------
mean(mse.simple)
mean(mse.complex)


## ----boxplotCV------------------------------------------------------------------------------------------
boxplot(mse.simple, mse.complex)



## ----10fold200Times-------------------------------------------------------------------------------------
set.seed(22)
##
mse.simple <- c()
mse.complex <- c()
##
for(i in 1:200){
  ## 1) prepare data
  ## randomly permute the rows:
  n <- nrow(d.trees)
  inds.permuted <- sample(1:n, replace = FALSE)
  d.trees.permuted <- d.trees[inds.permuted, ]
  
  ## create 10 (roughly) equally sized folds:
  ## K = number of folds 
  K <- 10
  folds <- cut(1:n, breaks = K, labels = FALSE)
  ##
  ## perform K fold cross validation:
  mse.simple.per.fold <- integer(K)
  mse.complex.per.fold <- integer(K)

  for(k in 1:K){
  ## take the Kth fold as test set and the other folds as training set  
    ind.test <- which(folds == k)
    d.trees.test <- d.trees.permuted[ind.test, ]
    d.trees.train <- d.trees.permuted[- ind.test, ]
  ##
  ## simple model ##
  ##
  ## 2) fit the model with "train" data
  lm.trees.1.train <- lm(formula = formula(lm.trees.1),
                       data = d.trees.train)
  ## 
  ## 3) make prediction on the test data
  predicted.trees.1.test <- predict(lm.trees.1.train,
                                newdata = d.trees.test)
  ##
  ## 4) compute MSE
  mse.simple.per.fold[k] <- mean((d.trees.test$growth.rate - predicted.trees.1.test)^2)
  ##
  ## complex model ##
  ##
  ## 2) fit the model with "train" data
  lm.trees.2.train <- lm(formula = formula(lm.trees.2),
                       data = d.trees.train)
  ## 
  ## 3) make prediction on the test data
  predicted.trees.2.test <- predict(lm.trees.2.train,
                                newdata = d.trees.test)
  ##
  ## 4) compute MSE
  mse.complex.per.fold[k] <- mean((d.trees.test$growth.rate - predicted.trees.2.test)^2)

  }
  
  ## we obtain the estimated test MSE for the 10-fold CV by averaging the 
  ## test MSE on all 10 folds
  mse.simple[i] <- mean(mse.simple.per.fold)
  mse.complex[i] <- mean(mse.complex.per.fold)
  
  }


## ----MEANRMSE10fold-------------------------------------------------------------------------------------
mean(mse.simple)
mean(mse.complex)


## ----boxplot10foldCV------------------------------------------------------------------------------------
boxplot(mse.simple, mse.complex)



## ----compromiseComplexityPredictivePerfomance, echo=FALSE-----------------------------------------------
curve(expr = -x^2 + 30, from = -5, to = 5,
      ylab = "performance",
      xlab = "model complexity",
      xaxt = "n")
Axis(x = c(-5,5), side = 1, labels = seq(0,100, length.out = 11),
     at = seq(-5,5, length.out = 11))
text(x = -3.3, y = 5, labels = "too simple models", col = "red")
text(x = 3.3, y = 5, labels = "too complex models", col = "red")
text(x = 0, y = 28, labels = "best models", col = "red")


## ----purlCV, include=FALSE, eval=FALSE------------------------------------------------------------------
knitr::purl("CrossValidation_Lab.Rmd")


## ----sessionINfo, size='footnotesize'-------------------------------------------------------------------
sessionInfo()

