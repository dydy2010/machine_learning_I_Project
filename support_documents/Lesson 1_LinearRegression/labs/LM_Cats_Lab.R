## ----getData--------------------------------------------
d.cats <- read.csv("../../Datasets/Cats.csv", 
                   header = TRUE, 
                   stringsAsFactors = TRUE)
##
str(d.cats)
head(d.cats)


## ----graphBwt, out.width='0.8\\linewidth'---------------
plot(Hwt ~ Bwt, data = d.cats,
     main = "Heart weight against body weight")


## ----graphSex, out.width='0.8\\linewidth'---------------
boxplot(Hwt ~ Sex, data = d.cats,
        main = "Heart weight against gender",
        ylab = "Hwt")


## ----graphBothPredictors--------------------------------
plot(Hwt ~ Bwt, data = d.cats,
     col = Sex, 
     pch = 19,
     main = "Heart weight against body weight")
##
legend("topleft", 
       pch = 19,
       legend = c("F","M"), 
       col = c("black", "red"))


## ----graphQplot-----------------------------------------
library(ggplot2)
qplot(y = Hwt, x = Bwt, 
      data = d.cats, 
      facets = ~ Sex)


## ----ggplotWithRegressionLine, echo=FALSE, message=FALSE----
ggplot(data = d.cats,
       mapping = aes(y = Hwt, x = Bwt)) +
  geom_point() +
  facet_grid(. ~ Sex) +
  geom_smooth(method = "lm", se = FALSE)


## ----fitRegression--------------------------------------
lm.cats <- lm(Hwt ~ Bwt, data = d.cats)
summary(lm.cats)


## ----coefRegression-------------------------------------
coef(lm.cats)


## ----visualiseSimpleRegression, echo=FALSE--------------
plot(Hwt ~ Bwt, data = d.cats,
     xlim = c(0,4), 
     ylim = c(-0.5, 20))
##
grid()
##
abline(a = coef(lm.cats)[1],
       b = coef(lm.cats)[2],
       col = "red", lty = "dashed")
## equivalent to:
# abline(lm.cats)






## ----addingSex------------------------------------------
lm.cats.2 <- lm(Hwt ~ Bwt + Sex, data = d.cats)


## ----summaryLmCats2-------------------------------------
summary(lm.cats.2)


## ----graphFitLmCats2, echo=FALSE------------------------
plot(Hwt ~ Bwt, data = d.cats,
     main = "Model 'lm.cats.2'",
     col = Sex)
##
legend("topleft", 
       pch = 19,
       legend = c("F","M"), 
       col = c("black", "red"))
##
abline(a = coef(lm.cats.2)[1],
       b = coef(lm.cats.2)["Bwt"])
abline(a = coef(lm.cats.2)[1] + coef(lm.cats.2)["SexM"] ,
       b = coef(lm.cats.2)["Bwt"],
       col = "red")


## ----coefLmCats2----------------------------------------
coef(lm.cats.2)


## ----interceptSexes-------------------------------------
## intercept females:
coef(lm.cats.2)["(Intercept)"] 
##
## intercept males:
coef(lm.cats.2)["(Intercept)"] + coef(lm.cats.2)["SexM"]




## ----fitCatsLmWithInteraction---------------------------
lm.cats.3 <- lm(Hwt ~ Bwt * Sex, data = d.cats)


## ----graphFitLmCats3, echo=FALSE------------------------
plot(Hwt ~ Bwt, data = d.cats,
     main = "Model 'lm.cats.3'",
     xlim = c(0, 4), 
     ylim = c(-5, 21),
     col = Sex)
##
grid()
##
abline(h = 0, lwd = 0.5)
abline(v = 0, lwd = 0.5)
##
abline(a = coef(lm.cats.3)[1],
       b = coef(lm.cats.3)["Bwt"])
abline(a = coef(lm.cats.3)[1] + coef(lm.cats.3)["SexM"] ,
       b = coef(lm.cats.3)["Bwt"] + coef(lm.cats.3)["Bwt:SexM"],
       col = "red")
##
legend(x = 0.1, y = 20, 
       pch = 19,
       legend = c("F","M"), 
       col = c("black", "red"))


## ----coefLmCats3----------------------------------------
coef(lm.cats.3)


## ----fitLmCats3Bis--------------------------------------
lm.cats.3.bis <- lm(Hwt ~ Bwt + Sex + Bwt:Sex, data = d.cats)




## ----confintLmCats3-------------------------------------
confint(lm.cats.3)


## ----GraphsCisLmCats3, out.width='0.7\\linewidth', echo=FALSE----
## 1) estimate CIs
cis.lm.cats.3 <- confint(lm.cats.3)
##
## 2) plot estimates
par(mar = c(4,5,2,2))
plot(y = 1:4,
     x = rev(coef(lm.cats.3)),
     xlim = c(-9, 7),
     xlab = "Estimated coefficients",
     ylab = "",
     axes = FALSE)
box()
axis(side = 2, at = 1:4,
     labels = rev(names(coef(lm.cats.3))), 
     las = 2)
axis(side = 1)
##
## 3) plot CIs
segments(x0 = rev(cis.lm.cats.3[, "2.5 %"]),
         x1 = rev(cis.lm.cats.3[, "97.5 %"]),
         y0 = 1:4,
         y1 = 1:4)
abline(v = 0, lty = "dashed")


## ----rSquared-------------------------------------------
## model with no interaction
formula(lm.cats.2)
summary(lm.cats.2)$r.squared
##
## model with interaction
formula(lm.cats.3)
summary(lm.cats.3)$r.squared


## ----adjustedRsquared-----------------------------------
summary(lm.cats.2)$adj.r.squared
summary(lm.cats.3)$adj.r.squared


## ----fittedValues---------------------------------------
fitted.cats <- fitted(lm.cats)
##
str(fitted.cats)
head(fitted.cats)


## ----fittedValuesPlot-----------------------------------
plot(Hwt ~ Bwt, data = d.cats,
     main = "Model 'lm.cats'",
     col = "darkgray")
##
points(fitted.cats ~ Bwt, 
       col = "purple",
       pch = 19,
       data = d.cats)
##
abline(lm.cats, col = "black")


## ----residuals------------------------------------------
resid.cats <- resid(lm.cats)
##
length(resid.cats)
head(resid.cats)


## ----randomResiduals------------------------------------
set.seed(20) ## for reproducibility
id <- sample(x = 1:144, size = 5)
resid.cats[id]
fitted.cats[id]


## ----plotResiduals--------------------------------------
plot(Hwt ~ Bwt, data = d.cats,
     main = "Model 'lm.cats'",
     col = "lightgray")
##
abline(lm.cats)
##
points(Hwt ~ Bwt, data = d.cats[id, ], col = "red")
##
segments(x0 = d.cats[id, "Bwt"], x1 = d.cats[id, "Bwt"],
         y0 = fitted.cats[id], y1 = d.cats[id, "Hwt"],
         col = "blue")


## ----PredictionsNewCats---------------------------------
## 1) create the new data
new.data.cats <- data.frame(Bwt = c(4, 2.5, 3))
##
## 2) make predictions
pred.new.cats <- predict(object = lm.cats, newdata = new.data.cats)
## 
## 3) display predictions
plot(Hwt ~ Bwt, 
     data = d.cats,
     xlim = c(2, 4))
abline(lm.cats)
##
points(x = new.data.cats$Bwt,
       y = pred.new.cats, 
       col = "purple",
       pch = 19, cex = 1.5)


## ----PredictionsNewCatsCIs------------------------------
pred.new.cats.ci <- predict(object = lm.cats, 
                            interval = "prediction",
        newdata = new.data.cats)
pred.new.cats.ci


## ----PredictionsNewCatsCIsGraphs------------------------
plot(Hwt ~ Bwt, 
     data = d.cats,
     xlim = c(2, 4))
abline(lm.cats)
##
points(x = new.data.cats$Bwt,
       y = pred.new.cats.ci[, "fit"], 
       col = "purple",
       pch = 19, cex = 1.5)
##
segments(x0 = new.data.cats$Bwt, 
         x1 = new.data.cats$Bwt,
         y0 = pred.new.cats.ci[, "lwr"],
         y1 = pred.new.cats.ci[, "upr"],
         lwd = 2,
         col = "purple")


## ----TreatmentContrastsThreeLevels----------------------
## 1) add the new level first
levels(d.cats$Sex)
##
levels(d.cats$Sex) <- c("F", "M", "Unknown")
##
levels(d.cats$Sex)
##
## 2) set the first 10 observations to unknown
d.cats$Sex[1:10] <- "Unknown"
##
## 3) fit a simple model 
lm.cats.Newgender <- lm(Hwt ~ Sex, data = d.cats)
##
## 4) look at the coefficients
coef(lm.cats.Newgender)


## ----changingReference----------------------------------
## 1) change reference level
levels(d.cats$Sex)
##
d.cats$Sex <- relevel(d.cats$Sex, ref = "M")
##
levels(d.cats$Sex)
##
## 2) refit model
lm.cats.relevelled <- lm(Hwt ~ Sex, data = d.cats)
coef(lm.cats.relevelled)

