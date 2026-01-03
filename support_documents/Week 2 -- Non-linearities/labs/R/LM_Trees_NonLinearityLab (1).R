## ----getData---------------------------------------------
d.trees <- read.csv2("../../Datasets/TreesChamagne2017_Lab_modified.csv",
                     stringsAsFactors = TRUE)
##
str(d.trees)
head(d.trees)


## ----diversitySite, message=FALSE------------------------
library(ggplot2)
gg.density.site <- ggplot(data = d.trees,
       mapping = aes(y = growth.rate,
                     x = density.site)) + 
  geom_point()
##
gg.density.site +
  geom_smooth()


## ----lmQuadraticEffects----------------------------------
## 1) model with a linear effect for density.site
lm.trees.1 <- lm(growth.rate ~ species + age + diversity.site +
                   density.site,
                 data = d.trees)
##
## 2) model with a quadratic effect for density.site
lm.trees.2 <- update(lm.trees.1, . ~ . + I(density.site^2))


## ----testinQuadratic-------------------------------------
anova(lm.trees.1, lm.trees.2)


## ----PlotDensitySiteQuadratic, echo=FALSE, message=FALSE----
gg.density.site + 
  geom_smooth(method = "lm", 
              formula = y ~ poly(x, degree = 2))
## NOTE
## This is NOT exactly what the model predicts... it is just the marginal estimation...for conditional we need to make predictions
## This concept is not explained in the course


## ----quadraticaViaPoly-----------------------------------
lm.trees.3 <- lm(growth.rate ~ species + age + diversity.site +
                           poly(density.site, degree = 2),
                 data = d.trees)


## ----simulateData, echo=FALSE----------------------------
set.seed(123)
n <- 100
##
x1 <- seq(from = -1, to = 20, length.out = n)
y1 <- -0.0057 * x1^3 + 0.1424 * x1^2
yy <- y1 + rnorm(n = n, mean = 0, sd = 0.5)
##
sim.data <- data.frame(y1, x1, yy)


## ----plotTrueAndObs, echo=FALSE--------------------------
## A) true relationship
gg.sim <- ggplot(data = sim.data,
                 mapping = aes(x = x1))
##
gg.true <- gg.sim +
  geom_line(mapping = aes(y = y1)) +
  ggtitle("True relationship")
##
## B) observed
gg.obs <- gg.sim +
  geom_point(mapping = aes(y = yy)) +
  ggtitle("Observations")
##
## C) both together
library(gridExtra)
grid.arrange(gg.true, gg.obs, ncol = 2)


## ----PlotSimDataLm, echo=FALSE, message=FALSE------------
gg.obs + 
  geom_smooth(method = "lm", 
              mapping = aes(y = yy),
              se = FALSE) +
  ggtitle("Linear effect of x1")


## ----plotSimDataQuadratic, echo=FALSE, message=FALSE-----
gg.obs + 
  geom_smooth(method = "lm", 
              formula = y ~ poly(x, degree = 2),
              mapping = aes(y = yy),
              se = FALSE) +
  ggtitle("Quadratic polynomial for x1")


## ----plotSimDataCubic, echo=FALSE, message=FALSE---------
gg.obs + 
  geom_smooth(method = "lm", 
              formula = y ~ poly(x, degree = 3),
              mapping = aes(y = yy),
              se = FALSE) +
  ggtitle("Cubic polynomial for x1")


## ----cubicLm---------------------------------------------
lm.cubic <- lm(yy ~ poly(x1, degree = 3), data = sim.data)
summary(lm.cubic)


## ----PredictionsOutside, echo=FALSE, message=FALSE, warning=FALSE----
d.pred.cubic.lm.outside <- data.frame(x1 = seq(from = -5,
                                       to = 25,
                                       length.out = 100))
pred.cubic.outside <- predict(lm.cubic, 
                              newdata = d.pred.cubic.lm.outside)
##
## Graph
gg.sim +
  geom_point(mapping = aes(y = sim.data$yy)) +
  geom_line(mapping = aes(y = pred.cubic.outside,
                          x = d.pred.cubic.lm.outside$x1),
            col = "red") +
    geom_smooth(method = "lm", 
              formula = y ~ poly(x, degree = 3),
              mapping = aes(y = yy),
              se = FALSE, col = "black")


## ----bSplinesLM, message=FALSE---------------------------
library(splines)
lm.regressionSplines <- lm(yy ~ bs(x1, df = 3), data = sim.data)
summary(lm.regressionSplines)


## ----plotBsFit, echo=FALSE-------------------------------
sim.data$fit.bs <- fitted(lm.regressionSplines)
ggplot(data = sim.data,
       mapping = aes(y = yy,
                     x = x1)) +
  geom_point() +
  geom_line(mapping = aes(y = fit.bs))


## ----graphPredictionsOutsideFittingRegion, echo=FALSE, message=FALSE----
require(mgcv)
gam.un.1 <- gam(yy ~ s(x1, fx = TRUE, k = 4), data = sim.data)
plot(gam.un.1, residuals = TRUE, cex = 3, xlim = c(-5, 25))
## This is not fully correct
## as we are not using a B-spline, but rather the thin plate regression splines...


## ----headOfBs, include=FALSE-----------------------------
## (this chunk is not included)
head(bs(sim.data$x1), df = 3)


## ----GraphChooseComplexits, echo=FALSE, message=FALSE----
gg.1 <- gg.obs + 
  geom_smooth(method = "lm", 
              formula = y ~ poly(x, degree = 1),
              mapping = aes(y = yy),
              se = FALSE) +
  ggtitle("Not complex enough (degree = 1)")
gg.1
##
gg.2 <- gg.obs + 
  geom_smooth(method = "lm", 
              formula = y ~ poly(x, degree = 2),
              mapping = aes(y = yy),
              se = FALSE) +
  ggtitle("Better, but not yet enough")
##
gg.3 <- gg.obs + 
  geom_smooth(method = "lm", 
              formula = y ~ poly(x, degree = 3),
              mapping = aes(y = yy),
              se = FALSE) +
  ggtitle("Quite Right (degree = 3)")
gg.3
##
gg.4 <- gg.obs + 
  geom_smooth(method = "lm", 
              formula = y ~ poly(x, degree = 25),
              mapping = aes(y = yy),
              se = FALSE) +
  ggtitle("Too complex  (degree = 25)")
gg.4
##
# grid.arrange(gg.1, gg.2, gg.3, gg.4, ncol = 2)


## ----fitGamTrees-----------------------------------------
library(mgcv)
gam.1 <- gam(yy ~ s(x1), data = sim.data)
summary(gam.1)


## ----effectOfx1------------------------------------------
plot(gam.1, residuals = TRUE, cex = 2)


## ----GAMTreeGrowth---------------------------------------
gam.trees.1 <- gam(growth.rate ~ species + 
                     s(density.site) + s(diversity.site), 
                   data = d.trees)
summary(gam.trees.1)


## ----plotGamDensity--------------------------------------
plot(gam.trees.1, residuals = TRUE, select = 1)


## ----gamTrees2-------------------------------------------
gam.trees.2 <- gam(growth.rate ~ species + 
                     s(density.site) + diversity.site, 
                   data = d.trees)
summary(gam.trees.2)


## ----ModelPancia-----------------------------------------
## 1) get data
d.pancia <- read.table("../../Datasets/Crescita.txt",
                       header = TRUE)
## 
## 2) declare Dates as such
d.pancia$MeasurementDate_AsDate <- as.Date(
  d.pancia$MeasurementDate,
  format = "%d-%m-%Y")
## check
head(d.pancia)
str(d.pancia)


## ----GraphPancia, message=FALSE--------------------------
gg.pancia <- ggplot(data = d.pancia,
       mapping = aes(y = Circonferenza,
                     x = MeasurementDate_AsDate)) +
  geom_point() +
  geom_smooth() +
  ylab("Girth [cm]") +
  ggtitle("Girth growth") 
gg.pancia


## ----gamGirth, include=FALSE-----------------------------
gam.girth <- gam(Circonferenza ~ s(as.numeric(MeasurementDate_AsDate)),
                 data = d.pancia) ## or
## equivalent to:
gam.girth <- gam(Circonferenza ~ s(Week),
                 data = d.pancia)
## note, however, that date is a more precise measurement than week
## in practice nothing changes here.
##
summary(gam.girth)


## ----plotGAmGirth, include=FALSE-------------------------
plot(gam.girth, residuals = TRUE, cex = 3)


## ----createDiversity2------------------------------------
set.seed(12)
d.trees$diversity.2 <- d.trees$diversity.site + 
  rnorm(n = nrow(d.trees), sd = 0.05)
##
plot(diversity.2 ~ diversity.site, data = d.trees)


## ----LmDiversity-----------------------------------------
lm.trees.4 <- lm( growth.rate ~ species + age + diversity.site, 
                  data = d.trees)
summary(lm.trees.4)


## ----LmDivesityAddSecond---------------------------------
lm.trees.5 <- update(lm.trees.4, . ~ . + diversity.2)
summary(lm.trees.5)


## ----CoeffsDiversity-------------------------------------
coef(lm.trees.4)["diversity.site"]
##
coef(lm.trees.5)["diversity.site"]


## ----SEModelsDiversity-----------------------------------
summary(lm.trees.4)$coefficients["diversity.site", ]
##
summary(lm.trees.5)$coefficients["diversity.site", ]


## ----vifLM4----------------------------------------------
library(car)
vif(lm.trees.4)


## ----vifLM5----------------------------------------------
vif(lm.trees.5)


## ----gamWithIneraction, eval=FALSE-----------------------
# gam.trees.3 <- gam(growth.rate ~ species +
#                      s(density.site, by = species),
#                    data = d.trees)
# summary(gam.trees.3)


## ----gamTensor, eval=FALSE-------------------------------
# gam.trees.4 <- gam(growth.rate ~ species +
#                      s(size) + s(X, Y), ## X and Y are coordinates
#                    data = d.trees)


## ----familyGam-------------------------------------------
gam.1
## or
# summary(gam.1)


## ----gamCheck--------------------------------------------
gam.check(gam.1)


## ----formulaTrees2---------------------------------------
formula(lm.trees.2)
# summary(lm.trees.2)


## ----twoFoldFormulae-------------------------------------
lm.two.fold <- lm(growth.rate ~ (species + age + diversity.site + density.site + density.site)^2, 
                  data = d.trees)
summary(lm.two.fold)


## ----sessionINfo, size='footnotesize'--------------------
sessionInfo()

