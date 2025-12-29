## ----getDataCats-----------------------------------------
d.cats <- read.csv("../../Datasets/Cats.csv", header = TRUE,
                   stringsAsFactors = TRUE)
##
head(d.cats)
str(d.cats)


## ----garphCAts, message=FALSE----------------------------
library(ggplot2)
ggplot(data = d.cats,
       mapping = aes(y = Heart.weight,
                     x = Body.weight)) +
  geom_point() +
  geom_smooth() +
  facet_wrap(. ~ Sex)


## ----catsLm----------------------------------------------
lm.cats <- lm(Heart.weight ~ Body.weight * Sex, data = d.cats)


## ----QQforCats-------------------------------------------
## 1) store the residuals in d.cats
d.cats$resid.lm.cats <- resid(lm.cats)
##
## 2) create the QQ-plot
qqnorm(resid(lm.cats))
qqline(resid(lm.cats))


## ----installPlgraphics, eval=FALSE-----------------------
# ## Install the package
# install.packages("plgraphics",
#                  repos = "http://R-forge.R-project.org")


## ----QQplotForCatsPlgraphics, message=FALSE--------------
library(plgraphics)
plregr(lm.cats, 
       plotselect = c(default = 0,
                      qq = 1),
       xvar = FALSE)


## ----TAplotCats, message=FALSE---------------------------
ggplot(mapping = aes(y = resid(lm.cats),
                     x = fitted(lm.cats))) +
  geom_abline(intercept = 0, slope = 0) +
  geom_point() + 
  geom_smooth()


## ----TACatsPlgraphics------------------------------------
plregr(lm.cats, 
       plotselect = c(resfit = 3,
                      default = 0,
                      absresfit = 0),
       xvar = FALSE)


## ----HomoscedastictiyCats, message=FALSE-----------------
ggplot(mapping = aes(y = abs(resid(lm.cats)),
                     x = fitted(lm.cats))) +
  geom_abline(intercept = 0, slope = 0) +
  geom_point() + 
  geom_smooth()


## ----HomoscedasticityCatsPlgraphics----------------------
plregr(lm.cats, 
       plotselect = c(default = 0,
                      absresfit = 2),
       xvar = FALSE)


## ----GraphCatsRegrLines----------------------------------
ggplot(data = d.cats,
       mapping = aes(y = Heart.weight,
                     x = Body.weight)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(. ~ Sex)


## ----CooksForCats----------------------------------------
## 1) compute Cook's distance 
cooks.dist.cats <- cooks.distance(lm.cats)
##
## 2) plot it
ggplot(data = d.cats,
       mapping = aes(y = Heart.weight,
                     x = Body.weight,
                     colour = cooks.dist.cats)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(. ~ Sex)


## ----LeveraPlotCats--------------------------------------
plot(lm.cats, which = 5)


## ----PlotLMCatsLM----------------------------------------
plot(lm.cats)


## ----graphResVsSex---------------------------------------
ggplot(data = d.cats,
       mapping = aes(y = resid.lm.cats,
                     x = Sex)) +
  geom_boxplot()


## ----ResAgainstBody.weight, message=FALSE----------------
ggplot(data = d.cats,
       mapping = aes(y = resid.lm.cats,
                     x = Body.weight)) +
  geom_point() +
  geom_smooth()


## ----ResAgainstBody.weightBySex, message=FALSE-----------
ggplot(data = d.cats,
       mapping = aes(y = resid.lm.cats,
                     x = Body.weight)) +
  geom_point() +
  geom_smooth() + 
  facet_wrap(. ~ Sex)


## ----simNormalErrors-------------------------------------
set.seed(7)
x <- gl(n = 2, k = 50, labels = c("A", "B"))
errors <- rnorm(n = 100, sd = 2)
y <- rep(x = c(8,20), each = 50) + errors
plot(y ~ x)


## ----histY-----------------------------------------------
par(mfrow = c(1,2))
hist(y, breaks = 20)
qqnorm(y)
qqline(y)


## ----QQForSimData----------------------------------------
lm.sim <- lm(y ~ x)
qqnorm(resid(lm.sim))
qqline(resid(lm.sim))


## ----sessionINfo, size='footnotesize'--------------------
sessionInfo()

