## ----getData---------------------------------------------
# d.farms <- read.table("http://stat.ethz.ch/Teaching/Datasets/WBL/farmsAus.dat",
#                       header = TRUE)
## or
d.farms <- readRDS("Farms_Australia.RDS")
str(d.farms)
head(d.farms)


## ----plotIndustry----------------------------------------
library(ggplot2)
ggplot(data = d.farms,
       mapping = aes(y = revenue,
                     x = industry)) + 
  geom_boxplot()


## ----plotRemoveOut, message=FALSE------------------------
library(dplyr) ## contains the filter() function
ggplot(data = filter(d.farms, revenue < 10^4),
       mapping = aes(y = revenue,
                     x = industry)) + 
  geom_boxplot()


## ----plotRemoveOut2--------------------------------------
ggplot(data = filter(d.farms, revenue < 4000),
       mapping = aes(y = revenue,
                     x = industry)) + 
  geom_boxplot()


## ----BoxpltoLogy-----------------------------------------
ggplot(data = d.farms,
       mapping = aes(y = log(revenue),
                     x = industry)) + 
  geom_boxplot()


## ----ggLogRevenuVsExpenditure----------------------------
ggplot(data = d.farms,
       mapping = aes(y = log(revenue),
                     x = expenditure)) + 
  geom_point()


## ----ggLogRevenuVsLogExpenditure-------------------------
ggplot(data = d.farms,
       mapping = aes(y = log(revenue),
                     x = log(expenditure))) + 
  geom_point()


## ----getOzone--------------------------------------------
# install.packages("gss")
library(gss)
data(ozone)
## OR
# ozone <- readRDS("Ozone.RDS")


## ----helpOzone, eval=FALSE-------------------------------
# ?ozone


## ----strOzone--------------------------------------------
str(ozone)
head(ozone)


## ----edaOzone, message=FALSE-----------------------------
# vdht
ggplot(data = ozone,
       mapping = aes(y = upo3,
                     x = vdht)) + 
  geom_point() +
  geom_smooth()
# 
# sbtp
ggplot(data = ozone,
       mapping = aes(y = upo3,
                     x = sbtp)) + 
  geom_point() +
  geom_smooth()
##
# ibth
ggplot(data = ozone,
       mapping = aes(y = upo3,
                     x = ibht)) + 
  geom_point() +
  geom_smooth()


## ----pairsOzone------------------------------------------
## all plots together (quick and dirty)
pairs(upo3 ~ vdht + sbtp + ibht, data = ozone, upper.panel = panel.smooth)


## ----lmOzone---------------------------------------------
lm.ozone <- lm(upo3 ~ vdht + sbtp + ibht, data = ozone)


## ----TAlmOzone-------------------------------------------
plot(lm.ozone, which = 1)
plot(lm.ozone, which = 3)


## ----LmOzoneQuad-----------------------------------------
lm.ozone.quad <- lm(upo3 ~ poly(vdht, degree = 2) +
                   poly(sbtp, degree = 2) + 
                   poly(ibht, degree = 2), data = ozone)
##
plot(lm.ozone.quad, which = 1)
plot(lm.ozone.quad, which = 3)


## ----edaLogOzone-----------------------------------------
pairs(log(upo3) ~ vdht + sbtp + ibht, data = ozone, 
      upper.panel = panel.smooth)


## ----plotLogOzone----------------------------------------
lm.ozone.log.quad <- update(lm.ozone.quad, log(upo3) ~ .)
formula(lm.ozone.log.quad )
##
##
plot(lm.ozone.log.quad, which = 1)
plot(lm.ozone.log.quad, which = 3)


## ----qqLmLogOZone, message=FALSE-------------------------
library(plgraphics)
plregr(lm.ozone.log.quad ,
       plotselect = c(default = 0,
                      qq = 2),
       xvar = FALSE)


## ----LeverageLogQuaOzone---------------------------------
plot(lm.ozone.log.quad, which = 5)


## ----summaryLmLOgOzon------------------------------------
summary(lm.ozone.log.quad)


## ----fitOzoneModel---------------------------------------
lm.ozone.simple <- lm(log(upo3) ~ sbtp, data = ozone)
summary(lm.ozone.simple)


## ----expCoeff--------------------------------------------
exp(coef(lm.ozone.simple)["sbtp"]) %>% 
  print(digits = 3)


## ----lmFarms---------------------------------------------
lm.farms <- lm(log(revenue)~ log(expenditure) + region + industry,
               data = d.farms)
summary(lm.farms)


## ----CoefFarms-------------------------------------------
coef(lm.farms)["log(expenditure)"]


## ----predictBack-----------------------------------------
exp(fitted(lm.ozone.log.quad)[1:5])


## ----echo=FALSE, message=FALSE---------------------------
## (this chunk is not echoed and messages are omitted)
##
gg.sbtp <- ggplot(data = ozone,
                  mapping = aes(y = log(upo3),
                                x = sbtp)) +
  geom_point() + geom_smooth()
gg.sbtp.log <- gg.sbtp + aes(x = log(sbtp))
##
require(gridExtra)
grid.arrange(gg.sbtp, gg.sbtp.log)


## ----sessionINfo, size='footnotesize'--------------------
sessionInfo()

