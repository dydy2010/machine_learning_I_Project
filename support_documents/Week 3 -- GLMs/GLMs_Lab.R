## ----SimPois, echo=FALSE-------------------------------------------
set.seed(1)
no.days <- 14
v.non.smoker <- c(rpois(n = no.days - 1, lambda = 0), 1)
## I added a single sigarette to the non-smoker because
## if you don't do so all p-values are 1...
v.smoker1_2 <- rpois(n = no.days, lambda = 5)
v.smoker.box <- rpois(n = no.days, lambda = 20)
d.smokers <- data.frame(no.cigarettes = 
                          c(v.non.smoker, v.smoker1_2, v.smoker.box),
                        person = gl(n = 3,
                                    k = no.days, 
                                    labels = c("non-smoker",
                                               "moderate smoker", 
                                               "heavy smoker")))
##
library(ggplot2)
set.seed(3)
# ggplot(data = d.smokers,
#        mapping = aes(y = no.cigarettes,
#                      x = person)) +
#   geom_jitter(width = 0.05, height = 0, alpha = 0.5) +
#   scale_y_continuous(breaks = c(0:5,1:3*10)) +
#   geom_hline(yintercept = 0)
##
ggplot(data = d.smokers,
       mapping = aes(y = no.cigarettes,
                     x = person)) +
  geom_boxplot() +
  # scale_y_continuous(breaks = c(0:5, 1:3*10)) +
  geom_hline(yintercept = 0)


## ----coefSmokers---------------------------------------------------
lm.smokers <- lm(no.cigarettes ~ person, data = d.smokers)
round(coef(lm.smokers), digits = 1)


## ----simObservations-----------------------------------------------
set.seed(3)
sim.data.smokers <- simulate(lm.smokers)
##
NROW(sim.data.smokers)
head(sim.data.smokers)
tail(sim.data.smokers)
##
ggplot(mapping = aes(y = sim.data.smokers$sim_1,
                     x = d.smokers$person)) +
  geom_boxplot() +
  geom_hline(yintercept = 0) +
  ylab("simulated no. of cigarettes\n(assuming normality)") +
  xlab("person")


## ----ExpResults----------------------------------------------------
exp(-5)
exp(-2)
exp(0)
exp(3)


## ----glmPoission---------------------------------------------------
glm.smokers <- glm(no.cigarettes ~ person, 
                   family = "poisson", ## we specify the distribution!
                   data = d.smokers)


## ----summaryPoisson------------------------------------------------
summary(glm.smokers)


## ----simPoissonFromGLM---------------------------------------------
set.seed(2)
sim.data.smokers.Poisson <- simulate(glm.smokers)
##
NROW(sim.data.smokers.Poisson)
head(sim.data.smokers.Poisson)
tail(sim.data.smokers.Poisson)
##
ggplot(mapping = aes(y = sim.data.smokers.Poisson$sim_1,
                     x = d.smokers$person)) +
  geom_boxplot() +
  geom_hline(yintercept = 0) +
  ylab("simulated no. of cigarettes\n(assuming Poisson dist)") +
  xlab("person")


## ----blissData-----------------------------------------------------
library(faraway)
data(bliss)


## ----strBliss------------------------------------------------------
str(bliss)
bliss


## ----computeMortality, message=FALSE-------------------------------
bliss$total.insects <- bliss$dead + bliss$alive
bliss$mortality.rate <- bliss$dead / bliss$total.insects
## 
## or with a more "modern" approach
library(dplyr)
bliss <- bliss %>% 
  mutate(mortality.rate = round(dead / (dead + alive), digits = 2))
##
bliss


## ----plotBliss-----------------------------------------------------
library(ggplot2)
ggplot(data = bliss,
       mapping = aes(y = mortality.rate,
                     x = conc)) + 
  geom_point()


## ----plotBlissWithLine, message=FALSE------------------------------
ggplot(data = bliss,
       mapping = aes(y = mortality.rate,
                     x = conc)) + 
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  ylim(0, 1) +
  geom_hline(yintercept = 0:1)


## ----curveInverseLogistic------------------------------------------
curve(expr = ilogit, from = -5, to = 5)
abline(h = c(0, 1), col = "gray")


## ----ilogit--------------------------------------------------------
ilogit(-20)
ilogit(-5)
ilogit(0)
ilogit(5)
ilogit(20)


## ----fittingGLmInsexts---------------------------------------------
glm.insects <- glm(cbind(dead, alive) ~ conc,  
                   family = "binomial",
                   data = bliss)


## ----summaryGLMinsects---------------------------------------------
summary(glm.insects)


## ----plotEffectConc, echo=FALSE------------------------------------
new.data = data.frame(conc = seq(0, 5, length.out = 100))
new.data$pred.insects <- predict(glm.insects, newdata = new.data,
                                 type = "response")
##
ggplot(data = bliss,
       mapping = aes(y = mortality.rate,
                     x = conc)) + 
  ylim(0,1) +
  geom_hline(yintercept = 0:1, col = "gray") +
  ##
  ## predictions for conc 0 --> 5
  geom_point(data = new.data,
               mapping = aes(
      y = pred.insects,
      x = conc)) +
  ##
  ## actual observations
  geom_point(col = "red", 
             size = 3)


## ----exampleLogistic, eval=FALSE-----------------------------------
## glm(survived.surgery ~ age + sex,
##     family = "binomial",
##     data = someSurgeryData)


## ----multinom, message=FALSE---------------------------------------
## Iris data
head(iris)
table(iris$Species)
##
library(nnet)
multinom.iris <- multinom(Species ~ Sepal.Length + Petal.Width, 
                          trace = FALSE,
                          data = iris)



## ----stab----------------------------------------------------------
d.stab <- read.table("../../Datasets/stability.dat", 
                     stringsAsFactors = TRUE,
                     header = TRUE)
str(d.stab)
head(d.stab)


## ----plotStab------------------------------------------------------
ggplot(data = d.stab,
       mapping = aes(y = perform,
                     x = stability)) + 
  geom_point()


## ----stabBinomGraph, message=FALSE, echo=FALSE---------------------
ggplot(data = d.stab,
       mapping = aes(y = perform,
                     x = stability)) + 
  geom_point() +
  geom_smooth(method = "glm", 
              se = FALSE,
              method.args = list(family = "binomial")) 


## ----logRegrStab---------------------------------------------------
glm.stab <- glm(perform ~ stability, 
                data = d.stab, 
                family = "binomial")
summary(glm.stab)


## ----fitteGLm------------------------------------------------------
fitted(glm.stab) %>% round(digits = 2)


## ----discretise----------------------------------------------------
fitted.stab.disc <- ifelse(fitted(glm.stab) < 0.5,
                           yes = 0, no = 1)
head(fitted.stab.disc)


## ----comapreObsFitted----------------------------------------------
d.obs.fit.stab <- data.frame(obs = d.stab$perform, 
                             fitted = fitted.stab.disc) 
head(d.obs.fit.stab)


## ----tableFotOBs---------------------------------------------------
table(d.obs.fit.stab$obs)
## 14 success and 13 failures in the real data
##
table(obs = d.obs.fit.stab$obs,
      fit = d.obs.fit.stab$fitted)


## ----confusionMatrixPrp--------------------------------------------
table(obs = d.obs.fit.stab$obs,
      fit = d.obs.fit.stab$fitted) %>% 
  prop.table() %>% 
  round(digits = 2)


## ----swappinggenderforsex, message=FALSE, echo=FALSE---------------
data("esdcomp")
colnames(esdcomp)[4] <- "sex"
# colnames(esdcomp)


## ----fittingTheComplainGLm-----------------------------------------
glm.complaints <- glm(complaints ~ . , data = esdcomp, 
                     family = "poisson")


## ----summaryGLMComplaints------------------------------------------
summary(glm.complaints)


## ----coefVisits----------------------------------------------------
coef(glm.complaints)["sexM"]


## ----expGender-----------------------------------------------------
exp(coef(glm.complaints)["sexM"]) %>% round(digits = 2)


## ----firstDoctor---------------------------------------------------
esdcomp[1, ]


## ----fittedForFirstDoctor------------------------------------------
fitted.first.doctor <- fitted(glm.complaints)[1]
fitted.first.doctor


## ----computeCompainsForFirstAsMale---------------------------------
first.doctor.as.male <- esdcomp[1, ]
first.doctor.as.male$sex <- "M"
first.doctor.as.male
##
pred.first.doc.male <- predict(glm.complaints, 
        type = "response", # ! important to set argument "type" to response!
        newdata = first.doctor.as.male)
pred.first.doc.male


## ----computingFirstDoctorByHand------------------------------------
fitted.first.doctor * exp(coef(glm.complaints)["sexM"])


## ----expBetaVisits-------------------------------------------------
exp.coef.visits <- exp(coef(glm.complaints)["visits"])
print(exp.coef.visits, digits = 5)


## ----rangeVisits---------------------------------------------------
range(esdcomp$visits)


## ----expBetaVisitsTimes100-----------------------------------------
coef.visits.50 <- coef(glm.complaints)["visits"] * 50
coef.visits.50
##
print(exp(coef.visits.50), digits = 5)


## ----plotEffectConcAgain, echo=FALSE-------------------------------
new.data = data.frame(conc = seq(0, 5, length.out = 100))
new.data$pred.insects <- predict(glm.insects, newdata = new.data,
                                 type = "response")
##
ggplot(data = bliss,
       mapping = aes(y = mortality.rate,
                     x = conc)) + 
  geom_point() +
  # geom_smooth(method = "lm", se = FALSE) +
  ylim(0,1) +
  geom_hline(yintercept = c(0,1)) +
  geom_line(data = new.data,
               mapping = aes(
      y = pred.insects,
      x = conc))


## ----coeffBinom----------------------------------------------------
coef(glm.insects)


## ----expCoeff------------------------------------------------------
exp(coef(glm.insects)["conc"])


## ----concPlotFited, echo=FALSE-------------------------------------
ggplot(data = bliss,
       mapping = aes(y = mortality.rate,
                     x = conc)) + 
  ylim(c(0, 1)) +
  geom_hline(yintercept = c(0,1)) +
  geom_segment(mapping = aes( x = 2,
                              y = 0,
                              xend = 2,
                              yend = 0.5),
                              colour = "red") +
  geom_segment(mapping = aes(x = 0,
                              y = 0.5,
                              xend = 2,
                              yend = 0.5),
                              colour = "red") +
  geom_segment(mapping = aes( x = 3,
                              y = 0,
                              xend = 3,
                              yend = 0.76),
                              colour = "red") +
  geom_segment(mapping = aes( x = 0,
                              y = 0.76,
                              xend = 3,
                              yend = 0.76),
                              colour = "red") +
    geom_line(data = new.data,
               mapping = aes(
      y = pred.insects,
      x = conc))


## ----expConcBeta---------------------------------------------------
exp(coef(glm.insects)["conc"])


## ----glmbinomialFactor---------------------------------------------
bliss$conc.asFactor <- cut(bliss$conc, breaks = c(-Inf, 0, 2, 4),
                           labels = c("no insecticide", "low conc", "high conc"))
levels(bliss$conc.asFactor)


## ----tableBliss----------------------------------------------------
with(bliss, table(conc, conc.asFactor))


## ----fittingBlissModel---------------------------------------------
glm.insects.factor <- glm(cbind(dead, alive) ~ conc.asFactor,
                          data = bliss,
                          family = "binomial")


## ----coefFactro----------------------------------------------------
coef(glm.insects.factor)


## ----coefFactrModelExp---------------------------------------------
exp(coef(glm.insects.factor))


## ----fittingGLmInsexts2--------------------------------------------
glm.insects <- glm(cbind(dead, alive) ~ conc,  
                   family = "binomial",
                   data = bliss)
summary(glm.insects)


## ----quasiModels---------------------------------------------------
quasi.glm.complaints <- glm(complaints ~ . , 
                                  data = esdcomp, 
                     family = "quasipoisson")


## ----summaryQuasiModel---------------------------------------------
summary(quasi.glm.complaints)


## ----fitGamGLm, message=FALSE--------------------------------------
library(mgcv)
gam.complaints <- gam(complaints ~ sex + 
                        s(visits) + 
                        s(revenue), 
                      family = "quasipoisson",
                      data = esdcomp)
##
summary(gam.complaints)


## ----purlFile, include=FALSE, eval=FALSE---------------------------
## ## (this chunk is not included nor evaluated)
## ##
## knitr::purl("GLMs_Lab.Rmd")


## ----sessionINfo, size='footnotesize'------------------------------
sessionInfo()

