## ----getData, results='hide'------------------
## (results are hidden)
##
d.trees <- read.csv2("../../Datasets/TreesChamagne2017_Lab.csv",
                     stringsAsFactors = TRUE)
##
str(d.trees)
head(d.trees)


## ----plotSpecies------------------------------
boxplot(growth.rate ~ species, data = d.trees)


## ----fitLm1-----------------------------------
lm.trees.1 <- lm(growth.rate ~ species, data = d.trees)
coef(lm.trees.1)


## ----pvaluesSpecies---------------------------
summary(lm.trees.1)


## ----fitLm0-----------------------------------
lm.trees.0 <- lm(growth.rate ~ 1, data = d.trees)
coef(lm.trees.0)


## ----compareLm0AndLm1-------------------------
anova(lm.trees.0, lm.trees.1)


## ----testQuercusVsPicea, message=FALSE--------
library(multcomp)
ph.test.1 <- glht(model = lm.trees.1, 
                  linfct = mcp(species = 
                                 c("Quercus - Picea = 0")))
summary(ph.test.1)


## ----addingVariables--------------------------
lm.trees.2 <- update(lm.trees.1, 
                     . ~ . + Density.tree.Class + SiteID)


## ----checkFormulaLM2--------------------------
formula(lm.trees.2)


## ----Drop1OnLM2-------------------------------
drop1(lm.trees.2, test = "F")


## ----getUniqueSiteIDHead----------------------
head(unique(d.trees$SiteID))


## ----siteAsFactor-----------------------------
d.trees$SiteID.fac <- factor(d.trees$SiteID)


## ----fixingSiteIdPredictor--------------------
lm.trees.3 <- update(lm.trees.2,
                     . ~ . + SiteID.fac - SiteID)
##
drop1(lm.trees.3, test = "F")


## ----addingAge--------------------------------
lm.trees.4 <- update(lm.trees.3, . ~ . + age)
##
drop1(lm.trees.4, test = "F")


## ----pvalueForAgeViaSummary-------------------
capture.output(summary(lm.trees.4))[c(10,11,62)]


## ----globalFtest------------------------------
anova(lm.trees.0, lm.trees.4)


## ----GlobalFTest------------------------------
tail(capture.output(summary(lm.trees.4)))


## ----SeqSS------------------------------------
formula(lm.trees.4)
anova(lm.trees.4)


## ----SeqSSWithSpeciesAtEnd--------------------
lm.trees.4.again <- lm(growth.rate ~ Density.tree.Class + SiteID.fac +
                         age + species, 
                       data = d.trees)
anova(lm.trees.4.again)


## ----testref, ref.label='testQuercusVsPicea'----


## ----createVectorConifersBroadleavedOne-------
levels(d.trees$species)
##
## vector of contrasts
v.Quercus_vs_Picea <- c(0, 0, -1, 1)
names(v.Quercus_vs_Picea) <-
  levels(d.trees$species)
##
v.Quercus_vs_Picea


## ----testQuercusVsPiceaVector, message=FALSE----
library(multcomp)
ph.test.Q_vs_P <- glht(model = lm.trees.1, 
                  linfct = mcp(species = v.Quercus_vs_Picea))
summary(ph.test.Q_vs_P)


## ----createVectorConifersBroadleaved----------
levels(d.trees$species)
##
## vector of contrasts
v.conifers_vs_broadleaved <- c(1/2, -1/2, -1/2, 1/2)
names(v.conifers_vs_broadleaved) <-
  levels(d.trees$species)
##
v.conifers_vs_broadleaved


## ----testConifersBroadleaved------------------
ph.test.conifers_vs_broadleaved <- glht(
  model = lm.trees.1,
  linfct = mcp(species = v.conifers_vs_broadleaved))
##
summary(ph.test.conifers_vs_broadleaved)


## ----THSDForSpecies---------------------------
ph.test.THSD <- glht(lm.trees.1,
                     linfct = mcp(species = "Tukey"))
summary(ph.test.THSD)


## ----graphTHSDForSpcies, out.width='0.9\\linewidth'----
## 1) change margins default
par("mar")
par(mar = c(5.1,7.5,4.1,2.1))
##
## 2) plot contrasts
plot(ph.test.THSD)


## ----matrixForPHContrasts---------------------
M.ph.test.mat.1 <- rbind(
  "Fagus vs Larix" = c(1, -1, 0, 0),
  "Picea vs Quercus" = c(0, 0, 1,-1),
  "Fagus vs Quercus & Picea" = c(1, 0, -1/2, -1/2),
  "F - L vs Q - P" = c(1, -1, 1, -1))
colnames(M.ph.test.mat.1) <- levels(d.trees$species)
##
M.ph.test.mat.1


## ----testingPHMAtrix--------------------------
ph.test.mat.1 <- glht(
  model = lm.trees.1,
  linfct = mcp(species = M.ph.test.mat.1))
##
summary(ph.test.mat.1)


## ----conifersYES------------------------------
d.trees$conifers.YES <- d.trees$species %in% c("Larix", "Picea")
##
table(d.trees$species,
      d.trees$conifers.YES)


## ----LmTreesConifersYes-----------------------
lm.trees.conifers <- update(lm.trees.1, . ~ . + conifers.YES) 
##
summary(lm.trees.conifers)


## ----addingInteractionSpeciesage--------------
lm.trees.5 <- update(lm.trees.4, 
                     . ~ . + age:species)
##
drop1(lm.trees.5, test = "F")


## ----ageEffect, message=FALSE-----------------
library(ggplot2)
ggplot(data = d.trees,
       mapping = aes(y = growth.rate,
                     x = age)) +
  geom_point() + ## adds observations
  geom_smooth(method = "lm") ## adds the regression line with CI


## ----ageEffectBySpecies-----------------------
ggplot(data = d.trees,
       mapping = aes(y = growth.rate,
                     x = age)) +
  geom_point() +
  geom_smooth(method = "lm") +
  facet_grid(. ~ species)


## ----lm.trees1again---------------------------
summary(lm.trees.1)
##
drop1(lm.trees.1, test = "F")


## ----relevelSpecies---------------------------
d.trees$species.relevelled <- relevel(d.trees$species, 
                                      ref = "Quercus")
##
lm.trees.1.relevelled <- update(lm.trees.1, 
                                . ~ . -species + species.relevelled)


## ----pValuesRelevelledLm1---------------------
summary(lm.trees.1.relevelled)
##
drop1(lm.trees.1.relevelled, test = "F")


## ----sessionINfo, size='footnotesize'---------
sessionInfo()

