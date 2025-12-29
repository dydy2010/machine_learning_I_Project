## ----getData---------------------------------------------
d.trees <- read.csv2("../../Datasets/TreesChamagne2017_Lab_modified.csv",
                     stringsAsFactors = TRUE)
##
str(d.trees)
head(d.trees)


## ----graphDiversitySite----------------------------------
library(ggplot2)
ggplot(data = d.trees,
       mapping = aes(y = growth.rate,
                     x = diversity.site)) +
  geom_point()


## ----graphDiversitySiteWithRegrLine----------------------
ggplot(data = d.trees,
       mapping = aes(y = growth.rate,
                     x = diversity.site)) +
  geom_point() +
  geom_smooth(method = "lm")


## ----graphDiversitySiteWithSmoother, message=FALSE-------
ggplot(data = d.trees,
       mapping = aes(y = growth.rate,
                     x = diversity.site)) +
  geom_point() +
  geom_smooth()


## ----graphDensity, message=FALSE-------------------------
ggplot(data = d.trees,
       mapping = aes(y = growth.rate,
                     x = density.site)) +
  geom_point() +
  geom_smooth()


## ----graphAge, message=FALSE-----------------------------
ggplot(data = d.trees,
       mapping = aes(y = growth.rate,
                     x = age)) +
  geom_point() +
  geom_smooth()


## ----GraphSpecies----------------------------------------
ggplot(data = d.trees,
       mapping = aes(y = growth.rate,
                     x = species)) +
  geom_boxplot()


## ----graphDiversityGrouped, message=FALSE----------------
ggplot(data = d.trees,
       mapping = aes(y = growth.rate,
                     x = diversity.site,
                     colour = species)) + ## ! new argument
  geom_point() +
  geom_smooth()


## ----graphDiversityPanelling, message=FALSE--------------
ggplot(data = d.trees,
       mapping = aes(y = growth.rate,
                     x = diversity.site)) +
  geom_point() +
  geom_smooth(se = FALSE) +
  facet_wrap(. ~ species) ## ! new argument


## ----graphDiversityPanellingRegressionLines--------------
ggplot(data = d.trees,
       mapping = aes(y = growth.rate,
                     x = diversity.site)) +
  geom_point() +
  geom_smooth(method = "lm") +
  facet_wrap(. ~ species) 


## ----lm0-------------------------------------------------
lm.trees.0 <- lm(growth.rate ~ species + 
                   age + density.site + diversity.site +
                   species:age + 
                   species:density.site + 
                   species:diversity.site,
                 data = d.trees)


## ----drop1InteractionLm0---------------------------------
drop1(lm.trees.0, test = "F")


## ----fitLmTrees1-----------------------------------------
lm.trees.1 <- update(lm.trees.0, . ~ . - species:diversity.site)
drop1(lm.trees.1, test = "F")


## ----sessionINfo, size='footnotesize'--------------------
sessionInfo()

