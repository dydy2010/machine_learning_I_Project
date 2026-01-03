## Extending the Linear Model 6: Generalised Linear Models Lab
## Load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols, glm
from statsmodels.discrete.discrete_model import Poisson, Logit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial, Poisson as PoissonFamily#, QuasiPoisson
from statsmodels.gam.api import GLMGam, BSplines
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix
from scipy.special import expit 
from sklearn.datasets import load_iris
from pygam import LogisticGAM, s

## Set seed for reproducibility
np.random.seed(1)

## Simulate data for smokers
no_days = 14
v_non_smoker = np.concatenate([np.random.poisson(lam = 0, size = no_days - 1), [1]])
v_smoker1_2 = np.random.poisson(lam = 5, size = no_days)
v_smoker_box = np.random.poisson(lam = 20, size = no_days)

d_smokers = pd.DataFrame({
    'no_cigarettes': np.concatenate([v_non_smoker, v_smoker1_2, v_smoker_box]),
    'person': np.repeat(['non-smoker', 'moderate smoker', 'heavy smoker'], no_days)
})
d_smokers.head()

## Clean figure object
plt.clf()
## Plotting with seaborn
sns.boxplot(data = d_smokers, x = 'person', y = 'no_cigarettes')
plt.axhline(0, color = 'gray', linestyle = '--')
plt.title('Number of Cigarettes per Person')

## This line of code is necessary only if you run line-by-line
plt.show()

## Modelling count data with a linear model
lm_smokers = ols('no_cigarettes ~ person', data = d_smokers).fit()
print("Linear Model Coefficients:")
print(np.round(lm_smokers.params, 1))

## Simulate observations from the linear model
np.random.seed(3)
sim_data_smokers = lm_smokers.predict(d_smokers) + np.random.normal(0,
                                                                    lm_smokers.resid.std(),
                                                                    size = len(d_smokers))

sim_data_smokers.head()
sim_data_smokers.tail()

## Plot simulated data
plt.clf()
sns.boxplot(x = d_smokers['person'], y = sim_data_smokers)
plt.axhline(0, color = 'gray', linestyle = '--')
plt.title('Simulated Number of Cigarettes (Assuming Normality)')

plt.show()

## Poisson GLM
glm_smokers = smf.glm('no_cigarettes ~ person', 
                  data = d_smokers, 
                  family = PoissonFamily()).fit()
print("Poisson GLM Summary:")
print(glm_smokers.summary())

## Simulate Poisson data from GLM
np.random.seed(2)
sim_data_smokers_poisson = np.random.poisson(lam = glm_smokers.predict(d_smokers))

sim_data_smokers_poisson_df = pd.DataFrame({'simulated_y': sim_data_smokers_poisson})
sim_data_smokers_poisson_df.head()
sim_data_smokers_poisson_df.tail()

## Plot simulated Poisson data
plt.clf()
sns.boxplot(x = d_smokers['person'], y = sim_data_smokers_poisson_df['simulated_y'])
plt.axhline(0, color = 'gray', linestyle = '--')
plt.title('Simulated Number of Cigarettes (Assuming Poisson Distribution)')

plt.show()

## Reproduce bliss data set
bliss = pd.DataFrame({
    "dead": [2, 8, 15, 23, 27],
    "alive": [28, 22, 15, 7, 3],  
    "conc": [0, 1, 2, 3, 4]
})

bliss["total_insects"] = bliss["dead"] + bliss["alive"]
bliss["mortality_rate"] = bliss["dead"] / bliss["total_insects"]

print(bliss)

## Plot simulated Poisson data
plt.clf()
sns.scatterplot(x = bliss['conc'], y = bliss['mortality_rate'])

plt.show()

plt.clf()
sns.scatterplot(x = 'conc', y = 'mortality_rate', data = bliss)
sns.lmplot(x = 'conc', y = 'mortality_rate', data = bliss, ci = None)

plt.show()

plt.clf()
# Generate x values from -5 to 5
x = np.linspace(-5, 5, 300)

# Compute the inverse logit function (logistic sigmoid)
y = expit(x)

# Plot the curve
plt.plot(x, y, label = r'$\sigma(x) = \frac{1}{1 + e^{-x}}$', color = 'blue')

# Add horizontal reference lines at y = 0 and y = 1
plt.axhline(y = 0, color = 'gray', linestyle = '--')
plt.axhline(y = 1, color = 'gray', linestyle = '--')

# Labels and title
plt.xlabel("x")
plt.ylabel("ilogit(x)")
plt.title("Inverse Logit Function")
plt.legend()

# Show the plot
plt.show()

glm_insects = smf.glm("mortality_rate ~ conc", 
                data = bliss, 
                family = sm.families.Binomial()).fit()
print(glm_insects.summary())

plt.clf()
# Generate new data for prediction
new_data = pd.DataFrame({"conc": np.linspace(0, 5, 100)})
new_data["pred_insects"] = glm_insects.predict(new_data)

# Plot
plt.figure(figsize = (8, 6))
plt.grid()

# Plot predictions
sns.scatterplot(x = new_data["conc"], y = new_data["pred_insects"], 
               color = "blue", s = 50, label = "Predictions")


# Plot actural observations on top
sns.scatterplot(x = bliss["conc"], y = bliss["mortality_rate"],
              color = "red", s = 100, label = "Actual observations")


# Horizontal reference lines
plt.axhline(0, color = "gray", linestyle = "--")
plt.axhline(1, color = "gray", linestyle = "--")

# Labels and limits
plt.ylim(0, 1)
plt.xlabel("Concentration")
plt.ylabel("Mortality Rate")
plt.title("Logistic Regression Predictions")

plt.legend()
plt.show()

glm_insects_binary = smf.glm("survived_surgery ~ age + sex", 
                             data = someSurgeryData, 
                             family = sm.families.Binomial()).fit()
print(glm_insects_binary.summary())

## Load iris dataset
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns = iris.feature_names)
df_iris["Species"] = iris.target  # Convert species to numerical labels

## Selecting the predictors and response variable
X = df_iris[["sepal length (cm)", "petal width (cm)"]]  # Equivalent to Sepal.Length and Petal.Width
y = df_iris["Species"]  # Target variable (0, 1, 2 for species)

## Fit multinomial logistic regression
multinom_iris = LogisticRegression(solver = "lbfgs", max_iter = 500)
multinom_iris.fit(X, y)

## Display model coefficients
print("Coefficients:\n", multinom_iris.coef_)
print("Intercept:\n", multinom_iris.intercept_)

## Read space- or tab-delimited file correctly
d_stab = pd.read_table("../../Datasets/stability.dat", sep='\\s+', header = 0)

# Display structure equivalent to str(d.stab)
print(d_stab.info())
print(d_stab.head())

plt.clf()
sns.scatterplot(y = 'perform', x = 'stability', data = d_stab)

plt.show()

## Fit logistic regression (equivalent to method = "glm", family = "binomial" in ggplot)
glm_stab = smf.glm("perform ~ stability", data = d_stab, 
                   family = sm.families.Binomial()).fit()

print(glm_stab.summary())

# Generate predictions
stability_range = np.linspace(d_stab["stability"].min(), d_stab["stability"].max(), 100)
pred_data = pd.DataFrame({"stability": stability_range})
pred_data["perform_pred"] = glm_stab.predict(pred_data)

# Plot
plt.figure(figsize = (8, 6))

# Scatter plot (equivalent to geom_point())
sns.scatterplot(data = d_stab, x = "stability", y = "perform",
                color = "blue", label = "Observed")

# Logistic regression curve (equivalent to geom_smooth(method = "glm", family = "binomial"))
plt.plot(pred_data["stability"], pred_data["perform_pred"], 
        color = "red", label = "Logistic Fit")

# Labels and title
plt.xlabel("Stability")
plt.ylabel("Perform")
plt.title("Logistic Regression Fit")

plt.legend()

plt.show()

fitted_stab = glm_stab.fittedvalues
##
binary_predictions = (fitted_stab >= 0.5).astype(int)
print(binary_predictions)
##
d_obs_fit_stab = pd.DataFrame({'obs': d_stab['perform'], 'fit': binary_predictions})
print(d_obs_fit_stab.head())
##
## Frequency count of a single categorical column
print(d_obs_fit_stab["obs"].value_counts())

contingency_table = pd.crosstab(d_obs_fit_stab["obs"], d_obs_fit_stab["fit"])

## Display the contingency table
print(contingency_table)

prop_table = contingency_table.div(contingency_table.sum().sum())
print(round(prop_table, 2))

## Load datasets (adjust the path if necessary)
esdcomp = pd.read_csv("../../Datasets/esdcomp.csv")

## Fit Poisson GLM
glm_complaints = smf.glm("complaints ~ visits + residency + gender + revenue + hours",
                      data = esdcomp, 
                      family = sm.families.Poisson()).fit()
print(glm_complaints.summary())

## Extract coefficient for sexM
coef_genderM = glm_complaints.params["gender[T.M]"]
print("Coefficient for gender M:", coef_genderM)

## Exponentiated coefficient
exp_coef_genderM = np.exp(coef_genderM).round(2)
print("Exponentiated coefficient for sexM:", exp_coef_genderM)

## First doctor data
first_doctor = esdcomp[0:1]
print("First doctor:", first_doctor)

## Fitted value for first doctor
fitted_first_doctor = glm_complaints.fittedvalues[0:1]
print("Fitted complaints for first doctor:", fitted_first_doctor)

## Compute predicted complaints if the first doctor was male
first_doctor_as_male = first_doctor
first_doctor_as_male.iloc[0, 3] = "M"
print(first_doctor_as_male)
pred_first_doc_male = glm_complaints.predict(first_doctor_as_male)
print("Predicted complaints if first doctor was male:", pred_first_doc_male)

## Manually computing expected complaints for first doctor as male
manual_pred = fitted_first_doctor * np.exp(coef_genderM)
print("Manually computed complaints for first doctor as male:", manual_pred)

## Exponentiate coefficient for visits
exp_coef_visits = np.exp(glm_complaints.params["visits"])
print("Exponentiated coefficient for visits:", exp_coef_visits.round(5))

## Range of visits
visits_range = (esdcomp["visits"].min(), esdcomp["visits"].max())
print("Range of visits:", visits_range)

## Increase number of visits by 50
coef_visits_50_add = glm_complaints.params["visits"] * 50
print(coef_visits_50_add.round(5))

## Exponentiate it
exp_coef_visits_50_add = np.exp(coef_visits_50_add)
print("Exponentiated coefficient for visits * 50:", exp_coef_visits_50_add.round(5))

plt.clf()
# Generate new data for prediction
new_data = pd.DataFrame({"conc": np.linspace(0, 5, 100)})
new_data["pred_insects"] = glm_insects.predict(new_data)

# Plot
plt.figure(figsize = (8, 6))
plt.grid()

# Plot predictions
sns.scatterplot(x = new_data["conc"], y = new_data["pred_insects"], 
               color = "blue", s = 50, label = "Predictions")


# Plot actural observations on top
sns.scatterplot(x = bliss["conc"], y = bliss["mortality_rate"],
              color = "red", s = 100, label = "Actual observations")


# Horizontal reference lines
plt.axhline(0, color = "gray", linestyle = "--")
plt.axhline(1, color = "gray", linestyle = "--")

# Labels and limits
plt.ylim(0, 1)
plt.xlabel("Concentration")
plt.ylabel("Mortality Rate")
plt.title("Logistic Regression Predictions")

plt.legend()

plt.show()

print(glm_insects.params)
##
print(np.exp(glm_insects.params['conc']).round(5))

plt.vlines(x = 2, ymin = 0, ymax = 0.5, color = 'red', linewidth = 0.8)
plt.hlines(y = 0.5, xmin = 0, xmax = 2, color = 'red', linewidth = 0.8)
##
plt.vlines(x = 3, ymin = 0, ymax = 0.77, color = 'red', linewidth = 0.8)
plt.hlines(y = 0.77, xmin = 0, xmax = 3, color = 'red', linewidth = 0.8)

plt.show()

bins = [-np.inf, 0, 2, 4]
labels = ["no insecticide", "low conc", "high conc"]

# Apply binning (equivalent to cut() in R)
bliss["conc_asFactor"] = pd.cut(bliss["conc"], bins=bins, labels=labels)

# Display factor levels (unique categories)
print(bliss["conc_asFactor"].cat.categories)

pd.crosstab(bliss["conc"], bliss["conc_asFactor"])

glm_insects_fac = smf.glm("mortality_rate ~ conc_asFactor", 
                data = bliss, 
                family = sm.families.Binomial()).fit()
print(glm_insects_fac.summary())

## smf.glm doesn't directly support quasi poisson models
