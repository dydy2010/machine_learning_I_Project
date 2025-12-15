# Machine Learning I - Group Project
**Predicting Student Success: A Multi-Model Machine Learning Approach**  
*Early Warning System for Student Dropout Risk*

---

## Project Overview

A fictional university has engaged our consulting firm to diagnose why students drop out. We analyze **4,424 students** from a Portuguese higher education institution using **six machine learning techniques** to predict student outcomes and identify drivers of academic success.

**Business Goal:** Build an early warning system to identify at-risk students so the university can intervene early with targeted support.

---

## Team Members

| Member | Responsibilities |
|--------|------------------|
| **Cyriel** | Linear Regression, GLM (Binomial), GLM (Poisson), GAM |
| **Ramiro** | Support Vector Machines, Report Improvement |
| **Dongyuan**   | Neural Networks, Report Improvement|

---

## Quick Start

### 1. Install R Packages

Run once in R console if packages are not installed:

```r
# Core Data Manipulation & Visualization
install.packages("tidyverse")
install.packages("ggplot2")
install.packages("scales")
install.packages("gridExtra")
install.packages("corrplot")

# Machine Learning Utilities
install.packages("caret")
install.packages("broom")

# Generalized Linear Models & GAM
install.packages("mgcv")
install.packages("gratia")
install.packages("pROC")

# Neural Networks
install.packages("nnet")
install.packages("neuralnet")
install.packages("ROCR")
install.packages("NeuralNetTools")

# Support Vector Machines
install.packages("e1071")
install.packages("kernlab")
```

### 2. Knit the Report

```r
rmarkdown::render("final_report_ml1_group.rmd")
```

**Output:** `final_report_ml1_group.html` — HTML with interactive TOC and code folding

---

## Project Structure

```
machine_learning_I_Project/
├── data/
│   ├── data_choosing_process/       # Raw data exploration
│   └── preprocessed_data.csv        # Cleaned data for modeling
│
├── analysis/
│   ├── Neural Network/              # Neural network templates & materials
│   └── Support Vector Machines/     # SVM templates
│
├── final_report_ml1_group.rmd       # Main deliverable (knit this)
├── Evaluation_and_Hints.pdf         # Course requirements
└── README.md                        # This file
```

---

## Report Structure

| Section | Model | Purpose |
|---------|-------|---------|
| 1-3 | — | Introduction, Data Overview, Methodology |
| 4 | Linear Regression | Predict 2nd semester grades |
| 5 | GLM (Binomial) | Binary classification (Graduate vs Dropout) |
| 6 | GLM (Poisson) | Count of approved courses |
| 7 | GAM | Capture non-linear relationships |
| 8 | Neural Networks | Deep learning with nnet & neuralnet |
| 9 | SVM | Support Vector Machine classification |
| 10-12 | — | Model Comparison, Recommendations, Limitations |

---

## Data Source

| Attribute | Value |
|-----------|-------|
| **Dataset** | Predict Students' Dropout and Academic Success |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/dataset/697 |
| **Size** | 4,424 students × 36 features |
| **Target** | Dropout / Enrolled / Graduate |

---

## Key Findings

1. **1st semester performance** is the strongest predictor across all models
2. **Financial stability** (tuition fees up to date) significantly increases success odds
3. **Non-linear patterns** captured by GAMs outperform linear models by 2-5%
4. **Neural networks** achieve ~87% accuracy for dropout prediction
5. **Early intervention** after 1st semester can change student trajectories

---

## Technical Requirements

- **R version:** 4.0+
- **RStudio:** Recommended for knitting
- **Output format:** HTML (primary), PDF (alternative)

---

## References

Realinho, V., Machado, J., Baptista, L., & Martins, M.V. (2022). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

---

**Course:** Machine Learning I | **Institution:** HSLU
