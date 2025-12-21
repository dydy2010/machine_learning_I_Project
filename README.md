# Machine Learning I - Group Project
**Predicting Student Success: A Multi-Model Machine Learning Approach**  
*Early Warning System for Student Dropout Risk*

---

## Project Overview

A fictional university has engaged our consulting firm to diagnose why students drop out so that they could take action. We analyze **4,424 students** from a Portuguese higher education institution student dataset, using **six machine learning techniques** to predict student dropout/graduate outcomes and identify drivers of academic success and failure.

**Business Goal:** Build an early warning method to identify at-risk students and provide actionable insights for intervention for our client.

---

## Team Members

| Member | Responsibilities |
|--------|------------------|
| **Cyriel**| Linear Regression, GLM (Binomial & Poisson), Exploratory Analysis|
| **Ramiro** | Support Vector Machines, Data Pre-processing |
| **Dongyuan Gao**  | Neural Networks, Report Integration & Editing, Cross Validation |

---

## Quick Start

### 1. Install R Packages before newly rendering the html

Run once in R console if packages are not installed:

```r
# Install all packages used in the report
install.packages(c(
  "tidyverse",    # data manipulation + plotting
  "ggplot2",      # plotting (explicit for clarity)
  "scales",       # scaling functions for plots
  "gridExtra",    # arrange multiple plots
  "corrplot",     # correlation plots
  "caret",        # ML utilities
  "broom",        # tidy model outputs
  "mgcv",         # GAM
  "gratia",       # GAM visualization
  "pROC",         # ROC curves
  "nnet",         # neural net (Lab 1 style)
  "neuralnet",    # neural net (Lab 2 style)
  "ROCR",         # ROC for neural nets
  "NeuralNetTools", # visualize neural networks
  "e1071",        # SVM implementation
  "kernlab"       # SVM kernels
))
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
├── final_report_ml1_group.rmd       # Main deliverable (knit this)
├── Evaluation_and_Hints.pdf         # Course requirements
└── README.md                        # This file
```

---

## Report Structure

| Section | Model | Purpose |
|---------|-------|---------|
| 1 | — | Introduction & Data Context |
| 2 | — | Exploratory Data Analysis |
| 3 | — | Modeling Approach & Strategy |
| 4 | Linear Regression | Predict 2nd semester grades |
| 5 | GLM (Binomial) | Binary classification (Graduate vs Dropout) |
| 6 | GLM (Poisson) | Count of approved courses |
| 7 | GAM | Capture non-linear relationships |
| 8 | Neural Networks | Dropout vs NotDropout early warning |
| 9 | SVM | Support Vector Machine classification |
| 10 | — | Model Comparison & Summary |
| 11 | — | Business Recommendations |
| 12 | — | Limitations & Future Work |
| 13 | — | Conclusions |

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
2. **Financial stability** (tuition fees up to date) significantly increases graduation rates
3. **Non-linear patterns** captured by GAMs outperform linear models by 2-5%
4. **Neural networks** score about 0.87 AUC when telling graduates apart from others (overall comparison)
5. **SVM** (linear kernel) achieves ~87% accuracy on Dropout vs Graduate classification (best kernel: radial at ~87.7%, slightly higher)
6. **Early intervention** after 1st semester, with targeted support, can significantly change student outcomes

---

## References

Realinho, V., Machado, J., Baptista, L., & Martins, M.V. (2022). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

## Generative AI Declaration & Guidelines

Generative AI was used as a supplementary tool on top of the assisting material provided in the Machine Learning1 lecture.

### Guidelines for Responsible and Effective Usage of Gen-AI
1. **Human-in-the-loop verification:** All AI-suggested content was treated as a draft. Text, code, or sources delivered by AI had to be critically reviewed and tested before inclusion.
2. **Emphasis on learning:** Gen-AI functioned as a learning companion rather than an automated code generator. After receiving suggestions, team members engaged in follow-up questions to grasp the underlying logic and method.
3. **Transparency:** The team openly acknowledged where and how AI was used in the workflow so the benefits and quality safeguards remained clear.

### Use Cases of AI Tools in the DBM Course Context
AI-based assistants were applied in the following specific cases:
1. **Brainstorming:** Elaborating initial ideas, structuring thoughts, and outlining coding approaches.
2. **Debugging & optimization:** Explaining error messages and helping improve self-developed scripts to raise efficiency.
3. **Proofreading:** Accelerating grammar and typo checks during documentation.
4. **Information acquisition:** Searching for methodological references or code documentation for R Markdown, or machine learning methods (always followed by human verification of credibility).

### Benefits and Challenges in Using Generative AI Tools
1. **Benefits:** Faster debugging (e.g., resolving R Markdown knitting errors or debugging issues), on-demand tutoring for advanced questions, and more time spent on analytical reasoning instead of routine fixes.
2. **Challenges:** The ease of getting answers can create a temptation to trust outputs blindly. The "human-in-the-loop" guideline ensured the team retained ownership, validating every AI-assisted contribution.

---

**Course:** Machine Learning I | **Institution:** HSLU
