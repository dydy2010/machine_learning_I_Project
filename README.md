# Machine Learning I - Group Project (needs to be rewritten and edited)
**HSLU ML1 Group Project - Student Dropout Prediction**

## Project Overview
We analyze student dropout and success patterns using multiple machine learning techniques. Our objective is to predict student outcomes (Dropout, Enrolled, Graduate) and identify the drivers behind academic success so the university client can intervene early.

## Team Members
- **Dongyuan Gao:** Linear Regression, GLM (Binomial), GLM (Poisson), GAM
- **Ramiro:** Neural Networks
- **Cyriel:** Support Vector Machines

**Team Composition:** 3 members (project requirement satisfied)

## Data Source
**Dataset:** Predict Students' Dropout and Academic Success  
**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success  
**Institution:** Portuguese higher education institution  
**Size:** 4,424 students with 36 features

### Data Context
The dataset includes:
- **Demographics:** Age, gender, nationality, marital status
- **Socioeconomic factors:** Parents' education/occupation, scholarship status
- **Academic background:** Previous qualifications, admission grades
- **Performance metrics:** Semester grades, credits, evaluations
- **Financial status:** Tuition fees, debtor status
- **Target variable:** Dropout, Enrolled, or Graduate

## Project Structure

```
machine_learning_I_Project/
│
├── data/
│   ├── input/                        # Raw data
│   ├── preprocessed_data.csv         # Cleaned data ready for modeling
│   └── preprocessing                 # Preprocessing script
│
├── analysis/
│   ├── Linear Regression/            # Dongyuan Gao
│   ├── Generalised Linear Model (Binomial)/
│   ├── Generalised Linear Model (Poisson)/
│   ├── Generalised Additive Model/
│   ├── Neural Network/               # Ramiro
│   └── Support Vector Machines/      # Cyriel
│
├── support_documents/
│   ├── MASTER_REPORT.rmd             # Comprehensive storyline & guide
│   ├── TEAM_GUIDE.md                 # Collaboration guidelines
│   └── QUICK_START.md                # Onboarding summary
│
├── final_report_ml1_group.rmd        # Consolidated deliverable
├── requirements.txt                  # R packages needed
├── Evaluation_and_Hints.pdf          # Project requirements
└── README.md                         # This file
```

## Research Objectives
1. Predict student dropout risk with high accuracy
2. Identify key factors influencing academic success
3. Compare multiple ML techniques (interpretability vs. accuracy trade-offs)
4. Provide actionable recommendations for the client institution

## Methods Used
1. **Linear Regression** - Predict continuous grade values
2. **GLM (Binomial)** - Binary classification with odds ratios
3. **GLM (Poisson)** - Predict count of failed courses
4. **Generalized Additive Model (GAM)** - Capture non-linear relationships
5. **Neural Networks** - Deep learning for complex patterns
6. **Support Vector Machines** - Robust classification with kernel tricks

## Technical Stack
- **Language:** R 4.x
- **Key Packages:**
  - `tidyverse` - Data manipulation and visualization
  - `ggplot2` - Advanced plotting
  - `e1071` / `kernlab` - SVM implementation
  - `mgcv` - GAM modeling
  - `nnet` / `neuralnet` - Neural networks
  - `caret` - ML utilities and evaluation

## Key Findings
- **1st semester performance** is the strongest predictor of graduation
- **Financial stability** (tuition fees up to date) dramatically increases success odds
- **Non-linear patterns** exist that simple models miss
- **Early intervention** (after 1st semester) can change outcomes

## How to Run
1. **Install R** (version 4.0+)
2. **Install required packages:**
   ```r
   install.packages(c("tidyverse", "ggplot2", "e1071", "caret", 
                      "kernlab", "mgcv", "nnet"))
   ```
3. **Open RStudio**
4. **Knit the report:**
   ```r
   rmarkdown::render("MASTER_REPORT.rmd")
   ```
5. **Output:** HTML file (or PDF) with complete analysis

## Output Format
- **Primary:** HTML with interactive table of contents and code folding
- **Alternative:** PDF (LaTeX-rendered)
- **Length:** Under 30 pages (as required)

## Academic Context
- **Course:** Machine Learning I
- **Institution:** HSLU (Lucerne University of Applied Sciences and Arts)
- **Semester:** [Add your semester]
- **Submission Date:** [Add your date]

## Deliverables
1. Master RMD stored in `support_documents/`
2. Knitted HTML/PDF report produced from `final_report_ml1_group.rmd`
3. README (this file)
4. Data preprocessing script and processed dataset
5. All analysis R Markdown files per model

## Contact
For questions about this project, contact:
- Person A: [email]
- Person B: [email]
- Person C: [email]

## References
- Realinho, V., Machado, J., Baptista, L., & Martins, M.V. (2022). Predict Students' Dropout and Academic Success. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

---

**Last Updated:** `r Sys.Date()`
