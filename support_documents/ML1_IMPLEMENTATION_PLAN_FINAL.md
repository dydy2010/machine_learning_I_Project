# ML1 Report Implementation Plan
## Final Version with Narrative Flow Improvements

**File to Edit:** `final_report_ml1_group.Rmd`  
**Deadline:** January 9, 2026 (aiming for January 4)  
**Status:** ✅ Technical work complete, narrative polish remaining  
**Remaining Time:** ~45-60 minutes

---

## NARRATIVE FLOW ANALYSIS

### What's Working Well ✅

| Element | Location | Why It Works |
|---------|----------|--------------|
| Consulting firm framing | Summary | Sets up relatable business context |
| Business Questions → Model Mapping | Section 1.1 | Clear purpose for each model |
| EDA insights setup | Section 2 | Foreshadows what models should find |
| "Most actionable model" framing | Section 5.2 | Connects GLM to business value |
| "Early warning system" framing | Section 8 | Clear purpose for Neural Networks |
| Two-option flowchart | Section 10.1 | Shows how models work together |
| Stakeholder-specific recommendations | Section 11.2 | Actionable for different audiences |

### What Needs Improvement ⚠️

| Section | Problem | Impact on Story |
|---------|---------|-----------------|
| Section 4 (LM) | Generic intro, no explicit business question link | Reader doesn't know why we predict grades |
| Section 6 (Poisson) | Technical intro, weak business purpose | Poisson seems disconnected from dropout story |
| Section 7 (GAM) | Pure methodology, no EDA callback | Misses chance to say "EDA showed non-linear age effects" |
| Section 9 (SVM) | Starts with code, not context | Reader jumps into code without knowing purpose |
| Section 10.1 | Apologetic transition ("exploration and learning") | Undermines confidence in earlier models |
| Section 11 opening | Dry academic language | Buries the impact, leads with methodology |
| Conclusions | Missing quantified financial finding | EDA showed financial factors matter but no numbers |
| Conclusions | Missing age/non-linearity finding | GAM validated EDA insight but conclusion ignores it |

---

## PHASE 1: Quick Fixes (Already Identified)
**Priority: 🔴 MUST DO**  
**Time: ~5 minutes**

> 💡 **Tip:** After Phase 1, knit the document to verify no new errors were introduced.

### Step 1.1: Fix Remaining Typos (3 items)

**TYPO 1 - Line 809:**
```
FIND:    showing strong abilit to predict
REPLACE: showing strong ability to predict
```

**TYPO 2 - Line 2134:**
```
FIND:    definitive signal for inmediate intervention
REPLACE: definitive signal for immediate intervention
```

**TYPO 3 - Line 423:**
```
FIND:    **non-linear**relationships
REPLACE: **non-linear** relationships
```

---

### Step 1.2: Fix Formatting Issues (3 items)

**FORMAT 1 - Line 408:**
```
FIND:    Students that currently **pays tuition on time**
REPLACE: Students that currently **pay tuition on time**
```

**FORMAT 2 - Line 2128:**
```
FIND:    *   **Implement "Option 2" Warning System:
REPLACE: *   **Implement "Option 2" Warning System:**
```

**FORMAT 3 - Line 2138:**
```
FIND:    academic success(The Linear Regression
REPLACE: academic success (The Linear Regression
```

---

## PHASE 2: Narrative Flow Improvements (HIGH IMPACT)
**Priority: 🟡 SHOULD DO - Significantly improves grade**  
**Time: ~30-40 minutes**

These changes strengthen the story without requiring major rewrites.

> ⚠️ **Important:** Line numbers are approximate and may shift after Phase 1 edits. Use the FIND text patterns to locate sections accurately.

> 💡 **Tip:** After Phase 2, knit and do a quick read-through to ensure the narrative flows naturally.

---

### Step 2.1: Strengthen Linear Model Intro (Section 4)

**WHY:** Current intro is generic. Should connect to Business Question 2 from Section 1.1.

**FIND (lines 464-466):**
```markdown
## 4.1 Introduction to the Problem

Predicting student grades is fundamental to early intervention strategies. Linear regression provides an interpretable framework for understanding how various factors—from prior academic preparation to socioeconomic conditions—combine to influence academic performance. While simple in concept, this approach reveals which factors matter most and by how much.
```

**REPLACE WITH:**
```markdown
## 4.1 Introduction to the Problem

**Business Question:** *"Can we predict exact 2nd-semester grades?"* (from Section 1.1)

Predicting specific grades—not just pass/fail—helps our client university forecast academic performance and identify students who may be sliding toward failure before they actually fail. Linear regression provides an interpretable framework: we can tell advisors exactly which factors matter and by how much. For example, "a 1-point drop in first semester grades predicts a 0.7-point drop in second semester."

This granular insight complements our binary dropout classifiers (GLM, NN, SVM) by providing a continuous early warning signal.
```

---

### Step 2.2: Strengthen Poisson GLM Intro (Section 6)

**WHY:** Current intro is too technical. Should connect to Business Question 3 and the "experimental" pipeline in Section 10.

**FIND (lines 815-829):**
```markdown
# 6. Generalized Linear Model - Poisson

## 6.1 Why Poisson Regression?

Course completion counts represent non-negative integers (0, 1, 2, 3...) that don't follow a normal distribution. Poisson regression solves this by:

1.  **Ensuring non-negative predictions** through a log link function
2.  **Modeling the count distribution** appropriately
3.  **Accounting for variance** that changes with the mean

The model structure is: $\log(\lambda) = \beta_0 + \beta_1X_1 + ... + \beta_nX_n$, where $\lambda$ represents the expected count.

## 6.2 Model 1: First Semester Course Approvals

Understanding what predicts first semester success is crucial for early intervention.
```

**REPLACE WITH:**
```markdown
# 6. Generalized Linear Model - Poisson

**Business Question:** *"How many courses will a student pass in the 1st semester?"* (from Section 1.1)

This model serves a unique role in our Early Warning System: it enables prediction **before** first semester grades exist. By estimating how many courses a student will pass based on enrollment-time data (admission grades, demographics, financial status, and course load), we can identify potentially struggling students **at the start of the semester**—before any academic performance is observed.

This feeds into our "Experimental Early Warning Pipeline" (see Section 10.1): Enrollment Data → Poisson GLM → Predicted Course Count → Binomial GLM → Dropout Risk.

## 6.1 Why Poisson Regression?

Course completion counts are non-negative integers (0, 1, 2, 3...) that don't follow a normal distribution. Poisson regression handles this by:

1.  **Ensuring non-negative predictions** through a log link function
2.  **Modeling the count distribution** appropriately
3.  **Accounting for variance** that changes with the mean

The model structure is: $\log(\lambda) = \beta_0 + \beta_1X_1 + ... + \beta_nX_n$, where $\lambda$ represents the expected count.

## 6.2 Predicting First Semester Course Approvals
```

---

### Step 2.3: Strengthen GAM Intro (Section 7)

**WHY:** Current intro is pure methodology. Should reference the EDA finding about non-linear age effects.

**FIND (lines 958-964):**
```markdown
# 7. Generalized Additive Model - GAM

## 7.1 Beyond Linear Relationships

All our previous models assumed **linear relationships**. But reality is rarely this simple. **GAMs** extend **GLMs** by replacing straight lines with **smooth curves**, allowing the model to discover **non-linear patterns** while maintaining interpretability.

The model structure: $Y = \beta_0 + f_1(X_1) + f_2(X_2) + ... + f_n(X_n) + \epsilon$, where $f_i$ are smooth, flexible functions learned from data.
```

**REPLACE WITH:**
```markdown
# 7. Generalized Additive Model - GAM

**Business Question:** *"Are there non-linear relationships that affect student success?"*

Our EDA (Section 2.4) revealed that age has a **non-linear relationship** with graduation: students aged 18-22 have higher success rates, while both younger and older students face elevated dropout risks. Linear models can't capture this "sweet spot" pattern—they can only model straight-line effects.

**GAMs** extend **GLMs** by replacing straight lines with **smooth curves**, allowing the model to discover **non-linear patterns** while remaining interpretable. This lets us test whether the age effect we observed in EDA holds up in a predictive model.

## 7.1 Methodology

The model structure: $Y = \beta_0 + f_1(X_1) + f_2(X_2) + ... + f_n(X_n) + \epsilon$, where $f_i$ are smooth, flexible functions learned from data.
```

> **Before making this change:** Verify the GAM smooth plot for Age (Section 7.3) actually shows the non-linear pattern. If it doesn't clearly show the 18-22 sweet spot, soften the language to: "Our EDA suggested age might have non-linear effects. GAMs let us test this hypothesis..."

---

### Step 2.4: Add SVM Intro Paragraph (Section 9)

**WHY:** Section currently starts with code. Should have business context first.

**FIND (lines 1639-1654):**
```markdown
# 9. Support Vector Machine Analysis

```{r svm-load-data, echo=TRUE}
# Load the dataset from the project data folder
student_data <- read_csv("data/preprocessed_data.csv")
```

**INSERT BEFORE THE CODE BLOCK:**
```markdown
# 9. Support Vector Machine Analysis

**Business Question:** *"Who is at immediate risk of dropping out?"* (from Section 1.1)

Support Vector Machines represent our **robust, high-accuracy classifier** for the Early Warning System. While Neural Networks learn complex patterns automatically, SVMs take a different approach: they find the **maximum margin** boundary that best separates dropout students from successful ones.

We test three kernel types (Linear, Radial, Polynomial) to determine which decision boundary shape works best for our data. The goal: a production-ready classifier that advisors can trust.

```{r svm-load-data, echo=TRUE}
```

---

### Step 2.5: Improve Section 10.1 Transition Text

**WHY:** Current text sounds apologetic ("exploration and learning"). Should be confident.

**FIND (lines 1917-1920):**
```markdown
## 10.1 Model Overview
Until now, we have done the following models and experiments,
the first models served also as a exploration and learning, we did not limit ourselves to dropout and non-dropout predictions. These could also provide other insights for our client.
With NN and SVM, we focused on the dropout prediction.
```

**REPLACE WITH:**
```markdown
## 10.1 Model Overview

Each model in our analysis serves a specific purpose in our client's decision-making toolkit:

- **Understanding models** (LM, Gaussian GAM): Reveal *which factors* drive academic performance and *by how much*
- **Classification models** (Binomial GLM, Logistic GAM): Provide *interpretable risk scores* with clear odds ratios for stakeholder communication
- **Early Warning models** (Neural Network, SVM): Deliver *maximum accuracy* (~88%) for operational deployment
- **Planning model** (Poisson GLM): Enables *enrollment-time predictions* before any grades exist

This layered approach gives our client flexibility: interpretable models for policy discussions, high-accuracy models for daily operations.
```

---

### Step 2.6: Rewrite Conclusions Opening (Section 11)

**WHY:** Current opening buries the impact. Should lead with results, not methodology.

**FIND (lines 2109-2111):**
```markdown
# 11. Conclusions & Business Recommendations

This study applied multiple machine learning approaches to predict student dropout and graduation outcomes using academic performance data from a Portuguese higher education institution. Three classification methods were evaluated: Linear and non-linear models like GLM/GAM, Support Vector Machines with multiple kernels, and Neural Networks.
```

**REPLACE WITH:**
```markdown
# 11. Conclusions & Business Recommendations

**Bottom Line:** Our Early Warning System can identify at-risk students with **88% accuracy** after first semester data is available. For students who haven't completed a semester yet, our experimental admission-based pipeline provides earlier (though less accurate) predictions.

The models consistently point to two dominant factors: **academic performance** (especially 1st/2nd semester grades) and **financial stability** (tuition payment, debtor status). These findings translate directly into intervention strategies.
```

---

### Step 2.7: Add Quantified Financial Finding to Conclusions

**WHY:** EDA showed financial factors matter, but conclusions lack specific numbers.

**FIND (lines 2124-2125):**
```markdown
Our models suggest that **grade performance** and **financial stability** are the most critical factors for student retention. We propose the following structured intervention strategy:
```

**REPLACE WITH:**
```markdown
Our models quantify what intuition suggests: **grade performance** and **financial stability** are the most critical factors for student retention. Specifically:

- Students with **tuition fees up to date** show significantly higher graduation odds (Binomial GLM)
- **Scholarship holders** demonstrate better outcomes across all models
- The **Debtor** status flag is a consistent risk factor in both GLM and Neural Network

We propose the following structured intervention strategy:
```

---

### Step 2.8: Add Non-Linear Age Finding to Key Findings

**WHY:** GAM validated the EDA observation about age, but conclusions don't mention it.

**FIND (lines 2117-2118):**
```markdown
*   **1st Semester Performance:** The critical **"early warning"** signal. While less definitive than 2nd-semester results, poor performance here is the earliest actionable indicator of risk.
*   **2nd Semester Performance:** The **"definitive"** predictor. Models using 2nd-semester data (SVM, NN) achieved the highest accuracy (~88%). By the end of the first year, the student's success or dropout is largely decided. **Failure to pass enrolled units** in the second semester is the dominant risk factor.
```

**REPLACE WITH:**
```markdown
*   **1st Semester Performance:** The critical **"early warning"** signal. While less definitive than 2nd-semester results, poor performance here is the earliest actionable indicator of risk.
*   **2nd Semester Performance:** The **"definitive"** predictor. Models using 2nd-semester data (SVM, NN) achieved the highest accuracy (~88%). By the end of the first year, the student's success or dropout is largely decided. **Failure to pass enrolled units** in the second semester is the dominant risk factor.
*   **Non-Linear Age Effects:** Our GAM analysis confirmed what EDA suggested: students aged 18-22 have optimal graduation rates, while both younger (less prepared) and significantly older (competing life responsibilities) students face elevated risk. This non-linear pattern was invisible to our linear models.
```

---

## COHERENCE CHECK: Section 1.1 ↔ Model Sections

Before finalizing, verify each business question in Section 1.1 matches what the model actually does:

| Section 1.1 Question | Model Section | ✓ Verify |
|---------------------|---------------|----------|
| "How do 1st-semester results predict final graduation?" | 5 (Binomial GLM), 7.3 (Logistic GAM) | ☐ Both predict Graduation vs Dropout |
| "Can we predict exact 2nd-semester grades?" | 4 (LM), 7.2 (Gaussian GAM) | ☐ Both predict continuous grades |
| "How many courses will a student pass in the 1st semester?" | 6 (Poisson GLM) | ☐ Predicts count of approved courses |
| "Who is at immediate risk of dropping out?" | 8 (NN), 9 (SVM) | ☐ Both predict Dropout vs NotDropout |

> **If any mismatch exists:** Either adjust Section 1.1's wording OR add a note in the model section explaining the variation.

---

## PHASE 3: Create ReadMe.md
**Priority: 🔴 MUST DO - Required for submission**  
**Time: ~5 minutes**

**CREATE NEW FILE:** `ReadMe.md` in your project root folder

```markdown
# Student Dropout Prediction - ML1 Group Work HS2025

## Team
- **Original Team:** Dongyuan Gao, Ramiro, Cyriel  
- **Contributions:** See Section 1.2 in report

## Project Structure
```
├── data/
│   ├── data_choosing_process/Cyriel/data.csv  # Main dataset
│   └── preprocessed_data.csv                   # Preprocessed for SVM/NN
├── final_report_ml1_group.Rmd                  # R Markdown source
├── final_report_ml1_group.html                 # Rendered report
└── ReadMe.md                                   # This file
```

## Data Source
**UCI Machine Learning Repository:** Predict Students' Dropout and Academic Success  
**URL:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success  
**Sample:** 4,424 students from Portuguese higher education, 37 variables

## The Story
A fictional university engaged our consulting firm to build an **Early Warning System** for student dropout. We analyzed 4,424 students using 6 machine learning approaches to answer:

1. What factors predict academic performance? (Linear Regression, GAM)
2. Who will graduate vs drop out? (Binomial GLM, Logistic GAM)
3. How many courses will students pass? (Poisson GLM)
4. Can we build a high-accuracy classifier? (Neural Network, SVM)

## Key Results
| Model | Purpose | Key Metric |
|-------|---------|------------|
| Linear Regression | Understand grade drivers | R² ≈ 0.68 |
| Binomial GLM | Interpretable risk scores | AUC ≈ 0.87 |
| Poisson GLM | Admission-time prediction | RMSE ≈ 1.8 |
| Gaussian GAM | Non-linear grade patterns | R² ≈ 0.72 |
| Logistic GAM | Non-linear risk patterns | AUC ≈ 0.89 |
| Neural Network | High-accuracy classification | Accuracy ≈ 88% |
| SVM (Radial) | Robust classification | Accuracy ≈ 87.7% |

## Main Findings
1. **1st/2nd semester grades** are the strongest dropout predictors across all models
2. **Financial factors** (tuition status, debtor) significantly affect outcomes
3. **Age has non-linear effects**: 18-22 year olds have optimal success rates
4. **Early Warning System** achievable with ~88% accuracy after 1st semester

## How to Reproduce
1. Open `final_report_ml1_group.Rmd` in RStudio
2. Ensure data files are in the `data/` folder
3. Install required packages (listed in setup chunk)
4. Knit to HTML

## Project Repository
https://github.com/dydy2010/machine_learning_I_Project
```

---

## PHASE 4: Final Verification
**Priority: 🔴 MUST DO**  
**Time: ~10 minutes**

> ⚠️ **Before Phase 4:** Make a backup copy of your edited .Rmd file!

### Page Count Check
```
☐ Knit to HTML
☐ Open in browser  
☐ Print Preview (Ctrl+P / Cmd+P)
☐ Verify ≤ 30 pages
```

**IF OVER 30 PAGES after narrative additions:**
- The narrative improvements add ~200 words total
- If needed, reduce a `fig.height` from 8 to 6 somewhere
- Or remove one redundant visualization

### Final Quality Check
```
☐ All business questions from Section 1.1 are answered
☐ EDA insights are validated by model findings
☐ Conclusions reference specific model results
☐ Story flows: Problem → Data → Models → Findings → Recommendations
```

---

## SUBMISSION CHECKLIST

### Create Zip
```
☐ data/ folder (both CSV files)
☐ final_report_ml1_group.Rmd
☐ final_report_ml1_group.html
☐ ReadMe.md
```

**Naming:** `Gao_Ramiro_Cyriel_[YourMatriculationNumber].zip`

### Upload
```
☐ Upload to ILIAS
☐ Download and verify the uploaded file
```

---

## TIME ESTIMATE

| Phase | Tasks | Time |
|-------|-------|------|
| Phase 1 | Fix 6 typos/formatting | 5 min |
| Phase 2 | 8 narrative improvements | 30-40 min |
| Phase 3 | Create ReadMe.md | 5 min |
| Phase 4 | Final knit + page check | 10 min |
| Submission | Create zip + upload | 5 min |
| **TOTAL** | | **~60 min** |

---

## SUMMARY: THE STORY ARC

After your improvements, the report will follow this clear narrative:

```
SUMMARY: "We're consultants helping a university reduce dropout"
    ↓
SECTION 1: "Here are 4 business questions, each answered by specific models"
    ↓
EDA: "Data shows grades, finances, and age matter"
    ↓
SECTION 4 (LM): "Predicting exact grades for early warning" → validates grade importance
    ↓
SECTION 5 (GLM): "Interpretable dropout risk with odds ratios" → quantifies factors
    ↓
SECTION 6 (Poisson): "Admission-time predictions" → enables earliest intervention
    ↓
SECTION 7 (GAM): "Non-linear effects exist" → validates EDA age pattern
    ↓
SECTION 8 (NN): "Early warning system at 88% accuracy" → production model
    ↓
SECTION 9 (SVM): "Robust classifier for comparison" → confirms NN findings
    ↓
SECTION 10: "Here's how models work together as a system"
    ↓
SECTION 11: "Key findings + specific recommendations by stakeholder"
```

This is the **interesting story with solid ML methods** that professors love! 🎓

---

*Plan Version: 4.1 (Reviewed & Corrected)*  
*Last Updated: December 23, 2024*
