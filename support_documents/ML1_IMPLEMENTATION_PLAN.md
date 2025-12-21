# ML1 Report Implementation Plan
## Step-by-Step Guide for Improvements

**File to Edit:** `final_report_ml1_group.Rmd`  
**Deadline:** January 9, 2026 (aiming for January 4)  
**Total Estimated Time:** 4-5 hours across 14 days

---

## PRE-WORK: Setup & Backup

### Step 0.1: Create Backup
```
☐ Copy final_report_ml1_group.Rmd → final_report_ml1_group_BACKUP.Rmd
☐ Copy final_report_ml1_group.html → final_report_ml1_group_BACKUP.html
```

### Step 0.2: Verify Current State
```
☐ Open RStudio
☐ Open final_report_ml1_group.Rmd
☐ Knit to HTML (verify it works before changes)
☐ Check current page count (Print Preview → count pages)
   Current pages: _____ (should be ~35)
```

### Step 0.3: Verify Data Files Exist
Your report uses TWO data files:
```
☐ data/data_choosing_process/Cyriel/data.csv (main data, line 190)
☐ data/preprocessed_data.csv (SVM section, line 1921)
```
Both must be included in final zip.

---

## PHASE 1: Critical Administrative Fixes
**Priority: 🔴 MUST DO - Prevents Penalties**  
**Time: ~15 minutes**

---

### Step 1.1: Verify Author List in YAML
**WHY:** Confirm all 3 team members are listed as authors (Requirement: Team size of 3).

**ACTION:**
Check the YAML header in `final_report_ml1_group.Rmd` (lines 1-14).
Ensure `author:` field lists: "Dongyuan Gao, Ramiro, Cyriel"

**VERIFICATION:**
```
☐ YAML Author field contains all 3 names
```

---

### Step 1.2: Add Team Contributions Table

**WHY:** Requirement says "each group member must take the lead on a model. This must be indicated in the pdf file"

**WHERE:** After Section 1.1 Motivation, before Section 1.2 Data Source

**FIND:**
```markdown
## 1.2 Data Source and Selection Process
```

**INSERT BEFORE IT:**
```markdown
## 1.2 Team Contributions

This project was completed collaboratively by three team members, with each member taking the lead on specific models:

| Team Member | Models Led | Other Contributions |
|-------------|------------|-------------------|
| **Cyriel** | Linear Regression, GLM (Binomial), GLM (Poisson), GAM | Exploratory Data Analysis |
| **Ramiro** | Support Vector Machines | Data Preprocessing |
| **Dongyuan Gao** | Neural Networks | Cross-validation, Report Integration & Editing |

```

**THEN:** Renumber the section:
- `## 1.2 Data Source` → `## 1.3 Data Source`
- `## 1.3 Feature Categories` → `## 1.4 Feature Categories`

**VERIFICATION:**
```
☐ Section 1.2 now says "Team Contributions"
☐ Table clearly shows who led which models
☐ Section 1.3 now says "Data Source and Selection Process"
☐ Knit and verify table renders correctly
```

---

## PHASE 2: Page Count Reduction
**Priority: 🔴 MUST DO - Hard Limit of 30 Pages**  
**Time: ~45 minutes**  
**Target: Cut ~5 pages**

---

### Step 2.1: Delete Binomial GLM Model 1 (High/Low Admission)

**WHY:** This model is disconnected from dropout storyline, saves ~1.5 pages

**FIND START:**
```markdown
## 5.2 Model 1: Predicting High vs Low Admission Scores
```

**FIND END:**
```markdown
## 5.3 Model 2: Predicting Graduation vs Dropout
```

**ACTION:** Delete everything FROM `## 5.2 Model 1:` UP TO (but not including) `## 5.3 Model 2:`

**THEN:** Renumber section:
- `## 5.3 Model 2: Predicting Graduation vs Dropout` → `## 5.2 Predicting Graduation vs Dropout`
- Also change `### 5.3 Binomial GLM:` → `### 5.2 Binomial GLM:`

**VERIFICATION:**
```
☐ Section 5.2 now starts with "Predicting Graduation vs Dropout"
☐ No references to "High vs Low Admission" remain in Section 5
☐ Knit and verify no errors (no missing variables)
```

---

### Step 2.2: Delete Poisson GLM Model 2 (2nd Semester)

**WHY:** One Poisson model is sufficient, saves ~0.5 pages

**FIND START:**
```markdown
## 6.3 Model 2: Second Semester Course Approvals
```

**FIND END:**
```markdown
## 6.4 Why Poisson Over Linear Regression?
```

**ACTION:** Delete everything FROM `## 6.3 Model 2:` UP TO (but not including) `## 6.4 Why Poisson`

**THEN:** Renumber:
- `## 6.4 Why Poisson` → `## 6.3 Why Poisson`

**VERIFICATION:**
```
☐ Section 6.2 still exists (1st Semester model)
☐ Section 6.3 is now "Why Poisson Over Linear Regression?"
☐ Knit and verify no errors
```

---

### Step 2.3: Delete GAM Poisson Model (Section 7.4)

**WHY:** Two GAM models sufficient to show non-linearity, saves ~1 page

**FIND START:**
```markdown
## 7.4 Model 3: Poisson GAM for Course Counts
```

**FIND END:**
```markdown
------------------------------------------------------------------------

# 8. Neural Network Analysis
```

**ACTION:** Delete everything FROM `## 7.4 Model 3:` UP TO (but not including) the divider line `------------------------------------------------------------------------` before Section 8

**VERIFICATION:**
```
☐ Section 7 now ends after 7.3 (Logistic GAM)
☐ Section 8 (Neural Network) still starts correctly
☐ Knit and verify no errors
```

---

### Step 2.4: Remove Extra SVM Decision Boundary Plots

**WHY:** One plot sufficient for illustration, saves ~1 page

**KEEP:** Linear Kernel plot (Section 9.5)

**FOR RADIAL KERNEL - FIND:**
```markdown
### Radial Kernel Results

```{r svm-plot-radial
```

**REPLACE THE PLOT CODE BLOCK with just a reference:**
```markdown
### Radial Kernel Results

*Decision boundary plot omitted for brevity; pattern similar to linear kernel.*

```{r svm-eval-radial
```

(Keep the confusion matrix and metrics code that follows)

---

**FOR POLYNOMIAL KERNEL - FIND the plot code after:**
```markdown
knitr::kable(metrics_poly, 
             caption = "Polynomial Kernel - Performance Metrics",
             align = 'c')

# svm-plot-poly
```

**DELETE** the entire `# svm-plot-poly` section (the plot code block including grid creation and plotting)

**VERIFICATION:**
```
☐ Linear kernel still has decision boundary plot
☐ Radial kernel has confusion matrix + metrics but no plot
☐ Polynomial kernel has confusion matrix + metrics but no plot
☐ Knit and verify
```

---

### Step 2.5: Update Model Overview Table (Section 10.1)

**WHY:** Table references deleted models

**FIND:**
```markdown
| **Binomial GLM 1** | High vs Low Admission Score | Previous Qualification, Age, Parents' Qualification | Accuracy ~ 70% |
```
**DELETE** this entire row.

**FIND:**
```markdown
| **Poisson GLM 2** | 2nd Sem Courses Approved (count) | 1st Sem Approved/Grade, Tuition Status | RMSE ~ 2.1 |
```
**DELETE** this entire row.

**FIND:**
```markdown
| **Poisson GAM** | 2nd Sem Courses Approved (count) | 1st Sem Approved (smooth), Age, GDP | R² ~ 0.65 |
```
**DELETE** this entire row.

**ALSO:** Rename remaining models for clarity:
- `| **Binomial GLM 2** |` → `| **Binomial GLM** |`
- `| **Poisson GLM 1** |` → `| **Poisson GLM** |`

**VERIFICATION:**
```
☐ Table now has 9 rows (was 12)
☐ No references to deleted models
☐ Knit and verify table displays correctly
```

---

### Step 2.6: Verify Page Count

```
☐ Knit to HTML
☐ Open in browser
☐ Print Preview (Ctrl+P / Cmd+P)
☐ Count pages: _____ (target: ≤30)
```

**IF STILL OVER 30 PAGES:**
- Option A: Reduce figure heights (change `fig.height=8` to `fig.height=6`)
- Option B: Remove one more visualization
- Option C: Trim verbose text in conclusions

---

## PHASE 3: Storyline & Content Improvements
**Priority: 🟡 SHOULD DO - Improves Grade**  
**Time: ~45 minutes**

---

### Step 3.1: Add Business Framing to Section 3.2

**WHY:** Creates coherent story connecting different model targets

**FIND:**
```markdown
## 3.2 Research Questions

Each model addresses specific questions:
```

**REPLACE ENTIRE SECTION 3.2 WITH:**
```markdown
## 3.2 Business Questions & Model Mapping

Our client university needs insights at **three decision points**:

**Decision Point 1: Understanding Performance Drivers**
*Question: "What factors influence student grades?"*

- **Linear Regression** → Quantifies how each factor affects 2nd semester grades
- **Gaussian GAM** → Reveals non-linear patterns (e.g., optimal age ranges)

**Decision Point 2: Early Dropout Risk Identification**
*Question: "Which students are at risk of dropping out?"*

- **Binomial GLM** → Provides interpretable odds ratios for stakeholder communication
- **Logistic GAM** → Captures non-linear risk patterns
- **Neural Network** → Early warning system achieving ~88% accuracy
- **SVM** → Robust classifier with ~87% accuracy

**Decision Point 3: Resource Planning**
*Question: "How many courses will students complete?"*

- **Poisson GLM** → Predicts course approval counts for capacity planning

This multi-model approach provides **actionable insights across the student lifecycle**.
```

**VERIFICATION:**
```
☐ Section 3.2 now titled "Business Questions & Model Mapping"
☐ Three decision points clearly explained
☐ Each model mapped to a business question
```

---

### Step 3.2: Fix Incomplete Sentences and Typos

**FIX 1 - FIND:**
```markdown
compare with 2nd best model ****
```
**REPLACE WITH:**
```markdown
compare with the second-best model to increase confidence
```

---

**FIX 2 - FIND:**
```markdown
For Uiversity Academic Advisors:
```
**REPLACE WITH:**
```markdown
For University Academic Advisors:
```

---

**FIX 3 - FIND:**
```markdown
Some system could be develped using standarized scale.
```
**REPLACE WITH:**
```markdown
Some system could be developed using a standardized scale.
```

---

### Step 3.3: Strengthen Linear Model Interpretation

**WHY:** Current interpretation says "strongest effect" but doesn't explain coefficient meaning

**FIND:**
```markdown
**Key Insights:** First semester grades have the strongest positive effect—students who perform well initially tend to continue succeeding.
```

**REPLACE WITH:**
```markdown
**Key Insights:** First semester grades have the strongest positive effect—students who perform well initially tend to continue succeeding. 

**Interpreting the Coefficients:** Since all features are standardized (mean=0, SD=1), each coefficient represents the expected change in 2nd semester grade for a 1 standard deviation increase in that predictor. For example, the coefficient of ~0.7 for 1st semester grade means students scoring 1 SD above average in the first semester can expect roughly 0.7 additional grade points in the second semester, holding other factors constant.
```

---

### Step 3.4: Add SVM Feature Selection Note

**WHY:** Addresses the weakness of using only 2 predictors

**FIND (after the marks dataframe creation):**
```markdown
# Remove rows with NA values if any exist
marks <- na.omit(marks)
```

**ADD AFTER:**
```markdown
```

**Note on Feature Selection:** For visualization clarity and to demonstrate SVM decision boundaries in 2D, we use only the top 2 predictors by F-statistic. A production system could include additional features for potentially higher accuracy, but the core insight—that academic performance metrics dominate—would remain unchanged.

---

## PHASE 4: Deliverables Preparation
**Priority: 🟡 REQUIRED for Submission**  
**Time: ~30 minutes**

---

### Step 4.1: Create ReadMe.md File

**CREATE NEW FILE:** `ReadMe.md` in your project root folder

```markdown
# Student Dropout Prediction - ML1 Group Work HS2025

## Team Contributions

| Member | Models Led | Contributions |
|--------|------------|---------------|
| **Cyriel**| Linear Regression, GLM (Binomial & Poisson), GAM | Exploratory Analysis |
| **Ramiro** | Support Vector Machines | Data Preprocessing |
| **Dongyuan Gao**  | Neural Networks | Cross-validation, <br>Report Integration & Editing |

## Project Structure
```
├── data/
│   ├── data_choosing_process/Cyriel/data.csv  # Main dataset
│   └── preprocessed_data.csv                   # Preprocessed for SVM
├── final_report_ml1_group.Rmd                  # R Markdown source
├── final_report_ml1_group.html                 # Rendered report
└── ReadMe.md                                   # This file
```

## Data Source
**UCI Machine Learning Repository:** Predict Students' Dropout and Academic Success  
**URL:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success  
**Institution:** Portuguese Higher Education  
**Sample:** 4,424 students, 37 variables

## Models Implemented
| Model | Target | Key Metric |
|-------|--------|------------|
| Linear Regression | 2nd Semester Grade | R² ≈ 0.68 |
| Binomial GLM | Graduation vs Dropout | AUC ≈ 0.87 |
| Poisson GLM | 1st Sem Courses Approved | RMSE ≈ 1.8 |
| Gaussian GAM | 2nd Semester Grade | R² ≈ 0.72 |
| Logistic GAM | Graduation vs Dropout | AUC ≈ 0.89 |
| Neural Network | Dropout vs Not Dropout | Accuracy ≈ 88% |
| SVM (3 kernels) | Graduation vs Dropout | Accuracy ≈ 87% |

## How to Reproduce
1. Open `final_report_ml1_group.Rmd` in RStudio  
2. Ensure data files are in the `data/` folder  
3. Install required packages (listed in the setup chunk)  
4. Knit to HTML

## Key Findings
1. First/second semester academic performance is the strongest dropout predictor
2. Financial factors (tuition status, debtor) significantly affect outcomes
3. NN and SVM achieve similar accuracy (~87-88%) for early warning systems
4. GAM reveals non-linear relationships that linear models miss
```

---

### Step 4.2: Final Knit & Page Verification

```
☐ Save all changes to Rmd
☐ Knit to HTML
☐ Verify no errors in console
☐ Open HTML in browser
☐ Print Preview → Verify ≤30 pages
☐ Check all plots render correctly
☐ Verify table of contents works
☐ Verify code folding works
```

---

### Step 4.3: Create Submission Zip

**NAMING FORMAT:** `Gao_Ramiro_Cyriel_[YourMatriculationNumber].zip`

**CONTENTS:**
```
☐ data/ folder (with both CSV files)
☐ final_report_ml1_group.Rmd
☐ final_report_ml1_group.html
☐ ReadMe.md
```

**VERIFICATION:**
```
☐ Zip file named correctly
☐ Unzip to new folder and verify all files present
☐ Open HTML from unzipped folder - verify it works
```

---

### Step 4.4: Upload to ILIAS

```
☐ Log in to ILIAS
☐ Navigate to ML1 course → Group Work submission folder
☐ Upload zip file
☐ Verify upload successful
☐ Download and verify the uploaded file
```

---

## FINAL CHECKLIST

### 🔴 Critical (Prevents Penalties)
```
☐ Team names confirmed in YAML
☐ Team Contributions table added (Section 1.2)
☐ Page count ≤ 30
☐ ReadMe.md included
☐ Zip naming correct
```

### 🟡 Important (Improves Grade)
```
☐ Binomial GLM Model 1 removed
☐ Poisson GLM Model 2 removed
☐ GAM Poisson removed
☐ Extra SVM plots removed
☐ Section 10.1 table updated
☐ Section 3.2 business framing added
☐ Typos fixed
☐ LM interpretation strengthened
```

### 🟢 Polish
```
☐ All sections knit without errors
☐ All visualizations render
☐ No broken cross-references
☐ Consistent formatting
```

---

## SCHEDULE

| Day | Date | Tasks | Est. Time | Checkpoint |
|-----|------|-------|-----------|------------|
| 1 | Dec 22 | Step 0 (backup), Steps 1.1-1.2 | 20 min | Knit works, labels visible |
| 2 | Dec 23 | Steps 2.1-2.2 (cut GLM models) | 25 min | Knit works, 2 sections gone |
| 3 | Dec 24 | Steps 2.3-2.4 (cut GAM, SVM plots) | 25 min | Knit works |
| 4 | Dec 25 | Step 2.5-2.6 (update table, check pages) | 20 min | ≤30 pages confirmed |
| 5 | Dec 26 | Steps 3.1-3.2 (storyline, typos) | 25 min | Section 3.2 rewritten |
| 6 | Dec 27 | Steps 3.3-3.4 (LM interp, SVM note) | 15 min | Content complete |
| 7 | Dec 28 | Step 4.1 (create ReadMe) | 15 min | ReadMe exists |
| 8 | Dec 29 | Steps 4.2 (final knit, full review) | 30 min | Everything works |
| 9-11 | Dec 30-Jan 1 | Buffer for issues | As needed | — |
| 12 | Jan 2 | Step 4.3 (create zip) | 15 min | Zip created |
| 13 | Jan 3 | Final review | 20 min | Ready to submit |
| 14 | Jan 4 | Step 4.4 (upload) | 10 min | ✅ SUBMITTED |

---

## TROUBLESHOOTING

### If knit fails after deletions:
1. Check for orphaned R objects (variables used but code deleted)
2. Look for unbalanced code chunks (``` missing)
3. Check cross-references to deleted sections

### If page count still > 30:
1. Reduce `fig.height` values globally
2. Add `results='hide'` to verbose output chunks
3. Use `echo=FALSE` on more code chunks

### If confused about what to delete:
- Always delete FROM section header TO next section header
- Never delete section dividers (---) unless sure
- Keep chunk labels unique after renumbering

---

*Plan Version: 2.0*  
*Last Updated: December 2024*
