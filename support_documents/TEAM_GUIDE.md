# ML1 Project - Team Collaboration Guide

## 📋 Team Structure & Responsibilities

### Person A (Completed - Needs R Conversion)
- ✅ Linear Regression
- ✅ GLM Binomial (Logistic Regression)
- ✅ GLM Poisson
- ✅ Generalized Additive Model (GAM)

**Status:** Analyses done in Python. **Action needed:** Convert to R code tomorrow.

### Person B
- 🔲 Neural Networks
- Package suggestions: `nnet`, `neuralnet`, or `keras`/`tensorflow`
- Focus on multi-layer architecture with dropout prediction

### Person C
- 🔲 Support Vector Machines (SVM)
- Package: `e1071` or `kernlab`
- Compare linear, polynomial, and RBF kernels

---

## 🎯 Storyline for High Grades

Our report tells a compelling story:

**Arc 1: The Problem** → Student dropout is costly for everyone  
**Arc 2: The Data** → Rich dataset with 4,424 students  
**Arc 3: The Investigation** → Six different ML lenses reveal different truths  
**Arc 4: The Discovery** → 1st semester + finances = key predictors  
**Arc 5: The Solution** → Actionable early warning system  

### What Makes This High-Quality:

1. ✅ **Uses ALL required methods** (LM, GLM, GAM, NN, SVM)
2. ✅ **Tells a business story** - not just technical metrics
3. ✅ **Brevity** - Structured to stay under 30 pages
4. ✅ **Interpretability** - Every model has business insights
5. ✅ **Model comparison** - Shows understanding of tradeoffs
6. ✅ **Real-world value** - Actionable recommendations
7. ✅ **Proper EDA** - Uses smoothers (LOESS), not regression lines
8. ✅ **Log-transforms** - If we had "amount" variables (we don't need here)

---

## 🔧 Workflow: How to Work Together

### Option 1: Separate RMD Files → Copy/Paste (Your Proposal) ✅ RECOMMENDED

**Pros:**
- ✅ Everyone works independently without merge conflicts
- ✅ Easy to develop and test each section
- ✅ Code_folding works naturally in HTML
- ✅ Final control over what goes in report

**Cons:**
- Manual copying at the end (but not a big deal)

**Process:**
1. Each person works in their own RMD file (current structure)
2. Test your code locally until it runs perfectly
3. When ready, copy YOUR section into `MASTER_REPORT.rmd`
4. Replace the placeholder comments (e.g., `>>> PERSON B: ...`)
5. One person compiles the final report

### Option 2: Quarto Book (You Asked About This)

**Pros:**
- Professional multi-page structure
- Automatic cross-references
- Each person has their own chapter file

**Cons:**
- ❌ Overkill for a 30-page report
- ❌ All team members need to learn Quarto syntax
- ❌ Requires coordination on `_quarto.yml` config
- ❌ **The project requirement is a SINGLE report file**, not a book

**Verdict:** **NOT recommended** for this project. Your copy/paste approach is better.

---

## 📄 Output Format: HTML vs PDF?

### HTML Output ✅ RECOMMENDED

**Advantages:**
- ✅ `code_folding: hide` → Clean report with "Show Code" buttons
- ✅ Interactive plots (if you use `plotly`)
- ✅ Floating table of contents (`toc_float: true`)
- ✅ Easy to check if under 30 pages (print preview)
- ✅ Modern, professional look
- ✅ No LaTeX errors to debug

**How to check page count:**
1. Open HTML in browser
2. File → Print Preview (or Cmd+P)
3. Count pages in preview
4. Submit HTML file as-is ✅ (requirements allow HTML!)

### PDF Output

**Advantages:**
- Universal format
- Guaranteed consistent rendering

**Disadvantages:**
- ❌ No code folding
- ❌ May require LaTeX troubleshooting
- ❌ Formatting issues with large tables/plots
- ❌ Less interactive

### My Recommendation: **HTML primary, PDF backup**

The YAML header in `MASTER_REPORT.rmd` has BOTH outputs configured:
```yaml
output:
  html_document:
    code_folding: hide
    toc_float: true
  pdf_document:
    toc: true
```

**Strategy:**
1. Develop everything for HTML (use `code_folding: hide`)
2. Before submission, knit to PDF too
3. If PDF looks good and stays under 30 pages → submit both
4. If PDF has issues → submit HTML only (✅ explicitly allowed!)

---

## 🎨 Code Visibility Strategy

The YAML setting `code_folding: hide` means:
- ✅ Code is hidden by default (clean report)
- ✅ Readers can click "Code" buttons to expand
- ✅ No need to manually hide chunks

**For specific chunks you NEVER want to show:**
```r
```{r chunk-name, include=FALSE}
# This chunk won't appear at all in output
```

**For chunks where you ONLY want output (no code button):**
```r
```{r chunk-name, echo=FALSE}
# This shows results but never shows code
```

**Default behavior with `code_folding: hide`:**
```r
```{r chunk-name}
# Code is hidden but expandable
```

---

## 📊 Data Loading - Important Note

All three team members should use:
```r
data <- read.csv("data/preprocessed_data.csv", stringsAsFactors = TRUE)
```

Use **relative paths** (not absolute) so code works on everyone's machine.

---

## ⏱️ Timeline Suggestion

### Day 1 (Tomorrow):
- **Person A:** Convert Python analyses to R in separate RMD files
- **Person B:** Start Neural Network analysis
- **Person C:** Start SVM analysis

### Day 2:
- **Person A:** Finish all four models, test in R Markdown
- **Person B:** Complete NN, evaluate performance
- **Person C:** Complete SVM with kernel comparison

### Day 3:
- **All:** Copy sections into `MASTER_REPORT.rmd`
- **One person:** Compile, check formatting, verify < 30 pages
- **All:** Review, polish writing, check interpretations

### Day 4:
- **All:** Final proofread
- **Submit:** HTML file (+ PDF if it works well)

---

## ✅ Pre-Submission Checklist

Before submitting, verify:

- [ ] All models implemented in R (not Python)
- [ ] Code runs from top to bottom without errors
- [ ] Under 30 pages (check HTML print preview)
- [ ] Target variable distribution shown
- [ ] EDA uses **smoothers** (LOESS), not regression lines
- [ ] Every model has:
  - [ ] Objective clearly stated
  - [ ] Performance metrics reported
  - [ ] Business interpretation provided
- [ ] Model comparison table complete
- [ ] Conclusions written (brief but insightful)
- [ ] ReadMe file created (data source, team members)
- [ ] .zip file named correctly (check ILIAS requirements)
- [ ] No package loading messages showing (code_folding helps)

---

## 🚀 Quick Start for Person B & C

### Person B (Neural Networks):

```r
# In your RMD file:
library(nnet)  # or library(keras)

nn_data <- read.csv("data/preprocessed_data.csv", stringsAsFactors = TRUE)

# Filter Graduate vs Dropout
nn_data <- nn_data %>%
  filter(Target %in% c("Graduate", "Dropout")) %>%
  mutate(Target_binary = ifelse(Target == "Graduate", 1, 0))

# Select features, normalize, train/test split
# Build model: nnet() or keras sequential model
# Train, evaluate, plot results
```

Key outputs needed:
- Accuracy, confusion matrix
- Training curves (loss/accuracy over epochs)
- Brief interpretation

### Person C (SVM):

```r
# In your RMD file:
library(e1071)

svm_data <- read.csv("data/preprocessed_data.csv", stringsAsFactors = TRUE)

# Filter Graduate vs Dropout
svm_data <- svm_data %>%
  filter(Target %in% c("Graduate", "Dropout"))

# Try different kernels:
svm_linear <- svm(Target ~ ., data = train, kernel = "linear")
svm_rbf <- svm(Target ~ ., data = train, kernel = "radial")
svm_poly <- svm(Target ~ ., data = train, kernel = "polynomial")

# Tune hyperparameters (cost, gamma)
# Compare performance
# Report best model
```

Key outputs needed:
- Kernel comparison table
- Accuracy of best model
- Brief interpretation

---

## 🎓 Grading Criteria (From Requirements)

To maximize your grade:

1. ✅ **Use all methods** - LM, GLM, GAM, NN, SVM (we do!)
2. ✅ **Be brief** - Under 30 pages, concise interpretations
3. ✅ **Interpret results** - Not just metrics, explain what they mean
4. ✅ **Write conclusions** - Brief paragraph at end
5. ✅ **Exploratory plots with smoothers** - Not regression lines!
6. ✅ **Creativity** - Our storyline (early warning system) is creative
7. ✅ **Engagement** - Non-Kaggle dataset shows effort (UCI is fine)
8. ✅ **Structure** - ReadMe, data source, context all included

---

## 📞 Communication Tips

- Each person should test their code in their own RMD file first
- Use consistent variable names (e.g., `data` for main dataset)
- Share your section early for feedback
- If stuck, share error messages with team
- One person should be "final compiler" to ensure consistency

---

## 🎯 Final Thoughts

**Your approach is solid:**
- ✅ Separate RMD files for development
- ✅ Copy/paste into master document
- ✅ HTML output with code_folding
- ✅ Clear team responsibilities

**Success factors:**
1. Person A converts to R tomorrow (priority!)
2. Everyone tests their code thoroughly
3. Integrate sections 2 days before deadline
4. Leave 1 day for polishing and proofreading

**You've got this! The master template is ready. Now each person fills in their section. Good luck! 🚀**
