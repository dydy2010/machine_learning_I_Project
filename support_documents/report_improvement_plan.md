# Report Improvement Plan (step-by-step)

1) **Add required Generative AI paragraph**
   - **File**: `final_report_ml1_group.rmd` (intro or a short subsection near the end).
   - **What**: Briefly state how generative AI was used (e.g., help with wording/code review), what was easy/hard, and cautions taken.
   - **Why**: Explicitly required by the project brief.

2) **Check page-length (<30 pages) and trim if needed**
   - **File**: `final_report_ml1_group.rmd` → knit → print preview of `final_report_ml1_group.html`.
   - **What**: Confirm page count; if close to 30, hide verbose outputs (e.g., long metric tables) or collapse chunks with `echo=FALSE`, `message=FALSE`, `warning=FALSE`.
   - **Why**: Requirement to stay under 30 pages.

3) **Polish tables for readability**
   - **File**: `final_report_ml1_group.rmd`.
   - **What**: Remove row-number columns (set `rownames(...) <- NULL`), align numeric columns right, and standardize captions (short, action-oriented). Fix duplicated labels in SVM metrics tables (currently shows Metric twice).
   - **Why**: Cleaner HTML output and easier grading.

4) **Consolidate repetitive outputs**
   - **File**: `final_report_ml1_group.rmd`.
   - **What**: Where multiple confusion matrices/metrics are shown, keep the key one(s) or summarize side-by-side with `kableExtra` to save space. Use `results='hide'` + short textual takeaway when output is long.
   - **Why**: Improves aesthetics and helps with the page-limit.

5) **Cross-method summary clarity**
   - **File**: `final_report_ml1_group.rmd`.
   - **What**: In the Model Comparison section, add a 2–3 bullet recap highlighting which method is preferred per outcome type and why (already partly present—tighten and bold key takeaways).
   - **Why**: Faster reviewer comprehension; aligns with “compare methods fairly” requirement.

6) **Check “amounts” transformations (if any)**
   - **File**: `final_report_ml1_group.rmd` data prep section.
   - **What**: Verify if any strictly positive “amount” variables are modeled with linear models; if yes, consider log-transforming and note it. If not applicable, add a one-line justification.
   - **Why**: Matches guidance on handling continuous amount variables.

7) **Submission checklist**
   - **File**: `README.md` (brief addendum).
   - **What**: Add a short checklist: data source, group members, models included, knit command, page-limit check, AI paragraph present.
   - **Why**: Satisfies “complementary info & project structure” guidance and eases submission.
