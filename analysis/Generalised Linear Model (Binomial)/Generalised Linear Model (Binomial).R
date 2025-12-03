################################################################################
# BINOMIAL GLM (GENERALIZED LINEAR MODEL) FOR ADMISSION ANALYSIS
################################################################################
# This script implements Binomial GLM (Logistic Regression) to predict:
# 1. High vs Low Admission Score
# 2. Graduate vs Dropout (Student Success)
#
# GLM with binomial family is equivalent to Logistic Regression
################################################################################

# Load required libraries
library(tidyverse)      # Data manipulation and visualization
library(caret)          # Machine learning tools
library(pROC)           # ROC curves
library(MASS)           # Statistical functions
library(gridExtra)      # Multiple plots
library(scales)         # Plot formatting

# Suppress warnings
options(warn = -1)

cat(paste(rep("=", 80), collapse = ""), "\n")
cat("BINOMIAL GLM FOR ADMISSION ANALYSIS\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

################################################################################
# STEP 1: LOAD AND PREPARE DATA
################################################################################
cat("[STEP 1] Loading data...\n")

# Load data
df <- read.csv2('/Users/cyrielvanhelleputte/PycharmProjects/machine_learning_I_Project/data/data_choosing_process/Cyriel/data.csv',
                encoding = 'UTF-8')

cat(sprintf("✓ Dataset loaded: %d rows, %d columns\n", nrow(df), ncol(df)))

################################################################################
# MODEL 1: PREDICT HIGH VS LOW ADMISSION SCORE
################################################################################
cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("MODEL 1: BINOMIAL GLM FOR HIGH VS LOW ADMISSION\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

cat("[STEP 2] Creating binary outcome: High Admission vs Low Admission...\n")

# Create binary target: 1 = High Admission (above median), 0 = Low Admission
median_admission <- median(df$Admission.grade, na.rm = TRUE)
df$High_Admission <- ifelse(df$Admission.grade >= median_admission, 1, 0)

cat(sprintf("Median admission grade: %.2f\n", median_admission))
cat("\nTarget distribution:\n")
cat(sprintf("  Low Admission (0):  %d students\n", sum(df$High_Admission == 0, na.rm = TRUE)))
cat(sprintf("  High Admission (1): %d students\n", sum(df$High_Admission == 1, na.rm = TRUE)))

# Select predictor features (things known BEFORE admission)
features_admission <- c(
  'Previous.qualification..grade.',
  'Age.at.enrollment',
  'Mother.s.qualification',
  'Father.s.qualification',
  'Application.order',
  'Scholarship.holder',
  'Gender',
  'Displaced',
  'International'
)

cat(sprintf("\n[STEP 3] Preparing features for admission model...\n"))
cat(sprintf("Using %d predictor variables\n", length(features_admission)))

# Create modeling dataset
df_admission <- df %>%
  select(all_of(c(features_admission, 'High_Admission'))) %>%
  mutate(across(everything(), as.numeric)) %>%
  na.omit()

cat(sprintf("✓ Complete cases: %d students\n", nrow(df_admission)))

# Split data (80-20 split)
set.seed(42)
train_index_adm <- createDataPartition(df_admission$High_Admission, p = 0.8, list = FALSE)
train_adm <- df_admission[train_index_adm, ]
test_adm <- df_admission[-train_index_adm, ]

cat(sprintf("Training set: %d samples\n", nrow(train_adm)))
cat(sprintf("Test set: %d samples\n", nrow(test_adm)))

# Standardize features
preproc_adm <- preProcess(train_adm[, features_admission], method = c("center", "scale"))
train_adm_scaled <- predict(preproc_adm, train_adm)
test_adm_scaled <- predict(preproc_adm, test_adm)

cat("\n[STEP 4] Training Binomial GLM (Logistic Regression)...\n")

# Train GLM with binomial family
glm_adm <- glm(High_Admission ~ .,
               data = train_adm_scaled,
               family = binomial(link = "logit"))

cat("✓ Model trained using GLM with binomial family\n")

cat("\n[STEP 5] Model Summary (Full Statistical Output)...\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
print(summary(glm_adm))
cat(paste(rep("=", 80), collapse = ""), "\n")

# Make predictions
pred_adm_prob <- predict(glm_adm, test_adm_scaled, type = "response")
pred_adm <- ifelse(pred_adm_prob > 0.5, 1, 0)

# Evaluate
accuracy_adm <- mean(pred_adm == test_adm_scaled$High_Admission)

cat(sprintf("\n[STEP 6] Model Performance...\n"))
cat(sprintf("Accuracy: %.4f (%.2f%%)\n", accuracy_adm, accuracy_adm * 100))

# Confusion Matrix
cm_adm <- confusionMatrix(factor(pred_adm, levels = c(0, 1)),
                          factor(test_adm_scaled$High_Admission, levels = c(0, 1)))
cat("\nClassification Report:\n")
print(cm_adm)

# Plot Confusion Matrix
cm_table_adm <- as.data.frame(table(Predicted = pred_adm, Actual = test_adm_scaled$High_Admission))
p1 <- ggplot(cm_table_adm, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 8, color = "white", fontface = "bold") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix - Admission Prediction",
       x = "Predicted", y = "Actual") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave("/mnt/user-data/outputs/glm_admission_confusion_matrix.png", p1, width = 8, height = 6, dpi = 300)
cat("\n✓ Saved: glm_admission_confusion_matrix.png\n")

# ROC Curve
roc_adm <- roc(test_adm_scaled$High_Admission, pred_adm_prob, quiet = TRUE)
auc_adm <- auc(roc_adm)

p2 <- ggroc(roc_adm, color = "darkorange", size = 1.5) +
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", color = "navy", size = 1) +
  labs(title = sprintf("ROC Curve - High Admission Prediction\nAUC = %.3f", auc_adm),
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5)) +
  coord_fixed()

ggsave("/mnt/user-data/outputs/glm_admission_roc_curve.png", p2, width = 10, height = 6, dpi = 300)
cat("✓ Saved: glm_admission_roc_curve.png\n")

# Coefficients and Odds Ratios
coef_df_adm <- data.frame(
  Feature = features_admission,
  Coefficient = coef(glm_adm)[-1],  # Exclude intercept
  Odds_Ratio = exp(coef(glm_adm)[-1])
) %>%
  arrange(desc(abs(Coefficient)))

cat("\n[STEP 7] Feature Coefficients and Odds Ratios...\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
print(coef_df_adm, row.names = FALSE)
cat(paste(rep("=", 80), collapse = ""), "\n")

# Visualize coefficients
p3 <- ggplot(coef_df_adm, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", aes(fill = Coefficient > 0), alpha = 0.7) +
  scale_fill_manual(values = c("red", "green"), guide = "none") +
  coord_flip() +
  labs(title = "GLM Coefficients - High Admission Prediction",
       x = "Feature", y = "Coefficient (Log-Odds)") +
  geom_hline(yintercept = 0, color = "black", size = 1) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave("/mnt/user-data/outputs/glm_admission_coefficients.png", p3, width = 10, height = 8, dpi = 300)
cat("\n✓ Saved: glm_admission_coefficients.png\n")

################################################################################
# MODEL 2: PREDICT GRADUATE VS DROPOUT (STUDENT SUCCESS)
################################################################################
cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("MODEL 2: BINOMIAL GLM FOR STUDENT SUCCESS (GRADUATE VS DROPOUT)\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

cat("[STEP 8] Creating binary outcome: Graduate (1) vs Dropout (0)...\n")

# Filter only Graduate and Dropout (exclude Enrolled)
df_success <- df %>%
  filter(Target %in% c('Graduate', 'Dropout')) %>%
  mutate(Success = ifelse(Target == 'Graduate', 1, 0))

cat("\nTarget distribution:\n")
cat(sprintf("  Dropout (0):  %d students\n", sum(df_success$Success == 0)))
cat(sprintf("  Graduate (1): %d students\n", sum(df_success$Success == 1)))

# Select predictor features
features_success <- c(
  'Previous.qualification..grade.',
  'Admission.grade',
  'Age.at.enrollment',
  'Mother.s.qualification',
  'Father.s.qualification',
  'Scholarship.holder',
  'Gender',
  'Debtor',
  'Tuition.fees.up.to.date',
  'Curricular.units.1st.sem..approved.',
  'Curricular.units.1st.sem..grade.',
  'Unemployment.rate',
  'Inflation.rate',
  'GDP'
)

cat(sprintf("\n[STEP 9] Preparing features for success model...\n"))
cat(sprintf("Using %d predictor variables\n", length(features_success)))

# Create modeling dataset
df_success_model <- df_success %>%
  select(all_of(c(features_success, 'Success'))) %>%
  mutate(across(everything(), as.numeric)) %>%
  na.omit()

cat(sprintf("✓ Complete cases: %d students\n", nrow(df_success_model)))

# Split data
set.seed(42)
train_index_suc <- createDataPartition(df_success_model$Success, p = 0.8, list = FALSE)
train_suc <- df_success_model[train_index_suc, ]
test_suc <- df_success_model[-train_index_suc, ]

cat(sprintf("Training set: %d samples\n", nrow(train_suc)))
cat(sprintf("Test set: %d samples\n", nrow(test_suc)))

# Standardize features
preproc_suc <- preProcess(train_suc[, features_success], method = c("center", "scale"))
train_suc_scaled <- predict(preproc_suc, train_suc)
test_suc_scaled <- predict(preproc_suc, test_suc)

cat("\n[STEP 10] Training Binomial GLM for Student Success...\n")

# Train GLM
glm_suc <- glm(Success ~ .,
               data = train_suc_scaled,
               family = binomial(link = "logit"))

cat("✓ Model trained using GLM with binomial family\n")

cat("\n[STEP 11] Model Summary...\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
print(summary(glm_suc))
cat(paste(rep("=", 80), collapse = ""), "\n")

# Make predictions
pred_suc_prob <- predict(glm_suc, test_suc_scaled, type = "response")
pred_suc <- ifelse(pred_suc_prob > 0.5, 1, 0)

# Evaluate
accuracy_suc <- mean(pred_suc == test_suc_scaled$Success)

cat(sprintf("\n[STEP 12] Model Performance...\n"))
cat(sprintf("Accuracy: %.4f (%.2f%%)\n", accuracy_suc, accuracy_suc * 100))

# Confusion Matrix
cm_suc <- confusionMatrix(factor(pred_suc, levels = c(0, 1)),
                          factor(test_suc_scaled$Success, levels = c(0, 1)))
cat("\nClassification Report:\n")
print(cm_suc)

# Plot Confusion Matrix
cm_table_suc <- as.data.frame(table(Predicted = pred_suc, Actual = test_suc_scaled$Success))
p4 <- ggplot(cm_table_suc, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 8, color = "white", fontface = "bold") +
  scale_fill_gradient(low = "lightgreen", high = "darkgreen") +
  labs(title = "Confusion Matrix - Student Success Prediction",
       x = "Predicted", y = "Actual") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave("/mnt/user-data/outputs/glm_success_confusion_matrix.png", p4, width = 8, height = 6, dpi = 300)
cat("\n✓ Saved: glm_success_confusion_matrix.png\n")

# ROC Curve
roc_suc <- roc(test_suc_scaled$Success, pred_suc_prob, quiet = TRUE)
auc_suc <- auc(roc_suc)

p5 <- ggroc(roc_suc, color = "darkgreen", size = 1.5) +
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", color = "navy", size = 1) +
  labs(title = sprintf("ROC Curve - Graduate Prediction\nAUC = %.3f", auc_suc),
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5)) +
  coord_fixed()

ggsave("/mnt/user-data/outputs/glm_success_roc_curve.png", p5, width = 10, height = 6, dpi = 300)
cat("✓ Saved: glm_success_roc_curve.png\n")

# Coefficients and Odds Ratios
coef_df_suc <- data.frame(
  Feature = features_success,
  Coefficient = coef(glm_suc)[-1],
  Odds_Ratio = exp(coef(glm_suc)[-1])
) %>%
  arrange(desc(abs(Coefficient)))

cat("\n[STEP 13] Feature Coefficients and Odds Ratios...\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
print(coef_df_suc, row.names = FALSE)
cat(paste(rep("=", 80), collapse = ""), "\n")

cat("\nInterpreting Odds Ratios:\n")
cat(paste(rep("-", 80), collapse = ""), "\n")
for (i in 1:min(5, nrow(coef_df_suc))) {
  row <- coef_df_suc[i, ]
  if (row$Odds_Ratio > 1) {
    pct_change <- (row$Odds_Ratio - 1) * 100
    cat(sprintf("✓ %s: %.1f%% increase in odds of graduating\n",
                row$Feature, pct_change))
  } else {
    pct_change <- (1 - row$Odds_Ratio) * 100
    cat(sprintf("✗ %s: %.1f%% decrease in odds of graduating\n",
                row$Feature, pct_change))
  }
}

# Visualize coefficients
p6 <- ggplot(coef_df_suc, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", aes(fill = Coefficient > 0), alpha = 0.7) +
  scale_fill_manual(values = c("red", "green"), guide = "none") +
  coord_flip() +
  labs(title = "GLM Coefficients - Graduate Prediction",
       x = "Feature", y = "Coefficient (Log-Odds)") +
  geom_hline(yintercept = 0, color = "black", size = 1) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5))

ggsave("/mnt/user-data/outputs/glm_success_coefficients.png", p6, width = 10, height = 10, dpi = 300)
cat("\n✓ Saved: glm_success_coefficients.png\n")

################################################################################
# SAVE PREDICTIONS AND RESULTS
################################################################################
cat("\n[STEP 14] Saving predictions and model results...\n")

# Save admission predictions
adm_predictions <- data.frame(
  Actual = test_adm_scaled$High_Admission,
  Predicted = pred_adm,
  Probability_High_Admission = pred_adm_prob
)
write.csv(adm_predictions, "/mnt/user-data/outputs/glm_admission_predictions.csv", row.names = FALSE)
cat("✓ Saved: glm_admission_predictions.csv\n")

# Save success predictions
suc_predictions <- data.frame(
  Actual = test_suc_scaled$Success,
  Predicted = pred_suc,
  Probability_Graduate = pred_suc_prob
)
write.csv(suc_predictions, "/mnt/user-data/outputs/glm_success_predictions.csv", row.names = FALSE)
cat("✓ Saved: glm_success_predictions.csv\n")

# Save coefficients
write.csv(coef_df_adm, "/mnt/user-data/outputs/glm_admission_coefficients.csv", row.names = FALSE)
write.csv(coef_df_suc, "/mnt/user-data/outputs/glm_success_coefficients.csv", row.names = FALSE)
cat("✓ Saved: coefficient files\n")

################################################################################
# FINAL SUMMARY
################################################################################
cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("FINAL SUMMARY\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

cat("MODEL 1: HIGH vs LOW ADMISSION PREDICTION\n")
cat(paste(rep("-", 80), collapse = ""), "\n")
cat(sprintf("✓ Accuracy: %.2f%%\n", accuracy_adm * 100))
cat(sprintf("✓ AUC-ROC: %.3f\n", auc_adm))
cat(sprintf("✓ Top predictor: %s\n", coef_df_adm$Feature[1]))
cat(sprintf("  Coefficient: %.4f\n", coef_df_adm$Coefficient[1]))

cat("\nMODEL 2: GRADUATE vs DROPOUT PREDICTION\n")
cat(paste(rep("-", 80), collapse = ""), "\n")
cat(sprintf("✓ Accuracy: %.2f%%\n", accuracy_suc * 100))
cat(sprintf("✓ AUC-ROC: %.3f\n", auc_suc))
cat(sprintf("✓ Top predictor: %s\n", coef_df_suc$Feature[1]))
cat(sprintf("  Coefficient: %.4f\n", coef_df_suc$Coefficient[1]))
cat(sprintf("  Odds Ratio: %.4f\n", coef_df_suc$Odds_Ratio[1]))

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("UNDERSTANDING BINOMIAL GLM\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("
Binomial GLM (Logistic Regression) Models Binary Outcomes:
- Uses logit link function: log(p/(1-p)) = β₀ + β₁X₁ + β₂X₂ + ...
- Coefficients are in log-odds scale
- Odds Ratio = exp(coefficient)
- Positive coefficient → increases odds of outcome
- Negative coefficient → decreases odds of outcome

Example: If Odds Ratio = 1.5, a 1-unit increase in predictor
increases odds of success by 50%
\n")

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("FILES CREATED\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("Visualizations:
  1. glm_admission_confusion_matrix.png
  2. glm_admission_roc_curve.png
  3. glm_admission_coefficients.png
  4. glm_success_confusion_matrix.png
  5. glm_success_roc_curve.png
  6. glm_success_coefficients.png

Data Files:
  7. glm_admission_predictions.csv
  8. glm_success_predictions.csv
  9. glm_admission_coefficients.csv
  10. glm_success_coefficients.csv
\n")

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("ANALYSIS COMPLETE!\n")
cat(paste(rep("=", 80), collapse = ""), "\n")