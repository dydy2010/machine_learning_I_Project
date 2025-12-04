################################################################################
# POISSON GLM (GENERALIZED LINEAR MODEL) FOR COUNT DATA
################################################################################
# This script implements Poisson GLM to predict count outcomes:
# 1. Number of courses approved in 1st semester
# 2. Number of courses approved in 2nd semester
#
# Poisson GLM is used when:
# - Outcome is a COUNT (0, 1, 2, 3, ...)
# - Data are non-negative integers
# - You want to model rates or frequencies
################################################################################

# Load required libraries
library(tidyverse)    # Data manipulation and visualization
library(caret)        # Machine learning workflows
library(MASS)         # For statistical models
library(broom)        # For tidy model outputs
library(gridExtra)    # For plot arrangements

# Set plot theme
theme_set(theme_minimal())

cat(rep("=", 80), "\n", sep = "")
cat("POISSON GLM FOR COUNT DATA PREDICTION\n")
cat(rep("=", 80), "\n", sep = "")

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
cat("\n[STEP 1] Loading data...\n")

df <- read.csv2('/Users/cyrielvanhelleputte/PycharmProjects/machine_learning_I_Project/data/data_choosing_process/Cyriel/data.csv',
                fileEncoding = 'UTF-8-BOM')
cat(sprintf("✓ Dataset loaded: %d rows, %d columns\n", nrow(df), ncol(df)))

cat("\n", rep("=", 80), "\n", sep = "")
cat("UNDERSTANDING COUNT DATA IN YOUR DATASET\n")
cat(rep("=", 80), "\n", sep = "")

# Identify count variables
count_vars <- c(
    'Curricular.units.1st.sem..credited.',
    'Curricular.units.1st.sem..enrolled.',
    'Curricular.units.1st.sem..evaluations.',
    'Curricular.units.1st.sem..approved.',
    'Curricular.units.1st.sem..without.evaluations.',
    'Curricular.units.2nd.sem..credited.',
    'Curricular.units.2nd.sem..enrolled.',
    'Curricular.units.2nd.sem..evaluations.',
    'Curricular.units.2nd.sem..approved.',
    'Curricular.units.2nd.sem..without.evaluations.'
)

cat("\nCount Variables (Perfect for Poisson GLM):\n")
for (var in count_vars[1:5]) {
    df[[var]] <- as.numeric(as.character(df[[var]]))
    cat(sprintf("\n%s:\n", var))
    cat(sprintf("  Range: %.0f to %.0f\n", min(df[[var]], na.rm = TRUE), max(df[[var]], na.rm = TRUE)))
    cat(sprintf("  Mean: %.2f\n", mean(df[[var]], na.rm = TRUE)))
    dist_vals <- head(sort(table(df[[var]]), decreasing = TRUE), 3)
    cat(sprintf("  Distribution: %s\n", paste(names(dist_vals), collapse = ", ")))
}

# ============================================================================
# MODEL 1: PREDICT NUMBER OF APPROVED COURSES (1ST SEMESTER)
# ============================================================================
cat("\n", rep("=", 80), "\n", sep = "")
cat("MODEL 1: POISSON GLM FOR 1ST SEMESTER APPROVED COURSES\n")
cat(rep("=", 80), "\n", sep = "")

cat("\n[STEP 2] Preparing data for Poisson GLM...\n")

# Target: Number of approved courses in 1st semester
TARGET_1 <- 'Curricular.units.1st.sem..approved.'

# Predictors: Things known BEFORE 1st semester
FEATURES_1 <- c(
    'Previous.qualification..grade.',
    'Admission.grade',
    'Age.at.enrollment',
    "Mother.s.qualification",
    "Father.s.qualification",
    'Scholarship.holder',
    'Gender',
    'Debtor',
    'Tuition.fees.up.to.date',
    'Curricular.units.1st.sem..enrolled.',
    'Unemployment.rate',
    'Inflation.rate',
    'GDP'
)

# Create modeling dataset
df_model1 <- df %>%
    select(all_of(c(FEATURES_1, TARGET_1))) %>%
    mutate(across(everything(), ~ as.numeric(as.character(.)))) %>%
    drop_na()

cat(sprintf("✓ Complete cases: %d students\n", nrow(df_model1)))

# Check target distribution
cat(sprintf("\nTarget Variable: %s\n", TARGET_1))
cat("Distribution of approved courses:\n")
print(head(table(df_model1[[TARGET_1]]), 10))
cat(sprintf("\nMean: %.2f\n", mean(df_model1[[TARGET_1]])))
cat(sprintf("Variance: %.2f\n", var(df_model1[[TARGET_1]])))
cat(sprintf("Mean ≈ Variance? %s\n",
    abs(mean(df_model1[[TARGET_1]]) - var(df_model1[[TARGET_1]])) < 1))

# Separate features and target
X1 <- df_model1 %>% select(all_of(FEATURES_1))
y1 <- df_model1[[TARGET_1]]

# Split data
set.seed(42)
trainIndex1 <- createDataPartition(y1, p = 0.8, list = FALSE)

X_train1 <- X1[trainIndex1, ]
X_test1 <- X1[-trainIndex1, ]
y_train1 <- y1[trainIndex1]
y_test1 <- y1[-trainIndex1]

cat(sprintf("\nTraining set: %d samples\n", length(y_train1)))
cat(sprintf("Test set: %d samples\n", length(y_test1)))

cat("\n[STEP 3] Training Poisson GLM Model...\n")

# Combine features and target for model training
train_data1 <- X_train1
train_data1[[TARGET_1]] <- y_train1

# Build formula
formula_1 <- as.formula(paste(TARGET_1, "~", paste(FEATURES_1, collapse = " + ")))

# Fit Poisson GLM
poisson_model1 <- glm(formula_1, data = train_data1, family = poisson(link = "log"))

cat("✓ Poisson GLM trained successfully!\n")

cat("\n[STEP 4] Model Summary (Full Statistical Output)...\n")
cat(rep("=", 80), "\n", sep = "")
print(summary(poisson_model1))
cat(rep("=", 80), "\n", sep = "")

# Make predictions
test_data1 <- X_test1
y_pred_train1 <- predict(poisson_model1, newdata = X_train1, type = "response")
y_pred_test1 <- predict(poisson_model1, newdata = X_test1, type = "response")

# Evaluate
cat("\n[STEP 5] Model Performance...\n")

# Training metrics
train_rmse1 <- sqrt(mean((y_train1 - y_pred_train1)^2))
train_mae1 <- mean(abs(y_train1 - y_pred_train1))

# Test metrics
test_rmse1 <- sqrt(mean((y_test1 - y_pred_test1)^2))
test_mae1 <- mean(abs(y_test1 - y_pred_test1))
test_r2_1 <- cor(y_test1, y_pred_test1)^2

cat("\nTRAINING SET:\n")
cat(sprintf("  RMSE: %.4f courses\n", train_rmse1))
cat(sprintf("  MAE:  %.4f courses\n", train_mae1))

cat("\nTEST SET:\n")
cat(sprintf("  RMSE: %.4f courses\n", test_rmse1))
cat(sprintf("  MAE:  %.4f courses\n", test_mae1))
cat(sprintf("  R²:   %.4f\n", test_r2_1))

# Coefficients and Rate Ratios
cat("\n[STEP 6] Interpreting Coefficients and Rate Ratios...\n")
cat(rep("=", 80), "\n", sep = "")

coef_summary1 <- summary(poisson_model1)$coefficients
coef_df1 <- data.frame(
    Feature = rownames(coef_summary1),
    Coefficient = coef_summary1[, "Estimate"],
    Rate_Ratio = exp(coef_summary1[, "Estimate"]),
    P_value = coef_summary1[, "Pr(>|z|)"]
)

# Sort by absolute coefficient
coef_df1$Abs_Coef <- abs(coef_df1$Coefficient)
coef_df1_sorted <- coef_df1 %>% arrange(desc(Abs_Coef))

cat("\nTop 10 Most Important Predictors:\n")
print(head(coef_df1_sorted[, c("Feature", "Coefficient", "Rate_Ratio", "P_value")], 11),
      row.names = FALSE)

cat("\n", rep("=", 80), "\n", sep = "")
cat("INTERPRETING RATE RATIOS:\n")
cat(rep("=", 80), "\n", sep = "")
cat("\nRate Ratio (RR) = exp(coefficient)\n")
cat("- RR > 1: Increases expected count\n")
cat("- RR < 1: Decreases expected count\n")
cat("- RR = 1: No effect\n\n")

for (i in 2:min(6, nrow(coef_df1_sorted))) {
    row <- coef_df1_sorted[i, ]
    if (row$Feature == "(Intercept)") next

    if (row$Rate_Ratio > 1) {
        pct_change <- (row$Rate_Ratio - 1) * 100
        direction <- "increases"
    } else {
        pct_change <- (1 - row$Rate_Ratio) * 100
        direction <- "decreases"
    }

    sig <- if (row$P_value < 0.001) "***" else if (row$P_value < 0.01) "**" else if (row$P_value < 0.05) "*" else ""
    cat(sprintf("%s: %s expected courses by %.1f%% %s\n", row$Feature, direction, pct_change, sig))
}

# Visualizations
cat("\n[STEP 7] Creating visualizations...\n")

# Actual vs Predicted
png("poisson_predictions_1st_sem.png", width = 1500, height = 600)
par(mfrow = c(1, 2))

# Training set
plot(y_train1, y_pred_train1,
     xlab = "Actual Approved Courses", ylab = "Predicted Approved Courses",
     main = sprintf("Training Set (Poisson GLM)\nRMSE = %.3f", train_rmse1),
     pch = 16, col = rgb(0, 0, 1, 0.5))
abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
legend("topleft", legend = "Perfect Prediction", col = "red", lty = 2, lwd = 2)
grid()

# Test set
plot(y_test1, y_pred_test1,
     xlab = "Actual Approved Courses", ylab = "Predicted Approved Courses",
     main = sprintf("Test Set (Poisson GLM)\nRMSE = %.3f", test_rmse1),
     pch = 16, col = rgb(0, 1, 0, 0.5))
abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
legend("topleft", legend = "Perfect Prediction", col = "red", lty = 2, lwd = 2)
grid()

dev.off()
cat("✓ Saved: poisson_predictions_1st_sem.png\n")

# Distribution comparison
png("poisson_distributions_1st_sem.png", width = 1500, height = 600)
par(mfrow = c(1, 2))

# Actual distribution
hist(y_test1, breaks = seq(0, max(y_test1) + 1, by = 1),
     xlab = "Number of Approved Courses", ylab = "Frequency",
     main = "Actual Distribution",
     col = rgb(0, 0, 1, 0.7), border = "black")
grid()

# Predicted distribution
hist(y_pred_test1, breaks = seq(0, max(y_test1) + 1, by = 1),
     xlab = "Number of Approved Courses", ylab = "Frequency",
     main = "Predicted Distribution",
     col = rgb(0, 1, 0, 0.7), border = "black")
grid()

dev.off()
cat("✓ Saved: poisson_distributions_1st_sem.png\n")

# Coefficient plot
png("poisson_coefficients_1st_sem.png", width = 1000, height = 1000)
par(mar = c(5, 12, 4, 2))

coef_plot_df <- coef_df1_sorted %>%
    filter(Feature != "(Intercept)") %>%
    head(13)

colors <- ifelse(coef_plot_df$Coefficient > 0, "green", "red")
barplot(coef_plot_df$Coefficient,
        names.arg = coef_plot_df$Feature,
        horiz = TRUE,
        las = 1,
        col = adjustcolor(colors, alpha.f = 0.7),
        xlab = "Coefficient (Log Rate)",
        main = "Poisson GLM Coefficients - 1st Semester Approved Courses")
abline(v = 0, col = "black", lwd = 1)
grid()

dev.off()
cat("✓ Saved: poisson_coefficients_1st_sem.png\n")

# ============================================================================
# MODEL 2: PREDICT NUMBER OF APPROVED COURSES (2ND SEMESTER)
# ============================================================================
cat("\n", rep("=", 80), "\n", sep = "")
cat("MODEL 2: POISSON GLM FOR 2ND SEMESTER APPROVED COURSES\n")
cat(rep("=", 80), "\n", sep = "")

cat("\n[STEP 8] Preparing data for 2nd semester model...\n")

TARGET_2 <- 'Curricular.units.2nd.sem..approved.'

# Predictors: Include 1st semester performance
FEATURES_2 <- c(
    'Previous.qualification..grade.',
    'Admission.grade',
    'Age.at.enrollment',
    'Scholarship.holder',
    'Debtor',
    'Tuition.fees.up.to.date',
    'Curricular.units.1st.sem..approved.',
    'Curricular.units.1st.sem..grade.',
    'Curricular.units.2nd.sem..enrolled.',
    'Unemployment.rate',
    'GDP'
)

# Create modeling dataset
df_model2 <- df %>%
    select(all_of(c(FEATURES_2, TARGET_2))) %>%
    mutate(across(everything(), ~ as.numeric(as.character(.)))) %>%
    drop_na()

cat(sprintf("✓ Complete cases: %d students\n", nrow(df_model2)))

# Separate features and target
X2 <- df_model2 %>% select(all_of(FEATURES_2))
y2 <- df_model2[[TARGET_2]]

# Split data
set.seed(42)
trainIndex2 <- createDataPartition(y2, p = 0.8, list = FALSE)

X_train2 <- X2[trainIndex2, ]
X_test2 <- X2[-trainIndex2, ]
y_train2 <- y2[trainIndex2]
y_test2 <- y2[-trainIndex2]

cat(sprintf("Training set: %d samples\n", length(y_train2)))
cat(sprintf("Test set: %d samples\n", length(y_test2)))

cat("\n[STEP 9] Training 2nd semester Poisson GLM...\n")

# Combine features and target
train_data2 <- X_train2
train_data2[[TARGET_2]] <- y_train2

# Build formula
formula_2 <- as.formula(paste(TARGET_2, "~", paste(FEATURES_2, collapse = " + ")))

# Fit Poisson GLM
poisson_model2 <- glm(formula_2, data = train_data2, family = poisson(link = "log"))

cat("✓ Model trained!\n")

cat("\n[STEP 10] Model Summary...\n")
cat(rep("=", 80), "\n", sep = "")
print(summary(poisson_model2))
cat(rep("=", 80), "\n", sep = "")

# Predictions and evaluation
y_pred_train2 <- predict(poisson_model2, newdata = X_train2, type = "response")
y_pred_test2 <- predict(poisson_model2, newdata = X_test2, type = "response")

test_rmse2 <- sqrt(mean((y_test2 - y_pred_test2)^2))
test_mae2 <- mean(abs(y_test2 - y_pred_test2))
test_r2_2 <- cor(y_test2, y_pred_test2)^2

cat("\n[STEP 11] Performance Metrics...\n")
cat(sprintf("Test RMSE: %.4f courses\n", test_rmse2))
cat(sprintf("Test MAE:  %.4f courses\n", test_mae2))
cat(sprintf("Test R²:   %.4f\n", test_r2_2))

# Coefficients
coef_summary2 <- summary(poisson_model2)$coefficients
coef_df2 <- data.frame(
    Feature = rownames(coef_summary2),
    Coefficient = coef_summary2[, "Estimate"],
    Rate_Ratio = exp(coef_summary2[, "Estimate"]),
    P_value = coef_summary2[, "Pr(>|z|)"]
)

coef_df2$Abs_Coef <- abs(coef_df2$Coefficient)
coef_df2_sorted <- coef_df2 %>% arrange(desc(Abs_Coef))

cat("\nTop Predictors for 2nd Semester:\n")
print(head(coef_df2_sorted[, c("Feature", "Coefficient", "Rate_Ratio", "P_value")], 11),
      row.names = FALSE)

# Visualizations for 2nd semester
png("poisson_predictions_2nd_sem.png", width = 1500, height = 600)
par(mfrow = c(1, 2))

# Predictions plot
plot(y_test2, y_pred_test2,
     xlab = "Actual Approved Courses (2nd Sem)", ylab = "Predicted Approved Courses",
     main = sprintf("2nd Semester - Poisson GLM\nRMSE = %.3f", test_rmse2),
     pch = 16, col = rgb(0.5, 0, 0.5, 0.5))
abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
legend("topleft", legend = "Perfect Prediction", col = "red", lty = 2, lwd = 2)
grid()

# Residuals
residuals2 <- y_test2 - y_pred_test2
plot(y_pred_test2, residuals2,
     xlab = "Predicted Approved Courses", ylab = "Residuals",
     main = "Residual Plot",
     pch = 16, col = rgb(0.5, 0, 0.5, 0.5))
abline(h = 0, col = "red", lwd = 2, lty = 2)
grid()

dev.off()
cat("\n✓ Saved: poisson_predictions_2nd_sem.png\n")

# ============================================================================
# SAVE RESULTS
# ============================================================================
cat("\n[STEP 12] Saving predictions and coefficients...\n")

# Save predictions
pred_df1 <- data.frame(
    Actual_1st_Sem = y_test1,
    Predicted_1st_Sem = y_pred_test1,
    Error = y_test1 - y_pred_test1
)
write.csv(pred_df1, "poisson_predictions_1st_sem.csv", row.names = FALSE)

pred_df2 <- data.frame(
    Actual_2nd_Sem = y_test2,
    Predicted_2nd_Sem = y_pred_test2,
    Error = y_test2 - y_pred_test2
)
write.csv(pred_df2, "poisson_predictions_2nd_sem.csv", row.names = FALSE)

# Save coefficients
write.csv(coef_df1_sorted, "poisson_coefficients_1st_sem.csv", row.names = FALSE)
write.csv(coef_df2_sorted, "poisson_coefficients_2nd_sem.csv", row.names = FALSE)

cat("✓ Saved all files\n")

# ============================================================================
# COMPARISON WITH LINEAR REGRESSION
# ============================================================================
cat("\n", rep("=", 80), "\n", sep = "")
cat("POISSON GLM vs LINEAR REGRESSION FOR COUNT DATA\n")
cat(rep("=", 80), "\n", sep = "")

# Train linear regression for comparison
lr_model <- lm(as.formula(paste(TARGET_1, "~", paste(FEATURES_1, collapse = " + "))),
               data = train_data1)
y_pred_lr <- predict(lr_model, newdata = X_test1)

lr_rmse <- sqrt(mean((y_test1 - y_pred_lr)^2))
lr_mae <- mean(abs(y_test1 - y_pred_lr))

cat("\nModel Comparison for 1st Semester Approved Courses:\n")
cat(sprintf("\n%-20s %-10s %-10s\n", "Model", "RMSE", "MAE"))
cat(rep("-", 40), "\n", sep = "")
cat(sprintf("%-20s %-10.4f %-10.4f\n", "Poisson GLM", test_rmse1, test_mae1))
cat(sprintf("%-20s %-10.4f %-10.4f\n", "Linear Regression", lr_rmse, lr_mae))

# Check for impossible predictions
n_negative <- sum(y_pred_lr < 0)
cat(sprintf("\nLinear Regression issues:\n"))
cat(sprintf("  Negative predictions: %d (impossible for counts!)\n", n_negative))
cat("\nPoisson GLM advantages:\n")
cat("  ✓ Predictions always non-negative\n")
cat("  ✓ Accounts for count distribution\n")
cat("  ✓ Better for rare events (low counts)\n")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
cat("\n", rep("=", 80), "\n", sep = "")
cat("FINAL SUMMARY\n")
cat(rep("=", 80), "\n", sep = "")

cat("\nMODEL 1: 1st Semester Approved Courses\n")
cat(rep("-", 80), "\n", sep = "")
cat(sprintf("✓ Test RMSE: %.4f courses\n", test_rmse1))
cat(sprintf("✓ Test MAE:  %.4f courses\n", test_mae1))
cat(sprintf("✓ Average prediction error: ±%.2f courses\n", test_mae1))
top_feat1 <- coef_df1_sorted[2, ]  # Skip intercept
cat(sprintf("✓ Top predictor: %s\n", top_feat1$Feature))
cat(sprintf("  Rate Ratio: %.4f\n", top_feat1$Rate_Ratio))

cat("\nMODEL 2: 2nd Semester Approved Courses\n")
cat(rep("-", 80), "\n", sep = "")
cat(sprintf("✓ Test RMSE: %.4f courses\n", test_rmse2))
cat(sprintf("✓ Test MAE:  %.4f courses\n", test_mae2))
cat(sprintf("✓ Average prediction error: ±%.2f courses\n", test_mae2))
top_feat2 <- coef_df2_sorted[2, ]
cat(sprintf("✓ Top predictor: %s\n", top_feat2$Feature))
cat(sprintf("  Rate Ratio: %.4f\n", top_feat2$Rate_Ratio))

cat("\n", rep("=", 80), "\n", sep = "")
cat("WHEN TO USE POISSON GLM\n")
cat(rep("=", 80), "\n", sep = "")
cat("
Use Poisson GLM when predicting:
✓ Counts (0, 1, 2, 3, ...)
✓ Number of events in fixed time/space
✓ Rare events
✓ Data where variance ≈ mean

Examples from your data:
- Number of courses approved
- Number of evaluations
- Number of enrolled courses
- Number of courses without evaluation

Advantages over Linear Regression:
✓ No negative predictions
✓ Models count distribution properly
✓ Interpretable rate ratios
✓ Accounts for discrete nature of counts
\n")

cat("\n", rep("=", 80), "\n", sep = "")
cat("FILES CREATED\n")
cat(rep("=", 80), "\n", sep = "")
cat("Visualizations:\n")
cat("  1. poisson_predictions_1st_sem.png\n")
cat("  2. poisson_distributions_1st_sem.png\n")
cat("  3. poisson_coefficients_1st_sem.png\n")
cat("  4. poisson_predictions_2nd_sem.png\n")
cat("\nData Files:\n")
cat("  5. poisson_predictions_1st_sem.csv\n")
cat("  6. poisson_predictions_2nd_sem.csv\n")
cat("  7. poisson_coefficients_1st_sem.csv\n")
cat("  8. poisson_coefficients_2nd_sem.csv\n")

cat("\n", rep("=", 80), "\n", sep = "")
cat("ANALYSIS COMPLETE!\n")
cat(rep("=", 80), "\n", sep = "")