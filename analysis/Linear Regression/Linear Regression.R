################################################################################
# LINEAR REGRESSION MODEL FOR PREDICTING STUDENT GRADES
################################################################################
# This script builds a linear regression model to predict student grades
# using various features from the student dataset.
################################################################################

library(tidyverse)
library(caret)
library(corrplot)
library(gridExtra)

# Set plot theme
theme_set(theme_minimal())

cat(rep("=", 80), "\n", sep = "")
cat("LINEAR REGRESSION MODEL FOR GRADE PREDICTION\n")
cat(rep("=", 80), "\n", sep = "")

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
cat("\n[STEP 1] Loading and exploring data...\n")

df <- read.csv2('/Users/cyrielvanhelleputte/PycharmProjects/machine_learning_I_Project/data/data_choosing_process/Cyriel/data.csv',
                fileEncoding = 'UTF-8-BOM')

cat(sprintf("✓ Dataset loaded: %d rows, %d columns\n", nrow(df), ncol(df)))

# Display basic info
cat("\nTarget variable options:\n")
cat("  - Curricular units 1st sem (grade)\n")
cat("  - Curricular units 2nd sem (grade)\n")
cat("  - Admission grade\n")

# ============================================================================
# STEP 2: PREPARE DATA FOR GRADE PREDICTION
# ============================================================================
cat("\n[STEP 2] Preparing data for grade prediction...\n")

TARGET <- 'Curricular.units.2nd.sem..grade.'

FEATURES <- c(
    'Previous.qualification..grade.',
    'Admission.grade',
    'Age.at.enrollment',
    "Mother.s.qualification",
    "Father.s.qualification",
    'Scholarship.holder',
    'Curricular.units.1st.sem..credited.',
    'Curricular.units.1st.sem..enrolled.',
    'Curricular.units.1st.sem..approved.',
    'Curricular.units.1st.sem..grade.',
    'Unemployment.rate',
    'Inflation.rate',
    'GDP'
)

df_model <- df %>%
    select(all_of(c(FEATURES, TARGET)))

df_model <- df_model %>%
    mutate(across(everything(), ~ as.numeric(as.character(.))))

cat(sprintf("Original dataset: %d rows\n", nrow(df_model)))

df_model <- df_model %>%
    drop_na()

cat(sprintf("After removing missing values: %d rows\n", nrow(df_model)))

df_model <- df_model %>%
    filter(get(TARGET) > 0)

cat(sprintf("After removing zero grades: %d rows\n", nrow(df_model)))

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ============================================================================
cat("\n[STEP 3] Exploratory Data Analysis...\n")

cat(sprintf("\nTarget Variable Statistics (%s):\n", TARGET))
print(summary(df_model[[TARGET]]))

cat(sprintf("\nFeature Correlations with %s:\n", TARGET))
correlations <- cor(df_model[FEATURES], df_model[[TARGET]])
correlations_sorted <- sort(correlations[,1], decreasing = TRUE)
print(correlations_sorted)

correlation_matrix <- cor(df_model[c(FEATURES, TARGET)])

png("correlation_heatmap.png", width = 1000, height = 800)
corrplot(correlation_matrix,
         method = "color",
         type = "upper",
         tl.col = "black",
         tl.srt = 45,
         tl.cex = 0.8,
         col = colorRampPalette(c("blue", "white", "red"))(200),
         addCoef.col = "black",
         number.cex = 0.5,
         title = "Feature Correlation Heatmap",
         mar = c(0, 0, 2, 0))
dev.off()
cat("\n✓ Saved: correlation_heatmap.png\n")

# ============================================================================
# STEP 4: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================
cat("\n[STEP 4] Splitting data into train and test sets...\n")

set.seed(42)

trainIndex <- createDataPartition(df_model[[TARGET]], p = 0.8, list = FALSE)

train_data <- df_model[trainIndex, ]
test_data <- df_model[-trainIndex, ]

X_train <- train_data %>% select(all_of(FEATURES))
y_train <- train_data[[TARGET]]

X_test <- test_data %>% select(all_of(FEATURES))
y_test <- test_data[[TARGET]]

cat(sprintf("Training set: %d samples\n", nrow(train_data)))
cat(sprintf("Testing set: %d samples\n", nrow(test_data)))

# ============================================================================
# STEP 5: FEATURE SCALING
# ============================================================================
cat("\n[STEP 5] Scaling features...\n")

preProc <- preProcess(X_train, method = c("center", "scale"))

X_train_scaled <- predict(preProc, X_train)
X_test_scaled <- predict(preProc, X_test)

cat("✓ Features scaled to have mean=0 and std=1\n")

# ============================================================================
# STEP 6: BUILD AND TRAIN LINEAR REGRESSION MODEL
# ============================================================================
cat("\n[STEP 6] Building and training Linear Regression model...\n")

train_data_scaled <- X_train_scaled
train_data_scaled[[TARGET]] <- y_train

formula_str <- paste(TARGET, "~", paste(FEATURES, collapse = " + "))
formula_obj <- as.formula(formula_str)

model <- lm(formula_obj, data = train_data_scaled)

cat("✓ Model trained successfully!\n")

cat("\nModel Equation:\n")
cat(sprintf("%s = %.4f\n", TARGET, coef(model)[1]))
for (i in 2:length(coef(model))) {
    coef_val <- coef(model)[i]
    sign <- ifelse(coef_val >= 0, "+", "")
    cat(sprintf("    %s %.4f × %s\n", sign, coef_val, names(coef(model))[i]))
}

# ============================================================================
# STEP 7: MAKE PREDICTIONS
# ============================================================================
cat("\n[STEP 7] Making predictions...\n")

y_train_pred <- predict(model, X_train_scaled)

test_data_scaled <- X_test_scaled
y_test_pred <- predict(model, test_data_scaled)

cat("✓ Predictions generated\n")

# ============================================================================
# STEP 8: EVALUATE MODEL PERFORMANCE
# ============================================================================
cat("\n[STEP 8] Evaluating model performance...\n")
cat("\n", rep("=", 60), "\n", sep = "")
cat("MODEL PERFORMANCE METRICS\n")
cat(rep("=", 60), "\n", sep = "")

train_r2 <- cor(y_train, y_train_pred)^2
train_rmse <- sqrt(mean((y_train - y_train_pred)^2))
train_mae <- mean(abs(y_train - y_train_pred))

cat("\nTRAINING SET:\n")
cat(sprintf("  R² Score:  %.4f\n", train_r2))
cat(sprintf("  RMSE:      %.4f\n", train_rmse))
cat(sprintf("  MAE:       %.4f\n", train_mae))

test_r2 <- cor(y_test, y_test_pred)^2
test_rmse <- sqrt(mean((y_test - y_test_pred)^2))
test_mae <- mean(abs(y_test - y_test_pred))

cat("\nTEST SET:\n")
cat(sprintf("  R² Score:  %.4f\n", test_r2))
cat(sprintf("  RMSE:      %.4f\n", test_rmse))
cat(sprintf("  MAE:       %.4f\n", test_mae))


cat("\n", rep("=", 60), "\n", sep = "")
cat("INTERPRETATION:\n")
cat(rep("=", 60), "\n", sep = "")
cat(sprintf("✓ R² Score of %.4f means the model explains %.2f%% of variance\n",
            test_r2, test_r2 * 100))
cat(sprintf("✓ On average, predictions are off by %.2f grade points (MAE)\n", test_mae))
cat(sprintf("✓ Root Mean Squared Error (RMSE) is %.2f grade points\n", test_rmse))

cat("\nPerforming 5-Fold Cross-Validation...\n")
train_control <- trainControl(method = "cv", number = 5)
cv_model <- train(formula_obj, data = train_data_scaled,
                  method = "lm", trControl = train_control)
cv_r2 <- cv_model$results$Rsquared
cat(sprintf("  Mean CV R²: %.4f\n", cv_r2))

# ============================================================================
# STEP 9: VISUALIZE RESULTS
# ============================================================================
cat("\n[STEP 9] Creating visualizations...\n")

png("predictions_scatter.png", width = 1500, height = 600)
par(mfrow = c(1, 2))

plot(y_train, y_train_pred,
     xlab = "Actual Grade", ylab = "Predicted Grade",
     main = sprintf("Training Set\nR² = %.4f", train_r2),
     pch = 16, col = rgb(0, 0, 1, 0.5))
abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
grid()
legend("topleft", legend = "Perfect Prediction", col = "red", lty = 2, lwd = 2)

plot(y_test, y_test_pred,
     xlab = "Actual Grade", ylab = "Predicted Grade",
     main = sprintf("Test Set\nR² = %.4f", test_r2),
     pch = 16, col = rgb(0, 1, 0, 0.5))
abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
grid()
legend("topleft", legend = "Perfect Prediction", col = "red", lty = 2, lwd = 2)

dev.off()
cat("✓ Saved: predictions_scatter.png\n")

residuals_train <- y_train - y_train_pred
residuals_test <- y_test - y_test_pred

png("residuals_analysis.png", width = 1500, height = 600)
par(mfrow = c(1, 2))

plot(y_test_pred, residuals_test,
     xlab = "Predicted Grade", ylab = "Residuals (Actual - Predicted)",
     main = "Residual Plot",
     pch = 16, col = rgb(0, 0, 1, 0.5))
abline(h = 0, col = "red", lwd = 2, lty = 2)
grid()

# Residual distribution
hist(residuals_test, breaks = 30,
     xlab = "Residuals", ylab = "Frequency",
     main = "Residual Distribution",
     col = rgb(0, 0, 1, 0.7), border = "black")
abline(v = 0, col = "red", lwd = 2, lty = 2)
grid()

dev.off()
cat("✓ Saved: residuals_analysis.png\n")

# Visualization 3: Feature Importance
feature_importance <- data.frame(
    Feature = names(coef(model))[-1],
    Coefficient = coef(model)[-1]
) %>%
    mutate(Abs_Coefficient = abs(Coefficient)) %>%
    arrange(desc(Abs_Coefficient))

png("feature_importance.png", width = 1000, height = 800)
par(mar = c(5, 12, 4, 2))
colors <- ifelse(feature_importance$Coefficient > 0, "green", "red")
barplot(feature_importance$Coefficient,
        names.arg = feature_importance$Feature,
        horiz = TRUE,
        las = 1,
        col = adjustcolor(colors, alpha.f = 0.7),
        xlab = "Coefficient Value",
        main = "Feature Importance (Coefficients)",
        xlim = c(min(feature_importance$Coefficient) * 1.2,
                 max(feature_importance$Coefficient) * 1.2))
abline(v = 0, col = "black", lwd = 1)
grid()
dev.off()
cat("✓ Saved: feature_importance.png\n")

# ============================================================================
# STEP 10: MAKE SAMPLE PREDICTIONS
# ============================================================================
cat("\n[STEP 10] Making sample predictions...\n")

# Show first 10 predictions
cat("\nSample Predictions (First 10 test cases):\n")
cat(rep("-", 70), "\n", sep = "")
cat(sprintf("%-15s %-18s %-10s\n", "Actual Grade", "Predicted Grade", "Error"))
cat(rep("-", 70), "\n", sep = "")

n_show <- min(10, length(y_test))
for (i in 1:n_show) {
    actual <- y_test[i]
    predicted <- y_test_pred[i]
    error <- actual - predicted
    cat(sprintf("%-15.2f %-18.2f %-10.2f\n", actual, predicted, error))
}

# ============================================================================
# STEP 11: SAVE MODEL AND PREDICTIONS
# ============================================================================
cat("\n[STEP 11] Saving results...\n")

# Save predictions to CSV
results_df <- data.frame(
    Actual_Grade = y_test,
    Predicted_Grade = y_test_pred,
    Error = y_test - y_test_pred,
    Absolute_Error = abs(y_test - y_test_pred)
)
write.csv(results_df, "grade_predictions.csv", row.names = FALSE)
cat("✓ Saved: grade_predictions.csv\n")

# Save model coefficients
model_info <- data.frame(
    Feature = c("Intercept", names(coef(model))[-1]),
    Coefficient = coef(model)
)
write.csv(model_info, "model_coefficients.csv", row.names = FALSE)
cat("✓ Saved: model_coefficients.csv\n")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
cat("\n", rep("=", 80), "\n", sep = "")
cat("FINAL SUMMARY\n")
cat(rep("=", 80), "\n", sep = "")
cat(sprintf("\n✓ Model Type: Linear Regression\n"))
cat(sprintf("✓ Target Variable: %s\n", TARGET))
cat(sprintf("✓ Number of Features: %d\n", length(FEATURES)))
cat(sprintf("✓ Training Samples: %d\n", nrow(train_data)))
cat(sprintf("✓ Test Samples: %d\n", nrow(test_data)))
cat(sprintf("\n✓ Test R² Score: %.4f\n", test_r2))
cat(sprintf("✓ Test RMSE: %.4f\n", test_rmse))
cat(sprintf("✓ Test MAE: %.4f\n", test_mae))

cat("\n", rep("=", 80), "\n", sep = "")
cat("Top 5 Most Important Features:\n")
cat(rep("=", 80), "\n", sep = "")
for (i in 1:min(5, nrow(feature_importance))) {
    direction <- ifelse(feature_importance$Coefficient[i] > 0, "positive", "negative")
    cat(sprintf("%d. %-45s (%s effect)\n", i, feature_importance$Feature[i], direction))
    cat(sprintf("   Coefficient: %.4f\n", feature_importance$Coefficient[i]))
}

cat("\n", rep("=", 80), "\n", sep = "")
cat("FILES CREATED:\n")
cat(rep("=", 80), "\n", sep = "")
cat("  1. correlation_heatmap.png - Feature correlations\n")
cat("  2. predictions_scatter.png - Actual vs Predicted grades\n")
cat("  3. residuals_analysis.png - Model error analysis\n")
cat("  4. feature_importance.png - Feature coefficients\n")
cat("  5. grade_predictions.csv - All predictions with errors\n")
cat("  6. model_coefficients.csv - Model parameters\n")

cat("\n", rep("=", 80), "\n", sep = "")
cat("ANALYSIS COMPLETE!\n")
cat(rep("=", 80), "\n", sep = "")