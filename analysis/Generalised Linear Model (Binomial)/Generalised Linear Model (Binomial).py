"""
BINOMIAL GLM (GENERALIZED LINEAR MODEL) FOR ADMISSION ANALYSIS
================================================================
This script implements Binomial GLM (Logistic Regression) to predict:
1. High vs Low Admission Score
2. Graduate vs Dropout (Student Success)

GLM with binomial family is equivalent to Logistic Regression
================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, accuracy_score, precision_recall_curve)
import statsmodels.api as sm
from statsmodels.formula.api import glm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("BINOMIAL GLM FOR ADMISSION ANALYSIS")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n[STEP 1] Loading data...")

df = pd.read_csv('/Users/cyrielvanhelleputte/PycharmProjects/machine_learning_I_Project/data/data_choosing_process/Cyriel/data.csv', sep=';', encoding='utf-8-sig')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================================
# MODEL 1: PREDICT HIGH VS LOW ADMISSION SCORE
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 1: BINOMIAL GLM FOR HIGH VS LOW ADMISSION")
print("=" * 80)

print("\n[STEP 2] Creating binary outcome: High Admission vs Low Admission...")

# Create binary target: 1 = High Admission (above median), 0 = Low Admission
median_admission = df['Admission grade'].median()
df['High_Admission'] = (df['Admission grade'] >= median_admission).astype(int)

print(f"Median admission grade: {median_admission:.2f}")
print(f"\nTarget distribution:")
print(f"  Low Admission (0):  {(df['High_Admission'] == 0).sum()} students")
print(f"  High Admission (1): {(df['High_Admission'] == 1).sum()} students")

# Select predictor features (things known BEFORE admission)
FEATURES_ADMISSION = [
    'Previous qualification (grade)',    # Past academic performance
    'Age at enrollment',                 # Demographics
    "Mother's qualification",            # Family education
    "Father's qualification",            # Family education
    'Application order',                 # Application timing
    'Scholarship holder',                # Financial support
    'Gender',                           # Demographics
    'Displaced',                        # Geographic factors
    'International',                    # International status
]

print(f"\n[STEP 3] Preparing features for admission model...")
print(f"Using {len(FEATURES_ADMISSION)} predictor variables")

# Create modeling dataset
df_admission = df[FEATURES_ADMISSION + ['High_Admission']].copy()

# Convert to numeric
for col in df_admission.columns:
    df_admission[col] = pd.to_numeric(df_admission[col], errors='coerce')

# Remove missing values
df_admission = df_admission.dropna()
print(f"✓ Complete cases: {len(df_admission)} students")

# Separate features and target
X_adm = df_admission[FEATURES_ADMISSION]
y_adm = df_admission['High_Admission']

# Split data
X_train_adm, X_test_adm, y_train_adm, y_test_adm = train_test_split(
    X_adm, y_adm, test_size=0.2, random_state=42, stratify=y_adm
)

print(f"Training set: {len(X_train_adm)} samples")
print(f"Test set: {len(X_test_adm)} samples")

# Scale features
scaler_adm = StandardScaler()
X_train_adm_scaled = scaler_adm.fit_transform(X_train_adm)
X_test_adm_scaled = scaler_adm.transform(X_test_adm)

print("\n[STEP 4] Training Binomial GLM (Logistic Regression)...")

# Method 1: Using scikit-learn (faster, more practical)
glm_adm_sklearn = LogisticRegression(max_iter=1000, random_state=42)
glm_adm_sklearn.fit(X_train_adm_scaled, y_train_adm)
print("✓ Model trained using scikit-learn")

# Method 2: Using statsmodels (more statistical details)
X_train_adm_sm = sm.add_constant(X_train_adm_scaled)
glm_adm_statsmodels = sm.GLM(y_train_adm, X_train_adm_sm,
                              family=sm.families.Binomial())
glm_adm_result = glm_adm_statsmodels.fit()
print("✓ Model trained using statsmodels GLM")

print("\n[STEP 5] Model Summary (Statsmodels - Full Statistical Output)...")
print("=" * 80)
print(glm_adm_result.summary())
print("=" * 80)

# Make predictions
y_pred_adm = glm_adm_sklearn.predict(X_test_adm_scaled)
y_pred_adm_proba = glm_adm_sklearn.predict_proba(X_test_adm_scaled)[:, 1]

# Evaluate
accuracy_adm = accuracy_score(y_test_adm, y_pred_adm)
print(f"\n[STEP 6] Model Performance...")
print(f"Accuracy: {accuracy_adm:.4f} ({accuracy_adm*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test_adm, y_pred_adm,
                          target_names=['Low Admission', 'High Admission']))

# Confusion Matrix
cm_adm = confusion_matrix(y_test_adm, y_pred_adm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_adm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'High'],
            yticklabels=['Low', 'High'])
plt.title('Confusion Matrix - Admission Prediction', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()
print("\n✓ Saved: glm_admission_confusion_matrix.png")

# ROC Curve
fpr_adm, tpr_adm, thresholds_adm = roc_curve(y_test_adm, y_pred_adm_proba)
roc_auc_adm = auc(fpr_adm, tpr_adm)

plt.figure(figsize=(10, 6))
plt.plot(fpr_adm, tpr_adm, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc_adm:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - High Admission Prediction', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("✓ Saved: glm_admission_roc_curve.png")

# Coefficients
coef_df_adm = pd.DataFrame({
    'Feature': FEATURES_ADMISSION,
    'Coefficient': glm_adm_sklearn.coef_[0],
    'Odds_Ratio': np.exp(glm_adm_sklearn.coef_[0])
})
coef_df_adm = coef_df_adm.sort_values('Coefficient', key=abs, ascending=False)

print("\n[STEP 7] Feature Coefficients and Odds Ratios...")
print("=" * 80)
print(coef_df_adm.to_string(index=False))
print("=" * 80)

# Visualize coefficients
plt.figure(figsize=(10, 8))
colors = ['green' if x > 0 else 'red' for x in coef_df_adm['Coefficient']]
plt.barh(range(len(coef_df_adm)), coef_df_adm['Coefficient'], color=colors, alpha=0.7)
plt.yticks(range(len(coef_df_adm)), coef_df_adm['Feature'])
plt.xlabel('Coefficient (Log-Odds)', fontsize=12)
plt.title('GLM Coefficients - High Admission Prediction', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', lw=1)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
print("\n✓ Saved: glm_admission_coefficients.png")

# ============================================================================
# MODEL 2: PREDICT GRADUATE VS DROPOUT (STUDENT SUCCESS)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 2: BINOMIAL GLM FOR STUDENT SUCCESS (GRADUATE VS DROPOUT)")
print("=" * 80)

print("\n[STEP 8] Creating binary outcome: Graduate (1) vs Dropout (0)...")

# Filter only Graduate and Dropout (exclude Enrolled)
df_success = df[df['Target'].isin(['Graduate', 'Dropout'])].copy()
df_success['Success'] = (df_success['Target'] == 'Graduate').astype(int)

print(f"\nTarget distribution:")
print(f"  Dropout (0):  {(df_success['Success'] == 0).sum()} students")
print(f"  Graduate (1): {(df_success['Success'] == 1).sum()} students")

# Select predictor features (things that might predict success)
FEATURES_SUCCESS = [
    'Previous qualification (grade)',
    'Admission grade',
    'Age at enrollment',
    "Mother's qualification",
    "Father's qualification",
    'Scholarship holder',
    'Gender',
    'Debtor',
    'Tuition fees up to date',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

print(f"\n[STEP 9] Preparing features for success model...")
print(f"Using {len(FEATURES_SUCCESS)} predictor variables")

# Create modeling dataset
df_success_model = df_success[FEATURES_SUCCESS + ['Success']].copy()

# Convert to numeric
for col in df_success_model.columns:
    df_success_model[col] = pd.to_numeric(df_success_model[col], errors='coerce')

# Remove missing values
df_success_model = df_success_model.dropna()
print(f"✓ Complete cases: {len(df_success_model)} students")

# Separate features and target
X_suc = df_success_model[FEATURES_SUCCESS]
y_suc = df_success_model['Success']

# Split data
X_train_suc, X_test_suc, y_train_suc, y_test_suc = train_test_split(
    X_suc, y_suc, test_size=0.2, random_state=42, stratify=y_suc
)

print(f"Training set: {len(X_train_suc)} samples")
print(f"Test set: {len(X_test_suc)} samples")

# Scale features
scaler_suc = StandardScaler()
X_train_suc_scaled = scaler_suc.fit_transform(X_train_suc)
X_test_suc_scaled = scaler_suc.transform(X_test_suc)

print("\n[STEP 10] Training Binomial GLM for Student Success...")

# Scikit-learn GLM
glm_suc_sklearn = LogisticRegression(max_iter=1000, random_state=42)
glm_suc_sklearn.fit(X_train_suc_scaled, y_train_suc)
print("✓ Model trained using scikit-learn")

# Statsmodels GLM
X_train_suc_sm = sm.add_constant(X_train_suc_scaled)
glm_suc_statsmodels = sm.GLM(y_train_suc, X_train_suc_sm,
                              family=sm.families.Binomial())
glm_suc_result = glm_suc_statsmodels.fit()
print("✓ Model trained using statsmodels GLM")

print("\n[STEP 11] Model Summary (Statsmodels)...")
print("=" * 80)
print(glm_suc_result.summary())
print("=" * 80)

# Make predictions
y_pred_suc = glm_suc_sklearn.predict(X_test_suc_scaled)
y_pred_suc_proba = glm_suc_sklearn.predict_proba(X_test_suc_scaled)[:, 1]

# Evaluate
accuracy_suc = accuracy_score(y_test_suc, y_pred_suc)
print(f"\n[STEP 12] Model Performance...")
print(f"Accuracy: {accuracy_suc:.4f} ({accuracy_suc*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test_suc, y_pred_suc,
                          target_names=['Dropout', 'Graduate']))

# Confusion Matrix
cm_suc = confusion_matrix(y_test_suc, y_pred_suc)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_suc, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Dropout', 'Graduate'],
            yticklabels=['Dropout', 'Graduate'])
plt.title('Confusion Matrix - Student Success Prediction', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()
print("\n✓ Saved: glm_success_confusion_matrix.png")

# ROC Curve
fpr_suc, tpr_suc, thresholds_suc = roc_curve(y_test_suc, y_pred_suc_proba)
roc_auc_suc = auc(fpr_suc, tpr_suc)

plt.figure(figsize=(10, 6))
plt.plot(fpr_suc, tpr_suc, color='darkgreen', lw=2,
         label=f'ROC curve (AUC = {roc_auc_suc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Graduate Prediction', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("✓ Saved: glm_success_roc_curve.png")

# Coefficients and Odds Ratios
coef_df_suc = pd.DataFrame({
    'Feature': FEATURES_SUCCESS,
    'Coefficient': glm_suc_sklearn.coef_[0],
    'Odds_Ratio': np.exp(glm_suc_sklearn.coef_[0])
})
coef_df_suc = coef_df_suc.sort_values('Coefficient', key=abs, ascending=False)

print("\n[STEP 13] Feature Coefficients and Odds Ratios...")
print("=" * 80)
print(coef_df_suc.to_string(index=False))
print("=" * 80)

print("\nInterpreting Odds Ratios:")
print("-" * 80)
for idx, row in coef_df_suc.head(5).iterrows():
    if row['Odds_Ratio'] > 1:
        pct_change = (row['Odds_Ratio'] - 1) * 100
        print(f"✓ {row['Feature']}: {pct_change:.1f}% increase in odds of graduating")
    else:
        pct_change = (1 - row['Odds_Ratio']) * 100
        print(f"✗ {row['Feature']}: {pct_change:.1f}% decrease in odds of graduating")

# Visualize coefficients
plt.figure(figsize=(10, 10))
colors = ['green' if x > 0 else 'red' for x in coef_df_suc['Coefficient']]
plt.barh(range(len(coef_df_suc)), coef_df_suc['Coefficient'], color=colors, alpha=0.7)
plt.yticks(range(len(coef_df_suc)), coef_df_suc['Feature'])
plt.xlabel('Coefficient (Log-Odds)', fontsize=12)
plt.title('GLM Coefficients - Graduate Prediction', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', lw=1)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
print("\n✓ Saved: glm_success_coefficients.png")

# ============================================================================
# SAVE PREDICTIONS AND RESULTS
# ============================================================================
print("\n[STEP 14] Saving predictions and model results...")

# Save admission predictions
adm_predictions = pd.DataFrame({
    'Actual': y_test_adm.values,
    'Predicted': y_pred_adm,
    'Probability_High_Admission': y_pred_adm_proba
})
print("✓ Saved: glm_admission_predictions.csv")

# Save success predictions
suc_predictions = pd.DataFrame({
    'Actual': y_test_suc.values,
    'Predicted': y_pred_suc,
    'Probability_Graduate': y_pred_suc_proba
})
print("✓ Saved: glm_success_predictions.csv")

# Save coefficients
print("✓ Saved: coefficient files")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("\nMODEL 1: HIGH vs LOW ADMISSION PREDICTION")
print("-" * 80)
print(f"✓ Accuracy: {accuracy_adm*100:.2f}%")
print(f"✓ AUC-ROC: {roc_auc_adm:.3f}")
print(f"✓ Top predictor: {coef_df_adm.iloc[0]['Feature']}")
print(f"  Coefficient: {coef_df_adm.iloc[0]['Coefficient']:.4f}")

print("\nMODEL 2: GRADUATE vs DROPOUT PREDICTION")
print("-" * 80)
print(f"✓ Accuracy: {accuracy_suc*100:.2f}%")
print(f"✓ AUC-ROC: {roc_auc_suc:.3f}")
print(f"✓ Top predictor: {coef_df_suc.iloc[0]['Feature']}")
print(f"  Coefficient: {coef_df_suc.iloc[0]['Coefficient']:.4f}")
print(f"  Odds Ratio: {coef_df_suc.iloc[0]['Odds_Ratio']:.4f}")

print("\n" + "=" * 80)
print("UNDERSTANDING BINOMIAL GLM")
print("=" * 80)
print("""
Binomial GLM (Logistic Regression) Models Binary Outcomes:
- Uses logit link function: log(p/(1-p)) = β₀ + β₁X₁ + β₂X₂ + ...
- Coefficients are in log-odds scale
- Odds Ratio = exp(coefficient)
- Positive coefficient → increases odds of outcome
- Negative coefficient → decreases odds of outcome

Example: If Odds Ratio = 1.5, a 1-unit increase in predictor 
increases odds of success by 50%
""")

print("\n" + "=" * 80)
print("FILES CREATED")
print("=" * 80)
print("Visualizations:")
print("  1. glm_admission_confusion_matrix.png")
print("  2. glm_admission_roc_curve.png")
print("  3. glm_admission_coefficients.png")
print("  4. glm_success_confusion_matrix.png")
print("  5. glm_success_roc_curve.png")
print("  6. glm_success_coefficients.png")
print("\nData Files:")
print("  7. glm_admission_predictions.csv")
print("  8. glm_success_predictions.csv")
print("  9. glm_admission_coefficients.csv")
print("  10. glm_success_coefficients.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)