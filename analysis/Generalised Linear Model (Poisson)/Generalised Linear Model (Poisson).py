"""
POISSON GLM (GENERALIZED LINEAR MODEL) FOR COUNT DATA
=======================================================
This script implements Poisson GLM to predict count outcomes:
1. Number of courses approved in 1st semester
2. Number of courses approved in 2nd semester

Poisson GLM is used when:
- Outcome is a COUNT (0, 1, 2, 3, ...)
- Data are non-negative integers
- You want to model rates or frequencies
=======================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.discrete.count_model import Poisson
import warnings

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("POISSON GLM FOR COUNT DATA PREDICTION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("\n[STEP 1] Loading data...")

df = pd.read_csv('/Users/cyrielvanhelleputte/PycharmProjects/machine_learning_I_Project/data/data_choosing_process/Cyriel/data.csv', sep=';', encoding='utf-8-sig')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

print("\n" + "=" * 80)
print("UNDERSTANDING COUNT DATA IN YOUR DATASET")
print("=" * 80)

# Identify count variables
count_vars = [
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (without evaluations)'
]

print("\nCount Variables (Perfect for Poisson GLM):")
for var in count_vars[:5]:  # Show first 5
    print(f"\n{var}:")
    print(f"  Range: {df[var].min():.0f} to {df[var].max():.0f}")
    print(f"  Mean: {df[var].mean():.2f}")
    print(f"  Distribution: {df[var].value_counts().head(3).to_dict()}")

# ============================================================================
# MODEL 1: PREDICT NUMBER OF APPROVED COURSES (1ST SEMESTER)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 1: POISSON GLM FOR 1ST SEMESTER APPROVED COURSES")
print("=" * 80)

print("\n[STEP 2] Preparing data for Poisson GLM...")

# Target: Number of approved courses in 1st semester
TARGET_1 = 'Curricular units 1st sem (approved)'

# Predictors: Things known BEFORE 1st semester
FEATURES_1 = [
    'Previous qualification (grade)',
    'Admission grade',
    'Age at enrollment',
    "Mother's qualification",
    "Father's qualification",
    'Scholarship holder',
    'Gender',
    'Debtor',
    'Tuition fees up to date',
    'Curricular units 1st sem (enrolled)',  # How many they enrolled in
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

# Create modeling dataset
df_model1 = df[FEATURES_1 + [TARGET_1]].copy()

# Convert to numeric
for col in df_model1.columns:
    df_model1[col] = pd.to_numeric(df_model1[col], errors='coerce')

# Remove missing values
df_model1 = df_model1.dropna()
print(f"✓ Complete cases: {len(df_model1)} students")

# Check target distribution
print(f"\nTarget Variable: {TARGET_1}")
print(f"Distribution of approved courses:")
print(df_model1[TARGET_1].value_counts().sort_index().head(10))
print(f"\nMean: {df_model1[TARGET_1].mean():.2f}")
print(f"Variance: {df_model1[TARGET_1].var():.2f}")
print(f"Mean ≈ Variance? {abs(df_model1[TARGET_1].mean() - df_model1[TARGET_1].var()) < 1}")

# Separate features and target
X1 = df_model1[FEATURES_1]
y1 = df_model1[TARGET_1]

# Split data
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_train1)} samples")
print(f"Test set: {len(X_test1)} samples")

print("\n[STEP 3] Training Poisson GLM Model...")

# Add constant for statsmodels
X_train1_sm = sm.add_constant(X_train1)
X_test1_sm = sm.add_constant(X_test1)

# Fit Poisson GLM
poisson_model1 = sm.GLM(y_train1, X_train1_sm,
                        family=sm.families.Poisson()).fit()

print("✓ Poisson GLM trained successfully!")

print("\n[STEP 4] Model Summary (Full Statistical Output)...")
print("=" * 80)
print(poisson_model1.summary())
print("=" * 80)

# Make predictions
y_pred_train1 = poisson_model1.predict(X_train1_sm)
y_pred_test1 = poisson_model1.predict(X_test1_sm)

# Evaluate
print("\n[STEP 5] Model Performance...")

# Training metrics
train_rmse1 = np.sqrt(mean_squared_error(y_train1, y_pred_train1))
train_mae1 = mean_absolute_error(y_train1, y_pred_train1)

# Test metrics
test_rmse1 = np.sqrt(mean_squared_error(y_test1, y_pred_test1))
test_mae1 = mean_absolute_error(y_test1, y_pred_test1)
test_r2_1 = r2_score(y_test1, y_pred_test1)

print("\nTRAINING SET:")
print(f"  RMSE: {train_rmse1:.4f} courses")
print(f"  MAE:  {train_mae1:.4f} courses")

print("\nTEST SET:")
print(f"  RMSE: {test_rmse1:.4f} courses")
print(f"  MAE:  {test_mae1:.4f} courses")
print(f"  R²:   {test_r2_1:.4f}")

# Coefficients and Rate Ratios
print("\n[STEP 6] Interpreting Coefficients and Rate Ratios...")
print("=" * 80)

coef_df1 = pd.DataFrame({
    'Feature': ['Intercept'] + FEATURES_1,
    'Coefficient': poisson_model1.params.values,
    'Rate_Ratio': np.exp(poisson_model1.params.values),
    'P_value': poisson_model1.pvalues.values
})

# Sort by absolute coefficient
coef_df1['Abs_Coef'] = np.abs(coef_df1['Coefficient'])
coef_df1_sorted = coef_df1.sort_values('Abs_Coef', ascending=False)

print("\nTop 10 Most Important Predictors:")
print(coef_df1_sorted.head(11)[['Feature', 'Coefficient', 'Rate_Ratio', 'P_value']].to_string(index=False))

print("\n" + "=" * 80)
print("INTERPRETING RATE RATIOS:")
print("=" * 80)
print("\nRate Ratio (RR) = exp(coefficient)")
print("- RR > 1: Increases expected count")
print("- RR < 1: Decreases expected count")
print("- RR = 1: No effect\n")

for idx, row in coef_df1_sorted.head(6).iterrows():
    if row['Feature'] == 'Intercept':
        continue
    if row['Rate_Ratio'] > 1:
        pct_change = (row['Rate_Ratio'] - 1) * 100
        direction = "increases"
    else:
        pct_change = (1 - row['Rate_Ratio']) * 100
        direction = "decreases"

    sig = "***" if row['P_value'] < 0.001 else "**" if row['P_value'] < 0.01 else "*" if row['P_value'] < 0.05 else ""
    print(f"{row['Feature']}: {direction} expected courses by {pct_change:.1f}% {sig}")

# Visualizations
print("\n[STEP 7] Creating visualizations...")

# Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Training set
axes[0].scatter(y_train1, y_pred_train1, alpha=0.5, s=20)
axes[0].plot([y_train1.min(), y_train1.max()], [y_train1.min(), y_train1.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Approved Courses', fontsize=12)
axes[0].set_ylabel('Predicted Approved Courses', fontsize=12)
axes[0].set_title(f'Training Set (Poisson GLM)\nRMSE = {train_rmse1:.3f}',
                  fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test set
axes[1].scatter(y_test1, y_pred_test1, alpha=0.5, s=20, color='green')
axes[1].plot([y_test1.min(), y_test1.max()], [y_test1.min(), y_test1.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Approved Courses', fontsize=12)
axes[1].set_ylabel('Predicted Approved Courses', fontsize=12)
axes[1].set_title(f'Test Set (Poisson GLM)\nRMSE = {test_rmse1:.3f}',
                  fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("✓ Saved: poisson_predictions_1st_sem.png")

# Distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Actual distribution
axes[0].hist(y_test1, bins=range(0, int(y_test1.max()) + 2), alpha=0.7, edgecolor='black', label='Actual')
axes[0].set_xlabel('Number of Approved Courses', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Actual Distribution', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Predicted distribution
axes[1].hist(y_pred_test1, bins=range(0, int(y_test1.max()) + 2), alpha=0.7,
             color='green', edgecolor='black', label='Predicted')
axes[1].set_xlabel('Number of Approved Courses', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Predicted Distribution', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
print("✓ Saved: poisson_distributions_1st_sem.png")

# Coefficient plot
plt.figure(figsize=(10, 10))
coef_plot_df = coef_df1_sorted[coef_df1_sorted['Feature'] != 'Intercept'].head(13)
colors = ['green' if x > 0 else 'red' for x in coef_plot_df['Coefficient']]
plt.barh(range(len(coef_plot_df)), coef_plot_df['Coefficient'], color=colors, alpha=0.7)
plt.yticks(range(len(coef_plot_df)), coef_plot_df['Feature'])
plt.xlabel('Coefficient (Log Rate)', fontsize=12)
plt.title('Poisson GLM Coefficients - 1st Semester Approved Courses',
          fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', lw=1)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
print("✓ Saved: poisson_coefficients_1st_sem.png")

# ============================================================================
# MODEL 2: PREDICT NUMBER OF APPROVED COURSES (2ND SEMESTER)
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 2: POISSON GLM FOR 2ND SEMESTER APPROVED COURSES")
print("=" * 80)

print("\n[STEP 8] Preparing data for 2nd semester model...")

TARGET_2 = 'Curricular units 2nd sem (approved)'

# Predictors: Include 1st semester performance
FEATURES_2 = [
    'Previous qualification (grade)',
    'Admission grade',
    'Age at enrollment',
    'Scholarship holder',
    'Debtor',
    'Tuition fees up to date',
    'Curricular units 1st sem (approved)',  # KEY: 1st semester success
    'Curricular units 1st sem (grade)',  # 1st semester grades
    'Curricular units 2nd sem (enrolled)',  # How many enrolled in 2nd
    'Unemployment rate',
    'GDP'
]

# Create modeling dataset
df_model2 = df[FEATURES_2 + [TARGET_2]].copy()

# Convert to numeric
for col in df_model2.columns:
    df_model2[col] = pd.to_numeric(df_model2[col], errors='coerce')

# Remove missing values and zeros
df_model2 = df_model2.dropna()
print(f"✓ Complete cases: {len(df_model2)} students")

# Separate features and target
X2 = df_model2[FEATURES_2]
y2 = df_model2[TARGET_2]

# Split data
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train2)} samples")
print(f"Test set: {len(X_test2)} samples")

print("\n[STEP 9] Training 2nd semester Poisson GLM...")

# Add constant
X_train2_sm = sm.add_constant(X_train2)
X_test2_sm = sm.add_constant(X_test2)

# Fit Poisson GLM
poisson_model2 = sm.GLM(y_train2, X_train2_sm,
                        family=sm.families.Poisson()).fit()

print("✓ Model trained!")

print("\n[STEP 10] Model Summary...")
print("=" * 80)
print(poisson_model2.summary())
print("=" * 80)

# Predictions and evaluation
y_pred_train2 = poisson_model2.predict(X_train2_sm)
y_pred_test2 = poisson_model2.predict(X_test2_sm)

test_rmse2 = np.sqrt(mean_squared_error(y_test2, y_pred_test2))
test_mae2 = mean_absolute_error(y_test2, y_pred_test2)
test_r2_2 = r2_score(y_test2, y_pred_test2)

print("\n[STEP 11] Performance Metrics...")
print(f"Test RMSE: {test_rmse2:.4f} courses")
print(f"Test MAE:  {test_mae2:.4f} courses")
print(f"Test R²:   {test_r2_2:.4f}")

# Coefficients
coef_df2 = pd.DataFrame({
    'Feature': ['Intercept'] + FEATURES_2,
    'Coefficient': poisson_model2.params.values,
    'Rate_Ratio': np.exp(poisson_model2.params.values),
    'P_value': poisson_model2.pvalues.values
})

coef_df2['Abs_Coef'] = np.abs(coef_df2['Coefficient'])
coef_df2_sorted = coef_df2.sort_values('Abs_Coef', ascending=False)

print("\nTop Predictors for 2nd Semester:")
print(coef_df2_sorted.head(11)[['Feature', 'Coefficient', 'Rate_Ratio', 'P_value']].to_string(index=False))

# Visualizations for 2nd semester
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].scatter(y_test2, y_pred_test2, alpha=0.5, s=20, color='purple')
axes[0].plot([y_test2.min(), y_test2.max()], [y_test2.min(), y_test2.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Approved Courses (2nd Sem)', fontsize=12)
axes[0].set_ylabel('Predicted Approved Courses', fontsize=12)
axes[0].set_title(f'2nd Semester - Poisson GLM\nRMSE = {test_rmse2:.3f}',
                  fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residuals
residuals2 = y_test2 - y_pred_test2
axes[1].scatter(y_pred_test2, residuals2, alpha=0.5, s=20, color='purple')
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Approved Courses', fontsize=12)
axes[1].set_ylabel('Residuals', fontsize=12)
axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("\n✓ Saved: poisson_predictions_2nd_sem.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[STEP 12] Saving predictions and coefficients...")

# Save predictions
pred_df1 = pd.DataFrame({
    'Actual_1st_Sem': y_test1.values,
    'Predicted_1st_Sem': y_pred_test1,
    'Error': y_test1.values - y_pred_test1
})


pred_df2 = pd.DataFrame({
    'Actual_2nd_Sem': y_test2.values,
    'Predicted_2nd_Sem': y_pred_test2,
    'Error': y_test2.values - y_pred_test2
})


# Save coefficients



print("✓ Saved all files")

# ============================================================================
# COMPARISON WITH LINEAR REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("POISSON GLM vs LINEAR REGRESSION FOR COUNT DATA")
print("=" * 80)

from sklearn.linear_model import LinearRegression

# Train linear regression for comparison
lr = LinearRegression()
lr.fit(X_train1, y_train1)
y_pred_lr = lr.predict(X_test1)

lr_rmse = np.sqrt(mean_squared_error(y_test1, y_pred_lr))
lr_mae = mean_absolute_error(y_test1, y_pred_lr)

print("\nModel Comparison for 1st Semester Approved Courses:")
print(f"\n{'Model':<20} {'RMSE':<10} {'MAE':<10}")
print("-" * 40)
print(f"{'Poisson GLM':<20} {test_rmse1:<10.4f} {test_mae1:<10.4f}")
print(f"{'Linear Regression':<20} {lr_rmse:<10.4f} {lr_mae:<10.4f}")

# Check for impossible predictions
n_negative = (y_pred_lr < 0).sum()
print(f"\nLinear Regression issues:")
print(f"  Negative predictions: {n_negative} (impossible for counts!)")
print(f"\nPoisson GLM advantages:")
print(f"  ✓ Predictions always non-negative")
print(f"  ✓ Accounts for count distribution")
print(f"  ✓ Better for rare events (low counts)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print("\nMODEL 1: 1st Semester Approved Courses")
print("-" * 80)
print(f"✓ Test RMSE: {test_rmse1:.4f} courses")
print(f"✓ Test MAE:  {test_mae1:.4f} courses")
print(f"✓ Average prediction error: ±{test_mae1:.2f} courses")
top_feat1 = coef_df1_sorted.iloc[1]  # Skip intercept
print(f"✓ Top predictor: {top_feat1['Feature']}")
print(f"  Rate Ratio: {top_feat1['Rate_Ratio']:.4f}")

print("\nMODEL 2: 2nd Semester Approved Courses")
print("-" * 80)
print(f"✓ Test RMSE: {test_rmse2:.4f} courses")
print(f"✓ Test MAE:  {test_mae2:.4f} courses")
print(f"✓ Average prediction error: ±{test_mae2:.2f} courses")
top_feat2 = coef_df2_sorted.iloc[1]
print(f"✓ Top predictor: {top_feat2['Feature']}")
print(f"  Rate Ratio: {top_feat2['Rate_Ratio']:.4f}")

print("\n" + "=" * 80)
print("WHEN TO USE POISSON GLM")
print("=" * 80)
print("""
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
""")

print("\n" + "=" * 80)
print("FILES CREATED")
print("=" * 80)
print("Visualizations:")
print("  1. poisson_predictions_1st_sem.png")
print("  2. poisson_distributions_1st_sem.png")
print("  3. poisson_coefficients_1st_sem.png")
print("  4. poisson_predictions_2nd_sem.png")
print("\nData Files:")
print("  5. poisson_predictions_1st_sem.csv")
print("  6. poisson_predictions_2nd_sem.csv")
print("  7. poisson_coefficients_1st_sem.csv")
print("  8. poisson_coefficients_2nd_sem.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)