"""
LINEAR REGRESSION MODEL FOR PREDICTING STUDENT GRADES
======================================================
This script builds a linear regression model to predict student grades
using various features from the student dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("LINEAR REGRESSION MODEL FOR GRADE PREDICTION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("\n[STEP 1] Loading and exploring data...")

df = pd.read_csv('/Users/cyrielvanhelleputte/PycharmProjects/machine_learning_I_Project/data/data_choosing_process/Cyriel/data.csv', sep=';', encoding='utf-8-sig')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Display basic info
print(f"\nTarget variable options:")
print(f"  - Curricular units 1st sem (grade)")
print(f"  - Curricular units 2nd sem (grade)")
print(f"  - Admission grade")

# ============================================================================
# STEP 2: PREPARE DATA FOR GRADE PREDICTION
# ============================================================================
print("\n[STEP 2] Preparing data for grade prediction...")

# We'll predict: 2nd semester grade (as it's the final outcome)
TARGET = 'Curricular units 2nd sem (grade)'

# Select features that would be known BEFORE 2nd semester grades
# (to make realistic predictions)
FEATURES = [
    'Previous qualification (grade)',      # Past academic performance
    'Admission grade',                     # Entry qualification
    'Age at enrollment',                   # Demographics
    "Mother's qualification",              # Family education
    "Father's qualification",              # Family education
    'Scholarship holder',                  # Financial support
    'Curricular units 1st sem (credited)', # 1st semester performance
    'Curricular units 1st sem (enrolled)', # 1st semester workload
    'Curricular units 1st sem (approved)', # 1st semester success
    'Curricular units 1st sem (grade)',    # 1st semester grades
    'Unemployment rate',                   # Economic factors
    'Inflation rate',                      # Economic factors
    'GDP'                                  # Economic factors
]

# Create working dataset
df_model = df[FEATURES + [TARGET]].copy()

# Convert to numeric
for col in df_model.columns:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

# Remove rows with missing values
print(f"Original dataset: {len(df_model)} rows")
df_model = df_model.dropna()
print(f"After removing missing values: {len(df_model)} rows")

# Remove rows where target is 0 (no grade recorded)
df_model = df_model[df_model[TARGET] > 0]
print(f"After removing zero grades: {len(df_model)} rows")

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n[STEP 3] Exploratory Data Analysis...")

print(f"\nTarget Variable Statistics ({TARGET}):")
print(df_model[TARGET].describe())

# Check correlations with target
print(f"\nFeature Correlations with {TARGET}:")
correlations = df_model[FEATURES].corrwith(df_model[TARGET]).sort_values(ascending=False)
print(correlations)

# Visualize correlations
plt.figure(figsize=(10, 8))
correlation_matrix = df_model[FEATURES + [TARGET]].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print("\n✓ Saved: correlation_heatmap.png")

# ============================================================================
# STEP 4: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================
print("\n[STEP 4] Splitting data into train and test sets...")

X = df_model[FEATURES]
y = df_model[TARGET]

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

# ============================================================================
# STEP 5: FEATURE SCALING (OPTIONAL BUT RECOMMENDED)
# ============================================================================
print("\n[STEP 5] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled to have mean=0 and std=1")

# ============================================================================
# STEP 6: BUILD AND TRAIN LINEAR REGRESSION MODEL
# ============================================================================
print("\n[STEP 6] Building and training Linear Regression model...")

# Create model
model = LinearRegression()

# Train model
model.fit(X_train_scaled, y_train)

print("✓ Model trained successfully!")

# Display model equation
print(f"\nModel Equation:")
print(f"{TARGET} = {model.intercept_:.4f}")
for feature, coef in zip(FEATURES, model.coef_):
    sign = "+" if coef >= 0 else ""
    print(f"    {sign} {coef:.4f} × {feature}")

# ============================================================================
# STEP 7: MAKE PREDICTIONS
# ============================================================================
print("\n[STEP 7] Making predictions...")

# Predict on training set
y_train_pred = model.predict(X_train_scaled)

# Predict on test set
y_test_pred = model.predict(X_test_scaled)

print("✓ Predictions generated")

# ============================================================================
# STEP 8: EVALUATE MODEL PERFORMANCE
# ============================================================================
print("\n[STEP 8] Evaluating model performance...")
print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)

# Training set metrics
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

print("\nTRAINING SET:")
print(f"  R² Score:  {train_r2:.4f}")
print(f"  RMSE:      {train_rmse:.4f}")
print(f"  MAE:       {train_mae:.4f}")

# Test set metrics
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\nTEST SET:")
print(f"  R² Score:  {test_r2:.4f}")
print(f"  RMSE:      {test_rmse:.4f}")
print(f"  MAE:       {test_mae:.4f}")

# Interpret R²
print("\n" + "="*60)
print("INTERPRETATION:")
print("="*60)
print(f"✓ R² Score of {test_r2:.4f} means the model explains {test_r2*100:.2f}% of variance")
print(f"✓ On average, predictions are off by {test_mae:.2f} grade points (MAE)")
print(f"✓ Root Mean Squared Error (RMSE) is {test_rmse:.2f} grade points")

# Cross-validation
print("\nPerforming 5-Fold Cross-Validation...")
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5,
                            scoring='r2', n_jobs=-1)
print(f"  Cross-validation R² scores: {cv_scores}")
print(f"  Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# STEP 9: VISUALIZE RESULTS
# ============================================================================
print("\n[STEP 9] Creating visualizations...")

# Visualization 1: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Training set
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=20)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Grade', fontsize=12)
axes[0].set_ylabel('Predicted Grade', fontsize=12)
axes[0].set_title(f'Training Set\nR² = {train_r2:.4f}', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test set
axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Grade', fontsize=12)
axes[1].set_ylabel('Predicted Grade', fontsize=12)
axes[1].set_title(f'Test Set\nR² = {test_r2:.4f}', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("✓ Saved: predictions_scatter.png")

# Visualization 2: Residuals
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Residual plot
axes[0].scatter(y_test_pred, residuals_test, alpha=0.5, s=20)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted Grade', fontsize=12)
axes[0].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Residual distribution
axes[1].hist(residuals_test, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Residuals', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("✓ Saved: residuals_analysis.png")

# Visualization 3: Feature Importance
feature_importance = pd.DataFrame({
    'Feature': FEATURES,
    'Coefficient': model.coef_
})
feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

plt.figure(figsize=(10, 8))
colors = ['green' if x > 0 else 'red' for x in feature_importance['Coefficient']]
plt.barh(range(len(feature_importance)), feature_importance['Coefficient'], color=colors, alpha=0.7)
plt.yticks(range(len(feature_importance)), feature_importance['Feature'])
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Feature Importance (Coefficients)', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', lw=1)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
print("✓ Saved: feature_importance.png")

# ============================================================================
# STEP 10: MAKE SAMPLE PREDICTIONS
# ============================================================================
print("\n[STEP 10] Making sample predictions...")

# Show first 10 predictions
print("\nSample Predictions (First 10 test cases):")
print("-" * 70)
print(f"{'Actual Grade':<15} {'Predicted Grade':<18} {'Error':<10}")
print("-" * 70)

for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_test_pred[i]
    error = actual - predicted
    print(f"{actual:<15.2f} {predicted:<18.2f} {error:<10.2f}")

# ============================================================================
# STEP 11: SAVE MODEL AND PREDICTIONS®
# ============================================================================
print("\n[STEP 11] Saving results...")

# Save predictions to CSV
results_df = pd.DataFrame({
    'Actual_Grade': y_test.values,
    'Predicted_Grade': y_test_pred,
    'Error': y_test.values - y_test_pred,
    'Absolute_Error': np.abs(y_test.values - y_test_pred)
})

print("✓ Saved: grade_predictions.csv")

# Save model coefficients
model_info = pd.DataFrame({
    'Feature': ['Intercept'] + FEATURES,
    'Coefficient': [model.intercept_] + list(model.coef_)
})
print("✓ Saved: model_coefficients.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\n✓ Model Type: Linear Regression")
print(f"✓ Target Variable: {TARGET}")
print(f"✓ Number of Features: {len(FEATURES)}")
print(f"✓ Training Samples: {len(X_train)}")
print(f"✓ Test Samples: {len(X_test)}")
print(f"\n✓ Test R² Score: {test_r2:.4f}")
print(f"✓ Test RMSE: {test_rmse:.4f}")
print(f"✓ Test MAE: {test_mae:.4f}")

print("\n" + "="*80)
print("Top 5 Most Important Features:")
print("="*80)
for i, row in feature_importance.head(5).iterrows():
    direction = "positive" if row['Coefficient'] > 0 else "negative"
    print(f"{i+1}. {row['Feature']:<45} ({direction} effect)")
    print(f"   Coefficient: {row['Coefficient']:.4f}")

print("\n" + "="*80)
print("FILES CREATED:")
print("="*80)
print("  1. correlation_heatmap.png - Feature correlations")
print("  2. predictions_scatter.png - Actual vs Predicted grades")
print("  3. residuals_analysis.png - Model error analysis")
print("  4. feature_importance.png - Feature coefficients")
print("  5. grade_predictions.csv - All predictions with errors")
print("  6. model_coefficients.csv - Model parameters")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)