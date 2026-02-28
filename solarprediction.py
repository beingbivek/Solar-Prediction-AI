import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. DATA LOADING & MERGING
print("Loading datasets...")
# Load datasets (Ensure these files are in your folder)
gen_df = pd.read_csv('Plant_1_Generation_Data.csv')
weather_df = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')

# Convert DATE_TIME to datetime for merging
gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], dayfirst=True)
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], dayfirst=True)

# Drop identifiers not needed for the model
gen_df.drop(columns=['PLANT_ID', 'SOURCE_KEY', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD'], inplace=True)
weather_df.drop(columns=['PLANT_ID', 'SOURCE_KEY'], inplace=True)

# Merge datasets on timestamp
df = pd.merge(gen_df, weather_df, on='DATE_TIME', how='inner')

# Feature Engineering: Extract 'Hour'
df['Hour'] = df['DATE_TIME'].dt.hour

# 2. PREPROCESSING
features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'Hour']
target = 'DC_POWER'

X = df[features]
y = df[target]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Data Prepared. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# 3. MODEL 1: LINEAR REGRESSION (BASELINE)
print("\n--- Training Model 1: Linear Regression (Baseline) ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)
print(f"Linear Regression RMSE: {lr_rmse:.2f}")
print(f"Linear Regression R2: {lr_r2:.4f}")

# 4. MODEL 2: RANDOM FOREST (WITH CROSS-VALIDATION)
print("\n--- Training Model 2: Random Forest (Optimized) ---")
print("Running GridSearchCV (This may take 1-2 minutes)...")

# Hyperparameters
param_grid = {
    'n_estimators': [50, 100],        # Number of trees
    'max_depth': [10, 20, None],      # Depth of trees
    'min_samples_split': [2, 5]       # Minimum samples to split a node
}

# Initialize Random Forest
rf = RandomForestRegressor(random_state=42)

# Grid Search with 3-Fold Cross Validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, n_jobs=-1, scoring='r2', verbose=1)

grid_search.fit(X_train, y_train)

# Get Best Model
best_rf = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"Best Hyperparameters found: {best_params}")

# Predict with Best Model
rf_pred = best_rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"Random Forest R2: {rf_r2:.4f}")

# 5. FINAL COMPARISON & VISUALIZATION
print("\n--- FINAL COMPARISON TABLE ---")
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'RMSE (Error)': [lr_rmse, rf_rmse],
    'R2 Score (Accuracy)': [lr_r2, rf_r2]
})
print(results)

# PLOT 1: Actual vs Predicted (Random Forest)
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=rf_pred, alpha=0.3, color='blue', label='Predicted')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Fit')
plt.xlabel('Actual DC Power (kW)')
plt.ylabel('Predicted DC Power (kW)')
plt.title('Random Forest: Actual vs Predicted')
plt.legend()
plt.show()

# PLOT 2: Feature Importance
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette='viridis')
plt.title('Feature Importance (What drives Solar Generation?)')
plt.xlabel('Importance Score')
plt.show()

# PLOT 3: Model Comparison Bar Chart
plt.figure(figsize=(8,5))
sns.barplot(x='Model', y='R2 Score (Accuracy)', data=results, palette='magma')
plt.ylim(0.8, 1.0) # Zoom in to show difference
plt.title('Model Comparison: Linear Regression vs Random Forest')
plt.show()

# Plot 4: Error Distribution Histogram

# Calculate Residuals (The difference between Actual and Predicted values)
# We use 'rf_pred' because Random Forest was our best model
residuals = y_test - rf_pred

# Plot the Histogram with a Kernel Density Estimate (KDE) line
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30, color='purple', edgecolor='black')

# Add a vertical red line at 0 (Perfect prediction line)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.title('Error Distribution (Residuals) - Random Forest Model', fontsize=16)
plt.xlabel('Prediction Error (kW) [Actual - Predicted]', fontsize=12)
plt.ylabel('Frequency (Count of Predictions)', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.show()