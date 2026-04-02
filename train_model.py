import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os
import json

# Load dataset
df = pd.read_csv('ice-cream.csv')

# Display dataset info
print("Dataset Overview:")
print(f"Total records: {len(df)}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset statistics:")
print(df[['Temperature', 'Rainfall', 'IceCreamsSold']].describe())

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Prepare features and target
# Using Temperature and Rainfall as features
X = df[['Temperature', 'Rainfall']].copy()
y = df['IceCreamsSold'].copy()

# Handle any missing values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate model
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print("\n===== MODEL PERFORMANCE =====")
print(f"Training R² Score: {train_r2:.4f}")
print(f"Testing R² Score: {test_r2:.4f}")
print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")
print(f"Training RMSE: {np.sqrt(train_mse):.4f}")
print(f"Testing RMSE: {np.sqrt(test_mse):.4f}")

print("\n===== MODEL COEFFICIENTS =====")
print(f"Temperature coefficient: {model.coef_[0]:.4f}")
print(f"Rainfall coefficient: {model.coef_[1]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save model using pickle
model_path = 'model/model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Save feature names and metadata for consistent predictions
metadata = {
    'feature_names': ['Temperature', 'Rainfall'],
    'coefficients': {
        'Temperature': float(model.coef_[0]),
        'Rainfall': float(model.coef_[1]),
        'Intercept': float(model.intercept_)
    },
    'training_r2': float(train_r2),
    'testing_r2': float(test_r2)
}

metadata_path = 'model/metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"\n✓ Model saved successfully at: {model_path}")
print(f"✓ Metadata saved successfully at: {metadata_path}")
print("\nYou can now run the Flask app with: python app.py")
