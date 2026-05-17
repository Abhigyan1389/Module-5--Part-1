# ============================================
# DATA PREPROCESSING FOR CUSTOMER CHURN DATASET
# ============================================

# Import Required Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ============================================
# LOAD DATASET
# ============================================

# Replace with your dataset filename
df = pd.read_csv("customer_churn_nn.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Dataset information
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# ============================================
# DEFINE FEATURES AND TARGET
# ============================================

# Assuming last column is target variable
target_col = df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

# ============================================
# ENCODE TARGET VARIABLE
# ============================================

# Convert categorical target into numeric
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

# ============================================
# IDENTIFY COLUMN TYPES
# ============================================

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric Columns:")
print(numeric_cols)

print("\nCategorical Columns:")
print(categorical_cols)

# ============================================
# HANDLE MISSING VALUES + SCALING
# ============================================

# Numeric preprocessing:
# 1. Fill missing values with median
# 2. Scale features using StandardScaler

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Alternative:
# ('scaler', MinMaxScaler())

# ============================================
# HANDLE CATEGORICAL DATA
# ============================================

# Categorical preprocessing:
# 1. Fill missing values with most frequent value
# 2. Apply One-Hot Encoding

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# ============================================
# COMBINE PREPROCESSING STEPS
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# ============================================
# SPLIT DATASET
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Set Shape:")
print(X_train.shape)

print("\nTesting Set Shape:")
print(X_test.shape)

# ============================================
# APPLY PREPROCESSING
# ============================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

# ============================================
# FINAL OUTPUT
# ============================================

print("\nPreprocessing Completed Successfully!")

print("\nProcessed Training Data Shape:")
print(X_train_processed.shape)

print("\nProcessed Testing Data Shape:")
print(X_test_processed.shape)
