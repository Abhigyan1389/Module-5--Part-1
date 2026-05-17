# ==========================================================
# FEED-FORWARD NEURAL NETWORK FOR CUSTOMER CHURN PREDICTION
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ==========================================================
# LOAD DATASET
# ==========================================================

# Replace with your dataset filename
df = pd.read_csv("customer_churn_nn.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================================================
# DEFINE FEATURES AND TARGET
# ==========================================================

# Assuming last column is the target variable
target_col = df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

# ==========================================================
# ENCODE TARGET VARIABLE
# ==========================================================

# Convert categorical target labels into numeric
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

# ==========================================================
# IDENTIFY NUMERIC & CATEGORICAL COLUMNS
# ==========================================================

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# ==========================================================
# PREPROCESSING PIPELINES
# ==========================================================

# Numeric preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# ==========================================================
# TRAIN-TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# APPLY PREPROCESSING
# ==========================================================

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Convert sparse matrix to array if needed
X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

# ==========================================================
# BUILD FEED-FORWARD NEURAL NETWORK
# ==========================================================

# Input dimension
input_dim = X_train.shape[1]

# Create model
model = Sequential()

# ----------------------------------------------------------
# INPUT LAYER + HIDDEN LAYER
# ----------------------------------------------------------

model.add(Dense(
    units=32,              # Number of neurons
    activation='relu',     # Activation function
    input_dim=input_dim
))

# ----------------------------------------------------------
# OUTPUT LAYER
# ----------------------------------------------------------

# Binary classification output
model.add(Dense(
    units=1,
    activation='sigmoid'
))

# ==========================================================
# COMPILE MODEL
# ==========================================================

model.compile(
    optimizer=Adam(learning_rate=0.001),   # Optimizer
    loss='binary_crossentropy',             # Loss function
    metrics=['accuracy']
)

# ==========================================================
# MODEL SUMMARY
# ==========================================================

print("\nModel Summary:")
model.summary()

# ==========================================================
# TRAIN MODEL
# ==========================================================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ==========================================================
# EVALUATE MODEL
# ==========================================================

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nTest Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))

# ==========================================================
# PREDICTIONS
# ==========================================================

y_pred_prob = model.predict(X_test)

# Convert probabilities to class labels
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nSample Predictions:")
print(y_pred[:10])

# ==========================================================
# INTERPRETATION
# ==========================================================

print("""
==========================================================
MODEL COMPONENTS USED
==========================================================

1. Input Layer:
   - Receives all processed features from the dataset

2. Hidden Layer:
   - 32 neurons
   - ReLU activation function

3. Output Layer:
   - 1 neuron
   - Sigmoid activation for binary classification

4. Loss Function:
   - Binary Crossentropy

5. Optimizer:
   - Adam optimizer

6. Evaluation Metric:
   - Accuracy
==========================================================
""")
