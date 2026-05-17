# ==========================================================
# NEURAL NETWORK EXPERIMENTS ON CUSTOMER CHURN DATASET
# ==========================================================

# This script runs multiple experiments by changing:
# 1. Number of Hidden Layers
# 2. Learning Rate
# 3. Batch Size
# 4. Activation Function

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

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

from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("customer_churn_nn.csv")

# ==========================================================
# FEATURES AND TARGET
# ==========================================================

target_col = df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target variable
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# ==========================================================
# IDENTIFY COLUMN TYPES
# ==========================================================

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# ==========================================================
# PREPROCESSING PIPELINES
# ==========================================================

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

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

# Convert sparse matrix to dense array
X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

# ==========================================================
# EXPERIMENT SETTINGS
# ==========================================================

hidden_layer_configs = [
    [32],
    [64, 32],
    [128, 64, 32]
]

learning_rates = [0.01, 0.001]

batch_sizes = [16, 32]

activation_functions = ['relu', 'tanh']

# ==========================================================
# STORE RESULTS
# ==========================================================

results = []

# ==========================================================
# RUN EXPERIMENTS
# ==========================================================

experiment_no = 1

for hidden_layers in hidden_layer_configs:

    for lr in learning_rates:

        for batch_size in batch_sizes:

            for activation in activation_functions:

                print("\n================================================")
                print(f"Experiment {experiment_no}")
                print("Hidden Layers :", hidden_layers)
                print("Learning Rate :", lr)
                print("Batch Size    :", batch_size)
                print("Activation    :", activation)

                # --------------------------------------------------
                # BUILD MODEL
                # --------------------------------------------------

                model = Sequential()

                # First hidden layer
                model.add(Dense(
                    hidden_layers[0],
                    activation=activation,
                    input_dim=X_train.shape[1]
                ))

                # Additional hidden layers
                for neurons in hidden_layers[1:]:
                    model.add(Dense(
                        neurons,
                        activation=activation
                    ))

                # Output layer
                model.add(Dense(1, activation='sigmoid'))

                # --------------------------------------------------
                # COMPILE MODEL
                # --------------------------------------------------

                model.compile(
                    optimizer=Adam(learning_rate=lr),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )

                # --------------------------------------------------
                # TRAIN MODEL
                # --------------------------------------------------

                history = model.fit(
                    X_train,
                    y_train,
                    epochs=20,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=0
                )

                # --------------------------------------------------
                # EVALUATE MODEL
                # --------------------------------------------------

                train_loss, train_acc = model.evaluate(
                    X_train,
                    y_train,
                    verbose=0
                )

                test_loss, test_acc = model.evaluate(
                    X_test,
                    y_test,
                    verbose=0
                )

                # --------------------------------------------------
                # STORE RESULTS
                # --------------------------------------------------

                results.append({
                    'Experiment': experiment_no,
                    'Hidden Layers': str(hidden_layers),
                    'Learning Rate': lr,
                    'Batch Size': batch_size,
                    'Activation': activation,
                    'Train Accuracy': round(train_acc, 4),
                    'Test Accuracy': round(test_acc, 4),
                    'Train Loss': round(train_loss, 4),
                    'Test Loss': round(test_loss, 4)
                })

                experiment_no += 1

# ==========================================================
# DISPLAY RESULTS
# ==========================================================

results_df = pd.DataFrame(results)

print("\n================================================")
print("ALL EXPERIMENT RESULTS")
print("================================================")

print(results_df)

# ==========================================================
# BEST MODEL
# ==========================================================

best_model = results_df.sort_values(
    by='Test Accuracy',
    ascending=False
).iloc[0]

print("\n================================================")
print("BEST MODEL CONFIGURATION")
print("================================================")

print(best_model)

# ==========================================================
# SAVE RESULTS
# ==========================================================

results_df.to_csv("nn_experiment_results.csv", index=False)

print("\nResults saved as 'nn_experiment_results.csv'")

# ==========================================================
# INTERPRETATION
# ==========================================================

print("""
=========================================================
INTERPRETATION OF EXPERIMENTS
=========================================================

1. Hidden Layers:
   - More hidden layers can improve learning capacity.
   - Too many layers may lead to overfitting.

2. Learning Rate:
   - High learning rate trains faster but may overshoot.
   - Low learning rate is more stable but slower.

3. Batch Size:
   - Smaller batch sizes improve generalization.
   - Larger batch sizes train faster.

4. Activation Function:
   - ReLU is efficient and commonly used.
   - Tanh may perform better on some datasets.

5. Best Model:
   - Selected based on highest test accuracy.
=========================================================
""")
