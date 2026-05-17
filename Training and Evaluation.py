# ============================================
# Customer Churn Prediction using Neural Network
# ============================================

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ============================================
# Load Dataset
# ============================================

# Replace with your dataset path
df = pd.read_csv("customer_churn_nn.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ============================================
# Define Features and Target
# ============================================

# Assuming last column is target
target_col = df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target labels if categorical
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# ============================================
# Identify Numeric and Categorical Columns
# ============================================

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# ============================================
# Preprocessing Pipelines
# ============================================

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# ============================================
# Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================
# Build Neural Network Model
# ============================================

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42
)

# ============================================
# Create Full Pipeline
# ============================================

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# ============================================
# Train Model
# ============================================

clf.fit(X_train, y_train)

# ============================================
# Predictions
# ============================================

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

y_train_prob = clf.predict_proba(X_train)
y_test_prob = clf.predict_proba(X_test)

# ============================================
# Evaluation Metrics
# ============================================

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

train_loss = log_loss(y_train, y_train_prob)
test_loss = log_loss(y_test, y_test_prob)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Training Accuracy : {train_acc:.4f}")
print(f"Testing Accuracy  : {test_acc:.4f}")

print(f"Training Loss     : {train_loss:.4f}")
print(f"Testing Loss      : {test_loss:.4f}")

# ============================================
# Classification Report
# ============================================

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_test_pred))

# ============================================
# Confusion Matrix
# ============================================

cm = confusion_matrix(y_test, y_test_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

plt.title("Confusion Matrix")
plt.show()

# ============================================
# Interpretation
# ============================================

print("\n========== INTERPRETATION ==========")

print("""
1. High training and testing accuracy indicate good model performance.

2. Small difference between training and testing accuracy suggests
   the model generalizes reasonably well.

3. Confusion matrix shows how many predictions were correct
   and where misclassifications occurred.

4. If churn class predictions are weak, the dataset may be imbalanced.

5. Performance can be improved using:
   - Hyperparameter tuning
   - Feature engineering
   - Class balancing techniques (SMOTE, class weights)
   - Deep learning frameworks like TensorFlow/Keras
""")
