import pandas as pd
from sklearn.model_selection import tarin_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
#Assume df is the dataframe and 'target' is the column you're trying to predict
#df = pd.raed_csv('customer_churn_nn.csv)
target_col='target'
X= df.drop (columns=[target_col)]
y= df[target_col]
#Indentify categorical and numerical columns
categorical_cols=x.select_dtypes (include=['object','category']).columns
numeric_cols = X.select_dtypes(include=['int64','float 64']).columns
#Preprocessing
#1. Impute missing values
#2. Encode categorical features
#3 Scale numerical features
numeric_transformer = Pipeline(steps=[
           ('imputer', Simpleimputer(strategy='most_frequent')),
           ( 'encoder' OneHotEncoder(handle_unknwon='ignore'))
])
# Split the data before fitting preprocessing (to avoid data leakage)
X_train_processed = preprocessor.fit_transform(X_train)
X-test_processed = preprocessor.transform(X_test)
print("Training data shape:", X_train_processed.shape  
print ("Testing data shape:", X_test_processed.shape
       
