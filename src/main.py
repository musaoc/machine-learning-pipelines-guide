"""
Machine Learning Pipelines Architecture Guide — Scikit-Learn
A practical reference guide and architectural template for creating reproducible, clean, and leak-free machine learning pipelines with Scikit-Learn.

Original Kaggle Notebook: https://www.kaggle.com/code/lazer999/ml-pipelines-simplified-for-everyone
Author: Muhammad Musa Khan (Kaggle Master: https://kaggle.com/lazer999)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# --- Smart Dataset Path Resolution ---
def _resolve_data_path(file_path):
    """Checks local and data/ directories if dataset path is missing."""
    if os.path.exists(file_path):
        return file_path
    base = os.path.basename(file_path)
    candidates = [
        base,
        os.path.join("data", base),
        os.path.join("..", "data", base),
        file_path.replace("/kaggle/input/", "data/"),
        file_path.replace("../input/", "data/"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return file_path

# --- Pipeline Execution ---

# --- Cell 1 ---
import pandas as pd
import seaborn as sns
# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn import set_config
set_config(display='diagram')

# --- Cell 2 ---
df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')

df_train.head()

# --- Cell 3 ---
df_train.info()

# --- Cell 4 ---
X=df_train.drop('Survived',axis=1)
y = df_train['Survived']

# --- Cell 5 ---
X_train,X_val,y_train,y_val= train_test_split(X,y,test_size=0.25,random_state=1)

# --- Cell 6 ---
cat_cols=X_train.select_dtypes('object').columns.to_list()
cat_cols

# --- Cell 7 ---
num_cols=X_train.select_dtypes(exclude='object').columns.to_list()
num_cols

# --- Cell 8 ---
# creating pipeline for numeric data preprocessing
numeric_preprocessor = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='mean')),
    ('scaling',StandardScaler()),
])

# --- Cell 9 ---
# creating pipeline for categorical data preprocessing

categorical_preprocessor = Pipeline(steps=[
     ('imputer',SimpleImputer(strategy='constant')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

# --- Cell 10 ---
preprocessor = ColumnTransformer([
    ('categorical',categorical_preprocessor,cat_cols),
    ('numeric',numeric_preprocessor,num_cols),

])

# --- Cell 11 ---
# adding our preprocessor and model in pipelines.
Pipe = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('model',LogisticRegression(C=1e9)),
])

# --- Cell 12 ---
Pipe.fit(X_train,y_train)

# --- Cell 13 ---
pred=Pipe.predict(X_val)
pred_train = Pipe.predict(X_train)
print('Accuracy on training data :',accuracy_score(pred_train,y_train))
print('Accuracy on validation data :',accuracy_score(pred,y_val))

# --- Cell 14 ---
sns.heatmap(confusion_matrix(pred,y_val),annot=True);

# --- Cell 15 ---
pred = Pipe.predict(df_test)

output = pd.DataFrame({'PassengerId': df_test.PassengerId,
                       'Survived': pred})
output.to_csv('submission.csv', index=False)

# --- Cell 16 ---
output.sample(5)



if __name__ == "__main__":
    print("Pipeline execution complete.")
