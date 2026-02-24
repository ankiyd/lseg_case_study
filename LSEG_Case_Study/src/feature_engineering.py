import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


def separate_target(df: pd.DataFrame, target_col: str = "Attrition"):
    """
    Separate features and target variable.
    """
    X = df.drop(columns=[target_col, "Employee_ID"])
    y = df[target_col]
    return X, y


def get_feature_types(df: pd.DataFrame):
    """
    Identify categorical and numeric columns.
    """
    categorical_cols = df.select_dtypes(include=["category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    return categorical_cols, numeric_cols


def build_preprocessor(categorical_cols, numeric_cols):
    """
    Create preprocessing pipeline:
    - One-hot encode categorical variables
    - Pass numeric variables as-is
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    return preprocessor


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def evaluate(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()