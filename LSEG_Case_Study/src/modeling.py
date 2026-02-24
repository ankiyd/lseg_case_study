import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Preprocessor Builder
# ---------------------------------------------------

def build_preprocessor(categorical_cols, numeric_cols, scale_numeric=False):
    """
    Builds preprocessing pipeline.
    scale_numeric=True for logistic regression.
    """
    num_transform = StandardScaler() if scale_numeric else "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            ("num", num_transform, numeric_cols)
        ]
    )

    return preprocessor

# Logistic Regression Models
# ---------------------------------------------------

def build_simple_logistic():
    return LogisticRegression(max_iter=1000)


def build_balanced_logistic():
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )


def build_l1_balanced_logistic():
    return LogisticRegression(
        solver="liblinear",
        penalty="l1",
        class_weight="balanced",
        max_iter=1000
    )

# Tree-Based Models
# ---------------------------------------------------

def build_decision_tree():
    return DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )


def build_gradient_boosting():
    return GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

# Train Pipeline
# ---------------------------------------------------

def train_pipeline(model, preprocessor, X_train, y_train):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

# Feature Importance
# ---------------------------------------------------

def extract_feature_importance(pipeline, categorical_cols, numeric_cols):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]

    cat_features = preprocessor.transformers_[0][1] \
        .get_feature_names_out(categorical_cols)

    feature_names = list(cat_features) + list(numeric_cols)

    if hasattr(model, "coef_"):
        importance = model.coef_[0]
    else:
        importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    return importance_df