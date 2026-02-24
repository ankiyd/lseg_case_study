import pandas as pd


# Ordinal columns already encoded numerically
ORDINAL_COLUMNS = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "WorkLifeBalance"
]


def load_data(hris_path: str):
    """
    Load HRIS dataset only.
    """
    return pd.read_csv(hris_path)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    constant_cols = df.columns[df.nunique() <= 1]
    df = df.drop(columns=constant_cols)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    return df


def convert_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert only true nominal variables to category.
    Ordinal columns remain numeric.
    """
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    return df

def auto_convert_float_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert float columns to int if they contain only whole numbers.
    """
    for col in df.select_dtypes(include="float").columns:
        if (df[col] % 1 == 0).all():
            df[col] = df[col].astype(int)
    return df


def full_clean_pipeline(hris_path: str):
    df = load_data(hris_path)
    df = clean_column_names(df)
    
    # Remove completely empty rows
    df = df.dropna(how="all")
    
    # Remove rows where Attrition is missing (critical field)
    df = df[df["Attrition"].notna()]

    # Drop rows with minor missing values (very small %)
    df = df.dropna(subset=["Age", "Gender"])
    
    df = drop_constant_columns(df)
    df = encode_target(df)
    df = convert_categorical(df)

    df = auto_convert_float_to_int(df)
    
    return df
