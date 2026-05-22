import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_data(data_path: str | Path) -> pd.DataFrame:
    """Loads dataset from the specified path."""
    return pd.read_csv(data_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data filtering (removing extreme outliers based on EDA)."""
    df_clean = df.copy()
    
    # Filter by price
    df_clean = df_clean[(df_clean["price"] >= 500) & (df_clean["price"] <= 100000)]
    
    # Filter by year
    df_clean = df_clean[df_clean["year"] >= 1980]
    
    # Filter by odometer
    df_clean = df_clean[(df_clean["odometer"] > 0) & (df_clean["odometer"] <= 500000)]
    
    # Drop technical columns if they exist
    cols_to_drop = ["id", "url", "region_url", "image_url", "VIN", "county", "size"]
    cols_to_drop = [c for c in cols_to_drop if c in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        
    return df_clean

def get_train_test_split(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    """Splits dataframe into train and test sets."""
    return train_test_split(df, test_size=test_size, random_state=random_state)