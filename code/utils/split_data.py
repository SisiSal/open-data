from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def split_data(df):
    X = df.drop(columns=['goal', 'match_id'])
    y = df['goal']
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print('After splitting data:')
    print(f"Original Goal Rate: {y.mean():.4f}")
    print(f"Train Goal Rate:    {y_train.mean():.4f}")
    print(f"Test Goal Rate:     {y_test.mean():.4f}")
    return X_train, X_test, y_train, y_test

def standardize_data(X_train, X_test):
    cols_to_scale = X_train.loc[:, (X_train.nunique() > 2)].columns.tolist()
    X_train_numeric = X_train[cols_to_scale].select_dtypes(include=['number'])
    cols_to_scale = X_train_numeric.columns.tolist()
    scaler = StandardScaler()
    scaler.fit(X_train[cols_to_scale])
    X_train_scaled = X_train.copy()
    X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_test_scaled = X_test.copy()
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    return X_train_scaled, X_test_scaled, scaler


def calculate_vif(df):
    """
    df: DataFrame of ONLY the features (no target variable)
    returns a DataFrame of VIF values
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                       for i in range(len(df.columns))]
    return vif_data