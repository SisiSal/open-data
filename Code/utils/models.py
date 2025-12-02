from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import numpy as np

#Model Evaluation Metrics
def evaluate_model(y_test_vec, y_pred, proba_test):
    """
    Evaluate model performance using various classification metrics.
    
    Parameters:
        y_test_vec (array-like): True labels
        y_pred (array-like): Predicted labels
        prob_test (array-like): Predicted probabilities for the positive class

    Returns:
        None: Prints out the evaluation metrics
    """
    # Confusion matrix and metrics
    # The rows correspond to the true classes, while the columns correspond to the predicted classes.
    # First row: True Negatives (TN), False Positives (FP)
    # Second row: False Negatives (FN), True Positives (TP)
    # Confusion matrix with fixed label order so we know [TN FP; FN TP]
    cm = confusion_matrix(y_test_vec, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Core metrics
    acc  = accuracy_score(y_test_vec, y_pred)
    prec = precision_score(y_test_vec, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_test_vec, y_pred, pos_label=1)          # sensitivity/TPR
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")      # specificity/TNR

    # Optional but often helpful
    f1   = f1_score(y_test_vec, y_pred, pos_label=1)
    bacc = balanced_accuracy_score(y_test_vec, y_pred)             # (rec + spec)/2

    print("Confusion matrix (test):\n", cm)
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Sensitivity:  {rec:.4f} (Recall)")
    print(f"Specificity:  {spec:.4f}")
    print(f"F1-score:     {f1:.4f}")
    print(f"Balanced Acc: {bacc:.4f}")
    print("\n" + "-" * 72 + "\n")

    # Precision–Recall curve (useful for imbalanced data)
    # This shows the trade-off between precision and recall for different thresholds.
    # Higher recall typically comes at the cost of lower precision, and vice versa.
    # Ideally, we want to find a balance between the two that minimizes false negatives and false positives, which corresponds to the top right of the curve.
    prec_curve, rec_curve, _ = precision_recall_curve(y_test_vec, proba_test)
    plt.figure()
    plt.plot(rec_curve, prec_curve)
    plt.title("Precision–Recall Curve — Logistic Regression")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.show()
    return None

#Logistic Regression Model
def log_reg_mod(train_df, test_df, target_col, drop_cols=None):
    """
    Fit a logistic regression using statsmodels GLM (Binomial) and predict on test data.
    
    Parameters:
        train_df (pd.DataFrame): Training dataset
        test_df (pd.DataFrame): Test dataset
        target_col (str): Name of the target variable (0/1)
        drop_cols (list): Optional list of non-feature columns to exclude (e.g., IDs)

    Returns:
        model (Pipeline or LogisticRegression): Fitted model
    """

    # Columns not to use as features
    if drop_cols is None:
        drop_cols = []

    non_features = drop_cols + [target_col]

    # Build feature list
    features = [c for c in train_df.columns if c not in non_features]

    feat_train = train_df[features]
    targ_train = train_df[target_col]
    feat_test = test_df[features]
    targ_test = test_df[target_col]

    # Initialize Logistic Regression model
    model = LogisticRegression(
                C=0.3593813663804626,
                penalty="l2",
                solver="lbfgs" #adjust parameters according to grid search results
                )

    # Fit model
    model.fit(feat_train, targ_train)

    # Predictions
    prob_test = model.predict_proba(feat_test)[:, 1]   # xG values
    preds_test = model.predict(feat_test)

    evaluate_model(targ_test, preds_test, prob_test)

    return model

#Random Forest Model
from sklearn.ensemble import RandomForestClassifier
def random_forest_mod(train_df, test_df, target_col, drop_cols=None):
    """
    Fit a Random Forest classifier and predict on test data.
    
    Parameters:
        train_df (pd.DataFrame): Training dataset
        test_df (pd.DataFrame): Test dataset
        target_col (str): Name of the target variable (0/1)
        drop_cols (list): Optional list of non-feature columns to exclude (e.g., IDs)
    Returns:
        model (RandomForestClassifier): Fitted Random Forest model
    """
    if drop_cols is None:
        drop_cols = []

    non_features = drop_cols + [target_col]

    # Build feature list
    features = [c for c in train_df.columns if c not in non_features]

    feat_train = train_df[features]
    targ_train = train_df[target_col]
    feat_test = test_df[features]
    targ_test = test_df[target_col]

    # Initialize Random Forest model
    model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
                )

    # Fit model
    model.fit(feat_train, targ_train)

    # Predictions
    prob_test = model.predict_proba(feat_test)[:, 1]   # xG values
    preds_test = model.predict(feat_test)

    evaluate_model(targ_test, preds_test, prob_test)

    return model

#XGBoost Model
import xgboost as xgb
def xgboost_mod(train_df, test_df, target_col, drop_cols=None):
    """
    Fit an XGBoost classifier and predict on test data.
    
    Parameters:
        train_df (pd.DataFrame): Training dataset
        test_df (pd.DataFrame): Test dataset
        target_col (str): Name of the target variable (0/1)
        drop_cols (list): Optional list of non-feature columns to exclude (e.g., IDs
    Returns:
        model (xgb.XGBClassifier): Fitted XGBoost model
    """
    if drop_cols is None:
        drop_cols = []

    non_features = drop_cols + [target_col]

    # Build feature list
    features = [c for c in train_df.columns if c not in non_features]

    feat_train = train_df[features]
    targ_train = train_df[target_col]
    feat_test = test_df[features]
    targ_test = test_df[target_col]

    # Initialize XGBoost model
    model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
                )

    # Fit model
    model.fit(feat_train, targ_train)

    # Predictions
    prob_test = model.predict_proba(feat_test)[:, 1]   # xG values
    preds_test = model.predict(feat_test)

    evaluate_model(targ_test, preds_test, prob_test)

    return model

#Neural Network Model
from sklearn.neural_network import MLPClassifier
def neural_network_mod(train_df, test_df, target_col, drop_cols=None):
    """
    Fit a Neural Network (MLPClassifier) and predict on test data.
    
    Parameters:
        train_df (pd.DataFrame): Training dataset
        test_df (pd.DataFrame): Test dataset
        target_col (str): Name of the target variable (0/1)
        drop_cols (list): Optional list of non-feature columns to exclude (e.g., IDs
    Returns:
        model (MLPClassifier): Fitted Neural Network model
    """
    if drop_cols is None:
        drop_cols = []

    non_features = drop_cols + [target_col]

    # Build feature list
    features = [c for c in train_df.columns if c not in non_features]

    feat_train = train_df[features]
    targ_train = train_df[target_col]
    feat_test = test_df[features]
    targ_test = test_df[target_col]

    # Initialize Neural Network model
    model = MLPClassifier(
                hidden_layer_sizes=(100,),
                activation='relu',
                solver='adam',
                max_iter=200,
                random_state=42
                )

    # Fit model
    model.fit(feat_train, targ_train)

    # Predictions
    prob_test = model.predict_proba(feat_test)[:, 1]   # xG values
    preds_test = model.predict(feat_test)

    evaluate_model(targ_test, preds_test, prob_test)

    return model

#Naive Bayes Model
from sklearn.naive_bayes import GaussianNB
def naive_bayes_mod(train_df, test_df, target_col, drop_cols=None):
    """
    Fit a Gaussian Naive Bayes classifier and predict on test data.
    
    Parameters:
        train_df (pd.DataFrame): Training dataset
        test_df (pd.DataFrame): Test dataset
        target_col (str): Name of the target variable (0/1)
        drop_cols (list): Optional list of non-feature columns to exclude (e.g., IDs
    Returns:
        model (GaussianNB): Fitted Naive Bayes model
    """
    if drop_cols is None:
        drop_cols = []

    non_features = drop_cols + [target_col]

    # Build feature list
    features = [c for c in train_df.columns if c not in non_features]

    feat_train = train_df[features]
    targ_train = train_df[target_col]
    feat_test = test_df[features]
    targ_test = test_df[target_col]

    # Initialize Naive Bayes model
    model = GaussianNB()

    # Fit model
    model.fit(feat_train, targ_train)

    # Predictions
    prob_test = model.predict_proba(feat_test)[:, 1]   # xG values
    preds_test = model.predict(feat_test)

    evaluate_model(targ_test, preds_test, prob_test)

    return model