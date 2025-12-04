from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#Model Evaluation Metrics
def evaluate_model(y_train_vec, y_test_vec, y_pred, proba_train, proba_test):
    """
    Evaluate model performance using various classification metrics.
    
    Parameters:
        y_train_vec (array-like): True labels for training data
        y_test_vec (array-like): True labels for test data
        y_pred (array-like): Predicted labels
        proba_test (array-like): Predicted probabilities for the test set
        proba_train (array-like): Predicted probabilities for the train set

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
    bacc = balanced_accuracy_score(y_test_vec, y_pred)            # (rec + spec)/2

    print("Confusion matrix (test):\n", cm)
    print(f"Accuracy:     {acc:.4f}")
    print(f"Precision:    {prec:.4f}")
    print(f"Sensitivity:  {rec:.4f} (Recall)")
    print(f"Specificity:  {spec:.4f}")
    print(f"F1-score:     {f1:.4f}")
    print(f"Balanced Acc: {bacc:.4f}")
    print("\n" + "-" * 72 + "\n")

    # Precision–Recall curve (useful for imbalanced data)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test_vec, proba_test)
    plt.figure()
    plt.plot(rec_curve, prec_curve)
    plt.title("Precision–Recall Curve — Logistic Regression")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.show()

    # Test ROC-AUC 
    auc_score = roc_auc_score(y_test_vec, proba_test)
    fpr, tpr, _ = roc_curve(y_test_vec, proba_test)
    print(f"Test ROC-AUC:   {auc_score:.4f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")  # diagonal line
    plt.title("ROC Curve — Logistic Regression")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Train ROC-AUC
    auc_train = roc_auc_score(y_train_vec, proba_train)
    print(f"Train ROC-AUC:  {auc_train:.4f}")

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
    lr_model = LogisticRegression(
                C=10000.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000
                )

    # Fit model
    lr_model.fit(feat_train, targ_train)

    # xG Predictions
    lr_train_df = train_df.copy()
    lr_test_df = test_df.copy()
    lr_train_df["lr_xG"] = lr_model.predict_proba(feat_train)[:, 1]
    lr_test_df["lr_xG"]  = lr_model.predict_proba(feat_test)[:, 1]
    
    # Get predicted classes
    preds_test = lr_model.predict(feat_test)
    
    feature_coef_df = pd.DataFrame({'Feature': feat_train.columns, 'Coefficient': lr_model.coef_[0]}).sort_values(
        'Coefficient', ascending=False)
    print(feature_coef_df)

    evaluate_model(targ_train, targ_test, preds_test, lr_train_df["lr_xG"], lr_test_df["lr_xG"])

    return lr_model, lr_train_df, lr_test_df

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
    rf_model = RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                random_state=42,
                criterion='entropy',
                min_samples_split=7
                )

    # Fit model
    rf_model.fit(feat_train, targ_train)

    # xG Predictions
    rf_train_df = train_df.copy()
    rf_test_df = test_df.copy()
    rf_train_df["rf_xG"] = rf_model.predict_proba(feat_train)[:, 1]
    rf_test_df["rf_xG"]  = rf_model.predict_proba(feat_test)[:, 1]
    
    # Get predicted classes
    preds_test = rf_model.predict(feat_test)

    for col, imp in zip(feat_train.columns, rf_model.feature_importances_):
        print(f"{col}: {imp:.4f}")

    importances = rf_model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': feat_train.columns, 'Gini Importance': importances}).sort_values(
        'Gini Importance', ascending=False)
    print(feature_imp_df)

    evaluate_model(targ_train, targ_test, preds_test, rf_train_df["rf_xG"], rf_test_df["rf_xG"])

    return rf_model, rf_train_df, rf_test_df

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
    gb_model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                min_child_weight=1,
                gamma=0.1,
                colsample_bytree=1,
                subsample=1.0
                )

    # Fit model
    gb_model.fit(feat_train, targ_train)

    # xG Predictions
    gb_train_df = train_df.copy()
    gb_test_df = test_df.copy()
    gb_train_df["gb_xG"] = gb_model.predict_proba(feat_train)[:, 1]
    gb_test_df["gb_xG"]  = gb_model.predict_proba(feat_test)[:, 1]
    
    # Get predicted classes
    preds_test = gb_model.predict(feat_test)

    evaluate_model(targ_train, targ_test, preds_test, gb_train_df["gb_xG"], gb_test_df["gb_xG"])

    return gb_model, gb_train_df, gb_test_df

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
    nn_model = MLPClassifier(
                hidden_layer_sizes=(100,),
                activation='relu',
                solver='adam',
                max_iter=200,
                random_state=42
                )

    # Fit model
    nn_model.fit(feat_train, targ_train)

    # xG Predictions
    nn_train_df = train_df.copy()
    nn_test_df = test_df.copy()
    nn_train_df["nn_xG"] = nn_model.predict_proba(feat_train)[:, 1]
    nn_test_df["nn_xG"]  = nn_model.predict_proba(feat_test)[:, 1]
    
    # Get predicted classes
    preds_test = nn_model.predict(feat_test)

    evaluate_model(targ_train, targ_test, preds_test, nn_train_df["nn_xG"], nn_test_df["nn_xG"])

    return nn_model, nn_train_df, nn_test_df
