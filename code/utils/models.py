from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score, precision_score,
    recall_score, f1_score, balanced_accuracy_score, log_loss, brier_score_loss
)
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import shap

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
    plt.title("Precision–Recall Curve")
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
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Train ROC-AUC
    auc_train = roc_auc_score(y_train_vec, proba_train)
    print(f"Train ROC-AUC:  {auc_train:.4f}")
    print("\n" + "-" * 72 + "\n")

    # Brier Score and Log Loss
    brier_test = brier_score_loss(y_test_vec, proba_test)
    ll_test = log_loss(y_test_vec, proba_test)
    brier_train = brier_score_loss(y_train_vec, proba_train)
    ll_train = log_loss(y_train_vec, proba_train)
    print(f"\nBrier Score (test): {brier_test:.4f}")
    print(f"Brier Score (train): {brier_train:.4f}")
    print(f"Log Loss (test):    {ll_test:.4f}")
    print(f"Log Loss (train):    {ll_train:.4f}")
    print("\n" + "-" * 72 + "\n")

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

    importances = rf_model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': feat_test.columns, 'Gini Importance': importances}).sort_values(
        'Gini Importance', ascending=False)
    print(feature_imp_df)

    evaluate_model(targ_train, targ_test, preds_test, rf_train_df["rf_xG"], rf_test_df["rf_xG"])

    return rf_model, rf_train_df, rf_test_df

#XGBoost Model
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
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            min_child_weight=3,
            gamma=0,
            colsample_bytree=0.5,
            subsample=0.8
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

    # Feature Importance
    importances = gb_model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': feat_test.columns, 'Feature Importance': importances}).sort_values(
        'Feature Importance', ascending=False)
    print(feature_imp_df)

    plt.figure(figsize=(9, 8))
    plt.barh(
        feature_imp_df["Feature"],
        feature_imp_df["Feature Importance"]
    )
    plt.xlabel("XGBoost Feature Importance")
    plt.gca().invert_yaxis()  # so highest importance appears at top
    plt.show()

    # SHAP Values
    explainer = shap.TreeExplainer(gb_model)
    shap_values = explainer.shap_values(feat_test)
    shap.summary_plot(shap_values, feat_test)
    shap.summary_plot(shap_values, feat_test, plot_type="bar")    

    mpl.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7
    })

    # features for dependence plots
    features_to_plot = [
        'angle_to_post',
        'dist_shot_keeper',
        'player_in_between',
        'dist_to_post',
        'shot_body_part_name_Head',
        'dist_goal_keeper',
        'poss_team_match_state_possession',
        'shot_deflected',
        'duration_buildup_shot'
    ]

    fig, axes = plt.subplots(3, 3, figsize=(20, 10))
    axes = axes.ravel()

    for i, feature in enumerate(features_to_plot):
        shap.dependence_plot(
            feature,
            shap_values,
            feat_test,
            ax=axes[i],
            show=False,
            interaction_index=None             
        )
        for label in axes[i].get_xticklabels():
            label.set_fontsize(6)

        for label in axes[i].get_yticklabels():
            label.set_fontsize(6)

        axes[i].xaxis.label.set_size(7)
        axes[i].yaxis.label.set_size(7)
        axes[i].title.set_size(8)

    # hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(pad=2.0)
    plt.show()

    evaluate_model(targ_train, targ_test, preds_test, gb_train_df["gb_xG"], gb_test_df["gb_xG"])

    return gb_model, gb_train_df, gb_test_df

#Neural Network Model
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
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha= 0.05,
            learning_rate='constant',
            early_stopping=True,
            max_iter=1000,
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
