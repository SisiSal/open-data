import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def tune_log_model(x, y):
    """
    Function for performing hyperparameter 
    tuning for logistic regression model.

    Args:
        x (numpy.ndarray): the feature value.
        y (numpy.ndarray): the target value.

    Returns:
        dict: containing parameter values
    """    
    ## define values for logistic model
    param_grid = [
        {'penalty':['l1','l2','elasticnet','none'],
        'C' : np.logspace(-4,4,20),
        'solver': ['lbfgs','newton-cg','liblinear','sag','saga'],
        'max_iter'  : [100,1000,2500,5000]
    }
    ]

    ## init LogisticRegression class
    log_model = LogisticRegression()

    ## perform grid-search
    gridsearch = GridSearchCV(
        estimator = log_model, 
        param_grid = param_grid,
        scoring = "roc_auc",
        cv = 3,
        verbose = 1,
        n_jobs = -1
    )
    best_model = gridsearch.fit(x, y)

    ## best score
    print("ROC-AUC :",best_model.best_score_,"\nBest Estimator:", best_model.best_estimator_)
    print(f'Accuracy - : {best_model.score(x,y):.3f}')

    return best_model.best_params_

def tune_random_forest(x, y):
    """
    Function for performing hyperparameter 
    tuning for random forest model.

    Args:
        x (numpy.ndarray): the feature value.
        y (numpy.ndarray): the target value.

    Returns:
        dict: containing parameter values
    """  
    ## init RandomForesetClassifier object
    clf = RandomForestClassifier(n_jobs=-1)

    ## make param grid
    param_gird = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [1, 2, 5, 7, 9, 11, 15],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 5, 7]
    }

    ## run grid search
    best_model = GridSearchCV(
        estimator=clf,
        param_grid=param_gird,
        scoring="roc_auc",
        n_jobs=-1,
        cv=5,
        verbose=1
    )
    best_model.fit(x, y)

    ## best score
    print("ROC-AUC :",best_model.best_score_)

    return best_model.best_params_

def tune_xg_boost(x, y):
    """
    Function for performing hyperparameter 
    tuning for xGBoost model.

    Args:
        x (numpy.ndarray): the feature value.
        y (numpy.ndarray): the target value.
    
    Returns:
        dict: containing parameter values
    """ 
    ## init xGBoost model
    clf = XGBClassifier()

    ## make param grid
    params={
        "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.3] ,
        "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight" : [ 1, 3, 5, 7 ],
        "gamma"            : [ 0, 0.1, 0.2 , 0.3, 0.4 ],
        "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7, 1 ]    
    }

    ## run randomized-search
    best_model = RandomizedSearchCV(
        estimator=clf,
        param_distributions=params,
        n_iter=125,
        scoring="roc_auc",
        n_jobs=-1,
        cv=5,
        verbose=1
    )
    best_model.fit(x, y)

    ## best score
    print("ROC-AUC :",best_model.best_score_)

    return best_model.best_params_