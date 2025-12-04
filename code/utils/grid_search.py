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
        {'penalty':['l1','l2'],
        'C' : np.logspace(-4,4,10),
        'solver': ['lbfgs','newton-cg','liblinear','saga'],
        'max_iter': [1000]
    }
    ]

    ## init LogisticRegression class
    log_model = LogisticRegression()

    ## perform grid-search
    clf = GridSearchCV(
        estimator = log_model, 
        param_grid = param_grid,
        scoring = "roc_auc",
        cv = 5,
        verbose = 1,
        n_jobs = -1
    )
    best_clf = clf.fit(x, y)

    ## best score
    print("ROC-AUC :",best_clf.best_score_,"\nBest Estimator:", best_clf.best_estimator_)
    print(f'Accuracy - : {best_clf.score(x,y):.3f}')

    return best_clf.best_params_

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
    ## make param grid
    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [1, 2, 5, 7, 9, 11, 15],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 5, 7]
    }

    ## init RandomForesetClassifier object
    rf_model = RandomForestClassifier(n_jobs=-1)

    ## run grid search
    clf = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=5,
        verbose=1
    )
    
    best_clf = clf.fit(x, y)

    ## best score
    print("ROC-AUC :",best_clf.best_score_,"\nBest Estimator:", best_clf.best_estimator_)
    print(f'Accuracy - : {best_clf.score(x,y):.3f}')

    return best_clf.best_params_

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
    xgboost_model = XGBClassifier()

    ## make param grid
    param_grid={
        'n_estimators'     : [100, 200, 300, 400],
        'gamma'            : [0, 0.1, 0.2 , 0.3, 0.4],
        'subsample'        : [0.6, 0.7, 0.8 , 0.9, 1.0],
        "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25] ,
        "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12],
        "min_child_weight" : [ 1, 3, 5, 7],
        "gamma"            : [ 0, 0.1, 0.2 , 0.3, 0.4],
        "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7, 1]    
    }

    ## run randomized-search
    clf = RandomizedSearchCV(
        estimator=xgboost_model,
        param_distributions=param_grid,
        n_iter=125,
        scoring="roc_auc",
        n_jobs=-1,
        cv=5,
        verbose=1
    )
    best_clf = clf.fit(x, y)

    ## best score
    print("ROC-AUC :",best_clf.best_score_,"\nBest Estimator:", best_clf.best_estimator_)
    print(f'Accuracy - : {best_clf.score(x,y):.3f}')

    return best_clf.best_params_

def tune_neural_network(x, y):
    """
    Function for performing hyperparameter 
    tuning for neural network model.

    Args:
        x (numpy.ndarray): the feature value.
        y (numpy.ndarray): the target value.
    Returns:
        dict: containing parameter values
    """ 
    ## init Neural Network model
    from sklearn.neural_network import MLPClassifier
    nn_model = MLPClassifier()

    ## make param grid
    param_grid = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }

    ## run grid search
    clf = GridSearchCV(
        estimator=nn_model,
        param_grid=param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=5,
        verbose=1
    )
    best_clf = clf.fit(x, y)

    ## best score
    print("ROC-AUC :",best_clf.best_score_,"\nBest Estimator:", best_clf.best_estimator_)
    print(f'Accuracy - : {best_clf.score(x,y):.3f}')

    return best_clf.best_params_

def tune_naive_bayes(x, y):
    """
    Function for performing hyperparameter 
    tuning for naive bayes model.

    Args:
        x (numpy.ndarray): the feature value.
        y (numpy.ndarray): the target value.
    Returns:
        dict: containing parameter values
    """
    ## init Naive Bayes model
    from sklearn.naive_bayes import GaussianNB
    nb_model = GaussianNB()

    ## make param grid
    param_grid = {
        'var_smoothing': np.logspace(0,-9, num=100)
    }

    ## run grid search
    clf = GridSearchCV(
        estimator=nb_model,
        param_grid=param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=5,
        verbose=1
    )
    best_clf = clf.fit(x, y)

    ## best score
    print("ROC-AUC :",best_clf.best_score_,"\nBest Estimator:", best_clf.best_estimator_)
    print(f'Accuracy - : {best_clf.score(x,y):.3f}')

    return best_clf.best_params_