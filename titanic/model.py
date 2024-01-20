import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

ALGORITHM_HYPERPARAMETERS = {
    "xgboost": {
        "learning_rate": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2],
        "n_estimators": [10, 25, 50, 100, 200],
        "max_depth": [1, 2, 3, 5, 10]
    },
    "sklearnboost": {
        "learning_rate": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2],
        "n_estimators": [10, 25, 50, 100, 200],
        "max_depth": [1, 2, 3, 5, 10]
    }
}

CLASSIFIERS = {
    "xgboost": XGBClassifier(),
    "sklearnboost": GradientBoostingClassifier()
}


def prepare_model(classifier, x_train, y_train):
    params = ALGORITHM_HYPERPARAMETERS[classifier]
    model = CLASSIFIERS[classifier]
    model_grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3)
    model_grid_search.fit(x_train, y_train)
    best_model = model_grid_search.best_estimator_

    return best_model
