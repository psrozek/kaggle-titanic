import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, ParameterGrid
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

ALGORITHM_HYPERPARAMETERS = {
    "xgboost": {
        "objective": ["binary:logistic"],
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
    "xgboost": XGBClassifier,
    "sklearnboost": GradientBoostingClassifier
}


class Model:
    def __init__(self, classifier: str):
        self.classifier_name = classifier

    def prepare_model(self, x_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier | GradientBoostingClassifier:
        params_grid = ParameterGrid(ALGORITHM_HYPERPARAMETERS[self.classifier_name])

        best_model = None
        best_f1_score = 0.0

        for params in params_grid:
            model = CLASSIFIERS[self.classifier_name](**params)
            predict_proba = cross_val_predict(model, x_train, y_train, cv=4, method="predict_proba")

            class_0_proba = predict_proba[y_train == 0][:, 0]
            class_1_proba = predict_proba[y_train == 1][:, 1]
            cutoff = np.mean((np.median(class_0_proba), np.median(class_1_proba)))
            y_predicted = np.array(predict_proba[:, 1] >= cutoff).astype(int)

            model_metrics = self._calculate_metrics(y_train, y_predicted)

            if model_metrics["f1_score"] >= best_f1_score:
                best_model = model
                best_f1_score = model_metrics["f1_score"]

        return best_model

    @staticmethod
    def _calculate_metrics(y_true, y_predicted) -> dict:
        return {
            "confusion_matrix": metrics.confusion_matrix(y_true, y_predicted).ravel(),
            "accuracy_score": metrics.accuracy_score(y_true, y_predicted),
            "precision_score": metrics.precision_score(y_true, y_predicted, zero_division=0.0),
            "recall_score": metrics.recall_score(y_true, y_predicted),
            "f1_score": metrics.f1_score(y_true, y_predicted),
            "roc_auc_score": metrics.roc_auc_score(y_true, y_predicted)
        }
