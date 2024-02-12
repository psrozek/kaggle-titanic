from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, ParameterGrid
from sklearn.ensemble import GradientBoostingClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

ALGORITHM_HYPERPARAMETERS = {
    "xgboost": {
        "objective": ["binary:logistic"],
        "booster": ["gbtree"],
        "learning_rate": [0.01, 0.1],
        "n_estimators": [25, 50],
        "max_depth": [2, 5],
        "min_child_weight": [2, 5],
        "subsample": [0.75, 1],
        "scale_pos_weight": [1.0, 1.2]
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
        self.cutoff = None

    def prepare_model(self, x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Any, dict, dict]:
        params_grid = ParameterGrid(ALGORITHM_HYPERPARAMETERS[self.classifier_name])

        best_model = None
        best_model_params = None
        best_model_metrics = None
        best_cutoff = None
        best_f1_score = 0.0

        for model_params in tqdm(params_grid):
            model = CLASSIFIERS[self.classifier_name](**model_params)
            predicted_proba = cross_val_predict(model, x_train, y_train, cv=4, method="predict_proba")
            model_metrics, cutoff = self.get_model_metrics(predicted_proba, y_train)

            if model_metrics["f1_score"] >= best_f1_score:
                best_cutoff = cutoff
                best_model = model
                best_model_params = model_params
                best_model_metrics = model_metrics
                best_auc_score = model_metrics["roc_auc_score"]
                best_f1_score = model_metrics["f1_score"]

        self.cutoff = best_cutoff
        return best_model, best_model_params, best_model_metrics

    def get_model_metrics(self, predicted_proba: np.ndarray, y_true: pd.Series) -> tuple[dict, float]:
        cutoff = self.cutoff
        if not self.cutoff:
            class_0_proba = predicted_proba[y_true == 0][:, 0]
            class_1_proba = predicted_proba[y_true == 1][:, 1]
            cutoff = np.mean((np.median(class_0_proba), np.median(class_1_proba)))

        y_predicted = np.array(predicted_proba[:, 1] >= cutoff).astype(int)
        return self._calculate_metrics(y_true, y_predicted), cutoff

    @staticmethod
    def retrain_model_on_whole_dataset(model: Any, x: pd.DataFrame, y: pd.Series) -> Any:
        return model.fit(x, y)

    @staticmethod
    def _calculate_metrics(y_true: pd.Series, y_predicted: np.ndarray) -> dict:
        return {
            "confusion_matrix": metrics.confusion_matrix(y_true, y_predicted).ravel(),
            "accuracy_score": metrics.accuracy_score(y_true, y_predicted),
            "precision_score": metrics.precision_score(y_true, y_predicted, zero_division=0.0),
            "recall_score": metrics.recall_score(y_true, y_predicted),
            "f1_score": metrics.f1_score(y_true, y_predicted),
            "roc_auc_score": metrics.roc_auc_score(y_true, y_predicted)
        }
