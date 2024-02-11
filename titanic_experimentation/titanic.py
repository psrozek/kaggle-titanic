import pandas as pd
from sklearn.model_selection import train_test_split

from titanic_experimentation.model import Model


class Titanic:
    def __init__(self, experiment_configuration: dict):
        self.predictions_column_name = experiment_configuration["predictions_column_name"]
        self.redundant_columns = experiment_configuration["redundant_columns"]
        self._initialize_model(experiment_configuration["classifier_name"])

    def _initialize_model(self, classifier_name: str):
        self.model = Model(classifier_name)

    def preprocess_data(self, dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        dataset = self._remove_redundant_columns(dataset)
        features = dataset.drop([self.predictions_column_name], axis=1)
        predictions = dataset[self.predictions_column_name]
        return features, predictions

    def _remove_redundant_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return dataset.drop(self.redundant_columns, axis=1)

    def run_experiment(self, train_dataset):
        features, predictions = self.preprocess_data(train_dataset)
        x_train, x_test, y_train, y_test = train_test_split(
            features, predictions, test_size=0.2, stratify=predictions
        )
        model = self.model.prepare_model(x_train, y_train)
        return model
