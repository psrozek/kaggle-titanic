from sklearn.model_selection import train_test_split

from titanic.model import Model


class Titanic:
    def __init__(self, experiment_configuration: dict):
        self._initialize_model(experiment_configuration["classifier"])

    def _initialize_model(self, classifier: str):
        self.model = Model(classifier)

    def preprocess_data(self, dataset):
        return dataset

    def run_experiment(self, train_dataset):
        train_dataset = self.preprocess_data(train_dataset)
        x_train, x_test, y_train, y_test = train_test_split(train_dataset, test_size=0.2)
        model = self.model.prepare_model(x_train, y_train)
