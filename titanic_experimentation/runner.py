import click
import json
import pandas as pd

from titanic_experimentation.titanic import Titanic

TRAIN_DATASET_PATH = "../data/train.csv"
TEST_DATASET_PATH = "../data/train.csv"
INDEX_COLUMN_NAME = "PassengerId"


def read_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH, index_col=INDEX_COLUMN_NAME)
    test_dataset = pd.read_csv(TEST_DATASET_PATH, index_col=INDEX_COLUMN_NAME)
    return train_dataset, test_dataset


def load_json_file(json_file_path: str) -> dict:
    with open(json_file_path) as file:
        return json.load(file)


# @click.command()
# @click.option("--experiment_configuration_path", type=str, required=True)
def run_titanic_pipeline(experiment_configuration_path: str):
    experiment_configuration = load_json_file(experiment_configuration_path)
    titanic_pipeline = Titanic(experiment_configuration)

    train_dataset, final_test_dataset = read_data()
    model = titanic_pipeline.run_experiment(train_dataset)

    # TODO: Return model and prepare here submission file


def main():
    run_titanic_pipeline("experiment_configuration.json")


if __name__ == '__main__':
    main()
