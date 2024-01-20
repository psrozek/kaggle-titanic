import click
import json
import pandas as pd

from titanic.titanic import Titanic

TRAIN_DATASET_PATH = "../data/train.csv"
TEST_DATASET_PATH = "../data/train.csv"


def read_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    test_dataset = pd.read_csv(TEST_DATASET_PATH)
    return train_dataset, test_dataset


def load_json_file(json_file_path: str) -> dict:
    with open(json_file_path) as file:
        return json.load(file)


@click.command()
@click.option("--experiment_configuration_path", type=str, required=True)
def run_titanic_pipeline(experiment_configuration_path: str):
    experiment_configuration = load_json_file(experiment_configuration_path)
    titanic_pipeline = Titanic(experiment_configuration)

    train_dataset, test_dataset = read_data()
    titanic_pipeline.run_experiment(train_dataset)

    # TODO: Return model and prepare here submission file


def main():
    # run_titanic_pipeline()
    read_data()


if __name__ == '__main__':
    main()
