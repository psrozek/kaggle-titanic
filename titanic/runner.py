import click
import pandas as pd

TRAIN_DATASET_PATH = "../data/train.csv"
TEST_DATASET_PATH = "../data/train.csv"


def read_data():
    train_dataset = pd.read_csv(TRAIN_DATASET_PATH)
    test_dataset = pd.read_csv(TEST_DATASET_PATH)
    return train_dataset, test_dataset


@click.command()
@click.option("--experiment_configuration_path", type=str, required=True)
def run_titanic(experiment_configuration_path: str):
    train_dataset, test_dataset = read_data()
    ...


def main():
    # run_titanic()
    read_data()

if __name__ == '__main__':
    main()
