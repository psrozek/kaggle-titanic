import factory
import pandas as pd
import pytest


class TitanicDataframeFactory(factory.DictFactory):
    @classmethod
    def build_df_batch(cls, size, **kwargs) -> pd.DataFrame:
        raw_rows = cls.build_batch(size=size, **kwargs)
        return cls._convert_to_df(raw_rows)

    @classmethod
    def _convert_to_df(cls, raw_rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(data=raw_rows)


@pytest.fixture
def passengers_factory() -> type[TitanicDataframeFactory]:
    class PassengersFactory(TitanicDataframeFactory):
        PassengerId = factory.Sequence(lambda n: n + 1)
        Survived = factory.Iterator([0, 1, 0])
        Pclass = factory.Iterator([3, 1, 2])
        Name = factory.Iterator(["First, John", "Second, Mary", "Third, John"])
        Sex = factory.Iterator(["male", "female", "male"])
        Age = factory.Iterator([22, 38, 35])
        SibSp = factory.Iterator([1, 1, 0])
        Parch = factory.Iterator([0, 2, 1])
        Ticket = factory.Iterator(["PP123", "54321", "A404"])
        Fare = factory.Iterator([7.25, 53.1, 16.7])
        Cabin = factory.Iterator(["C85", "E46", "nan"])
        Embarked = factory.Iterator(["S", "C", "Q"])


        @classmethod
        def _converted_rows_to_df(cls, raw_rows: list[dict]) -> pd.DataFrame:
            return pd.DataFrame(data=raw_rows).set_index("NetID")

    return PassengersFactory
