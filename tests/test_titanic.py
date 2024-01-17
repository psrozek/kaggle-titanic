def test_passengers_factory(passengers_factory):
    assert passengers_factory.create() == {
        "PassengerId": 1,
        "Survived": 0,
        "Pclass": 3,
        "Name": "First, John",
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "PP123",
        "Fare": 7.25,
        "Cabin": "C85",
        "Embarked": "S",
    }
    assert passengers_factory.create() == {
        "PassengerId": 2,
        "Survived": 1,
        "Pclass": 1,
        "Name": "Second, Mary",
        "Sex": "female",
        "Age": 38,
        "SibSp": 1,
        "Parch": 2,
        "Ticket": "54321",
        "Fare": 53.1,
        "Cabin": "E46",
        "Embarked": "C",
    }