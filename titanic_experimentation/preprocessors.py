from sklearn.impute import SimpleImputer, KNNImputer


class Imputer:
    possible_imputers = ["mean", "median", "knn"]

    def __init__(self, imputer_type, n_neighbors_for_knn=None):
        self.n_neighbors_for_knn = n_neighbors_for_knn
        self.imputer = self._create_proper_imputer(imputer_type)

    def _create_proper_imputer(self, imputer_type: str):
        if imputer_type not in self.possible_imputers:
            print(f"Imputer {imputer_type} has not been implemented yet!")
        elif imputer_type == "knn":
            assert self.n_neighbors_for_knn, "Please define n_neighbors first!"
            return KNNImputer(n_neighbors=self.n_neighbors_for_knn)
        else:
            return SimpleImputer(strategy=imputer_type)

    def fit(self, data):
        return self.imputer.fit(data)

    def fit_transform(self, data):
        return self.imputer.fit_transform(data)

    def transform(self, data):
        return self.imputer.transform(data)
