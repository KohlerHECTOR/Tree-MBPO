import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler


class TransitionTreeModel:
    def __init__(self, max_leaf_nodes: int = 256):
        self.model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)

    def fit(self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray):
        self.model.fit(np.concatenate((S, A), axis=1), Snext)

    def predict(self, s: np.ndarray, a: np.ndarray):
        return self.model.predict(np.concatenate(s, a).reshape(-1, 1))[0]


class RewardTreeModel:
    def __init__(self, max_leaf_nodes: int = 256):
        self.model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes)

    def fit(self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray, R: np.ndarray):
        self.model.fit(np.concatenate((S, A, Snext), axis=1), R)

    def predict(self, s: np.ndarray, a: np.ndarray, snext: np.ndarray):
        return self.model.predict(np.concatenate(s, a, snext).reshape(-1, 1))


class DoneTreeModel:
    def __init__(self, max_leaf_nodes: int = 256, with_rus: bool = True):
        self.model = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
        if with_rus:
            self.rus = RandomUnderSampler()

    def fit(
        self,
        S: np.ndarray,
        A: np.ndarray,
        R: np.ndarray,
        Snext: np.ndarray,
        Term: np.ndarray,
    ):
        Train_Transi = np.concatenate((S, A, R.reshape(-1, 1), Snext), axis=1)
        Target_Transi = Term
        if self.rus:
            Train_Transi, Target_Transi = self.rus.fit_resample(
                Train_Transi, Target_Transi
            )
        self.model.fit(Train_Transi, Target_Transi)

    def predict(self, s: np.ndarray, a: np.ndarray, r: np.ndarray, snext: np.ndarray):
        return self.model.predict(
            np.concatenate(s, a, r.reshape(-1, 1), snext).reshape(-1, 1)
        )
