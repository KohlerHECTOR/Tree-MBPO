import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from imblearn.under_sampling import RandomUnderSampler
from abc import ABC, abstractmethod


class TransitionModel:
    def __init__(self, model, model_kwargs):
        self.model = model(**model_kwargs)

    def fit(self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray):
        self.model.fit(np.concatenate((S, A), axis=1), Snext)

    def predict(self, s: np.ndarray, a: np.ndarray):
        return self.model.predict(np.concatenate((s, a)).reshape(1, -1))[0]


class RewardModel:
    def __init__(self, model, model_kwargs):
        self.model = model(**model_kwargs)

    def fit(self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray, R: np.ndarray):
        self.model.fit(np.concatenate((S, A, Snext), axis=1), R)

    def predict(self, s: np.ndarray, a: np.ndarray, snext: np.ndarray):
        return self.model.predict(np.concatenate((s, a, snext)).reshape(1, -1))[0]


class DoneModel:
    def __init__(self, model, model_kwargs, with_rus: bool = True):
        self.model = model(**model_kwargs)
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
        if self.rus and len(np.unique(Target_Transi)) > 1:
            Train_Transi, Target_Transi = self.rus.fit_resample(
                Train_Transi, Target_Transi
            )
        self.model.fit(Train_Transi, Target_Transi)

    def predict(self, s: np.ndarray, a: np.ndarray, r: np.ndarray, snext: np.ndarray):
        return self.model.predict(
            np.concatenate((s, a, np.array([r]), snext)).reshape(1, -1)
        )[0]


class TransitionTreeModel(TransitionModel):
    def __init__(self, max_leaf_nodes: int = 1024):
        super().__init__(
            model=DecisionTreeRegressor, model_kwargs={"max_leaf_nodes": max_leaf_nodes}
        )
    
    def fit(self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray):
        self.model.max_leaf_nodes = int(self.model.max_leaf_nodes * 1.2)
        self.model.fit(np.concatenate((S, A), axis=1), Snext)

class RewardTreeModel(RewardModel):
    def __init__(self, max_leaf_nodes: int = 1024):
        super().__init__(
            model=DecisionTreeRegressor, model_kwargs={"max_leaf_nodes": max_leaf_nodes}
        )
    
    def fit(self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray, R: np.ndarray):
        self.model.max_leaf_nodes = int(self.model.max_leaf_nodes * 1.2)
        self.model.fit(np.concatenate((S, A, Snext), axis=1), R)


class DoneTreeModel(DoneModel):
    def __init__(self, max_leaf_nodes: int = 1024):
        super().__init__(
            model=DecisionTreeClassifier, model_kwargs={"max_leaf_nodes": max_leaf_nodes}
        )
    
    def fit(
        self,
        S: np.ndarray,
        A: np.ndarray,
        R: np.ndarray,
        Snext: np.ndarray,
        Term: np.ndarray,
    ):
        self.model.max_leaf_nodes = int(self.model.max_leaf_nodes * 1.2)
        
        Train_Transi = np.concatenate((S, A, R.reshape(-1, 1), Snext), axis=1)
        Target_Transi = Term
        if self.rus and len(np.unique(Target_Transi)) > 1:
            Train_Transi, Target_Transi = self.rus.fit_resample(
                Train_Transi, Target_Transi
            )
        self.model.fit(Train_Transi, Target_Transi)

class TransitionMLPModel(TransitionModel):
    def __init__(self):
        super().__init__(model=MLPRegressor, model_kwargs={})
    def fit(self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray):
        self.model.fit(np.concatenate((S, A), axis=1), Snext)

class RewardMLPModel(RewardModel):
    def __init__(self):
        super().__init__(model=MLPRegressor, model_kwargs={})

    def fit(self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray, R: np.ndarray):
        self.model.fit(np.concatenate((S, A, Snext), axis=1), R)

class DoneMLPModel(DoneModel):
    def __init__(self):
        super().__init__(model=MLPClassifier, model_kwargs={})
    
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
        if self.rus and len(np.unique(Target_Transi)) > 1:
            Train_Transi, Target_Transi = self.rus.fit_resample(
                Train_Transi, Target_Transi
            )
        self.model.fit(Train_Transi, Target_Transi)
