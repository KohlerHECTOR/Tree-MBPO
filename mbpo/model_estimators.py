import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from copy import deepcopy

class TransitionModel:
    def __init__(self, model, model_kwargs):
        self.model = model(**model_kwargs)

    def fit(self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray):
        self.model.fit(np.concatenate((S, A), axis=1), Snext)

    def predict(self, s: np.ndarray, a: np.ndarray):
        return self.model.predict(np.concatenate((s, a)).reshape(1, -1))[0]
    

class FullTransitionModel:
    def __init__(self, model, model_kwargs):
        self.model = model(**model_kwargs)

    def fit(self, S: np.ndarray, A: np.ndarray, R: np.ndarray, Snext: np.ndarray):
        self.model.fit(np.concatenate((S, A), axis=1), np.concatenate((R, Snext), axis=1))

    def predict(self, s: np.ndarray, a: np.ndarray):
        rsnext = self.model.predict(np.concatenate((s, a)).reshape(1, -1))[0]
        return rsnext[0], rsnext[1:]


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
    def __init__(self, max_depth: int = 10):
        super().__init__(
            model=DecisionTreeRegressor, model_kwargs={"max_depth":max_depth}
        )


class FullTransitionTreeModel(FullTransitionModel):
    def __init__(self, max_depth: int = 10):
        super().__init__(
            model=DecisionTreeRegressor, model_kwargs={"max_depth":max_depth}
        )

class FullTransitionTreeCVModel(FullTransitionModel):
    def __init__(self):
        super().__init__(model=GridSearchCV, model_kwargs={"estimator":DecisionTreeRegressor(), "param_grid":{'max_depth':range(5,15)}, "n_jobs":4})
        self.cv = deepcopy(self.model)

    def fit(self, S: np.ndarray, A: np.ndarray, R: np.ndarray, Snext: np.ndarray):
        self.cv.fit(np.concatenate((S, A), axis=1), np.concatenate((R, Snext), axis=1))
        self.be = deepcopy(self.cv.best_estimator_)
        self.cv = deepcopy(self.model)
    
    def predict(self, s: np.ndarray, a: np.ndarray):
        rsnext = self.be.predict(np.concatenate((s, a)).reshape(1, -1))[0]
        return rsnext[0], rsnext[1:]


class RewardTreeModel(RewardModel):
    def __init__(self, max_depth: int = 2048):
        super().__init__(
            model=DecisionTreeRegressor, model_kwargs={"max_depth":max_depth}
        )


class DoneTreeModel(DoneModel):
    def __init__(self, max_depth: int = 2048):
        super().__init__(
            model=DecisionTreeClassifier,
            model_kwargs={"max_depth":max_depth},
        )


class TransitionMLPModel(TransitionModel):
    def __init__(self):
        super().__init__(model=MLPRegressor, model_kwargs={"hidden_layer_sizes":[200, 200, 200, 200]})

class FullTransitionMLPModel(FullTransitionModel):
    def __init__(self):
        super().__init__(model=MLPRegressor, model_kwargs={"hidden_layer_sizes":[200, 200, 200, 200]})


class RewardMLPModel(RewardModel):
    def __init__(self):
        super().__init__(model=MLPRegressor, model_kwargs={"hidden_layer_sizes":[200, 200, 200, 200]})

    def fit(self, S: np.ndarray, A: np.ndarray, Snext: np.ndarray, R: np.ndarray):
        self.model.fit(np.concatenate((S, A, Snext), axis=1), R.ravel())


class DoneMLPModel(DoneModel):
    def __init__(self):
        super().__init__(model=MLPClassifier, model_kwargs={"hidden_layer_sizes":[200, 200, 200, 200]})

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
        self.model.fit(Train_Transi, Target_Transi.ravel())
