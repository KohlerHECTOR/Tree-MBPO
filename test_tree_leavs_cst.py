from sklearn.tree import DecisionTreeRegressor
from mbpo.real_data_collection import init_rng_data
import gymnasium as  gym
from mbpo import FullTransitionTreeModel
import numpy as np
dict_seen_leaves = {}
clf1 = FullTransitionTreeModel(max_leaf_nodes=4096)

env = gym.make("InvertedPendulum-v4")

S, A, R, Snext, Term = init_rng_data(env, 10_000)
clf1.fit(S, A, R, Snext)
leaves1 = clf1.model.predict(np.concatenate((S, A), axis = 1))
dif_leaves1 = np.unique(leaves1, axis=0)

clf2 = FullTransitionTreeModel(max_leaf_nodes=4096)
Sn, An, Rn, Snextn, Termn = init_rng_data(env, 10_000)

S = np.concatenate((S, Sn), axis = 1)
A = np.concatenate((A, An), axis = 1)
R = np.concatenate((R, Rn), axis = 1)
Snext = np.concatenate((Snext, Snextn), axis = 1)
clf2.fit(S, A, R, Snext)

leaves2 = clf2.model.predict(np.concatenate((S, A), axis = 1))
dif_leaves2 = np.unique(leaves2, axis=0)


leaves_tot = np.concatenate((dif_leaves1, dif_leaves2), axis=1)
print(len(np.unique(leaves_tot, axis=0)), len(dif_leaves1), len(dif_leaves2))


