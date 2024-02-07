###### For Tree-Based-Exploration see: https://github.com/KohlerHECTOR/TREX-Tree-Reward-EXploration/tree/main
## Only Continuous actions


Install scikit-learn and SB3

```pip3 install -r requirements.txt```

![trees-mlp](https://github.com/KohlerHECTOR/MBPO-Scikit-Stable/blob/main/mbpo_schematics_rdme/evals.png?raw=true)

![trees-mlp-times](https://github.com/KohlerHECTOR/MBPO-Scikit-Stable/blob/main/mbpo_schematics_rdme/times.png?raw=true)

![trees](https://github.com/KohlerHECTOR/MBPO-Scikit-Stable/blob/main/mbpo_schematics_rdme/evals-gsteps.png?raw=true)


### Available Models are Decision Trees, best CV Trees, and MLPs
### Available Policy Optim Algos are SAC and TD3

Launch MBPO for 100 iterations on InvertedPendulum with Decision Trees as Model estimators and SAC as policy optim.
Results are saved in 'Experience_Results/pendul-tree-sac/':

```python3 experience.py InvertedPendulum-v4 tree sac 100 pendul-tree-sac```

Launch MBPO for 100 iterations on InvertedPendulum with 2x64 MLP as Model estimators and SAC as policy optim.
Results are saved in 'Experience_Results/pendul-mlp-sac/':

```python3 experience.py InvertedPendulum-v4 mlp sac 100 pendul-mlp-sac```

Save Plots of comparisons 'Experience_Results/Comparison-date-time/':

```python3 compare_experiences.py pendul-tree-sac pendul-mlp-sac```

Save Plots of results in 'Experience_Results/pendul-tree-sac/':

```python3 plot_experience.py pendul-tree-sac```

MBPO: https://arxiv.org/abs/1906.08253

![MBPO-structure](https://github.com/KohlerHECTOR/MBPO-Scikit-Stable/blob/main/mbpo_schematics_rdme/mbpo-structure.png?raw=true)
![MBPO-rollout](https://github.com/KohlerHECTOR/MBPO-Scikit-Stable/blob/main/mbpo_schematics_rdme/mbpo-rollout.png?raw=true)
