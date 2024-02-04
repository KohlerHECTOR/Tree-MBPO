## Only Continuous actions


Install scikit-learn and SB3

```pip3 install -r requirements.txt```

### Available Models are Decision Trees, best CV Trees, and MLPs
### Available Policy Optim Algos are SAC and TD3

Launch MBPO for 15 iterations on InvertedPendulum with best Cross validated Decision Trees as Model estimators and SAC as policy optim.
Results are saved in 'Experience_Results/pendul-cvtree-sac/':

```python3 experience.py InvertedPendulum-v4 cvtree sac 15 pendul-cvtree-sac```

Launch MBPO for 15 iterations on InvertedPendulum with 4x200 MLP as Model estimators and SAC as policy optim.
Results are saved in 'Experience_Results/pendul-mlp-sac/':

```python3 experience.py InvertedPendulum-v4 mlp sac 15 pendul-mlp-sac```

Save Plots of comparisons 'Experience_Results/Comparison-date-time/':

```python3 compare_experiences.py pendul-cvtree-sac pendul-mlp-sac```

Save Plots of results in 'Experience_Results/pendul-cvtree-sac/':

```python3 plot_experience.py pendul-cvtree-sac```

MBPO: https://arxiv.org/abs/1906.08253

![MBPO-structure](https://github.com/KohlerHECTOR/MBPO-Scikit-Stable/blob/main/mbpo_schematics_rdme/mbpo-structure.png?raw=true)
![MBPO-rollout](https://github.com/KohlerHECTOR/MBPO-Scikit-Stable/blob/main/mbpo_schematics_rdme/mbpo-rollout.png?raw=true)
