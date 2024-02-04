## Only Continuous actions


Install scikit-learn and SB3

```pip3 install -r requirements.txt```

### Available Models are Decision Trees and MLPs
### Available Policy Optim Algos are SAC and TD3

Launch MBPO for 20 iterations on Hopper with Decision Trees as Model estimators and SAC as policy optim.
Results are saved in 'Experience_Results/hopper-tree-sac/':

```python3 experience.py Hopper-v4 tree sac 20 hopper-tree-sac```


Launch MBPO for 20 iterations on Hopper with MLPs as Model estimators and SAC as policy optim.
Results are saved in 'Experience_Results/hopper-mlp-sac/':

```python3 experience.py Hopper-v4 mlp sac 2O hopper-mlp-sac```

Save Plots of comparisons 'Experience_Results/Comparison-date-time/':

```python3 compare_experiences.py hopper-tree-sac hopper-mlp-sac```

![MBPO-mlp](https://github.com/KohlerHECTOR/MBPO-Scikit-Stable/blob/main/mbpo_schematics_rdme/times-tree-mlp.png?raw=true)

Save Plots of results in 'Experience_Results/hopper-tree-sac/':

```python3 plot_experience.py hopper-tree-sac```

MBPO: https://arxiv.org/abs/1906.08253

![MBPO-structure](https://github.com/KohlerHECTOR/MBPO-Scikit-Stable/blob/main/mbpo_schematics_rdme/mbpo-structure.png?raw=true)
![MBPO-rollout](https://github.com/KohlerHECTOR/MBPO-Scikit-Stable/blob/main/mbpo_schematics_rdme/mbpo-rollout.png?raw=true)
