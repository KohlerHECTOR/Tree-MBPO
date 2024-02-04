## Only Continuous actions


Install scikit-learn and SB3

```pip3 install -r requirements.txt```

### Available Models are Decision Trees and MLPs

Launch MBPO for 50 iterations on Hopper with Decision Trees as Model estimators and SAC as policy optim.
Results are saved in 'MBPO-Tree-Model-Hopper/':

```python3 experience.py Hopper-v4 tree 50 MBPO-Tree-Model-Hopper```


Launch MBPO for 50 iterations on Hopper with MLPs as Model estimators and SAC as policy optim.
Results are saved in 'MBPO-MLP-Model-Hopper/':

```python3 experience.py Hopper-v4 mlp 5O MBPO-MLP-Model-Hopper```

Save Plots of results in 'MBPO-Tree-Model-Hopper/':
```python3 plot_experience.py MBPO-Tree-Model-Hopper```

MBPO: https://arxiv.org/abs/1906.08253

![MBPO-structure](https://github.com/KohlerHECTOR/MBPO-Scikit-Stable/blob/main/mbpo_schematics_rdme/mbpo-structure.png?raw=true)
![MBPO-rollout](https://github.com/KohlerHECTOR/MBPO-Scikit-Stable/blob/main/mbpo_schematics_rdme/mbpo-rollout.png?raw=true)
![MBPO-mlp](https://github.com/KohlerHECTOR/MBPO-Scikit-Stable/blob/main/mbpo_schematics_rdme/times.png?raw=true)