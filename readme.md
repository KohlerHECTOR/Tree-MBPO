## Only Continuous actions


Install scikit-learn and SB3

```pip3 install -r requirements.txt```

### Available Models are Decision Tress and MLPs

Launch MBPO for 500 iterations on Hopper with Decision Trees as Model estimators.
Results are saved in 'MBPO-Tree-Model-Hopper/':

```python3 experience.py Hopper-v4 tree 500 MBPO-Tree-Model-Hopper```


Launch MBPO for 500 iterations on Hopper with MLPs as Model estimators.
Results are saved in 'MBPO-MLP-Model-Hopper/':

```python3 experience.py Hopper-v4 tree 50O MBPO-MLP-Model-Hopper```

Save Plots of results in 'MBPO-Tree-Model-Hopper/':
```python3 plot_experience.py MBPO-Tree-Model-Hopper```

MBPO: https://arxiv.org/abs/1906.08253

