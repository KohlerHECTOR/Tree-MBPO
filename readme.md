Install scikit-learn and SB3

```pip3 install -r requirements.txt```

### Available Models are Decision Tress and MLPs

Launch MBPO for 10 iterations on MountainCar with Decision Trees as Model estimators.
Results are saved in 'MBPO-Tree-Model-MountainCar/':

```python3 experience.py MountainCarContinuous-v0 tree 10 MBPO-Tree-Model-MountainCar```


Launch MBPO for 10 iterations on MountainCar with MLPs as Model estimators.
Results are saved in 'MBPO-MLP-Model-MountainCar/':

```python3 experience.py MountainCarContinuous-v0 tree 10 MBPO-MLP-Model-MountainCar```

Save Plots of results in 'MBPO-Tree-Model-MountainCar/':
```python3 plot_experience.py MBPO-Tree-Model-MountainCar```

MBPO: https://arxiv.org/abs/1906.08253

