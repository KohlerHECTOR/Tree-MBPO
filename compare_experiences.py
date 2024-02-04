import matplotlib.pyplot as plt
import sys
import numpy as np
import time
import os

args = sys.argv[1:]

timestr = time.strftime("%Y%m%d-%H%M%S")
compare_folder = "Experience_Results/Comparison-"+timestr
os.makedirs(compare_folder, exist_ok=True)


### Evals
for exp_name in args:
    evals = np.loadtxt("Experience_Results/" + exp_name + "/evals")
    plt.plot(np.arange(len(evals)), evals, label=exp_name)
plt.ylabel("Avg Cumulative Reward on Real Env")
plt.xlabel("Iter")
plt.grid()
plt.legend()
plt.savefig(compare_folder + "/evals.png")
plt.clf()
### Times
for exp_name in args:
    evals = np.loadtxt("Experience_Results/" + exp_name + "/evals")
    times = np.loadtxt("Experience_Results/" + exp_name + "/times")
    plt.plot(times, evals, label=exp_name)
plt.ylabel("Avg Cumulative Reward on Real Env")
plt.xlabel("Second")
plt.grid()
plt.legend()
plt.savefig(compare_folder + "/times.png")