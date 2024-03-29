import matplotlib.pyplot as plt
import numpy as np
import sys

args = sys.argv

exp_name = args[1]
exp_name = "Experience_Results/" + exp_name
evals = np.loadtxt(exp_name + "/evals")
times = np.loadtxt(exp_name + "/times")


plt.plot(np.arange(len(evals)), evals, label="Avg Cum Reward MBPO")
plt.grid()
plt.legend()
plt.ylabel("Avg Cumulative Reward on Real Env")
plt.xlabel("Iter")
plt.title(exp_name)
plt.savefig(exp_name + "/evals.png")
plt.clf()
plt.plot(times, evals, label="Avg Cum Rew MBPO")
plt.grid()
plt.legend()
plt.ylabel("Avg Cumulative Reward on Real Env")
plt.xlabel("Second")
plt.title(exp_name)
plt.savefig(exp_name + "/times.png")
