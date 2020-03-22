#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize
import math
import datetime

N_ = 1000000.0
k_ = 0.20
b_ = -50
n_days = 100

def fit_curve(x, N, k, b):
    return N / (1.0 + np.exp(-1.0*k*(x+b)))

x_ = np.arange(n_days)
y_ = np.zeros(n_days)

y_precise = fit_curve(x_, N_, k_, b_) 

for idx in range(n_days):
    y_[idx] = np.random.poisson(y_precise[idx], 1)[0]

print(y_precise)
print(y_)

colors = ['b', 'r', 'g', 'm', 'c', 'tab:orange','k']
mpl.rcParams['lines.markersize']=8
fig, ax = plt.subplots(1,1,figsize=(20,16))
testN = [20, 30, 40, 45, 50, 60]
ax.scatter(x_, y_, label='toy data'+r'$, N = \frac{%.0f'%N_+'}{1+e^{-%.2f'%k_+'(t%.2f'%b_+')}}$', marker="o", color='k')
for ic in range(len(testN)):
    params =  [0,0,0]
    params_covariance = [0,0,0]
    param0 = [y_[testN[ic]], 0.21, -40.0]
    params, params_covariance = optimize.curve_fit(fit_curve, x_[:testN[ic]], y_[:testN[ic]], maxfev=10000, p0=param0)
    label_legend='fit %d'%testN[ic]+' points'+r'$, N = \frac{%.0f'%params[0]+'}{1+e^{-%.2f'%params[1]+'(t%.2f'%params[2]+')}}$'
    print(params)
    ax.plot(x_, fit_curve(x_, params[0], params[1], params[2]), color=colors[ic], label=label_legend)

ax.grid()    
plt.xticks(fontsize=20)
plt.yticks(fontsize=30)
plt.legend(loc="upper left", fontsize=28, framealpha=0.0)

plt.yscale("log")
plt.ylim(100, 100.0*y_.max())
fig.savefig("plots/test_logY.pdf",bbox_inches='tight')
fig.savefig("plots/test_logY.png")

plt.yscale("linear")
plt.ylim(0, 1.4*y_.max())
fig.savefig("plots/test.pdf",bbox_inches='tight')
fig.savefig("plots/test.png")
