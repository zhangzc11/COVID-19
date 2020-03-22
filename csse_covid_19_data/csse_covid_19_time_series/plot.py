#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import optimize
import math
import datetime


countries = ["China", "Italy", "Spain", "Germany", "France", "US", "Outside-China", "World"]

data = []
x_title = []
with open('time_series_19-covid-Confirmed.csv') as f:
    lines = f.readlines()
    for line in lines:
        line_items = line.strip('\n').replace(", ", "-").split(',')
        if line_items[0] == "Province/State":
            x_title = line_items[4:]
            continue
        line_items_int_str = []
        for idx in range(1, len(line_items)):
            if line_items[idx] == "":
                continue
            if idx == 1:
                line_items_int_str.append(line_items[idx])
            elif idx > 3:
                line_items_int_str.append(int(line_items[idx]))
        data.append(line_items_int_str)
#print(x_title)
#print(data[11])

y_countries = np.zeros((len(countries), len(x_title)))
for idata in data:
    np_data = np.array(idata[1:])
    for ic in range(len(countries)):
        if idata[0] == countries[ic]:
            y_countries[ic] += np_data
        if countries[ic] == "World":
            y_countries[ic] += np_data
        if countries[ic] == "Outside-China" and idata[0] != "China":
            y_countries[ic] += np_data           
#print(y_countries[:3])


d0_threshold = 5000
y_countries_d0 = np.zeros((len(countries), len(x_title)))
idx_d0 = 0
for idate in range(len(y_countries[0])):
    if y_countries[0][idate] >= d0_threshold:
        idx_d0 = idate
        break
#print(idx_d0)
xmax_countries_d0 = np.zeros(len(countries))
for ic in range(0,len(countries)):
    idx_this = 0
    idx_d0_this = 0
    for idate in range(len(y_countries[ic])):
        if y_countries[ic][idate] < d0_threshold:
            idx_d0_this += 1
            continue
        if idx_this + idx_d0 >= len(y_countries[ic]):
            continue
        y_countries_d0[ic][idx_this + idx_d0] = y_countries[ic][idate]
        idx_this += 1
    for idate in range(idx_d0):
        y_countries_d0[ic][idate] = y_countries[ic][idx_d0_this - idx_d0 + idate]
    xmax_countries_d0[ic] = idx_this + idx_d0
#print(xmax_countries_d0)
#print(y_countries_d0[:3])

def fit_curve(x, N, k, b):
    return N / (1.0 + np.exp(-1.0*k*(x+b)))
    #return np.log(N/(1.0 + np.exp(-1.0*k*(x+b))))

plot_countries = ["China", "Italy", "Spain", "US", "Outside-China"]
colors = ['b', 'r', 'g', 'm', 'k', 'm','c']
mpl.rcParams['lines.markersize']=8
fig, ax = plt.subplots(1,1,figsize=(20,16))
for ic in range(len(countries)):
    if countries[ic] not in plot_countries:
        continue
    idx_plot = plot_countries.index(countries[ic])
    params =  [0,0,0]
    params_covariance = [0,0,0]
    label_legend=countries[ic]
    param0 = [y_countries[ic][-1], 0.21, -40.0]
    if countries[ic] != "China":
        param0[0] = 10.0*y_countries[ic][-1]
    if countries[ic] in ["Test"]:
        params, params_covariance = optimize.curve_fit(fit_curve, np.arange(len(x_title)-7), y_countries[ic][:-7], maxfev=10000, p0=param0)
    else:
        params, params_covariance = optimize.curve_fit(fit_curve, np.arange(len(x_title)), y_countries[ic], maxfev=10000, p0=param0)

    label_legend=countries[ic]+r'$, N = \frac{%.0f'%params[0]+'}{1+e^{-%.2f'%params[1]+'(t%.2f'%params[2]+')}}$'
    print(params)
    x_title_this = x_title.copy()
    
    for idx in range(14):
        s_pre = x_title_this[-1].split('/')
        d_pre = datetime.date(int('20'+s_pre[2]), int(s_pre[0]), int(s_pre[1]))
        d_this = d_pre+datetime.timedelta(days=1)
        s_this = str(d_this.month)+"/"+str(d_this.day)+'/20'
        x_title_this.append(s_this)
        
    ax.plot(x_title_this, fit_curve(np.arange(len(x_title)+14), params[0], params[1], params[2]), color=colors[idx_plot])
    
    
    if countries[ic] == "China":
        ax.scatter(x_title, y_countries[ic], label=label_legend, marker="o", color=colors[idx_plot])
    else:
        ax.scatter(x_title, y_countries[ic], label=label_legend, marker="v", color=colors[idx_plot])
        ax.scatter(x_title[:int(xmax_countries_d0[ic])], y_countries_d0[ic][:int(xmax_countries_d0[ic])], label=countries[ic]+", %d"%(len(x_title) - int(xmax_countries_d0[ic]))+" days shift", marker="o", color=colors[idx_plot])
        #ax.plot(x_title[:int(xmax_countries_d0[ic])], y_countries_d0[ic][:int(xmax_countries_d0[ic])], label=countries[ic]+", %d"%(len(x_title) - int(xmax_countries_d0[ic]))+" days shift", marker="o", color=colors[idx_plot])
        ax.plot(x_title[:int(xmax_countries_d0[ic])], fit_curve(np.arange(int(xmax_countries_d0[ic])), params[0], params[1], params[2]+(len(x_title) - int(xmax_countries_d0[ic]))), color=colors[idx_plot])

#ax.xaxis.set_major_locator(plt.MultipleLocator(7))
ax.yaxis.set_major_locator(plt.MaxNLocator(10))
ax.grid()    

every_nth = 7
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)
        
#plt.xticks(fontsize=23)
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=30)
plt.legend(loc="upper left", fontsize=28, framealpha=0.0, ncol=2, columnspacing=0)

plt.yscale("log")
plt.ylim(100, 100.0*y_countries_d0.max())
fig.savefig("plots/COVID19_logY_d"+str(len(x_title))+".pdf",bbox_inches='tight')
fig.savefig("plots/COVID19_logY_d"+str(len(x_title))+".png")

plt.yscale("linear")
plt.ylim(0, 1.4*y_countries_d0.max())
plt.xlim(0, len(x_title)+10)
fig.savefig("plots/COVID19_d"+str(len(x_title))+".pdf",bbox_inches='tight')
fig.savefig("plots/COVID19_d"+str(len(x_title))+".png")

