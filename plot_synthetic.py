import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from constants import *
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

i_0_sets = [[0.2, 0.4], [0.4, 0.6]]


for i_0_set in i_0_sets:

    with open('csv\\x_infected_{}.csv'.format(i_0_set), newline='') as f:
        reader = csv.reader(f)
        x_infected = list(reader)
        x_infected = [item for sublist in x_infected for item in sublist]
        x_infected = [float(x) for x in x_infected]
        x_infected = np.array(x_infected)

    with open('csv\\infected_mean_{}.csv'.format(i_0_set), newline='') as f:
        reader = csv.reader(f)
        infected_mean = list(reader)
        infected_mean = [item for sublist in infected_mean for item in sublist]
        infected_mean = [float(x) for x in infected_mean]
        infected_mean = np.array(infected_mean)

    with open('csv\\infected_std_{}.csv'.format(i_0_set), newline='') as f:
        reader = csv.reader(f)
        infected_std = list(reader)
        infected_std = [item for sublist in infected_std for item in sublist]
        infected_std = [float(x) for x in infected_std]
        infected_std = np.array(infected_std)

    with open('csv\\x_valid_{}.csv'.format(i_0_set), newline='') as f:
        reader = csv.reader(f)
        x_valid = list(reader)
        x_valid = [item for sublist in x_valid for item in sublist]
        x_valid = [float(x) for x in x_valid]
        x_valid = np.array(x_valid)

    with open('csv\\x_train_{}.csv'.format(i_0_set), newline='') as f:
        reader = csv.reader(f)
        x_train = list(reader)
        x_train = [item for sublist in x_train for item in sublist]
        x_train = [float(x) for x in x_train]
        x_train = np.array(x_train)

    with open('csv\\known_infected_prelock_{}.csv'.format(i_0_set), newline='') as f:
        reader = csv.reader(f)
        known_infected_prelock = list(reader)
        known_infected_prelock = [item for sublist in known_infected_prelock for item in sublist]
        known_infected_prelock = [float(x) for x in known_infected_prelock]
        known_infected_prelock = np.array(known_infected_prelock)

    with open('csv\\noise_std_{}.csv'.format(i_0_set), newline='') as f:
        reader = csv.reader(f)
        noise_std = list(reader)
        noise_std = [item for sublist in noise_std for item in sublist]
        noise_std = [float(x) for x in noise_std]
        noise_std = np.array(noise_std)

    with open('csv\\valid_infected_{}.csv'.format(i_0_set), newline='') as f:
        reader = csv.reader(f)
        valid_infected = list(reader)
        valid_infected = [item for sublist in valid_infected for item in sublist]
        valid_infected = [float(x) for x in valid_infected]
        valid_infected = np.array(valid_infected)

    marker = '.'
    plt.figure(figsize=(6, 5))

    ax1 = plt.gca()
    ax1.xaxis.set_tick_params(labelsize=ticksize)
    ax1.yaxis.set_tick_params(labelsize=ticksize)


    def fmt(x, pos):
        f=ticker.ScalarFormatter(useOffset=False, useMathText=True)
        x = "${}$".format(f._formatSciNotation('%1.10e' % x))
        if '\\times' in x:
            x = x.replace('\\times','\\cdot')
        return x

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(fmt))

    ax1.plot(x_infected, infected_mean, linewidth=3, label='Infected - Predicted', color=blue, zorder=1)
    ax1.fill_between(x=x_infected, y1=infected_mean + 2 * infected_std,
                     y2=infected_mean - 2 * infected_std, alpha=0.3, color=blue, zorder=2,)
    ax1.scatter(x_valid, valid_infected, label='Validation points', color=red, marker=marker, zorder=3)
    ax1.errorbar(x=x_train, y=known_infected_prelock, yerr=noise_std, label='Training points', color=green,
                 fmt=marker, zorder=4)


    ax1.set_xlabel('$t$', fontsize=labelsize - 3)
    ax1.set_ylabel('$I(t)$', fontsize=labelsize - 3)

    handles, labels = ax1.get_legend_handles_labels()
    handles = [handles[0], handles[2], handles[1]]
    labels = [labels[0], labels[2], labels[1]]

    ax1.legend(handles, labels, loc='best', fontsize=labelsize - 10)

    plt.xticks(fontsize=ticksize - 3)
    plt.yticks(fontsize=ticksize - 3)
    plt.tight_layout()

    plt.show()