import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from constants import *
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

areas = ['Switzerland', 'Italy', 'Spain']


for i, area in enumerate(areas):

    with open('csv\\sensitivity_infected_prelock_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        infected_prelock = list(reader)
        infected_prelock = [item for sublist in infected_prelock for item in sublist]
        infected_prelock = [float(x) for x in infected_prelock]
        infected_prelock = np.array(infected_prelock)

    with open('csv\\sensitivity_infected_postlock_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        infected_postlock = list(reader)
        infected_postlock = [item for sublist in infected_postlock for item in sublist]
        infected_postlock = [float(x) for x in infected_postlock]
        infected_postlock = np.array(infected_postlock)

    with open('csv\\sensitivity_x_valid_prelock_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        x_valid_prelock = list(reader)
        x_valid_prelock = [item for sublist in x_valid_prelock for item in sublist]
        x_valid_prelock = [float(x) for x in x_valid_prelock]
        x_valid_prelock = np.array(x_valid_prelock)

    with open('csv\\sensitivity_x_recovered_prelock_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        x_recovered_prelock = list(reader)
        x_recovered_prelock = [item for sublist in x_recovered_prelock for item in sublist]
        x_recovered_prelock = [float(x) for x in x_recovered_prelock]
        x_recovered_prelock = np.array(x_recovered_prelock)

    with open('csv\\sensitivity_x_infected_prelock_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        x_infected_prelock = list(reader)
        x_infected_prelock = [item for sublist in x_infected_prelock for item in sublist]
        x_infected_prelock = [float(x) for x in x_infected_prelock]
        x_infected_prelock = np.array(x_infected_prelock)

    with open('csv\\sensitivity_recovered_postlock_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        recovered_postlock = list(reader)
        recovered_postlock = [item for sublist in recovered_postlock for item in sublist]
        recovered_postlock = [float(x) for x in recovered_postlock]
        recovered_postlock = np.array(recovered_postlock)

    with open('csv\\sensitivity_valid_recovered_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        valid_recovered = list(reader)
        valid_recovered = [item for sublist in valid_recovered for item in sublist]
        valid_recovered = [float(x) for x in valid_recovered]
        valid_recovered = np.array(valid_recovered)

    with open('csv\\sensitivity_valid_infected_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        valid_infected = list(reader)
        valid_infected = [item for sublist in valid_infected for item in sublist]
        valid_infected = [float(x) for x in valid_infected]
        valid_infected = np.array(valid_infected)

    with open('csv\\sensitivity_infected_mean_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        infected_mean = list(reader)
        infected_mean = [item for sublist in infected_mean for item in sublist]
        infected_mean = [float(x) for x in infected_mean]
        infected_mean = np.array(infected_mean)

    with open('csv\\sensitivity_infected_std_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        infected_std = list(reader)
        infected_std = [item for sublist in infected_std for item in sublist]
        infected_std = [float(x) for x in infected_std]
        infected_std = np.array(infected_std)

    with open('csv\\sensitivity_recovered_mean_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        recovered_mean = list(reader)
        recovered_mean = [item for sublist in recovered_mean for item in sublist]
        recovered_mean = [float(x) for x in recovered_mean]
        recovered_mean = np.array(recovered_mean)

    with open('csv\\sensitivity_recovered_std_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        recovered_std = list(reader)
        recovered_std = [item for sublist in recovered_std for item in sublist]
        recovered_std = [float(x) for x in recovered_std]
        recovered_std = np.array(recovered_std)

    with open('csv\\sensitivity_x_train_prelock_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        x_train_prelock = list(reader)
        x_train_prelock = [item for sublist in x_train_prelock for item in sublist]
        x_train_prelock = [float(x) for x in x_train_prelock]
        x_train_prelock = np.array(x_train_prelock)

    with open('csv\\sensitivity_x_postlock_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        x_postlock = list(reader)
        x_postlock = [item for sublist in x_postlock for item in sublist]
        x_postlock = [float(x) for x in x_postlock]
        x_postlock = np.array(x_postlock)

    with open('csv\\sensitivity_noise_std_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        noise_std = list(reader)
        noise_std = [item for sublist in noise_std for item in sublist]
        noise_std = [float(x) for x in noise_std]
        noise_std = np.array(noise_std)

    with open('csv\\sensitivity_recovered_prelock_{}.csv'.format(area), newline='') as f:
        reader = csv.reader(f)
        recovered_prelock = list(reader)
        recovered_prelock = [item for sublist in recovered_prelock for item in sublist]
        recovered_prelock = [float(x) for x in recovered_prelock]
        recovered_prelock = np.array(recovered_prelock)

    marker = '.'

    def fmt(x, pos):
        f=ticker.ScalarFormatter(useOffset=False, useMathText=True)
        x = "${}$".format(f._formatSciNotation('%1.10e' % x))
        if '\\times' in x:
            x = x.replace('\\times','\\cdot')
        return x

    plt.figure(figsize=(6,5))


    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(fmt))

    marker = '.'
    ax1 = plt.gca()
    ax1.xaxis.set_tick_params(labelsize=ticksize-3)
    ax1.yaxis.set_tick_params(labelsize=ticksize-3)
    ax1.plot(x_infected_prelock, infected_mean, label='Infected - Predicted', color=blue)
    ax1.errorbar(x=x_train_prelock, y=infected_prelock, yerr=noise_std, label='Training', color=green,
                 fmt=marker)
    ax1.scatter(x_valid_prelock, valid_infected, label='Validation', color=red, marker=marker)

    ax1.fill_between(x=x_infected_prelock, y1=infected_mean + 2 * infected_std,
                     y2=infected_mean - 2 * infected_std, alpha=0.3, color=blue)
    ax1.scatter(x_postlock, infected_postlock, marker=marker, label='Lockdown Ease', color=orange)


    handles, labels = ax1.get_legend_handles_labels()

    handles = [handles[0], handles[3], handles[1], handles[2]]
    labels = [labels[0], labels[3], labels[1], labels[2]]

    if area=='Switzerland':
        ax1.legend(handles, labels, loc='upper right', fontsize=labelsize - 15)
    else:
        ax1.legend(handles, labels, loc='lower left', fontsize=labelsize - 15)
        ax1.set_xlabel('$days$', fontsize=labelsize-5)
    ax1.set_ylabel('$I(t)$', fontsize=labelsize-5)
    plt.axvline(x=x_postlock[0], color='black', linestyle='--')

    plt.gcf().subplots_adjust(left=0.35, bottom=0.25)

    plt.savefig('sensitivity_{}_i'.format(area))


    plt.figure(figsize=(6,5))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(fmt))

    ax2 = plt.gca()
    ax2.xaxis.set_tick_params(labelsize=ticksize-3)
    ax2.yaxis.set_tick_params(labelsize=ticksize-3)
    ax2.scatter(x=x_train_prelock, y=recovered_prelock, label='Training',
                color=green, marker=marker)
    ax2.fill_between(x=x_recovered_prelock, y1=recovered_mean + 2 * recovered_std,
                     y2=recovered_mean - 2 * recovered_std, alpha=0.3, color=blue)

    ax2.scatter(x_valid_prelock, valid_recovered, label='Validation', color=red, marker=marker)
    ax2.scatter(x_postlock, recovered_postlock, marker=marker, label='Lockdown Ease', color=orange)
    ax2.plot(x_recovered_prelock, recovered_mean, label='Recovered - Predicted', color=blue)
    plt.axvline(x=x_postlock[0], color='black', linestyle='--')

    ax2.legend(handles, labels, loc='best', fontsize=legendsize-15)
    ax2.set_xlabel('$days$', fontsize=labelsize-5)
    ax2.set_ylabel('$R(t)$', fontsize=labelsize-5)

    handles, labels = ax2.get_legend_handles_labels()

    handles = [handles[0], handles[3], handles[1], handles[2]]
    labels = [labels[0], labels[3], labels[1], labels[2]]
    plt.gcf().subplots_adjust(left=0.35, bottom=0.25)
    plt.savefig('sensitivity_{}_r'.format(area))


    plt.show()