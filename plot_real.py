import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from constants import *
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

area = 'Italy'

with open('csv\\infected_prelock_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    infected_prelock = list(reader)
    infected_prelock = [item for sublist in infected_prelock for item in sublist]
    infected_prelock = [float(x) for x in infected_prelock]
    infected_prelock = np.array(infected_prelock)

with open('csv\\infected_postlock_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    infected_postlock = list(reader)
    infected_postlock = [item for sublist in infected_postlock for item in sublist]
    infected_postlock = [float(x) for x in infected_postlock]
    infected_postlock = np.array(infected_postlock)

with open('csv\\x_valid_prelock_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    x_valid_prelock = list(reader)
    x_valid_prelock = [item for sublist in x_valid_prelock for item in sublist]
    x_valid_prelock = [float(x) for x in x_valid_prelock]
    x_valid_prelock = np.array(x_valid_prelock)

with open('csv\\x_recovered_prelock_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    x_recovered_prelock = list(reader)
    x_recovered_prelock = [item for sublist in x_recovered_prelock for item in sublist]
    x_recovered_prelock = [float(x) for x in x_recovered_prelock]
    x_recovered_prelock = np.array(x_recovered_prelock)

with open('csv\\x_infected_prelock_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    x_infected_prelock = list(reader)
    x_infected_prelock = [item for sublist in x_infected_prelock for item in sublist]
    x_infected_prelock = [float(x) for x in x_infected_prelock]
    x_infected_prelock = np.array(x_infected_prelock)

with open('csv\\recovered_postlock_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    recovered_postlock = list(reader)
    recovered_postlock = [item for sublist in recovered_postlock for item in sublist]
    recovered_postlock = [float(x) for x in recovered_postlock]
    recovered_postlock = np.array(recovered_postlock)

with open('csv\\valid_recovered_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    valid_recovered = list(reader)
    valid_recovered = [item for sublist in valid_recovered for item in sublist]
    valid_recovered = [float(x) for x in valid_recovered]
    valid_recovered = np.array(valid_recovered)

with open('csv\\valid_infected_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    valid_infected = list(reader)
    valid_infected = [item for sublist in valid_infected for item in sublist]
    valid_infected = [float(x) for x in valid_infected]
    valid_infected = np.array(valid_infected)

with open('csv\\r_hat_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    r_hat = list(reader)
    r_hat = [item for sublist in r_hat for item in sublist]
    r_hat = [float(x) for x in r_hat]
    r_hat = np.array(r_hat)

with open('csv\\i_hat_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    i_hat = list(reader)
    i_hat = [item for sublist in i_hat for item in sublist]
    i_hat = [float(x) for x in i_hat]
    i_hat = np.array(i_hat)

with open('csv\\x_train_prelock_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    x_train_prelock = list(reader)
    x_train_prelock = [item for sublist in x_train_prelock for item in sublist]
    x_train_prelock = [float(x) for x in x_train_prelock]
    x_train_prelock = np.array(x_train_prelock)

with open('csv\\x_train_postlock_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    x_train_postlock = list(reader)
    x_train_postlock = [item for sublist in x_train_postlock for item in sublist]
    x_train_postlock = [float(x) for x in x_train_postlock]
    x_train_postlock = np.array(x_train_postlock)

with open('csv\\x_postlock_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    x_postlock = list(reader)
    x_postlock = [item for sublist in x_postlock for item in sublist]
    x_postlock = [float(x) for x in x_postlock]
    x_postlock = np.array(x_postlock)

with open('csv\\recovered_prelock_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    recovered_prelock = list(reader)
    recovered_prelock = [item for sublist in recovered_prelock for item in sublist]
    recovered_prelock = [float(x) for x in recovered_prelock]
    recovered_prelock = np.array(recovered_prelock)

plt.figure(figsize=(22, 8))

plt.subplot(1, 2, 1)
marker = 'o'
plt.scatter(x_train_prelock, infected_prelock, marker=marker, label='Training', color=green)
plt.scatter(x_valid_prelock, valid_infected, marker=marker, label='Validation', color=red)
plt.scatter(x_postlock, infected_postlock, marker=marker, label='Lockdown Ease', color=orange)
plt.plot(x_infected_prelock, i_hat, label='Infected - Predicted', color=blue)
plt.legend(loc='best', fontsize=legendsize)
plt.xlabel(r'$days$', fontsize=labelsize)
plt.ylabel(r'$I(t)$', fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
handles, labels = plt.gca().get_legend_handles_labels()

handles = [handles[0], handles[2], handles[1]]
labels = [labels[0], labels[2], labels[1]]

plt.subplot(1, 2, 2)
plt.scatter(x_train_prelock, recovered_prelock, marker=marker, label='Training', color=green)
plt.scatter(x_valid_prelock, valid_recovered, marker=marker, label='Validation', color=red)
plt.scatter(x_postlock, recovered_postlock, marker=marker, label='Lockdown Ease', color=orange)
plt.plot(x_recovered_prelock, r_hat, label='Recovered - Predicted', color=blue)

plt.legend(loc='best', fontsize=legendsize)
plt.xlabel('$days$', fontsize=labelsize)
plt.ylabel('$R(t)$', fontsize=labelsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
handles, labels = plt.gca().get_legend_handles_labels()

handles = [handles[0], handles[2], handles[1]]
labels = [labels[0], labels[2], labels[1]]

plt.tight_layout()
plt.show()