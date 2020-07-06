import csv
from constants import *
import numpy as np

without_sm  = 'C:\\Users\\aless\\Desktop\\parameters_estimation_repo\\csv\\without_sm.csv'
with open(without_sm, newline='') as f:
    reader = csv.reader(f)
    no_sm = list(reader)[0]
    no_sm = np.array([float(x) for x in no_sm])

with_sm = 'C:\\Users\\aless\\Desktop\\parameters_estimation_repo\\csv\\with_sm.csv'
with open(with_sm, newline='') as f:
    reader = csv.reader(f)
    sm = list(reader)[0]
    sm = np.array([float(x) for x in sm])

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

plt.figure(figsize=(10, 5))
plt.yscale('log')
plt.xscale('log')
plt.minorticks_off()
plt.plot(no_sm, color=red, label='Linear activation')
plt.plot(sm, color=green, label='Softmax activation')
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.xlabel('$Epochs$', fontsize=legendsize)
plt.ylabel(r'$L$', fontsize=legendsize)
plt.legend(loc='best', fontsize=legendsize)
plt.show()