import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from real_data_countries import selected_countries_populations
from constants import *
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

area='Italy'

with open('csv\\s_hat_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    s_hat = list(reader)
    s_hat = [item for sublist in s_hat for item in sublist]
    s_hat = [float(x) for x in s_hat]
    s_hat = np.array(s_hat)

with open('csv\\i_hat_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    i_hat = list(reader)
    i_hat = [item for sublist in i_hat for item in sublist]
    i_hat = [float(x) for x in i_hat]
    i_hat = np.array(i_hat)

with open('csv\\r_hat_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    r_hat = list(reader)
    r_hat = [item for sublist in r_hat for item in sublist]
    r_hat = [float(x) for x in r_hat]
    r_hat = np.array(r_hat)


with open('csv\\p_hat_{}.csv'.format(area), newline='') as f:
    reader = csv.reader(f)
    p_hat = list(reader)
    p_hat = [item for sublist in p_hat for item in sublist]
    p_hat = [float(x) for x in p_hat]
    p_hat = np.array(p_hat)

# Convert to [0-1]
i_hat = i_hat / selected_countries_populations[area]
r_hat = r_hat / selected_countries_populations[area]

sum = s_hat + i_hat + r_hat + p_hat

t = np.array(range(len(sum))) / 5

def fmt(x, pos):
    f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    x = "${}$".format(f._formatSciNotation('%2.10e' % x))
    if '\\times' in x:
        x = x.replace('\\times', '\\cdot')
    return x

plt.figure(figsize=(6, 5))
# plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
plt.plot(t, p_hat, linewidth=3, label='$P$', color=blue, zorder=1)
plt.plot(t, sum, linewidth=3, label='$S + I + R + P$', color=red, zorder=2)
plt.ylim([0, 1.1])
plt.xlabel('$t$', fontsize=labelsize - 5)
plt.ylabel('Network solutions', fontsize=labelsize - 5)
plt.xticks([0, 10, 20], fontsize=ticksize - 3)
plt.yticks(fontsize=ticksize - 3)
plt.legend(loc='best', fontsize=labelsize - 15)
plt.gcf().subplots_adjust(left=0.35, bottom=0.25)
plt.savefig('sirp_hats.png')
plt.show()