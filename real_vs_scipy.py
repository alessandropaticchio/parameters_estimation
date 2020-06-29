import numpy as np
import matplotlib.pyplot as plt
from utils import SIRP_solution, get_data_dict
from real_data_countries import countries_dict_prelock, countries_dict_postlock, selected_countries_populations, selected_countries_rescaling

# Here I compare solution provided by Scipy with real data

t_final = 20
time_unit = 0.25
area = 'Italy'
scaled = True

multiplication_factor = 10

# Both data will have the shape of a multidimensional array [S(t), I(t), R(t)]
data_prelock = get_data_dict(area=area, data_dict=countries_dict_prelock, time_unit=time_unit, skip_every=0,
                             cut_off=1.5e-3, scaled=scaled, populations=selected_countries_populations, rescaling=selected_countries_rescaling, multiplication_factor=multiplication_factor)
data_postlock = get_data_dict(area=area, data_dict=countries_dict_postlock, time_unit=time_unit, skip_every=0,
                              cut_off=0., scaled=scaled, populations=selected_countries_populations, rescaling=selected_countries_rescaling, multiplication_factor=multiplication_factor)

infected_prelock = np.array([traj[1] for traj in list(data_prelock.values())])
infected_postlock = np.array([traj[1] for traj in list(data_postlock.values())])


recovered_prelock = np.array([traj[2] for traj in list(data_prelock.values())])
recovered_postlock = np.array([traj[2] for traj in list(data_postlock.values())])



# Total confirmed cases
confirmed_prelock = infected_prelock + recovered_prelock
confirmed_postlock = infected_postlock + recovered_postlock

x_postlock = np.array(list(data_postlock.keys())) + list(data_prelock.keys())[-1] + time_unit

# Scipy solver solution
beta = 0.5
gamma = 0.2

# Fix the initial conditions as the first element of the infected and recovered data
alpha = 0.964
i_0 = infected_prelock[0]
r_0 = recovered_prelock[0]

if scaled:
    p_0 = alpha * (1 - i_0 - r_0)
    s_0 = (1 - alpha) * (1 - i_0 - r_0)
else:
    p_0 = alpha * (selected_countries_populations[area] - i_0 - r_0)
    s_0 = (1 - alpha) * (selected_countries_populations[area] - i_0 - r_0)

# Solve the equation using Scipy
t = np.linspace(0, t_final*4, t_final*4)
s_p, i_p, r_p, p_p = SIRP_solution(t, s_0, i_0, r_0, p_0, beta, gamma)


plt.figure(figsize=(15, 5))
plt.title('Comparison of trend\n'
          'Real vs .Scipy')
plt.subplot(1, 3, 1)
plt.plot(t, i_p, label='Infected - Scipy')
plt.plot(list(data_prelock.keys()), infected_prelock, label='Infected - Real', color='red')
plt.plot(x_postlock, infected_postlock, label='Infected - Real - PostLockdown', color='orange')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('I(t)')

plt.subplot(1, 3, 2)
plt.plot(t, r_p, label='Recovered - Scipy')
plt.plot(list(data_prelock.keys()), recovered_prelock, label='Recovered - Real', color='red')
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('R(t)')

plt.show()