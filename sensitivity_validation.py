import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from models import SIRNetwork
from data_fitting import fit
from utils import get_syntethic_data, get_data_dict
from real_data_countries import countries_dict_prelock, countries_dict_postlock, selected_countries_populations, \
    selected_countries_rescaling
from constants import *
from torch.utils.tensorboard import SummaryWriter
from shutil import rmtree
from tqdm import tqdm
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

if __name__ == '__main__':
    t_0 = 0
    t_final = 20

    # Compute the interval in which the equation parameters and the initial conditions should vary
    i_0_set = [0.01, 0.02]
    r_0_set = [0.001, 0.006]
    p_0_set = [0.9, 0.97]
    betas = [0.7, 0.9]
    gammas = [0.15, 0.3]

    # Model parameters
    initial_conditions_set = []
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(i_0_set)
    initial_conditions_set.append(r_0_set)
    initial_conditions_set.append(p_0_set)

    # How many solutions I want to generate
    n_draws = 10

    # How many times I want to fit a single trajectory, getting the best result
    n_trials = 10

    fit_epochs = 100

    # Init model
    sirp = SIRNetwork(input=6, layers=4, hidden=50, output=4)

    model_name = 'i_0={}_r_0={}_p_0={}_betas={}_gammas={}.pt'.format(i_0_set, r_0_set, p_0_set,
                                                                     betas,
                                                                     gammas)

    checkpoint = torch.load(
        ROOT_DIR + '/models/SIRP_bundle_total/{}'.format(model_name))

    # Load the model
    sirp.load_state_dict(checkpoint['model_state_dict'])

    writer_dir = 'runs/' + 'real_{}'.format(model_name)

    # Check if the writer directory exists, if yes delete it and overwrite
    if os.path.isdir(writer_dir):
        rmtree(writer_dir)

    writer = SummaryWriter(writer_dir)

    mode = 'real'

    if mode == 'real':
        time_unit = 0.25
        area = 'Switzerland'
        multiplication_factor = 10
        data_prelock = get_data_dict(area, data_dict=countries_dict_prelock, time_unit=time_unit, skip_every=0,
                                     multiplication_factor=multiplication_factor,
                                     cut_off=1.5e-3, populations=selected_countries_populations,
                                     rescaling=selected_countries_rescaling)
        # Real data postlockdown
        data_postlock = get_data_dict(area, data_dict=countries_dict_postlock, time_unit=time_unit,
                                      skip_every=1, cut_off=0., populations=selected_countries_populations,
                                      multiplication_factor=multiplication_factor,
                                      rescaling=selected_countries_rescaling)
        susceptible_weight = 0.
        infected_weight = 1.
        recovered_weight = 1.
        passive_weight = 0.
        force_init = True
    else:
        # Synthetic data
        exact_i_0 = 0.015
        exact_r_0 = 0.008
        exact_p_0 = 0.9
        exact_beta = 0.4
        exact_gamma = 0.35
        data_prelock = get_syntethic_data(sirp, t_final=t_final, i_0=exact_i_0, r_0=exact_r_0, p_0=exact_p_0,
                                          exact_beta=exact_beta,
                                          exact_gamma=exact_gamma,
                                          size=10)
        susceptible_weight = 1.
        infected_weight = 1.
        recovered_weight = 1.
        passive_weight = 1.
        force_init = False

    # Generate validation set by taking the last time units
    validation_data = {}
    valid_times = []
    valid_infected = []
    valid_recovered = []
    train_val_split = 0.2
    max_key = max(data_prelock.keys())
    keys = list(data_prelock.keys())

    for k in keys:
        if train_val_split == 0.:
            break

        if k >= max_key * (1 - train_val_split):
            valid_times.append(k)
            if mode == 'real':
                valid_infected.append(data_prelock[k][0])
                valid_recovered.append(data_prelock[k][1])
                validation_data[k] = [data_prelock[k][0], data_prelock[k][1]]
            else:
                valid_infected.append(data_prelock[k][1])
                valid_recovered.append(data_prelock[k][2])
                validation_data[k] = [data_prelock[k][0], data_prelock[k][1], data_prelock[k][2], data_prelock[k][3]]
            del data_prelock[k]

    if mode == 'real':
        total_population = selected_countries_populations[area]
    else:
        total_population = 10e6

    optimal_set = []

    for i in tqdm(range(n_draws), desc='Fitting...'):
        exact_points_tmp = copy.deepcopy(data_prelock)
        min_loss = 1000

        # Inject some noise in the infected.
        # Noise is gaussian noise with mean 0 and std=(sqrt(Infected(t)) * N_total ) / N_total
        for t, v in exact_points_tmp.items():
            if mode == 'real':
                i = v[0]
            else:
                i = v[1]
            scale = np.sqrt((i * total_population)) / total_population
            noise = np.random.normal(loc=0, scale=scale)
            noisy_infected = i + noise
            if mode == 'real':
                exact_points_tmp[t] = [noisy_infected, v[1]]
            else:
                exact_points_tmp[t] = [v[0], noisy_infected, v[2], v[3]]

        # Fit n_trials time and take the best fitting
        for j in range(n_trials):
            # Search optimal params
            optimal_i_0, optimal_r_0, optimal_p_0, optimal_beta, optimal_gamma, val_losses = fit(sirp,
                                                                                                 init_bundle=initial_conditions_set,
                                                                                                 betas=betas,
                                                                                                 gammas=gammas,
                                                                                                 lr=1e-3, n_batches=10,
                                                                                                 known_points=exact_points_tmp,
                                                                                                 writer=writer,
                                                                                                 epochs=fit_epochs,
                                                                                                 mode=mode,
                                                                                                 validation_data=validation_data,
                                                                                                 susceptible_weight=susceptible_weight,
                                                                                                 recovered_weight=recovered_weight,
                                                                                                 infected_weight=infected_weight,
                                                                                                 passive_weight=passive_weight,
                                                                                                 force_init=force_init
                                                                                                 )

            optimal_s_0 = 1 - (optimal_i_0 + optimal_r_0 + optimal_p_0)

            if val_losses[-1] <= min_loss:
                optimal_subset = [optimal_i_0, optimal_r_0, optimal_p_0, optimal_beta, optimal_gamma]
                min_loss = val_losses[-1]

        optimal_set.append(optimal_subset)

    # Collection of optimal solutions
    overall_infected = []
    overall_recovered = []

    # Collectio of optimal initial conditions and parameters
    optimal_betas = []
    optimal_gammas = []
    optimal_i_0s = []
    optimal_r_0s = []
    optimal_p_0s = []

    # Let's generate the solutions
    for set in optimal_set:
        i_0 = set[0]
        r_0 = set[1]
        p_0 = set[2]
        beta = set[3]
        gamma = set[4]


        s_hat, i_hat, r_hat, p_hat, de_loss, t = sirp.solve(i_0=i_0, r_0=r_0, p_0=p_0,
                                                            beta=beta, gamma=gamma, t_0=0,
                                                            t_final=t_final)


        overall_infected.append(i_hat)
        overall_recovered.append(r_hat)

        optimal_i_0s.append(set[0].item())
        optimal_r_0s.append(set[1].item())
        optimal_p_0s.append(set[2].item())
        optimal_betas.append(set[3].item())
        optimal_gammas.append(set[4].item())

    exact_times = range(0, t_final)

    # Summarize the solutions with their means and std deviations

    i_0_mean = np.mean(optimal_i_0s) * total_population
    i_0_std = np.std(optimal_i_0s) * total_population

    r_0_mean = np.mean(optimal_r_0s) * total_population
    r_0_std = np.std(optimal_r_0s) * total_population

    p_0_mean = np.mean(optimal_p_0s) * total_population
    p_0_std = np.std(optimal_p_0s) * total_population

    betas_mean = np.mean(optimal_betas)
    betas_std = np.std(optimal_betas)

    gammas_mean = np.mean(optimal_gammas)
    gammas_std = np.std(optimal_gammas)

    infected_mean = np.mean(overall_infected, axis=0) * total_population
    infected_std = np.std(overall_infected, axis=0) * total_population

    recovered_mean = np.mean(overall_recovered, axis=0) * total_population
    recovered_std = np.std(overall_recovered, axis=0) * total_population

    noise_std = [2 * np.sqrt((v[0] * total_population)) for v in data_prelock.values()]

    x_infected_prelock = t
    x_recovered_prelock = t

    x_train_prelock = np.array(list(data_prelock.keys()))
    x_valid_prelock = np.array(valid_times)

    if mode == 'real':
        infected_prelock = [traj[0] for traj in list(data_prelock.values())]
        recovered_prelock = [traj[1] for traj in list(data_prelock.values())]
        infected_postlock = [traj[0] for traj in list(data_postlock.values())]
        recovered_postlock = [traj[1] for traj in list(data_postlock.values())]
        x_infected_prelock = x_infected_prelock / time_unit
        x_recovered_prelock = x_recovered_prelock / time_unit
        x_train_prelock = x_train_prelock / time_unit
        x_valid_prelock = x_valid_prelock / time_unit
        infected_postlock = np.array(infected_postlock) * total_population
        recovered_postlock = np.array(recovered_postlock) * total_population
        exact_i_0, exact_r_0, exact_p_0, exact_beta, exact_gamma = [0.] * 5
        # Shift after the lockdown
        x_postlock = (np.array(list(data_postlock.keys()))) / time_unit + x_valid_prelock[-1] + 1
    else:
        infected_prelock = [traj[1] for traj in list(data_prelock.values())]
        recovered_prelock = [traj[2] for traj in list(data_prelock.values())]

    valid_infected = np.array(valid_infected) * total_population
    valid_recovered = np.array(valid_recovered) * total_population
    infected_prelock = np.array(infected_prelock) * total_population
    recovered_prelock = np.array(recovered_prelock) * total_population

    title = 'Comparison between real data and predicted data\n' \
            'Estimated vs Real\n' \
            'I(0) = {:.6f} +- {:.3f} - {:.6f} \nR(0) = {:.6f}  +- {:.3f} -  {:.6f}\nP(0) = {:.6f}  +- {:.3f} -  {:.6f}\n' \
            'Beta = {:.6f} +- {:.3f} - {:.6f} \nGamma = {:.6f}  +- {:.3f} - {:.6f}\n'.format(i_0_mean, i_0_std,
                                                                                             exact_i_0,
                                                                                             r_0_mean, r_0_std,
                                                                                             exact_r_0,
                                                                                             p_0_mean, p_0_std,
                                                                                             exact_p_0,
                                                                                             betas_mean, betas_std,
                                                                                             exact_beta,
                                                                                             gammas_mean, gammas_std,
                                                                                             exact_gamma)

    plt.figure(figsize=(22, 8))
    plt.subplot(1, 2, 1)
    marker = 'o'
    print(title)
    ax1 = plt.gca()
    ax1.xaxis.set_tick_params(labelsize=ticksize)
    ax1.yaxis.set_tick_params(labelsize=ticksize)
    ax1.errorbar(x=x_train_prelock, y=infected_prelock, yerr=noise_std, label='Training', color=green,
                 fmt=marker)
    ax1.fill_between(x=x_infected_prelock, y1=infected_mean + 2 * infected_std,
                     y2=infected_mean - 2 * infected_std, alpha=0.3, color=blue)
    ax1.scatter(x_valid_prelock, valid_infected, label='Validation', color=red, marker=marker)
    if mode == 'real':
        ax1.scatter(x_postlock, infected_postlock, marker=marker, label='Lockdown Ease', color=orange)
    ax1.plot(x_infected_prelock, infected_mean, label='Infected - Predicted', color=blue)

    ax1.legend(loc='best', fontsize=legendsize)
    ax1.set_xlabel('t (days)', fontsize=labelsize)
    ax1.set_ylabel('I(t)', fontsize=labelsize)

    plt.tight_layout()

    plt.subplot(1, 2, 2)
    ax2 = plt.gca()
    ax2.xaxis.set_tick_params(labelsize=ticksize)
    ax2.yaxis.set_tick_params(labelsize=ticksize)
    ax2.scatter(x=x_train_prelock, y=recovered_prelock, label='Training',
                color=green, marker=marker)
    ax2.fill_between(x=x_recovered_prelock, y1=recovered_mean + 2 * recovered_std,
                     y2=recovered_mean - 2 * recovered_std, alpha=0.3, color=blue)

    ax2.scatter(x_valid_prelock, valid_recovered, label='Validation', color=red, marker=marker)
    if mode == 'real':
        ax2.scatter(x_postlock, recovered_postlock, marker=marker, label='Lockdown Ease', color=orange)
    ax2.plot(x_recovered_prelock, recovered_mean, label='Recovered - Predicted', color=blue)

    ax2.legend(loc='best', fontsize=legendsize)
    ax2.set_xlabel('t (days)', fontsize=labelsize)
    ax2.set_ylabel('R(t)', fontsize=labelsize)

    plt.tight_layout()

    plt.show()

    if n_draws == 1:
        print('Optimal parameters: Beta = {:.3f} | Gamma = {:.3f}'.format(optimal_subset[2].item(),
                                                                          optimal_subset[3].item()))

    print('Estimated Beta: {:.3f} +/- {:.3f} | Estimated Gamma: {:.3f} +- {:.3f}'.format(betas_mean, betas_std,
                                                                                         gammas_mean,
                                                                                         gammas_std))
