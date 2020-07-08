import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import os
import datetime
from models import SIRNetwork
from data_fitting import fit
from utils import get_syntethic_data, get_data_dict
from real_data_countries import countries_dict_prelock, selected_countries_populations
from training import train_bundle
from constants import ROOT_DIR, red, blue, green, ticksize, labelsize
from torch.utils.tensorboard import SummaryWriter
from shutil import rmtree
from tqdm import tqdm

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

if __name__ == '__main__':

    t_0 = 0
    t_final = 20

    # Compute the interval in which the equation parameters and the initial conditions should vary
    i_0_set = [0.2, 0.4]
    r_0_set = [0.1, 0.3]
    betas = [0., 0.4]
    gammas = [0.4, 0.7]
    # i_0_set = [0.4, 0.6]
    # r_0_set = [0.1, 0.3]
    # betas = [0.45, 0.65]
    # gammas = [0.05, 0.15]
    initial_conditions_set = []
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(i_0_set)
    initial_conditions_set.append(r_0_set)

    # Model parameters
    train_size = 2000
    decay = 0.0
    hack_trivial = False
    epochs = 3000
    lr = 8e-4

    # How many solutions I want to generate
    n_draws = 10

    # How many times I want to fit a single trajectory, getting the best result
    n_trials = 1

    fit_epochs = 1000

    # Init model
    sir = SIRNetwork(input=5, layers=4, hidden=50)

    model_name = 'i_0={}_r_0={}_betas={}_gammas={}.pt'.format(i_0_set, r_0_set,
                                                              betas,
                                                              gammas)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))
    except FileNotFoundError:
        # Train
        optimizer = torch.optim.Adam(sir.parameters(), lr=lr)
        writer = SummaryWriter(
            'runs/' + '{}'.format(model_name))
        sir, train_losses, run_time, optimizer = train_bundle(sir, initial_conditions_set, t_final=t_final,
                                                              epochs=epochs,
                                                              num_batches=10, hack_trivial=hack_trivial,
                                                              train_size=train_size, optimizer=optimizer,
                                                              decay=decay,
                                                              writer=writer, betas=betas,
                                                              gammas=gammas)
        # Save the model
        torch.save({'model_state_dict': sir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))

    # Load the model
    sir.load_state_dict(checkpoint['model_state_dict'])

    writer_dir = 'runs/' + 'real_{}'.format(model_name)

    # Check if the writer directory exists, if yes delete it and overwrite
    if os.path.isdir(writer_dir):
        rmtree(writer_dir)

    writer = SummaryWriter(writer_dir)

    mode = 'fake'

    if mode == 'real':
        time_unit = 0.25
        area = 'US'
        data_prelock = get_data_dict(area, data_dict=countries_dict_prelock, time_unit=time_unit, skip_every=0,
                                     cut_off=1.5e-3, populations=selected_countries_populations)

        # If I'm fitting real data, I only fit Infected.
        # I also know the initial condition of I, so I can force it.
        susceptible_weight = 0.
        infected_weight = 1.
        recovered_weight = 0.
        force_init = True
    else:
        # Synthetic data
        exact_i_0 = 0.25
        exact_r_0 = 0.15
        exact_beta = 0.2
        exact_gamma = 0.5
        # exact_i_0 = 0.5
        # exact_r_0 = 0.2
        # exact_beta = 0.55
        # exact_gamma = 0.1
        data_prelock = get_syntethic_data(sir, t_final=t_final, i_0=exact_i_0, r_0=exact_r_0, exact_beta=exact_beta,
                                          exact_gamma=exact_gamma,
                                          size=20)
        susceptible_weight = 1.
        infected_weight = 1.
        recovered_weight = 1.
        force_init = False

    validation_data = {}
    valid_times = []
    valid_infected = []
    valid_recovered = []
    train_val_split = 0.2

    if mode == 'real':
        # Generate validation set by taking the last time units
        max_key = max(data_prelock.keys())
        val_keys = list(data_prelock.keys())

        for k in val_keys:
            if train_val_split == 0.:
                break

            if k >= max_key * (1 - train_val_split):
                valid_times.append(k)
                valid_infected.append(data_prelock[k][1])
                valid_recovered.append(data_prelock[k][2])
                validation_data[k] = [data_prelock[k][0], data_prelock[k][1], data_prelock[k][2]]
                del data_prelock[k]

    else:
        # Generate validation set
        max_key = max(data_prelock.keys())
        step = int(1 / train_val_split)
        val_keys = list(data_prelock.keys())[1::step]

        for k in val_keys:
            if train_val_split == 0.:
                break

            valid_times.append(k)
            valid_infected.append(data_prelock[k][1])
            valid_recovered.append(data_prelock[k][2])
            validation_data[k] = [data_prelock[k][0], data_prelock[k][1], data_prelock[k][2]]
            del data_prelock[k]

    if mode == 'real':
        total_population = selected_countries_populations[area]
    else:
        total_population = 10e6

    optimal_set = []

    loss_mode = 'mse'
    n_batches = 4

    for i in tqdm(range(n_draws), desc='Fitting...'):
        exact_points_tmp = copy.deepcopy(data_prelock)
        min_loss = 1000

        # Inject some noise in the infected. Noise is gaussian noise with mean 0 and std=(sqrt(Infected(t)) * N_total ) / N_total
        for t, v in exact_points_tmp.items():
            scale = np.sqrt((v[1] * total_population)) / total_population
            noise = np.random.normal(loc=0, scale=scale)
            noisy_infected = v[1] + noise
            exact_points_tmp[t] = [1 - (noisy_infected + v[2]), noisy_infected, v[2]]

        # Fit n_trials time and take the best fitting
        for j in range(n_trials):
            # Search optimal params
            optimal_i_0, optimal_r_0, optimal_beta, optimal_gamma, min_val_loss, val_losses = fit(sir,
                                                                                    init_bundle=initial_conditions_set,
                                                                                    betas=betas,
                                                                                    gammas=gammas,
                                                                                    lr=1e-1,
                                                                                    known_points=data_prelock,
                                                                                    writer=writer,
                                                                                    loss_mode=loss_mode,
                                                                                    epochs=fit_epochs,
                                                                                    verbose=False,
                                                                                    n_batches=n_batches,
                                                                                    susceptible_weight=susceptible_weight,
                                                                                    recovered_weight=recovered_weight,
                                                                                    infected_weight=infected_weight,
                                                                                    force_init=force_init,
                                                                                    validation_data=validation_data)
            s_0 = 1 - (optimal_i_0 + optimal_r_0)

            optimal_s_0 = 1 - (optimal_i_0 + optimal_r_0)

            if min_val_loss <= min_loss:
                optimal_subset = [optimal_i_0, optimal_r_0, optimal_beta, optimal_gamma]
                min_loss = min_val_loss

        optimal_set.append(optimal_subset)

    # Collection of optimal solutions
    overall_infected = []
    overall_recovered = []

    # Collectio of optimal initial conditions and parameters
    optimal_betas = []
    optimal_gammas = []
    optimal_i_0s = []
    optimal_r_0s = []

    # Let's generate the solutions
    for set in optimal_set:
        single_infected_line = []
        single_recovered_line = []
        single_initial_conditions = [1 - (set[0] + set[1]), set[0], set[1]]

        for t in range(t_final):
            t_tensor = torch.Tensor([t]).reshape(-1, 1)
            t_tensor.requires_grad = True

            _, i_single, r_single = sir.parametric_solution(t_tensor, single_initial_conditions, beta=set[2],
                                                            gamma=set[3],
                                                            mode='bundle_total')
            single_infected_line.append(i_single.item())
            single_recovered_line.append(r_single.item())

        overall_infected.append(single_infected_line)
        overall_recovered.append(single_recovered_line)

        optimal_i_0s.append(set[0].item())
        optimal_r_0s.append(set[1].item())
        optimal_betas.append(set[2].item())
        optimal_gammas.append(set[3].item())

    exact_times = range(0, t_final)

    # Summarize the solutions with their means and std deviations

    i_0_mean = np.mean(optimal_i_0s) * total_population
    i_0_std = np.std(optimal_i_0s) * total_population

    r_0_mean = np.mean(optimal_r_0s) * total_population
    r_0_std = np.std(optimal_r_0s) * total_population

    betas_mean = np.mean(optimal_betas)
    betas_std = np.std(optimal_betas)

    gammas_mean = np.mean(optimal_gammas)
    gammas_std = np.std(optimal_gammas)

    infected_mean = np.mean(overall_infected, axis=0) * total_population
    infected_std = np.std(overall_infected, axis=0) * total_population

    recovered_mean = np.mean(overall_recovered, axis=0) * total_population
    recovered_std = np.std(overall_recovered, axis=0) * total_population

    noise_std = [2 * np.sqrt((v[1] * total_population)) for v in data_prelock.values()]

    known_infected_prelock = np.array([traj[1] for traj in list(data_prelock.values())]) * total_population
    known_recovered_prelock = np.array([traj[2] for traj in list(data_prelock.values())]) * total_population

    valid_infected = np.array(valid_infected) * total_population
    valid_recovered = np.array(valid_recovered) * total_population

    fig = plt.figure(figsize=(15, 5))
    marker = '.'

    x_infected = np.array(range(len(infected_mean)))
    x_recovered = np.array(range(len(recovered_mean)))

    x_train = np.array(list(data_prelock.keys()))
    x_valid = np.array(valid_times)

    if mode == 'real':
        x_infected = x_infected / time_unit
        x_recovered = x_recovered / time_unit
        x_train = x_train / time_unit
        x_valid = x_valid / time_unit

    title = 'Comparison between real data and predicted data\n' \
            'Estimated vs Real\n' \
            'I(0) = {:.6f} +- {:.3f} - {:.6f} \n R(0) = {:.6f}  +- {:.3f} -  {:.6f}\n' \
            'Beta = {:.6f} +- {:.3f} - {:.6f} \n Gamma = {:.6f}  +- {:.3f} - {:.6f}\n'.format(i_0_mean, i_0_std,
                                                                                              exact_i_0,
                                                                                              r_0_mean, r_0_std,
                                                                                              exact_r_0,
                                                                                              betas_mean, betas_std,
                                                                                              exact_beta,
                                                                                              gammas_mean, gammas_std,
                                                                                              exact_gamma)

    plt.figure(figsize=(8, 5))


    plt.title(title)
    ax1 = plt.gca()
    ax1.xaxis.set_tick_params(labelsize=ticksize)
    ax1.yaxis.set_tick_params(labelsize=ticksize)
    ax1.plot(x_infected, infected_mean, linewidth=3, label='Infected - Predicted', color=blue, zorder=1)
    ax1.fill_between(x=x_infected, y1=infected_mean + 2 * infected_std,
                     y2=infected_mean - 2 * infected_std, alpha=0.3, color=blue, zorder=2,)
    ax1.scatter(x_valid, valid_infected, label='Validation points', color=red, marker=marker, zorder=3)
    ax1.errorbar(x=x_train, y=known_infected_prelock, yerr=noise_std, label='Training points', color=green,
                 fmt=marker, zorder=4)
    ax1.legend(loc='best')

    ax1.set_xlabel('$t$', fontsize=labelsize - 5)
    ax1.set_ylabel('$I(t)$', fontsize=labelsize - 5)

    handles, labels = ax1.get_legend_handles_labels()
    handles = [handles[0], handles[2], handles[1]]
    labels = [labels[0], labels[2], labels[1]]

    ax1.legend(handles, labels, loc='best')

    plt.xticks(fontsize=ticksize - 5)
    plt.yticks(fontsize=ticksize - 5)
    plt.tight_layout()

    ts = datetime.datetime.now().timestamp()

    plt.savefig(ROOT_DIR + '/plots/sensitivity_val_{}.png'.format(ts))

    # plt.figure(figsize=(8, 5))
    # plt.title(title)
    # ax2 = plt.gca()
    # ax2.xaxis.set_tick_params(labelsize=ticksize)
    # ax2.yaxis.set_tick_params(labelsize=ticksize)
    # ax2.plot(x_recovered, recovered_mean, linewidth=3, label='Predicted', color=blue, zorder=1)
    # ax2.fill_between(x=x_recovered, y1=recovered_mean + 2 * recovered_std,
    #                  y2=recovered_mean - 2 * recovered_std, alpha=0.3, color=blue, zorder=2)
    # ax2.scatter(x=x_train, y=known_recovered_prelock, label='Training points',
    #             color=green, marker=marker, zorder=3)
    # ax2.scatter(x_valid, valid_recovered, label='Validation points', color=red, marker=marker, zorder=4)
    # ax2.legend(loc='best')
    # ax2.set_xlabel('$t$', fontsize=labelsize)
    # ax2.set_ylabel('$R(t)$', fontsize=labelsize)
    #
    # plt.xticks(fontsize=ticksize / 1.5)
    # plt.yticks(fontsize=ticksize / 1.5)
    # plt.tight_layout()

    plt.show()

    if n_draws == 1:
        print('Optimal parameters: Beta = {:.3f} | Gamma = {:.3f}'.format(optimal_subset[2].item(),
                                                                          optimal_subset[3].item()))

    print('Estimated Beta: {:.3f} +/- {:.3f} | Estimated Gamma: {:.3f} +- {:.3f}'.format(betas_mean, betas_std,
                                                                                         gammas_mean,
                                                                                         gammas_std))
