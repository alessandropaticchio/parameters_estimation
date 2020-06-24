from models import SIRNetwork
from data_fitting import fit
from utils import get_syntethic_data, get_data_dict
from real_data_countries import countries_dict_prelock, countries_dict_postlock, selected_countries_populations, \
    selected_countries_rescaling
from training import train_bundle
from torch.utils.tensorboard import SummaryWriter
from shutil import rmtree
from losses import sir_loss
from torch.utils.data import DataLoader
from constants import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import os

if __name__ == '__main__':
    # If mode == real, it will fit real data, otherwise synthetic data.
    # Later the data to fit are specified
    mode = 'real'

    t_0 = 0
    t_final = 20

    # The interval in which the equation parameters and the initial conditions should vary
    e_0_set = [0.08, 0.1]
    i_0_set = [0.01, 0.2]
    r_0_set = [0., 0.001]
    d_0_set = [0., 0.001]
    betas = [0.004, 0.01]
    gammas = [0.15, 0.25]
    lams = [0.08, 0.15]
    deltas = [0.15, 0.25]

    # Model parameters
    initial_conditions_set = []
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(e_0_set)
    initial_conditions_set.append(i_0_set)
    initial_conditions_set.append(r_0_set)
    initial_conditions_set.append(d_0_set)

    # How many times I want to fit the trajectory, getting the best result
    n_trials = 10

    # Model parameters
    train_size = 2000
    decay = 1e-4
    hack_trivial = False
    epochs = 3000
    lr = 8e-4

    # Init model
    seird = SIRNetwork(input=9, layers=4, hidden=50, output=5)

    model_name = 'e_0={}_i_0={}_r_0={}_d_0={}_betas={}_gammas={}_lams={}_deltas={}.pt'.format(e_0_set, i_0_set, r_0_set,
                                                                                              d_0_set,
                                                                                              betas,
                                                                                              gammas, lams, deltas)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SEIRD_bundle_total/{}'.format(model_name))
    except FileNotFoundError:
        # Train
        optimizer = torch.optim.Adam(seird.parameters(), lr=lr)
        writer = SummaryWriter('runs/{}'.format(model_name))
        seird, train_losses, run_time, optimizer = train_bundle(seird, initial_conditions_set, t_final=t_final,
                                                                epochs=epochs, model_name=model_name,
                                                                num_batches=10, hack_trivial=hack_trivial,
                                                                train_size=train_size, optimizer=optimizer,
                                                                decay=decay,
                                                                writer=writer, betas=betas, gammas=gammas, lams=lams,
                                                                deltas=deltas)
        # Save the model
        torch.save({'model_state_dict': seird.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SEIRD_bundle_total/{}'.format(model_name))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SEIRD_bundle_total/{}'.format(model_name))

    # Load the model
    seird.load_state_dict(checkpoint['model_state_dict'])

    if mode == 'real':
        area = 'Italy'
        time_unit = 0.25
        cut_off = 1.5e-3
        multiplication_factor = 10
        # Real data prelockdown
        data_prelock = get_data_dict(area, data_dict=countries_dict_prelock, time_unit=time_unit,
                                     skip_every=1, cut_off=cut_off, populations=selected_countries_populations,
                                     multiplication_factor=multiplication_factor,
                                     rescaling=selected_countries_rescaling)
        # Real data postlockdown
        data_postlock = get_data_dict(area, data_dict=countries_dict_postlock, time_unit=time_unit,
                                      skip_every=1, cut_off=0., populations=selected_countries_populations,
                                      multiplication_factor=multiplication_factor,
                                      rescaling=selected_countries_rescaling)
        susceptible_weight = 0.
        exposed_weight = 0.
        recovered_weight = 1.
        infected_weight = 1.
        deaths_weight = 1.
        force_init = False
    else:
        # Synthetic data
        # TODO This has to be update to SEIRD
        exact_e_0 = 0.1
        exact_i_0 = 0.2
        exact_r_0 = 0.3
        exact_beta = 0.9
        exact_gamma = 0.2
        synthetic_data = get_syntethic_data(seird, t_final=t_final, e_0=exact_e_0, i_0=exact_i_0, r_0=exact_r_0,
                                            exact_beta=exact_beta,
                                            exact_gamma=exact_gamma,
                                            size=10, selection_mode='equally_spaced')
        susceptible_weight = 1.
        susceptible_weight = 1.
        recovered_weight = 1.
        force_init = False

    # Generate validation set by taking the last time units
    valid_times = []
    valid_infected = []
    valid_recovered = []
    valid_deaths = []
    train_val_split = 0.2
    max_key = max(data_prelock.keys())
    keys = list(data_prelock.keys())

    for k in keys:
        if train_val_split == 0.:
            break

        if k >= max_key * (1 - train_val_split):
            valid_times.append(k)
            valid_infected.append(data_prelock[k][0])
            valid_recovered.append(data_prelock[k][1])
            valid_deaths.append(data_prelock[k][2])
            del data_prelock[k]

    fit_epochs = 100

    min_loss = 1000
    loss_mode = 'mse'

    # Fit n_trials time and take the best fitting
    for i in range(n_trials):
        print('Fit no. {}\n'.format(i + 1))
        e_0, i_0, r_0, d_0, beta, gamma, lam, delta, rnd_init, traj_mse = fit(seird,
                                                                              init_bundle=initial_conditions_set,
                                                                              betas=betas,
                                                                              gammas=gammas,
                                                                              lams=lams,
                                                                              deltas=deltas,
                                                                              steps=train_size, lr=1e-1,
                                                                              known_points=data_prelock,
                                                                              writer=None,
                                                                              loss_mode=loss_mode,
                                                                              epochs=fit_epochs,
                                                                              verbose=True,
                                                                              infected_weight=infected_weight,
                                                                              recovered_weight=recovered_weight,
                                                                              deaths_weight=deaths_weight,
                                                                              force_init=force_init)

        s_0 = 1 - (e_0 + i_0 + r_0 + d_0)

        if traj_mse < min_loss:
            optimal_s_0, optimal_e_0, optimal_i_0, optimal_r_0, optimal_d_0, optimal_beta, optimal_gamma, optimal_lam, optimal_delta = s_0, e_0, i_0, r_0, d_0, beta, gamma, lam, delta
            min_loss = traj_mse

    optimal_initial_conditions = [optimal_s_0, optimal_e_0, optimal_i_0, optimal_r_0, optimal_d_0]

    # Let's save the predicted trajectories in the known points
    optimal_traj = []  # Solution of the network with the optimal found set of params

    # Generate points between 0 and t_final
    grid = torch.arange(0, t_final, out=torch.FloatTensor()).reshape(-1, 1)
    t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)
    s_hat = []
    e_hat = []
    i_hat = []
    r_hat = []
    d_hat = []
    for i, t in enumerate(t_dl, 0):
        # Network solutions
        s, e, i, r, d = seird.parametric_solution(t, optimal_initial_conditions, beta=optimal_beta, gamma=optimal_gamma,
                                                  lam=optimal_lam, delta=optimal_delta)
        s_hat.append(s.item())
        e_hat.append(e.item())
        i_hat.append(i.item())
        r_hat.append(r.item())
        d_hat.append(d.item())

    # Let's compute the MSE over all the known points, not only the subset used for training
    traj_mse = 0.0

    i_traj_real = []
    r_traj_real = []
    d_traj_real = []

    # Validation
    if mode == 'real':

        x_infected_prelock = np.array(range(len(i_hat))) / time_unit
        x_recovered_prelock = np.array(range(len(r_hat))) / time_unit
        x_deaths_prelock = np.array(range(len(d_hat))) / time_unit

        x_train_prelock = np.array(list(data_prelock.keys())) / time_unit
        x_valid_prelock = np.array(valid_times) / time_unit

        _, new_cases_prelock = get_data_dict(area, data_dict=countries_dict_prelock, time_unit=time_unit,
                                             skip_every=1, cut_off=cut_off, return_new_cases=True,
                                             populations=selected_countries_populations,
                                             rescaling=selected_countries_rescaling)
        _, new_cases_postlock = get_data_dict(area, data_dict=countries_dict_postlock, time_unit=time_unit,
                                              skip_every=1, cut_off=0., return_new_cases=True,
                                              populations=selected_countries_populations,
                                              rescaling=selected_countries_rescaling)
        new_cases = np.concatenate((new_cases_prelock, new_cases_postlock))

        # Shift after the lockdown
        x_train_postlock = (np.array(list(data_postlock.keys()))) / time_unit + x_valid_prelock[-1] + 1

        x_new_cases = np.concatenate((x_train_prelock, x_valid_prelock, x_train_postlock))

        traj_mse = traj_mse / len(valid_times)

        known_infected_exact_prelock = [traj[0] for traj in list(data_prelock.values())]
        known_recovered_exact_prelock = [traj[1] for traj in list(data_prelock.values())]
        known_deaths_exact_prelock = [traj[2] for traj in list(data_prelock.values())]

        known_infected_exact_postlock = [traj[0] for traj in list(data_postlock.values())]
        known_recovered_exact_postlock = [traj[1] for traj in list(data_postlock.values())]
        known_deaths_exact_postlock = [traj[2] for traj in list(data_postlock.values())]

        # Rescale everything to total population
        total_population = selected_countries_populations[area]
        i_hat = np.array(i_hat) * total_population
        r_hat = np.array(r_hat) * total_population
        d_hat = np.array(d_hat) * total_population
        valid_infected = np.array(valid_infected) * total_population
        valid_recovered = np.array(valid_recovered) * total_population
        valid_deaths = np.array(valid_deaths) * total_population
        known_infected_exact_prelock = np.array(known_infected_exact_prelock) * total_population
        known_infected_exact_postlock = np.array(known_infected_exact_postlock) * total_population
        known_recovered_exact_prelock = np.array(known_recovered_exact_prelock) * total_population
        known_recovered_exact_postlock = np.array(known_recovered_exact_postlock) * total_population
        known_deaths_exact_prelock = np.array(known_deaths_exact_prelock) * total_population
        known_deaths_exact_postlock = np.array(known_deaths_exact_postlock) * total_population

        plt.figure(figsize=(25, 8))

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        style.use('ggplot')

        plt.subplot(1, 3, 1)
        marker = 'o'
        plt.scatter(x_valid_prelock, valid_infected, marker=marker, label='Validation', color=red)
        plt.plot(x_infected_prelock, i_hat, label='Infected Predicted', color=blue)
        plt.scatter(x_train_prelock, known_infected_exact_prelock, marker=marker, label='Training', color=green)
        plt.scatter(x_train_postlock, known_infected_exact_postlock, marker=marker, label='Post Lockdown', color=orange)
        # plt.bar(x_new_cases, new_cases, label='New Cases', color=magenta)
        plt.title('Comparison between real deaths and predicted deaths\n'
                  'Optimal E(0) = {:.6f} | I(0) = {:.6f} | R(0) = {:.6f} | D(0) = {:.6f}\n '
                  'Beta = {:.6f} | Gamma = {:.6f} | Lam = {:.6f} | Delta = {:.6f}\n'.format(
            optimal_e_0.item(),
            optimal_i_0.item(),
            optimal_r_0.item(),
            optimal_d_0.item(),
            optimal_beta.item(),
            optimal_gamma.item(),
            optimal_lam.item(),
            optimal_delta.item(), fontsize=labelsize))
        plt.legend(loc='best')
        plt.xlabel('t (days)', fontsize=labelsize)
        plt.ylabel('I(t)', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

        plt.subplot(1, 3, 2)
        plt.scatter(x_valid_prelock, valid_recovered, marker=marker, label='Validation', color=red)
        plt.plot(x_recovered_prelock, r_hat, label='Recovered - Predicted', color=blue)
        plt.scatter(x_train_prelock, known_recovered_exact_prelock, marker=marker, label='Training', color=green)
        plt.scatter(x_train_postlock, known_recovered_exact_postlock, marker=marker, label='Post Lockdown',
                    color='orange')
        plt.title('Comparison between real deaths and predicted deaths\n'
                  'Optimal E(0) = {:.6f} | I(0) = {:.6f} | R(0) = {:.6f} | D(0) = {:.6f}\n '
                  'Beta = {:.6f} | Gamma = {:.6f} | Lam = {:.6f} | Delta = {:.6f}\n'.format(
            optimal_e_0.item(),
            optimal_i_0.item(),
            optimal_r_0.item(),
            optimal_d_0.item(),
            optimal_beta.item(),
            optimal_gamma.item(),
            optimal_lam.item(),
            optimal_delta.item(), fontsize=labelsize))
        plt.legend(loc='best')
        plt.xlabel('t (days)', fontsize=labelsize)
        plt.ylabel('R(t)', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

        plt.subplot(1, 3, 3)
        plt.scatter(x_valid_prelock, valid_deaths, marker=marker, label='Validation', color=red)
        plt.plot(x_deaths_prelock, d_hat, label='Recovered - Predicted', color=blue)
        plt.scatter(x_train_prelock, known_deaths_exact_prelock, marker=marker, label='Training', color=green)
        plt.scatter(x_train_postlock, known_deaths_exact_postlock, marker=marker, label='Post Lockdown', color='orange')
        plt.title('Comparison between real deaths and predicted deaths\n'
                  'Optimal E(0) = {:.6f} | I(0) = {:.6f} | R(0) = {:.6f} | D(0) = {:.6f}\n '
                  'Beta = {:.6f} | Gamma = {:.6f} | Lam = {:.6f} | Delta = {:.6f}\n'.format(
            optimal_e_0.item(),
            optimal_i_0.item(),
            optimal_r_0.item(),
            optimal_d_0.item(),
            optimal_beta.item(),
            optimal_gamma.item(),
            optimal_lam.item(),
            optimal_delta.item(), fontsize=labelsize))
        plt.legend(loc='best')
        plt.xlabel('t (days)', fontsize=labelsize)
        plt.ylabel('D(t)', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

        plt.tight_layout()
        plt.show()

    else:
        x_infected = np.array(range(len(i_hat)))
        x_recovered = np.array(range(len(r_hat)))
        x_train = np.array(list(data_prelock.keys()))
        x_valid = np.array(valid_times)
        synthetic_infected = [traj[1] for traj in list(data_prelock.values())]

        title = 'Comparison between real data and predicted data\n' \
                'Estimated vs Real\n' \
                'I(0) = {:.6f} - {:.6f} \n R(0) = {:.6f}  -  {:.6f}\n' \
                'Beta = {:.6f} - {:.6f} \n Gamma = {:.6f}  - {:.6f}\n'.format(optimal_i_0.item(),
                                                                              exact_i_0,
                                                                              optimal_r_0.item(),
                                                                              exact_r_0,
                                                                              optimal_beta.item(),
                                                                              exact_beta,
                                                                              optimal_gamma.item(),
                                                                              exact_gamma)

        plt.figure(figsize=(8, 5))
        plt.scatter(x_valid, valid_infected, label='Validation', color=red)
        plt.plot(x_infected, i_hat, label='Infected Predicted', color=blue)
        plt.scatter(x_train, synthetic_infected, label='Training', color=green)
        plt.title(title)
        plt.legend(loc='best')
        plt.xlabel('t', fontsize=labelsize)
        plt.ylabel('I(t)', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.tight_layout()

        plt.show()
