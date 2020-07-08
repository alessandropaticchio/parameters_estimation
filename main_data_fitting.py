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

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

if __name__ == '__main__':
    # If mode == real, it will fit real data, otherwise synthetic data.
    # Later the data to fit are specified
    mode = 'synthetic'

    t_0 = 0
    t_final = 20

    # The interval in which the equation parameters and the initial conditions should vary
    # i_0_set = [0.4, 0.6]
    # r_0_set = [0.1, 0.3]
    # betas = [0.45, 0.65]
    # gammas = [0.05, 0.15]
    i_0_set = [0.2, 0.4]
    r_0_set = [0.1, 0.3]
    betas = [0., 0.4]
    gammas = [0.4, 0.7]
    initial_conditions_set = []
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(i_0_set)
    initial_conditions_set.append(r_0_set)

    # How many times I want to fit the trajectory, getting the best result
    n_trials = 1
    fit_epochs = 300

    # Model parameters
    train_size = 2000
    decay = 1e-4
    hack_trivial = False
    epochs = 3000
    lr = 8e-4

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
        writer = SummaryWriter('runs/{}'.format(model_name))
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

    writer_dir = 'runs/' + 'fitting_{}'.format(model_name)

    # Check if the writer directory exists, if yes delete it and overwrite
    if os.path.isdir(writer_dir):
        rmtree(writer_dir)
    writer = SummaryWriter(writer_dir)

    if mode == 'real':
        area = 'Italy'
        time_unit = 0.25
        cut_off = 1e-1
        # Real data prelockdown
        data_prelock = get_data_dict(area, data_dict=countries_dict_prelock, time_unit=time_unit,
                                     skip_every=1, cut_off=cut_off, populations=selected_countries_populations,
                                     rescaling=selected_countries_rescaling)
        # Real data postlockdown
        data_postlock = get_data_dict(area, data_dict=countries_dict_postlock, time_unit=time_unit,
                                      skip_every=1, cut_off=0., populations=selected_countries_populations,
                                      rescaling=selected_countries_rescaling)
        susceptible_weight = 1.
        recovered_weight = 1.
        infected_weight = 1.
        force_init = False
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
        recovered_weight = 1.
        infected_weight = 1.
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


    min_loss = 1000
    loss_mode = 'mse'
    n_batches = 4

    # Fit n_trials time and take the best fitting
    for i in range(n_trials):
        print('Fit no. {}\n'.format(i + 1))
        i_0, r_0, beta, gamma, val_losses = fit(sir,
                                                init_bundle=initial_conditions_set,
                                                betas=betas,
                                                gammas=gammas,
                                                lr=1e-1,
                                                known_points=data_prelock,
                                                writer=writer,
                                                loss_mode=loss_mode,
                                                epochs=fit_epochs,
                                                verbose=True,
                                                n_batches=n_batches,
                                                susceptible_weight=susceptible_weight,
                                                recovered_weight=recovered_weight,
                                                infected_weight=infected_weight,
                                                force_init=force_init,
                                                validation_data=validation_data)
        s_0 = 1 - (i_0 + r_0)

        if val_losses[-1] < min_loss:
            optimal_s_0, optimal_i_0, optimal_r_0, optimal_beta, optimal_gamma = s_0, i_0, r_0, beta, gamma
            min_loss = val_losses[-1]

    optimal_initial_conditions = [optimal_s_0, optimal_i_0, optimal_r_0]

    # Let's save the predicted trajectories in the known points
    optimal_traj = []  # Solution of the network with the optimal found set of params

    optimal_de = 0.
    for t, v in data_prelock.items():
        t_tensor = torch.Tensor([t]).reshape(-1, 1)
        t_tensor.requires_grad = True

        s_optimal, i_optimal, r_optimal = sir.parametric_solution(t_tensor, optimal_initial_conditions,
                                                                  beta=optimal_beta,
                                                                  gamma=optimal_gamma,
                                                                  mode='bundle_total')

        optimal_de += sir_loss(t_tensor, s_optimal, i_optimal, r_optimal, optimal_beta, optimal_gamma)
        optimal_traj.append([s_optimal, i_optimal, r_optimal])

    # Exact solution subset
    exact_sub_traj = [data_prelock[t] for t in data_prelock.keys()]

    optimal_mse = 0.
    for idx, p in enumerate(exact_sub_traj):
        exact_s, exact_i, exact_r = p
        optimal_mse += (exact_s - optimal_traj[idx][0]) ** 2 + (exact_i - optimal_traj[idx][1]) ** 2 + (
                exact_r - optimal_traj[idx][2]) ** 2

    optimal_mse /= len(optimal_traj)

    # Generate points between 0 and t_final
    grid = torch.arange(0, t_final, out=torch.FloatTensor()).reshape(-1, 1)
    t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)
    s_hat = []
    i_hat = []
    r_hat = []
    for i, t in enumerate(t_dl, 0):
        # Network solutions
        s, i, r = sir.parametric_solution(t, optimal_initial_conditions, beta=optimal_beta, gamma=optimal_gamma,
                                          mode='bundle_total')
        s_hat.append(s.item())
        i_hat.append(i.item())
        r_hat.append(r.item())

    # Let's compute the MSE over all the known points, not only the subset used for training
    traj_mse = 0.0

    i_traj_real = []
    r_traj_real = []

    if mode == 'real':
        total_population = selected_countries_populations[area]
    else:
        total_population = 1e7

    # Validation
    if mode == 'real':

        # Save data for future plots

        x_infected_prelock = np.array(range(len(i_hat))) / time_unit
        x_recovered_prelock = np.array(range(len(r_hat))) / time_unit
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

        synthetic_infected = [traj[1] for traj in list(data_prelock.values())]
        known_recovered_prelock = [traj[2] for traj in list(data_prelock.values())]

        known_infected_postlock = [traj[1] for traj in list(data_postlock.values())]
        known_recovered_postlock = [traj[2] for traj in list(data_postlock.values())]

        # Rescale everything to total population
        i_hat = np.array(i_hat) * total_population
        r_hat = np.array(r_hat) * total_population
        valid_infected = np.array(valid_infected) * total_population
        synthetic_infected = np.array(synthetic_infected) * total_population
        known_infected_postlock = np.array(known_infected_postlock) * total_population
        known_recovered_prelock = np.array(known_recovered_prelock) * total_population
        known_recovered_postlock = np.array(known_recovered_postlock) * total_population

        plt.figure(figsize=(30, 8))

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        style.use('ggplot')

        plt.subplot(1, 2, 1)
        plt.bar(x_valid_prelock, valid_infected, label='Validation', color=red)
        plt.plot(x_infected_prelock, i_hat, label='Infected Predicted', color=blue)
        plt.bar(x_train_prelock, synthetic_infected, label='Training', color=green)
        plt.bar(x_train_postlock, known_infected_postlock, label='Post Lockdown', color=orange)
        plt.bar(x_new_cases, new_cases, label='New Cases', color=magenta)
        plt.title('Comparison between real recovered and predicted infected\n'
                  'Optimal I(0) = {:.6f} | R(0) = {:.6f} \n Beta = {:.6f} | Gamma = {:.6f} \n'.format(
            optimal_i_0.item(),
            optimal_r_0.item(),
            optimal_beta.item(),
            optimal_gamma.item()))
        plt.legend(loc='best')
        plt.xlabel('$t (days)$', fontsize=labelsize)
        plt.ylabel('$I(t)$', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

        plt.subplot(1, 2, 2)
        plt.bar(x_valid_prelock, valid_recovered, label='Validation', color=red)
        plt.plot(x_recovered_prelock, r_hat, label='Recovered - Predicted', color=blue)
        plt.bar(x_train_prelock, known_recovered_prelock, label='Infected', color=green)
        plt.bar(x_train_postlock, known_recovered_postlock, label='Post Lockdown', color='orange')
        plt.title('Comparison between real recovered and predicted recovered\n'
                  'Optimal I(0) = {:.6f} | R(0) = {:.6f} \n Beta = {:.6f} | Gamma = {:.6f} \n'.format(
            optimal_i_0.item(),
            optimal_r_0.item(),
            optimal_beta.item(),
            optimal_gamma.item()))
        plt.legend(loc='best')
        plt.xlabel('$t (days)$', fontsize=labelsize)
        plt.ylabel('$R(t)$', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)

        plt.tight_layout()
        plt.show()

 

    else:
        synthetic_infected = [traj[1] for traj in list(data_prelock.values())]
        known_recovered_prelock = [traj[2] for traj in list(data_prelock.values())]

        synthetic_infected = np.array(synthetic_infected) * total_population
        valid_infected = np.array(valid_infected) * total_population

        i_hat = np.array(i_hat) * total_population
        r_hat = np.array(r_hat) * total_population
        x_infected = np.array(range(len(i_hat)))
        x_recovered = np.array(range(len(r_hat)))

        x_train = np.array(list(data_prelock.keys()))
        x_valid = np.array(valid_times)

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
        plt.plot(x_infected, i_hat, label='Infected Predicted', color=blue)
        plt.scatter(x_train, synthetic_infected, label='Training', color=green)
        plt.scatter(x_valid, valid_infected, label='Validation', color=red)
        plt.title(title)
        plt.legend(loc='best')
        plt.xlabel('$t$', fontsize=labelsize)
        plt.ylabel('$I(t)$', fontsize=labelsize)
        plt.xticks(fontsize=ticksize / 1.5)
        plt.yticks(fontsize=ticksize / 1.5)
        plt.tight_layout()

        plt.savefig(ROOT_DIR + '\synthetic_2.png')

        plt.show()
