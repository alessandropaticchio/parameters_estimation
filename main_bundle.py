from training import train_bundle
from constants import ROOT_DIR, red, green, blue, orange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from real_data_regions import *
from losses import sir_loss
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import SIRNetwork
from utils import SIRP_solution

if __name__ == '__main__':
    # If resume_training is True, it will also load the optimizer and resume training
    resume_training = False

    # Equation parameters
    t_0 = 0
    t_final = 20

    # The intervals in which the equation parameters and the initial conditions should vary
    i_0_set = [0.01, 0.02]
    r_0_set = [0.004, 0.009]
    p_0_set = [0.9, 0.97]
    betas = [0.4, 0.6]
    gammas = [0.05, 0.15]

    # Model parameters
    initial_conditions_set = []
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(i_0_set)
    initial_conditions_set.append(r_0_set)
    initial_conditions_set.append(p_0_set)

    train_size = 1000
    decay = 1e-2
    hack_trivial = 0
    epochs = 20000
    lr = 8e-4

    # Init model
    sirp = SIRNetwork(input=6, layers=4, hidden=50, output=4)

    model_name = 'i_0={}_r_0={}_p_0={}_betas={}_gammas={}.pt'.format(i_0_set, r_0_set, p_0_set,
                                                                     betas,
                                                                     gammas)
    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIRP_bundle_total/{}'.format(model_name))
    except FileNotFoundError:
        # Train
        optimizer = torch.optim.Adam(sirp.parameters(), lr=lr)
        writer = SummaryWriter('runs/{}'.format(model_name))
        sirp, train_losses, run_time, optimizer = train_bundle(sirp, initial_conditions_set, t_final=t_final,
                                                               epochs=epochs, model_name=model_name,
                                                               num_batches=10, hack_trivial=hack_trivial,
                                                               train_size=train_size, optimizer=optimizer,
                                                               decay=decay,
                                                               writer=writer, betas=betas, gammas=gammas)
        # Save the model
        torch.save({'model_state_dict': sirp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIRP_bundle_total/{}'.format(model_name))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIRP_bundle_total/{}'.format(model_name))

        import csv
        with open(ROOT_DIR + '/csv/train_losses_{}.csv'.format(model_name), 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(train_losses)

    # Load the model
    sirp.load_state_dict(checkpoint['model_state_dict'])

    if resume_training:
        additional_epochs = 10000
        optimizer = torch.optim.Adam(sirp.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        writer = SummaryWriter('runs/' + 'resume_{}'.format(model_name))
        sirp, train_losses, run_time, optimizer = train_bundle(sirp, initial_conditions_set, t_final=t_final,
                                                               epochs=additional_epochs, model_name=model_name,
                                                               num_batches=10, hack_trivial=hack_trivial,
                                                               train_size=train_size, optimizer=optimizer,
                                                               decay=decay,
                                                               writer=writer, betas=betas, gammas=gammas)
        # Save the model
        torch.save({'model_state_dict': sirp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIRP_bundle_total/{}'.format(model_name))


    # Equation parameters
    beta = 0.55
    gamma = 0.1013
    i_0 = 0.015
    r_0 = 0.005757
    p_0 = 0.94
    s_0 = 1 - (i_0 + r_0 + p_0)

    # Scipy solver solution
    t = np.linspace(0, t_final, t_final)
    s_p, i_p, r_p, p_p = SIRP_solution(t, s_0, i_0, r_0, p_0, beta, gamma)

    s_hat, i_hat, r_hat, p_hat, de_loss = sirp.solve(i_0=i_0, r_0=r_0, p_0=p_0, beta=beta, gamma=gamma, t_0=0, t_final=t_final)

    print('DE Loss: {:.15E} | LogLoss = {}'.format(de_loss, np.log(de_loss.item())))

    # Plot network solutions
    plt.figure(figsize=(8, 5))
    #plt.plot(range(len(s_p)), s_p, label='Susceptible - Scipy', linestyle='--', color=blue, linewidth=1.)
    plt.plot(range(len(i_p)), i_p, label='Infected - Scipy', linestyle='--', color=red, linewidth=1.)
    plt.plot(range(len(r_p)), r_p, label='Recovered - Scipy', linestyle='--', color=green, linewidth=1.)
    #plt.plot(range(len(s_hat)), s_hat, label='Susceptible', linestyle='-', color=blue, linewidth=1.)
    plt.plot(range(len(i_hat)), i_hat, label='Infected', linestyle='-', color=red, linewidth=1.)
    plt.plot(range(len(r_hat)), r_hat, label='Recovered', linestyle='-', color=green, linewidth=1.)
    plt.title('Solving SIR model with Beta = {} | Gamma = {}\n'
              'Starting conditions: S0 = {:.4f} | I0 = {:.4f} | R0 = {:.4f} | P0 = {:.4f} \n'
              'Model trained on bundle: I(0) in {} | R(0) in {} | P(0) in {} \n'
              'Beta in {} | Gamma in {}'.format(round(beta, 4),
                                                round(gamma, 4),
                                                s_0, i_0, r_0, p_0,
                                                i_0_set, r_0_set, p_0_set,
                                                betas,
                                                gammas))

    plt.xlabel('Time')
    plt.ylabel('S(t), I(t), R(t), P(t)')
    plt.legend(loc='lower right')
    plt.show()
