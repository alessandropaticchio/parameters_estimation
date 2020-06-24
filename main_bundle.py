from training import train_bundle
from constants import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from real_data_regions import *
from losses import seird_loss
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import SIRNetwork
from utils import SEIRD_solution

if __name__ == '__main__':
    # If resume_training is True, it will also load the optimizer and resume training
    resume_training = False

    # Equation parameters
    t_0 = 0
    t_final = 20

    # The intervals in which the equation parameters and the initial conditions should vary
    e_0_set = [0.08, 0.1]
    i_0_set = [0.01, 0.2]
    r_0_set = [0., 0.001]
    d_0_set = [0., 0.001]
    betas = [0.004, 0.01]
    gammas = [0.15, 0.25]
    lams = [0.05, 0.15]
    deltas = [0.15, 0.25]


    # Model parameters
    initial_conditions_set = []
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(e_0_set)
    initial_conditions_set.append(i_0_set)
    initial_conditions_set.append(r_0_set)
    initial_conditions_set.append(d_0_set)

    train_size = 2000
    decay = 1e-2
    hack_trivial = 0
    epochs = 5000
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

        import csv

        with open(ROOT_DIR + '/csv/train_losses_{}.csv'.format(model_name), 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(train_losses)

    # Load the model
    seird.load_state_dict(checkpoint['model_state_dict'])

    if resume_training:
        additional_epochs = 10000
        optimizer = torch.optim.Adam(seird.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        writer = SummaryWriter('runs/' + 'resume_{}'.format(model_name))
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

    # Equation parameters
    beta = 0.008267
    gamma = 0.17161
    lam = 0.08111
    delta = 0.08
    e_0 = 0.081020
    i_0 = 0.01592
    r_0 = 0.00095917
    d_0 = 0.0009
    s_0 = 1 - (e_0 + i_0 + r_0 + d_0)

    # Scipy solver solution
    t = np.linspace(0, t_final, t_final)
    s_p, e_p, i_p, r_p, d_p = SEIRD_solution(t, s_0, e_0, i_0, r_0, d_0, beta, gamma, lam, delta)

    s_hat, e_hat, i_hat, r_hat, d_hat, de_loss = seird.solve(e_0=e_0, i_0=i_0, r_0=r_0, d_0=d_0, beta=beta, gamma=gamma,
                                                             lam=lam, delta=delta, t_0=0, t_final=t_final)

    print('DE Loss: {:.15E} | LogLoss = {}'.format(de_loss, np.log(de_loss.item())))

    # Plot network solutions
    plt.figure(figsize=(8, 5))
    # plt.plot(range(len(s_p)), s_p, label='Susceptible - Scipy', linestyle='--', color=blue, linewidth=1.)
    plt.plot(range(len(e_p)), e_p, label='Exposed - Scipy', linestyle='--', color=orange, linewidth=1.)
    plt.plot(range(len(i_p)), i_p, label='Infected - Scipy', linestyle='--', color=red, linewidth=1.)
    plt.plot(range(len(r_p)), r_p, label='Recovered - Scipy', linestyle='--', color=green, linewidth=1.)
    plt.plot(range(len(d_p)), d_p, label='Deaths - Scipy', linestyle='--', color=purple, linewidth=1.)
    # plt.plot(range(len(s_hat)), s_hat, label='Susceptible', linestyle='-', color=blue, linewidth=1.)
    plt.plot(range(len(e_hat)), e_hat, label='Exposed', linestyle='-', color=orange, linewidth=1.)
    plt.plot(range(len(i_hat)), i_hat, label='Infected', linestyle='-', color=red, linewidth=1.)
    plt.plot(range(len(r_hat)), r_hat, label='Recovered', linestyle='-', color=green, linewidth=1.)
    plt.plot(range(len(d_hat)), d_hat, label='Deaths', linestyle='-', color=purple, linewidth=1.)
    plt.title('Solving SIR model with Beta = {} | Gamma = {} | Lam = {} | Delta = {}\n'
              'Starting conditions: S0 = {:.4f} | E0 = {:.4f} | I0 = {:.4f} | R0 = {:.4f} | D0 = {:.4f}\n'
              'Bundle: E(0) in {} | I(0) in {} | R(0) in {} | D(0) in {}\n'
              'Beta in {} | Gamma in {} | Lam in {} | Delta in {}'.format(round(beta, 4),
                                                                          round(gamma, 4),
                                                                          round(lam, 4), round(delta, 4),
                                                                          s_0, e_0, i_0, r_0, d_0,
                                                                          e_0_set, i_0_set, r_0_set, d_0_set,
                                                                          betas,
                                                                          gammas,
                                                                          lams, deltas))
    plt.xlabel('Time')
    plt.ylabel('S(t), E(t), I(t), R(t), D(t)')
    plt.legend(loc='lower right')
    plt.show()
