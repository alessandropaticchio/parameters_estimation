from constants import ROOT_DIR
from models import SIRNetwork
from training import train_bundle
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import SEIR_solution
from real_data_regions import *
import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # File to apply finetuning on a pretrained model

    source_e_0_set = [0.08, 0.1]
    source_i_0_set = [0.01, 0.2]
    source_r_0_set = [0., 0.001]
    source_betas = [0.004, 0.01]
    source_gammas = [0.15, 0.4]
    source_lams = [0.01, 0.03]

    initial_conditions_set = []
    t_0 = 0
    t_final = 20
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(source_e_0_set)
    initial_conditions_set.append(source_i_0_set)
    initial_conditions_set.append(source_r_0_set)
    # Init model
    seir = SIRNetwork(input=7, layers=4, hidden=50, output=4)
    lr = 8e-4

    source_model_name = 'e_0={}_i_0={}_r_0={}_betas={}_gammas={}_lams={}.pt'.format(source_e_0_set, source_i_0_set,
                                                                                    source_r_0_set,
                                                                                    source_betas,
                                                                                    source_gammas,
                                                                                    source_lams, )

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(ROOT_DIR + '/models/SEIR_bundle_total/{}'.format(source_model_name))
    except FileNotFoundError:
        # Train
        optimizer = torch.optim.Adam(seir.parameters(), lr=lr)
        source_epochs = 20000
        source_hack_trivial = 0
        source_train_size = 2000
        source_decay = 1e-2
        writer = SummaryWriter('runs/{}_scratch'.format(source_model_name))
        seir, train_losses, run_time, optimizer = train_bundle(seir, initial_conditions_set, t_final=t_final,
                                                               epochs=source_epochs, model_name=source_model_name,
                                                               num_batches=10, hack_trivial=source_hack_trivial,
                                                               train_size=source_train_size, optimizer=optimizer,
                                                               decay=source_decay,
                                                               writer=writer, betas=source_betas,
                                                               gammas=source_gammas, lams=source_lams)
        # Save the model
        torch.save({'model_state_dict': seir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SEIR_bundle_total/{}'.format(source_model_name))

        # Load the checkpoint
        checkpoint = torch.load(ROOT_DIR + '/models/SIR_bundle_total/{}'.format(source_model_name))

    # Target model
    target_e_0_set = [0.08, 0.1]
    target_i_0_set = [0.01, 0.2]
    target_r_0_set = [0., 0.001]
    target_betas = [0.004, 0.01]
    target_gammas = [0.15, 0.25]
    target_lams = [0.05, 0.09]

    target_model_name = 'e_0={}_i_0={}_r_0={}_betas={}_gammas={}_lams={}.pt'.format(target_e_0_set, target_i_0_set,
                                                                                    target_r_0_set,
                                                                                    target_betas,
                                                                                    target_gammas, target_lams)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/{}'.format(target_model_name))

    except FileNotFoundError:
        print('Finetuning...')
        # Load old model
        seir.load_state_dict(checkpoint['model_state_dict'])
        # Train
        initial_conditions_set = []
        t_0 = 0
        t_final = 20
        initial_conditions_set.append(t_0)
        initial_conditions_set.append(target_e_0_set)
        initial_conditions_set.append(target_i_0_set)
        initial_conditions_set.append(target_r_0_set)
        optimizer = torch.optim.Adam(seir.parameters(), lr=lr)
        target_epochs = 1
        target_hack_trivial = 0
        target_train_size = 2000
        target_decay = 1e-3
        writer = SummaryWriter('runs/{}_finetuned'.format(target_model_name))
        seir, train_losses, run_time, optimizer = train_bundle(seir, initial_conditions_set, t_final=t_final,
                                                               epochs=target_epochs, model_name=target_model_name,
                                                               num_batches=10, hack_trivial=target_hack_trivial,
                                                               train_size=target_train_size, optimizer=optimizer,
                                                               decay=target_decay,
                                                               writer=writer, betas=target_betas, lams=target_lams,
                                                               gammas=target_gammas)

        # Save the model
        torch.save({'model_state_dict': seir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SEIR_bundle_total/{}'.format(target_model_name))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SEIR_bundle_total/{}'.format(target_model_name))

    # Load fine-tuned model
    seir.load_state_dict(checkpoint['model_state_dict'])

    # Test between 0 and t_final
    grid = torch.arange(0, t_final, out=torch.FloatTensor()).reshape(-1, 1)
    t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)
    s_hat = []
    i_hat = []
    r_hat = []

    # Equation parameters
    beta = 6.15
    gamma = 6.14
    lam = 0.94
    e_0 = 0.2
    i_0 = 0.001509
    r_0 = 0.0000728
    s_0 = 1 - (e_0 + i_0 + r_0)

    # Scipy solver solution
    t = np.linspace(0, t_final, t_final)
    s_p, e_p, i_p, r_p = SEIR_solution(t, s_0, e_0, i_0, r_0, beta, gamma, lam)

    s_hat, e_hat, i_hat, r_hat, de_loss = seir.solve(e_0=e_0, i_0=i_0, r_0=r_0, beta=beta, gamma=gamma, lam=lam, t_0=0,
                                                     t_final=t_final)

    # Plot network solutions
    plt.figure(figsize=(8, 5))
    # plt.plot(range(len(s_hat)), s_hat, label='Susceptible')
    plt.plot(range(len(i_hat)), i_hat, label='Infected')
    plt.plot(range(len(r_hat)), r_hat, label='Recovered')
    # plt.plot(range(len(s_p)), s_p, label='Susceptible - Scipy', linestyle='--')
    plt.plot(range(len(i_p)), i_p, label='Infected - Scipy', linestyle='--')
    plt.plot(range(len(r_p)), r_p, label='Recovered - Scipy', linestyle='--')
    plt.title('Solving bundle SEIR model with Beta = {} | Gamma = {} | Lam = {} \n'
              'Starting conditions: S0 = {} | E0 = {} | I0 = {:.3f} | R0 = {:.3f} \n'
              'E(0) in {} | I(0) in {} | R(0) in {} \n'
              'Betas = {} | Gammas = {}\n'
              .format(beta,
                      gamma,
                      lam,
                      s_0, e_0, i_0, r_0, target_e_0_set,
                      target_i_0_set, target_r_0_set,
                      target_betas,
                      target_gammas))
    plt.legend(loc='lower right')
    plt.xlabel('t')
    plt.ylabel('S(t), E(t), I(t), R(t)')
    plt.show()
