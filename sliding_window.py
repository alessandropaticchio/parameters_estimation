from constants import ROOT_DIR
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from models import SIRNetwork
from utils import SIRP_solution
from utils import get_syntethic_data, get_data_dict
from real_data_countries import countries_dict_prelock, countries_dict_postlock, selected_countries_populations, \
    selected_countries_rescaling
from training import train_bundle
from torch.utils.tensorboard import SummaryWriter
import math

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def truncate(number, digits):
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def find_bundle_edges(area='Italy', mode='pre'):
    time_unit = 0.25
    cut_off = 1.5e-3
    multiplication_factor = 10

    if mode == 'pre':
        # Real data pre-lockdown
        data = get_data_dict(area, data_dict=countries_dict_prelock, time_unit=time_unit,
                             skip_every=1, cut_off=cut_off, populations=selected_countries_populations,
                             multiplication_factor=multiplication_factor,
                             rescaling=selected_countries_rescaling)
    elif mode == 'post':
        # Real data post-lockdown
        data = get_data_dict(area, data_dict=countries_dict_postlock, time_unit=time_unit,
                             skip_every=1, cut_off=0., populations=selected_countries_populations,
                             multiplication_factor=multiplication_factor,
                             rescaling=selected_countries_rescaling)

    # Find bundles edges
    i_sorted = sorted(data.values(), key=lambda x: x[0])
    r_sorted = sorted(data.values(), key=lambda x: x[1])
    i_min, i_max = truncate(i_sorted[0][0], 3), truncate(i_sorted[-1][0], 3)
    r_min, r_max = truncate(r_sorted[0][1], 3), truncate(r_sorted[-1][1], 3)

    return [[i_min, i_max], [r_min, r_max]]


def train_window_bundle(area, mode, resume_training=False):
    # Equation parameters
    t_0 = 0
    t_final = 20

    # The intervals in which the equation parameters and the initial conditions should vary
    edges = find_bundle_edges(area, mode)
    i_0_set = edges[0]
    r_0_set = edges[1]
    p_0_set = [0.9, 0.97]
    betas = [0.4, 0.6]
    gammas = [0.1, 0.2]

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
                                                                     betas, gammas)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIRP_bundle_total/{}_{}'.format(area, mode))
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
                   ROOT_DIR + '/models/SIRP_bundle_total/{}_{}'.format(area, mode))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIRP_bundle_total/{}_{}'.format(area, mode))

        import csv
        with open(ROOT_DIR + '/csv/train_losses_{}_{}.csv'.format(area, mode), 'w', newline='') as myfile:
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
                   ROOT_DIR + '/models/SIRP_bundle_total/{}_{}'.format(area, mode))


if __name__ == '__main__':
    train_window_bundle(area='Italy', mode='pre')
    train_window_bundle(area='Italy', mode='post')



