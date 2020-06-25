from models import SIRNetwork
from constants import *
from real_data_countries import countries_dict_prelock, selected_countries_populations, selected_countries_rescaling
from utils import get_syntethic_data, get_data_dict
from tqdm import tqdm
import torch
import numpy as np

if __name__ == '__main__':

    t_0 = 0
    t_final = 20

    # The interval in which the equation parameters and the initial conditions should vary
    e_0_set = [0.08, 0.1]
    i_0_set = [0.01, 0.2]
    r_0_set = [0.003, 0.009]
    betas = [0.004, 0.01]
    gammas = [0.15, 0.25]
    lams = [0.05, 0.09]

    # Model parameters
    initial_conditions_set = []
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(e_0_set)
    initial_conditions_set.append(i_0_set)
    initial_conditions_set.append(r_0_set)

    # Init model
    seir = SIRNetwork(input=7, layers=4, hidden=50, output=4)

    model_name = 'e_0={}_i_0={}_r_0={}_betas={}_gammas={}_lams={}.pt'.format(e_0_set, i_0_set, r_0_set,
                                                                             betas,
                                                                             gammas, lams)

    checkpoint = torch.load(ROOT_DIR + '/models/SEIR_bundle_total/{}'.format(model_name))

    # Load the model
    seir.load_state_dict(checkpoint['model_state_dict'])

    e_0 = 0.07
    i_0 = 0.015
    r_0 = 0.006

    # Generate grid of parameters
    size = 20
    betas = np.linspace(betas[0], betas[1], size)
    gammas = np.linspace(gammas[0], gammas[1], size)
    lams = np.linspace(betas[0], lams[1], size)

    # Get data
    area = 'Italy'
    time_unit = 0.25
    cut_off = 1.5e-3
    multiplication_factor = 10
    # Real data prelockdown
    data_prelock = get_data_dict(area, data_dict=countries_dict_prelock, time_unit=time_unit,
                                 skip_every=1, cut_off=cut_off, populations=selected_countries_populations,
                                 multiplication_factor=multiplication_factor,
                                 rescaling=selected_countries_rescaling)

    i_real = np.array([traj[1] for traj in list(data_prelock.values())])
    r_real = np.array([traj[2] for traj in list(data_prelock.values())])

    optimal_mse = 1000
    optimal_beta = None
    optimal_gamma = None
    optimal_lam = None

    for beta in tqdm(betas, desc='Grid search'):
        for gamma in gammas:
            for lam in lams:
                _, _, i, r, _ = seir.solve(e_0=e_0, i_0=i_0, r_0=r_0, beta=beta, gamma=gamma, lam=lam, t_0=t_0, t_final=t_final, size=len(i_real))

                # Compute MSE between the solution and real data
                i = np.array(i)
                r = np.array(r)

                n = len(i)
                mse = (sum((i_real - i) ** 2) + sum((r_real - r) ** 2)) / n

                if mse < optimal_mse:
                    optimal_mse = mse
                    optimal_beta = beta
                    optimal_gamma = gamma
                    optimal_lam = lam

    print('Optimal Beta, Gamma, Lam: {:.4f}, {:.4f}, {:.4f} - with MSE: {:.4f}'.format(optimal_beta, optimal_gamma,
                                                                                       optimal_lam, optimal_mse))


