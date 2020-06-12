from training import train_bundle
from constants import ROOT_DIR
from torch.utils.tensorboard import SummaryWriter
from models import SIRNetwork
from utils import get_syntethic_data
from matplotlib.patches import Rectangle
from data_fitting import fit
from nclmap import nlcmap
from random import uniform
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    # Equation parameters
    t_0 = 0
    t_final = 20

    # Compute the interval in which the equation parameters and the initial conditions should vary
    betas = [0.4, 0.8]
    gammas = [0.3, 0.7]
    i_0_set = [0.2, 0.4]
    r_0_set = [0.1, 0.3]
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

    n_draws = 200
    n_trials = 3
    exact_size = 4

    params_scores = {}
    init_scores = {}

    loss_mode = 'mse'

    # Load models to generate trajectories
    exact_beta = (betas[1] + betas[0]) / 2
    exact_gamma = (gammas[1] + gammas[0]) / 2


    init_surroundings = [[[0., 0.2], [0.3, 0.5]], [[0.2, 0.4], [0.3, 0.5]], [[0.4, 0.6], [0.3, 0.5]],
                         [[0., 0.2], [0.1, 0.3]], [[0.2, 0.4],[0.1, 0.3]], [[0.4, 0.6], [0.1, 0.3]],
                         [[0., 0.2], [0., 0.1]], [[0.2, 0.4], [0., 0.1]], [[0.4, 0.6], [0., 0.1]]]

    models_surroundings = {}

    for inits in init_surroundings:
        i_0_surroundings = inits[0]
        r_0_surroundings = inits[1]

        model_name = 'i_0={}_r_0={}_betas={}_gammas={}.pt'.format(i_0_surroundings, r_0_surroundings,
                                                                  betas,
                                                                  gammas)

        sir = SIRNetwork(input=5, layers=4, hidden=50)

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))

        # Load the model
        sir.load_state_dict(checkpoint['model_state_dict'])
        models_surroundings[tuple(i_0_surroundings + r_0_surroundings)] = sir

    # Fix beta, gamma and compute scores as a function of initial conditions
    for n in tqdm(range(n_draws), desc='Computing scores '):

        exact_i_0 = uniform(0, 0.6)
        exact_r_0 = uniform(0, 0.5)

        bundle_found = False

        # Get the bundle you should generate the trajectories from
        for bundle in init_surroundings:
            i_0_surroundings = bundle[0]
            r_0_surroundings = bundle[1]
            if exact_i_0 >= i_0_surroundings[0] and exact_i_0 <= i_0_surroundings[1] \
                    and exact_r_0 >= r_0_surroundings[0] and exact_r_0 <= r_0_surroundings[1]:
                sir = models_surroundings[tuple(i_0_surroundings + r_0_surroundings)]
                bundle_found = True

        assert bundle_found

        exact_beta = torch.Tensor([exact_beta]).reshape(-1, 1)
        exact_gamma = torch.Tensor([exact_gamma]).reshape(-1, 1)
        exact_i_0 = torch.Tensor([exact_i_0]).reshape(-1, 1)
        exact_r_0 = torch.Tensor([exact_r_0]).reshape(-1, 1)

        synthetic_data = get_syntethic_data(model=sir, t_final=t_final, i_0=exact_i_0, r_0=exact_r_0,
                                            exact_beta=exact_beta, exact_gamma=exact_gamma,
                                            size=exact_size)

        min_loss = 1000

        initial_conditions_set = [t_0, i_0_set, r_0_set]

        for n in range(n_trials):
            # Search optimal params
            fit_epochs = 100
            optimal_i_0, optimal_r_0, optimal_beta, optimal_gamma, rnd_init, loss = fit(sir,
                                                                                        init_bundle=initial_conditions_set,
                                                                                        betas=betas,
                                                                                        gammas=gammas,
                                                                                        steps=train_size, lr=1e-3,
                                                                                        known_points=synthetic_data,
                                                                                        writer=None,
                                                                                        epochs=fit_epochs,
                                                                                        verbose=False,
                                                                                        loss_mode=loss_mode)

            # Compute the score
            score = (optimal_beta.item() - exact_beta) ** 2 + (optimal_gamma.item() - exact_gamma) ** 2 + (
                    optimal_i_0.item() - exact_i_0) ** 2 + (optimal_r_0.item() - exact_r_0) ** 2

            if loss < min_loss:
                init_scores[(exact_i_0.item(), exact_r_0.item())] = score.item()
                min_loss = loss

    # Load models to generate trajectories
    exact_i_0 = (i_0_set[1] + i_0_set[0]) / 2
    exact_r_0 = (r_0_set[1] + r_0_set[0]) / 2

    params_surroundings = [[[0., 0.4], [0.7, 1.0]], [[0.4, 0.8], [0.7, 1.0]], [[0.8, 1.0], [0.7, 1.0]],
                           [[0., 0.4], [0.3, 0.7]], [[0.4, 0.8], [0.3, 0.7]], [[0.8, 1.0], [0.3, 0.7]],
                           [[0., 0.4], [0., 0.3]], [[0.4, 0.8], [0., 0.3]], [[0.8, 1.0], [0., 0.3]]]

    models_surroundings = {}

    for params in params_surroundings:
        betas_surroundings = params[0]
        gammas_surroundings = params[1]

        model_name = 'i_0={}_r_0={}_betas={}_gammas={}.pt'.format(i_0_set, r_0_set,
                                                                  betas_surroundings,
                                                                  gammas_surroundings)

        sir = SIRNetwork(input=5, layers=4, hidden=50)

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))
        # Load the model
        sir.load_state_dict(checkpoint['model_state_dict'])
        models_surroundings[tuple(betas_surroundings + gammas_surroundings)] = sir

    # Fix initial conditions and compute scores as a function of parameters
    for n in tqdm(range(n_draws), desc='Computing scores '):
        exact_beta = uniform(0., 1.)
        exact_gamma = uniform(0., 1.)

        bundle_found = False

        # Get the bundle you should generate the trajectories from
        for bundle in params_surroundings:
            betas_surroundings = bundle[0]
            gammas_surroundings = bundle[1]
            if exact_beta >= betas_surroundings[0] and exact_beta <= betas_surroundings[1] and exact_gamma >= \
                    gammas_surroundings[0] and exact_gamma <= gammas_surroundings[1]:
                sir = models_surroundings[tuple(betas_surroundings + gammas_surroundings)]
                bundle_found = True

        assert bundle_found

        exact_beta = torch.Tensor([exact_beta]).reshape(-1, 1)
        exact_gamma = torch.Tensor([exact_gamma]).reshape(-1, 1)
        exact_i_0 = torch.Tensor([exact_i_0]).reshape(-1, 1)
        exact_r_0 = torch.Tensor([exact_r_0]).reshape(-1, 1)

        synthetic_data = get_syntethic_data(model=sir, t_final=t_final, i_0=exact_i_0, r_0=exact_r_0,
                                            exact_beta=exact_beta, exact_gamma=exact_gamma,
                                            size=exact_size)

        min_loss = 1000

        initial_conditions_set = [t_0, i_0_set, r_0_set]

        for n in range(n_trials):
            # Search optimal params
            fit_epochs = 100
            optimal_i_0, optimal_r_0, optimal_beta, optimal_gamma, rnd_init, loss = fit(sir,
                                                                                        init_bundle=initial_conditions_set,
                                                                                        betas=betas,
                                                                                        gammas=gammas,
                                                                                        steps=train_size, lr=1e-3,
                                                                                        known_points=synthetic_data,
                                                                                        writer=None,
                                                                                        epochs=fit_epochs,
                                                                                        verbose=False,
                                                                                        loss_mode=loss_mode)

            # Compute the score
            score = (optimal_beta.item() - exact_beta) ** 2 + (optimal_gamma.item() - exact_gamma) ** 2 + (
                    optimal_i_0.item() - exact_i_0) ** 2 + (optimal_r_0.item() - exact_r_0) ** 2

            if loss < min_loss:
                params_scores[(exact_beta.item(), exact_gamma.item())] = score.item()
                min_loss = loss

    points_beta_gamma = list(params_scores.keys())
    points_beta_gamma = [list(p) for p in points_beta_gamma]

    points_init = list(init_scores.keys())
    points_init = [list(p) for p in points_init]

    params_scores = list(params_scores.values())
    init_scores = list(init_scores.values())

    beta_bundle = [p[0] for p in points_beta_gamma]
    gammas_bundle = [p[1] for p in points_beta_gamma]
    i_0_bundle = [p[0] for p in points_init]
    r_0_bundle = [p[1] for p in points_init]

    plt.figure(figsize=(15, 8))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.subplot(1, 2, 1)
    plt.title('Visualization of score \n'
              'with respect to Beta and Gamma\n'
              'I(0) in  {} | R(0) in {}\n'
              ' Beta in  {} | Gamma in {}'.format(i_0_set, r_0_set, betas, gammas), fontsize=18)
    plt.tricontourf(beta_bundle, gammas_bundle, params_scores,
                    cmap=nlcmap(plt.cm.RdYlGn_r, levels=[1, 5, 7, 10]))
    param_rectangle = Rectangle(xy=(betas[0], gammas[0]), width=betas[1] - betas[0],
                                height=gammas[1] - gammas[0], alpha=0.2)
    plt.gca().add_patch(param_rectangle)
    plt.axvline(betas[0], linestyle='--')
    plt.axvline(betas[1], linestyle='--')
    plt.axhline(gammas[0], linestyle='--')
    plt.axhline(gammas[1], linestyle='--')
    plt.xlabel('Beta', fontsize=15)
    plt.ylabel('Gamma', fontsize=15)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=13)

    plt.subplot(1, 2, 2)
    plt.title('Visualization of score \n'
              'with respect to Beta and Gamma\n'
              'I(0) in  {} | R(0) in {}\n'
              ' Beta in  {} | Gamma in {}'.format(i_0_set, r_0_set, betas, gammas), fontsize=18)
    plt.tricontourf(i_0_bundle, r_0_bundle, init_scores,
                    cmap=nlcmap(plt.cm.RdYlGn_r, levels=[1, 5, 7, 10]))
    param_rectangle = Rectangle(xy=(i_0_set[0], r_0_set[0]), width=i_0_set[1] - i_0_set[0],
                                height=r_0_set[1] - r_0_set[0], alpha=0.2)
    plt.gca().add_patch(param_rectangle)
    plt.axvline(i_0_set[0], linestyle='--')
    plt.axvline(i_0_set[1], linestyle='--')
    plt.axhline(r_0_set[0], linestyle='--')
    plt.axhline(r_0_set[1], linestyle='--')
    plt.xlabel('I(0)', fontsize=15)
    plt.ylabel('R(0)', fontsize=15)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=13)
    plt.show()
