from constants import ROOT_DIR
from random import uniform
from models import SIRNetwork
from losses import sir_loss
from torch.utils.data import DataLoader
from nclmap import nlcmap
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# File to analyze the trend of the loss as a function of initial conditions and parameters,
# within and outside the bundle

if __name__ == '__main__':
    t_0 = 0
    t_final = 20

    # The intervals in which the equation parameters and the initial conditions should vary
    betas = [0.4, 0.8]
    gammas = [0.3, 0.7]
    i_0_set = [0.2, 0.4]
    r_0_set = [0.1, 0.3]
    initial_conditions_set = []
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(i_0_set)
    initial_conditions_set.append(r_0_set)

    # Init model
    sir = SIRNetwork(input=5, layers=4, hidden=50)

    model_name = '(2)_i_0={}_r_0={}_betas={}_gammas={}.pt'.format(i_0_set, r_0_set,
                                                              betas,
                                                              gammas)
    checkpoint = torch.load(
        ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))
    # Load the model
    sir.load_state_dict(checkpoint['model_state_dict'])

    n_draws = 1000

    # Will save the loss as a function of params
    params_losses = {}
    # Will save the loss as a function of initial conditions
    init_losses = {}

    # To sample initial conditions and parameters outside from the bundle
    betas_std = 0.3 * betas[1]
    gammas_std = 0.3 * gammas[1]
    i_0_std = 0.3 * i_0_set[1]
    r_0_std = 0.3 * r_0_set[1]

    for n in tqdm(range(n_draws), desc='Computing losses'):
        current_loss = 0.

        beta = uniform(max(0, betas[0] - betas_std), max(0, betas[1] + betas_std))
        gamma = uniform(max(0, gammas[0] - gammas_std), max(0, gammas[1] + gammas_std))
        i_0 = uniform(max(0, i_0_set[0] - i_0_std), max(0, i_0_set[1] + i_0_std))
        r_0 = uniform(max(0, r_0_set[0] - r_0_std), max(0, r_0_set[1] + r_0_std))
        s_0 = 1 - (i_0 + r_0)

        # Generate points between 0 and t_final
        grid = torch.arange(0, t_final, out=torch.FloatTensor()).reshape(-1, 1)
        t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)

        s_0 = torch.Tensor([s_0]).reshape(-1, 1)
        i_0 = torch.Tensor([i_0]).reshape(-1, 1)
        r_0 = torch.Tensor([r_0]).reshape(-1, 1)
        beta = torch.Tensor([beta]).reshape(-1, 1)
        gamma = torch.Tensor([gamma]).reshape(-1, 1)
        initial_conditions = [s_0, i_0, r_0]

        for i, t in enumerate(t_dl, 0):
            t.requires_grad = True
            # Network solutions
            s, i, r = sir.parametric_solution(t, initial_conditions, beta=beta, gamma=gamma,
                                              mode='bundle_total')
            current_loss += sir_loss(t, s, i, r, beta, gamma)

            params_losses[(beta.item(), gamma.item())] = current_loss
            init_losses[(i_0.item(), r_0.item())] = current_loss

    beta_gamma = list(params_losses.keys())
    beta_gamma = [list(p) for p in beta_gamma]
    beta_sampled = [p[0] for p in beta_gamma]
    gamma_sampled = [p[1] for p in beta_gamma]

    inits = list(init_losses.keys())
    inits = [list(p) for p in inits]
    i_0_sampled = [p[0] for p in inits]
    r_0_sampled = [p[1] for p in inits]

    losses = list(params_losses.values())
    losses = [l.item() for l in losses]
    log_losses = np.log(losses)

    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)

    plt.tricontourf(beta_sampled, gamma_sampled, log_losses, cmap=nlcmap(plt.cm.RdYlGn_r, levels=[1, 5, 10]))

    plt.title('Visualization of DE loss \n'
              'with respect to Beta and Gamma\n'
              'I(0) in  {} | R(0) in {}\n'
              ' Beta in  {} | Gamma in {}'.format(i_0_set, r_0_set, betas, gammas), fontsize=18)
    plt.xlabel('Beta', fontsize=25)
    plt.ylabel('Gamma', fontsize=25)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    param_rectangle = Rectangle(xy=(betas[0], gammas[0]), width=betas[1] - betas[0], height=gammas[1] - gammas[0],
                                alpha=0.2)
    plt.gca().add_patch(param_rectangle)
    plt.axvline(betas[0], linestyle='--')
    plt.axvline(betas[1], linestyle='--')
    plt.axhline(gammas[0], linestyle='--')
    plt.axhline(gammas[1], linestyle='--')
    plt.xlim(max(0, betas[0] - betas_std), max(0, betas[1] + betas_std))
    plt.ylim(max(0, gammas[0] - gammas_std), max(0, gammas[1] + gammas_std))
    clb = plt.colorbar()
    clb.ax.tick_params(labelsize=15)
    clb.set_label('LogLoss', labelpad=-40, y=1.05, rotation=0, fontsize=18)

    plt.subplot(1, 2, 2)
    plt.tricontourf(i_0_sampled, r_0_sampled, log_losses, cmap=nlcmap(plt.cm.RdYlGn_r, levels=[1, 5, 10]))
    plt.title('Visualization of DE loss \n'
              'with respect to I(0) and R(0)\n'
              'I(0) in  {} | R(0) in {}\n'
              ' Beta in  {} | Gamma in {}'.format(i_0_set, r_0_set, betas, gammas), fontsize=18)
    plt.xlabel('I(0)', fontsize=25)
    plt.ylabel('R(0)', fontsize=25)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    param_rectangle = Rectangle(xy=(i_0_set[0], r_0_set[0]), width=i_0_set[1] - i_0_set[0],
                                height=r_0_set[1] - r_0_set[0], alpha=0.2)
    plt.gca().add_patch(param_rectangle)
    plt.axvline(i_0_set[0], linestyle='--')
    plt.axvline(i_0_set[1], linestyle='--')
    plt.axhline(r_0_set[0], linestyle='--')
    plt.axhline(r_0_set[1], linestyle='--')
    plt.xlim(max(0, i_0_set[0] - i_0_std), max(0, i_0_set[1] + i_0_std))
    plt.ylim(max(0, r_0_set[0] - r_0_std), max(0, r_0_set[1] + r_0_std))
    clb = plt.colorbar()
    clb.ax.tick_params(labelsize=15)
    clb.set_label('LogLoss', labelpad=-40, y=1.05, rotation=0, fontsize=18)
    plt.show()
