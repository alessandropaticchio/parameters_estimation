from constants import ROOT_DIR
from random import uniform
from models import SIRNetwork
from losses import sir_loss
from torch.utils.data import DataLoader
from nclmap import nlcmap
from tqdm import tqdm
from constants import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from matplotlib.patches import Rectangle
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# File to analyze the trend of the loss as a function of initial conditions and parameters,
# within and outside the bundle, in this case by using 5 different models trained on the same bundle.
# The losses are averaged to avoid fluctuations.

if __name__ == '__main__':
    within_bundle = False
    max_model_index = 5

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

    if within_bundle:
        betas_std = 0
        gammas_std = 0
        i_0_std = 0
        r_0_std = 0
    else:
        betas_std = 0.3 * betas[1]
        gammas_std = 0.3 * gammas[1]
        i_0_std = 0.3 * i_0_set[1]
        r_0_std = 0.3 * r_0_set[1]

    try:
        import csv

        with open('csv\\i_0_sampled.csv', newline='') as f:
            reader = csv.reader(f)
            i_0_sampled = list(reader)
            i_0_sampled = [item for sublist in i_0_sampled for item in sublist]
            i_0_sampled = [float(x) for x in i_0_sampled]

        with open('csv\\r_0_sampled.csv', newline='') as f:
            reader = csv.reader(f)
            r_0_sampled = list(reader)
            r_0_sampled = [item for sublist in r_0_sampled for item in sublist]
            r_0_sampled = [float(x) for x in r_0_sampled]

        with open('csv\\beta_sampled.csv', newline='') as f:
            reader = csv.reader(f)
            beta_sampled = list(reader)
            beta_sampled = [item for sublist in beta_sampled for item in sublist]
            beta_sampled = [float(x) for x in beta_sampled]


        with open('csv\\gamma_sampled.csv', newline='') as f:
            reader = csv.reader(f)
            gamma_sampled = list(reader)
            gamma_sampled = [item for sublist in gamma_sampled for item in sublist]
            gamma_sampled = [float(x) for x in gamma_sampled]

        with open('csv\\log_params_losses.csv', newline='') as f:
            reader = csv.reader(f)
            log_params_losses = list(reader)
            log_params_losses = [item for sublist in log_params_losses for item in sublist]
            log_params_losses = [float(x) for x in log_params_losses]

        with open('csv\\log_inits_losses.csv', newline='') as f:
            reader = csv.reader(f)
            log_inits_losses = list(reader)
            log_inits_losses = [item for sublist in log_inits_losses for item in sublist]
            log_inits_losses = [float(x) for x in log_inits_losses]

    except:
        # Init model
        sir = SIRNetwork(input=5, layers=4, hidden=50)

        models = []
        for model_index in range(1, max_model_index + 1, 1):
            model_name = '({})_i_0={}_r_0={}_betas={}_gammas={}.pt'.format(model_index, i_0_set, r_0_set,
                                                                           betas,
                                                                           gammas)
            # Init model
            sir = SIRNetwork(input=5, layers=4, hidden=50)
            checkpoint = torch.load(
                ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))
            # Load the model
            sir.load_state_dict(checkpoint['model_state_dict'])
            models.append(sir)

        n_draws = 300
        steps = 20

        # Will save the loss as a function of initial conditions
        init_losses = {}

        # Fix beta, gamma and compute loss as a function of initial conditions
        for n in tqdm(range(n_draws), desc='Computing losses'):
            current_loss = 0.

            beta = (betas[1] + betas[0]) / 2
            gamma = (gammas[1] + gammas[0]) / 2
            i_0 = uniform(max(0, i_0_set[0] - i_0_std), max(0, i_0_set[1] + i_0_std))
            r_0 = uniform(max(0, r_0_set[0] - r_0_std), max(0, r_0_set[1] + r_0_std))
            s_0 = 1 - (i_0 + r_0)

            # Generate points between 0 and t_final
            grid = torch.linspace(t_0, t_final, steps=steps).reshape(-1, 1)
            t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)

            s_0 = torch.Tensor([s_0]).reshape(-1, 1)
            i_0 = torch.Tensor([i_0]).reshape(-1, 1)
            r_0 = torch.Tensor([r_0]).reshape(-1, 1)
            beta = torch.Tensor([beta]).reshape(-1, 1)
            gamma = torch.Tensor([gamma]).reshape(-1, 1)
            initial_conditions = [s_0, i_0, r_0]

            for i, t in enumerate(t_dl, 0):
                t.requires_grad = True
                avg_loss = 0.
                # Network solutions
                for sir in models:
                    s, i, r = sir.parametric_solution(t, initial_conditions, beta=beta, gamma=gamma,
                                                      mode='bundle_total')
                    avg_loss += sir_loss(t, s, i, r, beta, gamma)
                avg_loss = avg_loss / max_model_index
                current_loss += avg_loss

                init_losses[(i_0.item(), r_0.item())] = current_loss

        inits = list(init_losses.keys())
        inits = [list(p) for p in inits]
        i_0_sampled = [p[0] for p in inits]
        r_0_sampled = [p[1] for p in inits]

        inits_losses = list(init_losses.values())
        inits_losses = [l.item() for l in inits_losses]
        log_inits_losses = np.log(inits_losses)

        params_losses = {}

        for n in tqdm(range(n_draws), desc='Computing losses'):
            current_loss = 0.

            beta = uniform(max(0, betas[0] - betas_std), max(0, betas[1] + betas_std))
            gamma = uniform(max(0, gammas[0] - gammas_std), max(0, gammas[1] + gammas_std))
            i_0 = (i_0_set[1] + i_0_set[0]) / 2
            r_0 = (r_0_set[1] + r_0_set[0]) / 2
            s_0 = 1 - (i_0 + r_0)

            # Generate points between 0 and t_final
            grid = torch.linspace(t_0, t_final, steps=steps).reshape(-1, 1)
            t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)

            s_0 = torch.Tensor([s_0]).reshape(-1, 1)
            i_0 = torch.Tensor([i_0]).reshape(-1, 1)
            r_0 = torch.Tensor([r_0]).reshape(-1, 1)
            beta = torch.Tensor([beta]).reshape(-1, 1)
            gamma = torch.Tensor([gamma]).reshape(-1, 1)
            initial_conditions = [s_0, i_0, r_0]

            for i, t in enumerate(t_dl, 0):
                t.requires_grad = True
                avg_loss = 0.
                # Network solutions
                for sir in models:
                    s, i, r = sir.parametric_solution(t, initial_conditions, beta=beta, gamma=gamma,
                                                      mode='bundle_total')
                    avg_loss += sir_loss(t, s, i, r, beta, gamma)
                avg_loss = avg_loss / max_model_index
                current_loss += avg_loss

                params_losses[(beta.item(), gamma.item())] = current_loss

        params = list(params_losses.keys())
        params = [list(p) for p in params]
        beta_sampled = [p[0] for p in params]
        gamma_sampled = [p[1] for p in params]

        params_losses = list(params_losses.values())
        params_losses = [l.item() for l in params_losses]
        log_params_losses = np.log(params_losses)

        # File saving
        import csv
        with open('csv\\beta_sampled.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(beta_sampled)

        with open('csv\\gamma_sampled.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(gamma_sampled)

        with open('csv\\i_0_sampled.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(i_0_sampled)

        with open('csv\\r_0_sampled.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(r_0_sampled)

        with open('csv\\log_inits_losses.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(log_inits_losses)

        with open('csv\\log_params_losses.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(log_params_losses)


    from matplotlib import ticker

    def fmt(x, pos):
        x = int(x)
        return r'$10^{{{}}}$'.format(x)


    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel('$I_{0}$', fontsize=labelsize+5)
    plt.ylabel('$R_{0}$', fontsize=labelsize+5)
    if within_bundle:
        plt.tricontourf(i_0_sampled, r_0_sampled, log_inits_losses,
                        cmap=nlcmap(plt.cm.YlGn_r, levels=[1, 2, 2.1, 2.2, 2.3, 2.4, 3]))
    if not within_bundle:
        plt.tricontourf(i_0_sampled, r_0_sampled, log_inits_losses,
                        cmap=plt.cm.RdYlGn_r)
        param_rectangle = Rectangle(xy=(i_0_set[0], r_0_set[0]), width=i_0_set[1] - i_0_set[0],
                                    height=r_0_set[1] - r_0_set[0], alpha=0.7, facecolor=None, ec='black', fill=False, lw=5, ls='--')
        plt.gca().add_patch(param_rectangle)
    plt.xlim(max(0, i_0_set[0] - i_0_std/3), max(0, i_0_set[1] + i_0_std/3))
    plt.ylim(max(0, r_0_set[0] - r_0_std/3), max(0, r_0_set[1] + r_0_std/3))
    clb = plt.colorbar(ticks=[-3, -4, -5, -6, -7, -8, -9, -10, -11, -12], format=ticker.FuncFormatter(fmt))
    clb.ax.tick_params(labelsize=ticksize)
    clb.set_label('L', labelpad=-62, y=1.05, rotation=0, fontsize=labelsize)
    # plt.show()
    plt.subplot(1, 2, 2)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel(r'$\beta$', fontsize=labelsize+5)
    plt.ylabel(r'$\gamma$', fontsize=labelsize+5)
    from matplotlib import ticker
    if within_bundle:
        plt.tricontourf(beta_sampled, gamma_sampled, log_params_losses,
                        cmap=nlcmap(plt.cm.YlGn_r, levels=[1, 2, 2.1, 2.2, 2.3, 2.4, 3]))
    if not within_bundle:
        plt.tricontourf(beta_sampled, gamma_sampled, log_params_losses, cmap=plt.cm.RdYlGn_r)
        param_rectangle = Rectangle(xy=(betas[0], gammas[0]), width=betas[1] - betas[0], height=gammas[1] - gammas[0],
                                    alpha=0.7, facecolor=None, ec='black', fill=False, lw=5, ls='--')
        plt.gca().add_patch(param_rectangle)
    plt.xlim(max(0, betas[0] - betas_std/3), max(0, betas[1] + betas_std/3))
    plt.ylim(max(0, gammas[0] - gammas_std/3), max(0, gammas[1] + gammas_std/3))
    clb = plt.colorbar(ticks=[-3, -4, -5, -6, -7, -8, -9, -10, -11, -12], format=ticker.FuncFormatter(fmt))
    clb.ax.tick_params(labelsize=ticksize)
    clb.set_label('Loss', labelpad=-66, y=1.05, rotation=0, fontsize=labelsize)
    plt.savefig(ROOT_DIR + '/multi_loss_analysis.png')
    plt.show()
