from tqdm import tqdm
from random import shuffle
import torch
import copy
import numpy as np
from numpy.random import uniform


def fit(model, init_bundle, betas, gammas, lams, known_points, steps, writer, epochs=100, lr=8e-4, loss_mode='mse',
        susceptible_weight=1., exposed_weight=1.,
        recovered_weight=1., force_init=False, verbose=False):
    model.eval()

    # Sample randomly initial conditions, beta and gamma
    e_0 = uniform(init_bundle[1][0], init_bundle[1][1], size=1)
    i_0 = uniform(init_bundle[2][0], init_bundle[2][1], size=1)
    r_0 = uniform(init_bundle[3][0], init_bundle[3][1], size=1)
    beta = uniform(betas[0], betas[1], size=1)
    gamma = uniform(gammas[0], gammas[1], size=1)
    lam = uniform(lams[0], lams[1], size=1)

    beta, gamma, lam = torch.Tensor([beta]).reshape(-1, 1), torch.Tensor([gamma]).reshape(-1, 1), torch.Tensor([lam]).reshape(-1, 1)

    # if force_init == True fix the initial conditions as the ones given and find only the params
    if force_init:
        i_0 = known_points[0][1]
        e_0 = torch.Tensor([e_0]).reshape(-1, 1)
        i_0 = torch.Tensor([i_0]).reshape(-1, 1)
        r_0 = torch.Tensor([r_0]).reshape(-1, 1)
        optimizer = torch.optim.SGD([beta, gamma, lam, e_0, r_0], lr=lr)
    else:
        e_0 = torch.Tensor([e_0]).reshape(-1, 1)
        i_0 = torch.Tensor([i_0]).reshape(-1, 1)
        r_0 = torch.Tensor([r_0]).reshape(-1, 1)
        optimizer = torch.optim.SGD([e_0, i_0, r_0, beta, gamma, lam], lr=lr)

    s_0 = 1 - (e_0 + i_0 + r_0)

    rnd_init = [round(e_0.item(), 5), round(i_0.item(), 5), round(r_0.item(), 5), round(beta.item(), 5),
                round(gamma.item(), 5), round(lam.item(), 5)]

    # Set requires_grad = True to the inputs to allow backprop
    e_0.requires_grad = True
    i_0.requires_grad = True
    r_0.requires_grad = True
    beta.requires_grad = True
    gamma.requires_grad = True

    initial_conditions = [s_0, e_0, i_0, r_0]

    known_t = copy.deepcopy(list(known_points.keys()))

    losses = []

    # Iterate for epochs to find best initial conditions, beta, and gamma that optimizes the MSE/Cross Entropy between
    # my prediction and the real data
    for epoch in tqdm(range(epochs), desc='Finding the best inputs', disable=not verbose):
        optimizer.zero_grad()

        loss = 0.

        # Take the time points and shuffle them
        shuffle(known_t)

        for t in known_t:
            target = known_points[t]

            t_tensor = torch.Tensor([t]).reshape(-1, 1)

            s_hat, e_hat, i_hat, r_hat = model.parametric_solution(t_tensor, initial_conditions, beta=beta, gamma=gamma,
                                                                   lam=lam,
                                                                   mode='bundle_total')

            i_target = target[1]
            r_target = target[2]

            if loss_mode == 'mse':
                loss_i = (i_target - i_hat).pow(2)
                loss_r = (r_target - r_hat).pow(2)
            elif loss_mode == 'cross_entropy':
                loss_i = - i_target * torch.log(i_hat + 1e-10)
                loss_r = - r_target * torch.log(r_hat + 1e-10)
            else:
                raise ValueError('Invalid loss specification!')

            # Regularization term to prevent r_0 to go negative in extreme cases:
            if loss_mode == 'cross_entropy':
                regularization = torch.exp(-0.01 * r_0)
            else:
                regularization = 0.

            # Weighting that regularizes how much we want to weight the Recovered/Susceptible curve
            loss_r = loss_r * recovered_weight

            loss += loss_i + loss_r + regularization

        loss = loss / len(known_points.keys())
        losses.append(loss)

        loss.backward(retain_graph=True)
        optimizer.step()

        # To prevent r_0 to go negative in extreme cases:
        if r_0 < 0.:
            r_0 = torch.clamp(r_0, 0, 10000)

        # Adjust s_0 after update of i_0 and r_0, and update initial_conditions
        s_0 = 1 - (e_0 + i_0 + r_0)
        initial_conditions = [s_0, e_0, i_0, r_0]

        if writer:
            writer.add_scalar('Loss/train', loss, epoch)

    return e_0, i_0, r_0, beta, gamma, lam, rnd_init, losses[-1]


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    ce = -np.sum(targets * np.log(predictions + epsilon))
    return ce
