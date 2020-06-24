import torch
import numpy as np
from torch.autograd import grad
from constants import device, dtype

def seird_loss(t, s, e, i, r, d, beta, gamma, lam, delta, decay=0):
    s_prime = dfx(t, s)
    e_prime = dfx(t, e)
    i_prime = dfx(t, i)
    r_prime = dfx(t, r)
    d_prime = dfx(t, d)

    N = 1

    loss_s = s_prime + (beta * i * s) / N
    loss_e = e_prime - (beta * i * s) / N + lam * e
    loss_i = i_prime - lam * e + (gamma + delta) * i
    loss_r = r_prime - gamma * i
    loss_d = d_prime - delta * i

    # Regularize to give more importance to initial points
    loss_s = loss_s * torch.exp(-decay * t)
    loss_e = loss_e * torch.exp(-decay * t)
    loss_i = loss_i * torch.exp(-decay * t)
    loss_r = loss_r * torch.exp(-decay * t)
    loss_d = loss_d * torch.exp(-decay * t)

    loss_s = (loss_s.pow(2)).mean()
    loss_e = (loss_e.pow(2)).mean()
    loss_i = (loss_i.pow(2)).mean()
    loss_r = (loss_r.pow(2)).mean()
    loss_d = (loss_d.pow(2)).mean()

    total_loss = loss_s + loss_e + loss_i + loss_r + loss_d

    return total_loss

def seir_loss(t, s, e, i, r, beta, gamma, lam, decay=0):
    s_prime = dfx(t, s)
    e_prime = dfx(t, e)
    i_prime = dfx(t, i)
    r_prime = dfx(t, r)

    N = 1

    loss_s = s_prime + (beta * i * s) / N
    loss_e = e_prime - (beta * i * s) / N + lam * e
    loss_i = i_prime - lam * e + gamma * i
    loss_r = r_prime - gamma * i

    # Regularize to give more importance to initial points
    loss_s = loss_s * torch.exp(-decay * t)
    loss_e = loss_e * torch.exp(-decay * t)
    loss_i = loss_i * torch.exp(-decay * t)
    loss_r = loss_r * torch.exp(-decay * t)

    loss_s = (loss_s.pow(2)).mean()
    loss_e = (loss_e.pow(2)).mean()
    loss_i = (loss_i.pow(2)).mean()
    loss_r = (loss_r.pow(2)).mean()

    total_loss = loss_s + loss_e + loss_i + loss_r

    return total_loss

def sir_loss(t, s, i, r, beta, gamma, sigma=0, decay=0):
    s_prime = dfx(t, s)
    i_prime = dfx(t, i)
    r_prime = dfx(t, r)

    N = 1

    loss_s = s_prime + (beta * i * s) / N
    loss_i = i_prime - (beta * i * s) / N + gamma * i
    loss_r = r_prime - gamma * i

    # Regularize to give more importance to initial points
    loss_s = loss_s * torch.exp(-decay * t)
    loss_i = loss_i * torch.exp(-decay * t)
    loss_r = loss_r * torch.exp(-decay * t)

    loss_s = (loss_s.pow(2)).mean()
    loss_i = (loss_i.pow(2)).mean()
    loss_r = (loss_r.pow(2)).mean()

    total_loss = loss_s + loss_i + loss_r

    return total_loss


def mse_loss(known, model, initial_conditions):
    mse_loss = 0.
    for t in known.keys():
        t_tensor = torch.Tensor([t]).reshape(-1, 1)
        s_hat, i_hat, r_hat = model.parametric_solution(t_tensor, initial_conditions)
        loss_s = (known[t][0] - s_hat).pow(2)
        loss_i = (known[t][1] - i_hat).pow(2)
        loss_r = (known[t][2] - i_hat).pow(2)

        mse_loss += loss_s + loss_i + loss_r
    return mse_loss


def trivial_loss(infected, hack_trivial):
    trivial_loss = 0.

    for i in infected:
        trivial_loss += i

    trivial_loss = hack_trivial * torch.exp(- (trivial_loss) ** 2)
    return trivial_loss

def dfx(x, f):
    # Calculate the derivative with auto-differentiation
    x = x.to(device)
    grad_outputs = torch.ones(x.shape, dtype=dtype)
    grad_outputs = grad_outputs.to(device)

    return grad([f], [x], grad_outputs=grad_outputs, create_graph=True)[0]


