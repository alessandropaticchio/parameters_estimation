from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import torch
from losses import seir_loss


# Define the sin() activation function
class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)


class SIRNetwork(torch.nn.Module):
    def __init__(self, activation=None, input=1, layers=2, hidden=10, output=3):

        self.n_output = output

        super(SIRNetwork, self).__init__()
        if activation is None:
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = activation

        self.fca = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            self.activation
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input, hidden),
            *[self.fca for _ in range(layers)],
            torch.nn.Linear(hidden, output)
        )

    def forward(self, x):
        x = self.ffn(x)
        s_N = (x[:, 0]).reshape(-1, 1)
        e_N = (x[:, 1]).reshape(-1, 1)
        i_N = (x[:, 2]).reshape(-1, 1)
        r_N = (x[:, 3]).reshape(-1, 1)
        return s_N, e_N, i_N, r_N

    def parametric_solution(self, t, initial_conditions, beta=None, gamma=None, lam=None):
        # Parametric solutions
        t_0 = 0
        s_0, e_0, i_0, r_0 = initial_conditions[0][:], initial_conditions[1][:], initial_conditions[2][:], \
                             initial_conditions[3][:]

        dt = t - t_0

        f = (1 - torch.exp(-dt))

        t_bundle = torch.cat([t, e_0, i_0, r_0, beta, gamma, lam], dim=1)

        N = self.forward(t_bundle)

        N1, N2, N3, N4 = N

        # Concatenate to go into softmax
        to_softmax = torch.cat([N1, N2, N3, N4], dim=1)
        softmax_output = softmax(to_softmax, dim=1)
        N1, N2, N3, N4 = softmax_output[:, 0], softmax_output[:, 1], softmax_output[:, 2], softmax_output[:, 3]
        N1, N2, N3, N4 = N1.reshape(-1, 1), N2.reshape(-1, 1), N3.reshape(-1, 1), N4.reshape(-1, 1)

        s_hat = (s_0 + f * (N1 - s_0))
        e_hat = (e_0 + f * (N2 - e_0))
        i_hat = (i_0 + f * (N3 - i_0))
        r_hat = (r_0 + f * (N4 - r_0))

        return s_hat, e_hat, i_hat, r_hat

    # Use the model to provide a solution with a given set of initial conditions and parameters
    def solve(self, e_0, i_0, r_0, beta, gamma, lam, t_0=0, t_final=20, size=20):
        s_0 = 1 - e_0 - i_0 - r_0

        # Test between 0 and t_final
        step = (t_final - t_0) / size
        grid = torch.arange(t_0, t_final, step=step, out=torch.FloatTensor()).reshape(-1, 1)
        t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)
        s_hat = []
        e_hat = []
        i_hat = []
        r_hat = []

        # Convert initial conditions, beta and gamma to tensor for prediction
        beta_t = torch.Tensor([beta]).reshape(-1, 1)
        gamma_t = torch.Tensor([gamma]).reshape(-1, 1)
        lam_t = torch.Tensor([lam]).reshape(-1, 1)
        s_0_t = torch.Tensor([s_0]).reshape(-1, 1)
        e_0_t = torch.Tensor([e_0]).reshape(-1, 1)
        i_0_t = torch.Tensor([i_0]).reshape(-1, 1)
        r_0_t = torch.Tensor([r_0]).reshape(-1, 1)
        initial_conditions_set = [s_0_t, e_0_t, i_0_t, r_0_t]

        de_loss = 0.
        for i, t in enumerate(t_dl, 0):
            t.requires_grad = True

            # Network solutions
            s, e, i, r = self.parametric_solution(t, initial_conditions_set, beta=beta_t, gamma=gamma_t, lam=lam_t)
            s_hat.append(s.item())
            e_hat.append(e.item())
            i_hat.append(i.item())
            r_hat.append(r.item())

            de_loss += seir_loss(t, s, e, i, r, beta_t, gamma_t, lam_t)

        return s_hat, e_hat, i_hat, r_hat, de_loss
