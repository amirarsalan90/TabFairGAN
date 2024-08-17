import torch
import torch.nn.functional as f
from torch import nn

class Generator(nn.Module):
    def __init__(self, input_dim, continuous_columns, discrete_columns):
        super(Generator, self).__init__()
        self._input_dim = input_dim
        self._discrete_columns = discrete_columns
        self._num_continuous_columns = len(continuous_columns)

        self.lin1 = nn.Linear(self._input_dim, self._input_dim)
        self.lin_numerical = nn.Linear(self._input_dim, self._num_continuous_columns)

        self.lin_cat = nn.ModuleDict()
        for key, value in self._discrete_columns.items():
            self.lin_cat[key] = nn.Linear(self._input_dim, value)

    def forward(self, x):
        x = torch.relu(self.lin1(x))
        # x = f.leaky_relu(self.lin1(x))
        # x_numerical = f.leaky_relu(self.lin_numerical(x))
        x_numerical = f.relu(self.lin_numerical(x))
        x_cat = []
        for key in self.lin_cat:
            x_cat.append(f.gumbel_softmax(self.lin_cat[key](x), tau=0.2))
        x_final = torch.cat((x_numerical, *x_cat), 1)
        return x_final


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self._input_dim = input_dim
        # self.dense1 = nn.Linear(109, 256)
        self.dense1 = nn.Linear(self._input_dim, self._input_dim)
        self.dense2 = nn.Linear(self._input_dim, self._input_dim)
        # self.dense3 = nn.Linear(256, 1)
        # self.drop = nn.Dropout(p=0.2)
        # self.activation = nn.Sigmoid()

    def forward(self, x):
        x = f.leaky_relu(self.dense1(x))
        # x = self.drop(x)
        # x = f.leaky_relu(self.dense2(x))
        x = f.leaky_relu(self.dense2(x))
        # x = self.drop(x)
        return x


class FairLossFunc(nn.Module):
    def __init__(self, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index):
        super(FairLossFunc, self).__init__()
        self._S_start_index = S_start_index
        self._Y_start_index = Y_start_index
        self._underpriv_index = underpriv_index
        self._priv_index = priv_index
        self._undesire_index = undesire_index
        self._desire_index = desire_index

    def forward(self, x, crit_fake_pred, lamda):
        G = x[:, self._S_start_index:self._S_start_index + 2]
        # print(x[0,64])
        I = x[:, self._Y_start_index:self._Y_start_index + 2]
        # disp = (torch.mean(G[:,1]*I[:,1])/(x[:,65].sum())) - (torch.mean(G[:,0]*I[:,0])/(x[:,64].sum()))
        # disp = -1.0 * torch.tanh(torch.mean(G[:,0]*I[:,1])/(x[:,64].sum()) - torch.mean(G[:,1]*I[:,1])/(x[:,65].sum()))
        # gen_loss = -1.0 * torch.mean(crit_fake_pred)
        disp = -1.0 * lamda * (torch.mean(G[:, self._underpriv_index] * I[:, self._desire_index]) / (
            x[:, self._S_start_index + self._underpriv_index].sum()) - torch.mean(
            G[:, self._priv_index] * I[:, self._desire_index]) / (
                                   x[:, self._S_start_index + self._priv_index].sum())) - 1.0 * torch.mean(
            crit_fake_pred)
        # print(disp)
        return disp



def get_gradient(crit, real, fake, epsilon):
    mixed_data = real * epsilon + fake * (1 - epsilon)

    mixed_scores = crit(mixed_data)

    gradient = torch.autograd.grad(
        inputs=mixed_data,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)

    return gen_loss


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp

    return crit_loss