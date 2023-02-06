import torch
import torch.nn.functional as f
from torch import nn
import pandas as pd
import numpy as np
from collections import OrderedDict

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import argparse


parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='command')
with_fairness = subparser.add_parser('with_fairness')
no_fairness = subparser.add_parser('no_fairness')

with_fairness.add_argument("df_name", help="Reference dataframe", type=str)
with_fairness.add_argument("S", help="Protected attribute", type=str)
with_fairness.add_argument("Y", help="Label (decision)", type=str)
with_fairness.add_argument("underprivileged_value", help="Value for underpriviledged group", type=str)
with_fairness.add_argument("desirable_value", help="Desired label (decision)", type=str)
with_fairness.add_argument("num_epochs", help="Total number of epochs", type=int)
with_fairness.add_argument("batch_size", help="the batch size", type=int)
with_fairness.add_argument("num_fair_epochs", help="number of fair training epochs", type=int)
with_fairness.add_argument("lambda_val", help="lambda parameter", type=float)
with_fairness.add_argument("fake_name", help="name of the produced csv file", type=str)
with_fairness.add_argument("size_of_fake_data", help="how many data records to generate", type=int)


no_fairness.add_argument("df_name", help="Reference dataframe", type=str)
no_fairness.add_argument("num_epochs", help="Total number of epochs", type=int)
no_fairness.add_argument("batch_size", help="the batch size", type=int)
no_fairness.add_argument("fake_name", help="name of the produced csv file", type=str)
no_fairness.add_argument("size_of_fake_data", help="how many data records to generate", type=int)

args = parser.parse_args()

if args.command == 'with_fairness':
    S = args.S
    Y = args.Y
    S_under = args.underprivileged_value
    Y_desire = args.desirable_value

    df = pd.read_csv(args.df_name)

    df[S] = df[S].astype(object)
    df[Y] = df[Y].astype(object)

elif args.command == 'no_fairness':
    df = pd.read_csv(args.df_name)


if args.command == "with_fairness":
    def get_ohe_data(df):
        df_int = df.select_dtypes(['float', 'integer']).values
        continuous_columns_list = list(df.select_dtypes(['float', 'integer']).columns)
        ##############################################################
        scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
        df_int = scaler.fit_transform(df_int)

        df_cat = df.select_dtypes('object')
        df_cat_names = list(df.select_dtypes('object').columns)
        numerical_array = df_int
        ohe = OneHotEncoder()
        ohe_array = ohe.fit_transform(df_cat)

        cat_lens = [i.shape[0] for i in ohe.categories_]
        discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))

        S_start_index = len(continuous_columns_list) + sum(
            list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(S)])
        Y_start_index = len(continuous_columns_list) + sum(
            list(discrete_columns_ordereddict.values())[:list(discrete_columns_ordereddict.keys()).index(Y)])

        if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(S)][0] == S_under:
            underpriv_index = 0
            priv_index = 1
        else:
            underpriv_index = 1
            priv_index = 0
        if ohe.categories_[list(discrete_columns_ordereddict.keys()).index(Y)][0] == Y_desire:
            desire_index = 0
            undesire_index = 1
        else:
            desire_index = 1
            undesire_index = 0

        final_array = np.hstack((numerical_array, ohe_array.toarray()))
        return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index

elif args.command == "no_fairness":
    def get_ohe_data(df):
        df_int = df.select_dtypes(['float', 'integer']).values
        continuous_columns_list = list(df.select_dtypes(['float', 'integer']).columns)
        ##############################################################
        scaler = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')
        df_int = scaler.fit_transform(df_int)

        df_cat = df.select_dtypes('object')
        df_cat_names = list(df.select_dtypes('object').columns)
        numerical_array = df_int
        ohe = OneHotEncoder()
        ohe_array = ohe.fit_transform(df_cat)

        cat_lens = [i.shape[0] for i in ohe.categories_]
        discrete_columns_ordereddict = OrderedDict(zip(df_cat_names, cat_lens))


        final_array = np.hstack((numerical_array, ohe_array.toarray()))
        return ohe, scaler, discrete_columns_ordereddict, continuous_columns_list, final_array


def get_original_data(df_transformed, df_orig, ohe, scaler):
    df_ohe_int = df_transformed[:, :df_orig.select_dtypes(['float', 'integer']).shape[1]]
    df_ohe_int = scaler.inverse_transform(df_ohe_int)
    df_ohe_cats = df_transformed[:, df_orig.select_dtypes(['float', 'integer']).shape[1]:]
    df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
    df_int = pd.DataFrame(df_ohe_int, columns=df_orig.select_dtypes(['float', 'integer']).columns)
    df_cat = pd.DataFrame(df_ohe_cats, columns=df_orig.select_dtypes('object').columns)
    return pd.concat([df_int, df_cat], axis=1)


if args.command == "with_fairness":
    def prepare_data(df, batch_size):
        ohe, scaler, discrete_columns, continuous_columns, df_transformed, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = get_ohe_data(df)
        input_dim = df_transformed.shape[1]
        X_train, X_test = train_test_split(df_transformed,test_size=0.1, shuffle=True)
        data_train = X_train.copy()
        data_test = X_test.copy()

        from torch.utils.data import TensorDataset
        from torch.utils.data import DataLoader
        data = torch.from_numpy(data_train).float()


        train_ds = TensorDataset(data)
        train_dl = DataLoader(train_ds, batch_size = batch_size, drop_last=True)
        return ohe, scaler, input_dim, discrete_columns, continuous_columns ,train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index

elif args.command == "no_fairness":
    def prepare_data(df, batch_size):
    #df = pd.concat([df_train, df_test], axis=0)

        ohe, scaler, discrete_columns, continuous_columns, df_transformed = get_ohe_data(df)


        input_dim = df_transformed.shape[1]

        #from sklearn.model_selection import train_test_split
        #################
        X_train, X_test = train_test_split(df_transformed,test_size=0.1, shuffle=True) #random_state=10)
        #X_train = df_transformed[:df_train.shape[0],:]
        #X_test = df_transformed[df_train.shape[0]:,:]

        data_train = X_train.copy()
        data_test = X_test.copy()

        from torch.utils.data import TensorDataset
        from torch.utils.data import DataLoader
        data = torch.from_numpy(data_train).float()


        train_ds = TensorDataset(data)
        train_dl = DataLoader(train_ds, batch_size = batch_size, drop_last=True)
        return ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test



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













device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")


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


display_step = 50


def train(df, epochs=500, batch_size=64, fair_epochs=10, lamda=0.5):
    if args.command == "with_fairness":
        ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = prepare_data(df, batch_size)
    elif args.command == "no_fairness":
        ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test = prepare_data(df, batch_size)

    generator = Generator(input_dim, continuous_columns, discrete_columns).to(device)
    critic = Critic(input_dim).to(device)
    if args.command == "with_fairness":
        second_critic = FairLossFunc(S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index).to(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    gen_optimizer_fair = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    crit_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # loss = nn.BCELoss()
    critic_losses = []
    cur_step = 0
    for i in range(epochs):
        # j = 0
        print("epoch {}".format(i + 1))
        ############################
        if i + 1 <= (epochs - fair_epochs):
            print("training for accuracy")
        if i + 1 > (epochs - fair_epochs):
            print("training for fairness")
        for data in train_dl:
            data[0] = data[0].to(device)
            crit_repeat = 4
            mean_iteration_critic_loss = 0
            for k in range(crit_repeat):
                # training the critic
                crit_optimizer.zero_grad()
                fake_noise = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake = generator(fake_noise)

                crit_fake_pred = critic(fake.detach())
                crit_real_pred = critic(data[0])

                epsilon = torch.rand(batch_size, input_dim, device=device, requires_grad=True)
                gradient = get_gradient(critic, data[0], fake.detach(), epsilon)
                gp = gradient_penalty(gradient)

                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda=10)

                mean_iteration_critic_loss += crit_loss.item() / crit_repeat
                crit_loss.backward(retain_graph=True)
                crit_optimizer.step()
            #############################
            if cur_step > 50:
                critic_losses += [mean_iteration_critic_loss]

            #############################
            if i + 1 <= (epochs - fair_epochs):
                # training the generator for accuracy
                gen_optimizer.zero_grad()
                fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake_2 = generator(fake_noise_2)
                crit_fake_pred = critic(fake_2)

                gen_loss = get_gen_loss(crit_fake_pred)
                gen_loss.backward()

                # Update the weights
                gen_optimizer.step()

            ###############################
            if i + 1 > (epochs - fair_epochs):
                # training the generator for fairness
                gen_optimizer_fair.zero_grad()
                fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=device).float()
                fake_2 = generator(fake_noise_2)

                crit_fake_pred = critic(fake_2)

                gen_fair_loss = second_critic(fake_2, crit_fake_pred, lamda)
                gen_fair_loss.backward()
                gen_optimizer_fair.step()
            cur_step += 1

    return generator, critic, ohe, scaler, data_train, data_test, input_dim


def train_plot(df, epochs, batchsize, fair_epochs, lamda):
    generator, critic, ohe, scaler, data_train, data_test, input_dim = train(df, epochs, batchsize, fair_epochs, lamda)
    return generator, critic, ohe, scaler, data_train, data_test, input_dim


if args.command == "with_fairness":
    generator, critic, ohe, scaler, data_train, data_test, input_dim = train_plot(df, args.num_epochs, args.batch_size, args.num_fair_epochs, args.lambda_val)
elif args.command == "no_fairness":
    generator, critic, ohe, scaler, data_train, data_test, input_dim = train_plot(df, args.num_epochs, args.batch_size, 0, 0)
fake_numpy_array = generator(torch.randn(size=(args.size_of_fake_data, input_dim), device=device)).cpu().detach().numpy()
fake_df = get_original_data(fake_numpy_array, df, ohe, scaler)
fake_df = fake_df[df.columns]
fake_df.to_csv(args.fake_name, index=False)
