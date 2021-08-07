import torch
import torch.nn.functional as f
from torch import nn
import pandas as pd
import numpy as np
import torch
from collections import OrderedDict

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




import argparse
parser = argparse.ArgumentParser()
parser.add_argument("df_name", help="Reference dataframe", type=str)
parser.add_argument("S", help="Protected attribute", type=str)
parser.add_argument("Y", help="Label (decision)", type=str)
parser.add_argument("underprivileged_value", help="Value for underpriviledged group", type=str)
parser.add_argument("desirable_value", help="Desired label (decision)", type=str)

parser.add_argument("num_epochs", help="Total number of epochs", type=int)
parser.add_argument("batch_size", help="the batch size", type=int)
parser.add_argument("num_fair_epochs", help="number of fair training epochs", type=int)
parser.add_argument("lambda_val", help="lambda parameter", type=float)
parser.add_argument("fake_name", help="name of the produced csv file", type=str)
args = parser.parse_args()

S = args.S
Y = args.Y
S_under = args.underprivileged_value
Y_desire = args.desirable_value

df = pd.read_csv(args.df_name)

df = df[df['race'].isin(['White','Black'])]
def binary_admission(first_pf):
    if first_pf == 1:
        return 'yes'
    if first_pf == 0:
        return 'no'
df['first_pf'] = df['first_pf'].apply(binary_admission)


df[S] = df[S].astype(object)
df[Y] = df[Y].astype(object)







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


def get_original_data(df_transformed, df_orig, ohe, scaler):
    # df_int = df_orig.select_dtypes(['float','integer'])
    df_ohe_int = df_transformed[:, :df_orig.select_dtypes(['float', 'integer']).shape[1]]
    df_ohe_int = scaler.inverse_transform(df_ohe_int)
    df_ohe_cats = df_transformed[:, df_orig.select_dtypes(['float', 'integer']).shape[1]:]
    df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
    # df_income = df_transformed[:,-1]
    # df_ohe_cats = np.hstack((df_ohe_cats, df_income.reshape(-1,1)))
    df_int = pd.DataFrame(df_ohe_int, columns=df_orig.select_dtypes(['float', 'integer']).columns)
    df_cat = pd.DataFrame(df_ohe_cats, columns=df_orig.select_dtypes('object').columns)
    return pd.concat([df_int, df_cat], axis=1)











def prepare_data(df, batch_size):
    #df = pd.concat([df_train, df_test], axis=0)

    ohe, scaler, discrete_columns, continuous_columns, df_transformed, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = get_ohe_data(df)


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
    return ohe, scaler, input_dim, discrete_columns, continuous_columns ,train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index


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
    ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index = prepare_data(
        df, batch_size)

    generator = Generator(input_dim, continuous_columns, discrete_columns).to(device)
    critic = Critic(input_dim).to(device)
    second_critic = FairLossFunc(S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index).to(
        device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    gen_optimizer_fair = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    crit_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # loss = nn.BCELoss()
    critic_losses = []
    generator_losses = []
    cur_step = 0
    for i in range(epochs):
        # j = 0
        #print("epoch {}".format(i + 1))
        ############################
        #if i + 1 <= (epochs - fair_epochs):
            #print("training for accuracy")
        #if i + 1 > (epochs - fair_epochs):
            #print("training for fairness")
        for data in train_dl:
            data[0] = data[0].to(device)
            # j += 1
            loss_of_epoch_G = 0
            loss_of_epoch_D = 0
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
            """
            # Keep track of the average generator loss
            #################################
            if cur_step > 50:
                if i + 1 <= (epochs - fair_epochs):
                    generator_losses += [gen_loss.item()]
                if i + 1 > (epochs - fair_epochs):
                    generator_losses += [gen_fair_loss.item()]

                    # print("cr step: {}".format(cur_step))
            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                crit_mean = sum(critic_losses[-display_step:]) / display_step
                print("Step {}: Generator loss: {}, critic loss: {}".format(cur_step, gen_mean, crit_mean))
                step_bins = 20
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss"
                )
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Critic Loss"
                )
                plt.legend()
                plt.show()
	    """
            cur_step += 1

    return generator, critic, ohe, scaler, data_train, data_test, input_dim







#generator, critic, ohe, scaler, data_train, data_test = train_plot(df, args.num_epochs,args.batch_size, args.num_fair_epochs, args.lambda_val)
def train_plot(df, epochs, batchsize, fair_epochs, lamda):
    generator, critic, ohe, scaler, data_train, data_test, input_dim = train(df, epochs, batchsize, fair_epochs, lamda)
    return generator, critic, ohe, scaler, data_train, data_test, input_dim








def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()


def multiple_runs(num_trainings, df, epochs, batchsize, fair_epochs, lamda, outFile):
    print(lamda)
    for i in range(num_trainings):
        print("training number: {}".format(i))
        if i == 0:
            first_line = "num_trainings: %d, num_epochs: %d, batchsize: %d, fair_epochs:%d, lamda:%f" % (
            num_trainings, epochs, batchsize, fair_epochs, lamda)
            print(first_line)
            print2file(first_line, outFile)
            # print2file(buf, outFile)

            second_line = "train_idx ,accuracy_original, accuracy_generated, difference_accuracy, f1_original, f1_generated, difference_f1, demographic_parity_data, demographic_parity_classifier"
            print(second_line)
            print2file(second_line, outFile)

        generator, critic, ohe, scaler, data_train, data_test, input_dim = train_plot(df=df, epochs=epochs, batchsize=batchsize,
                                                                           fair_epochs=fair_epochs, lamda=lamda)
        data_train_x = data_train[:, :-2]
        data_train_y = np.argmax(data_train[:, -2:], axis=1)

        data_test_x = data_test[:, :-2]
        data_test_y = np.argmax(data_test[:, -2:], axis=1)

        df_generated = generator(torch.randn(size=(df.shape[0], input_dim), device=device)).cpu().detach().numpy()
        df_generated_x = df_generated[:, :-2]
        df_generated_y = np.argmax(df_generated[:, -2:], axis=1)

        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        clf = clf.fit(data_train_x, data_train_y)
        from sklearn.metrics import accuracy_score
        # print("accuracy original = {}".format(accuracy_score(clf.predict(data_test_x), data_test_y)))
        accuracy_original = accuracy_score(clf.predict(data_test_x), data_test_y)
        from sklearn.metrics import f1_score
        f1_original = f1_score(clf.predict(data_test_x), data_test_y)

        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        clf = clf.fit(df_generated_x, df_generated_y)
        from sklearn.metrics import accuracy_score
        # print("accuracy generated = {}".format(accuracy_score(clf.predict(data_test_x), data_test_y)))
        accuracy_generated = accuracy_score(clf.predict(data_test_x), data_test_y)
        from sklearn.metrics import f1_score
        f1_generated = f1_score(clf.predict(data_test_x), data_test_y)

        difference_accuracy = accuracy_original - accuracy_generated
        difference_f1 = f1_original - f1_generated

        female_mask = df_generated_x[:, 5] == 1
        male_mask = 1 - female_mask
        male_mask = male_mask == 1
        rich_mask = df_generated_y == 1
        female_odds = ((female_mask & rich_mask).sum()) / (female_mask.sum())
        male_odds = ((male_mask & rich_mask).sum()) / (male_mask.sum())
        demographic_parity_data = female_odds - male_odds

        predictions = clf.predict(data_test_x)
        female_mask = data_test_x[:, 5] == 1
        male_mask = 1 - female_mask
        male_mask = male_mask == 1
        prediction_mask = predictions == 1
        female_odds = ((female_mask & prediction_mask).sum()) / (female_mask.sum())
        male_odds = ((male_mask & prediction_mask).sum()) / (male_mask.sum())
        demographic_parity_classifier = female_odds - male_odds

        # print("\nfairness original data: male_odds:{} \t female odds:{}".format(male_odds, female_odds))

        buf = '%d, %f, %f , %f, %f, %f, %f, %f, %f' % (
        i, accuracy_original, accuracy_generated, difference_accuracy, f1_original, f1_generated, difference_f1,
        demographic_parity_data, demographic_parity_classifier)
        print(buf)
        print2file(buf, outFile)



outfile = "results.log"
multiple_runs(num_trainings=5, df=df, epochs=args.num_epochs, batchsize=args.batch_size, fair_epochs=args.num_fair_epochs, lamda=args.lambda_val, outFile=outfile)
