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
from tqdm.auto import tqdm


from .utils import get_ohe_data_fair, get_ohe_data_nofair, get_original_data, prepare_data_fair, prepare_data_nofair

from .modules import Generator, Critic, FairLossFunc, get_gradient, gradient_penalty, get_gen_loss, get_crit_loss


display_step = 50


class TFG:

    def __init__(self, df: pd.DataFrame, epochs: int, batch_size: int, device: str = None, fairness_config: dict = None) -> None:
        self.df = df
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = "cpu" if device is None else device

        # Fairness configuration
        self.fair_epochs = fairness_config.get('fair_epochs', 0) if fairness_config else 0
        
        if fairness_config and self.fair_epochs > 0:
            self.lamda = fairness_config.get('lamda')
            self.S = fairness_config.get('S')
            self.Y = fairness_config.get('Y')
            self.S_under = fairness_config.get('S_under')
            self.Y_desire = fairness_config.get('Y_desire')

            # Perform type checking
            if not isinstance(self.lamda, float):
                raise TypeError("When fair_epochs is > 0, 'lamda' must be a float.")
            if not isinstance(self.S, str):
                raise TypeError("When fair_epochs is > 0, 'S' must be a string.")
            if not isinstance(self.Y, str):
                raise TypeError("When fair_epochs is > 0, 'Y' must be a string.")
            if not isinstance(self.S_under, str):
                raise TypeError("When fair_epochs is > 0, 'S_under' must be a string.")
            if not isinstance(self.Y_desire, str):
                raise TypeError("When fair_epochs is > 0, 'Y_desire' must be a string.")
        else:
            self.lamda = None
            self.S = None
            self.Y = None
            self.S_under = None
            self.Y_desire = None


    def prepare_data(self) -> None:
        if self.fair_epochs > 0:
            self.ohe, self.scaler, self.input_dim, self.discrete_columns, self.continuous_columns, self.train_dl, self.data_train, self.data_test, self.S_start_index, self.Y_start_index, self.underpriv_index, self.priv_index, self.undesire_index, self.desire_index = prepare_data_fair(self.df, self.batch_size, self.S, self.Y, self.S_under, self.Y_desire)

        else:
            self.ohe, self.scaler, self.input_dim, self.discrete_columns, self.continuous_columns, self.train_dl, self.data_train, self.data_test = prepare_data_nofair(self.df, self.batch_size)
    

    def train(self) -> None:

        self.prepare_data()

        self.generator = Generator(self.input_dim, self.continuous_columns, self.discrete_columns).to(self.device)

        self.critic = Critic(self.input_dim).to(self.device)

        if self.fair_epochs > 0:
            self.second_critic = FairLossFunc(self.S_start_index, self.Y_start_index, self.underpriv_index, self.priv_index, self.undesire_index, self.desire_index).to(self.device)
        
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.gen_optimizer_fair = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.crit_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0002, betas=(0.5, 0.999))


        critic_losses = []
        cur_step = 0
        with tqdm(total=self.epochs, desc="Training Progress", ncols=100) as pbar:
            for i in range(self.epochs):
                print("epoch {}".format(i + 1))
                ############################
                if i + 1 <= (self.epochs - self.fair_epochs):
                    pbar.set_postfix_str("Training for accuracy")
                    #print("training for accuracy")
                if i + 1 > (self.epochs - self.fair_epochs):
                    pbar.set_postfix_str("Training for fairness")
                    #print("training for fairness")
                for data in self.train_dl:
                    data[0] = data[0].to(self.device)
                    crit_repeat = 4
                    mean_iteration_critic_loss = 0
                    for k in range(crit_repeat):
                        # training the critic
                        self.crit_optimizer.zero_grad()
                        fake_noise = torch.randn(size=(self.batch_size, self.input_dim), device=self.device).float()
                        fake = self.generator(fake_noise)

                        crit_fake_pred = self.critic(fake.detach())
                        crit_real_pred = self.critic(data[0])

                        epsilon = torch.rand(self.batch_size, self.input_dim, device=self.device, requires_grad=True)
                        gradient = get_gradient(self.critic, data[0], fake.detach(), epsilon)
                        gp = gradient_penalty(gradient)

                        crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda=10)

                        mean_iteration_critic_loss += crit_loss.item() / crit_repeat
                        crit_loss.backward(retain_graph=True)
                        self.crit_optimizer.step()
                    #############################
                    if cur_step > 50:
                        critic_losses += [mean_iteration_critic_loss]

                    #############################
                    if i + 1 <= (self.epochs - self.fair_epochs):
                        # training the generator for accuracy
                        self.gen_optimizer.zero_grad()
                        fake_noise_2 = torch.randn(size=(self.batch_size, self.input_dim), device=self.device).float()
                        fake_2 = self.generator(fake_noise_2)
                        crit_fake_pred = self.critic(fake_2)

                        gen_loss = get_gen_loss(crit_fake_pred)
                        gen_loss.backward()

                        # Update the weights
                        self.gen_optimizer.step()

                    ###############################
                    if i + 1 > (self.epochs - self.fair_epochs):
                        # training the generator for fairness
                        self.gen_optimizer_fair.zero_grad()
                        fake_noise_2 = torch.randn(size=(self.batch_size, self.input_dim), device=self.device).float()
                        fake_2 = self.generator(fake_noise_2)

                        crit_fake_pred = self.critic(fake_2)

                        gen_fair_loss = self.second_critic(fake_2, crit_fake_pred, self.lamda)
                        gen_fair_loss.backward()
                        self.gen_optimizer_fair.step()
                    cur_step += 1

                pbar.update(1)
        

    def generate_fake_df(self, num_rows: int) -> pd.DataFrame:
        with torch.no_grad():
            fake_numpy_array = self.generator(torch.randn(size=(num_rows, self.input_dim), device=self.device)).cpu().detach().numpy()

            fake_df = get_original_data(fake_numpy_array, self.df, self.ohe, self.scaler)

            fake_df = fake_df[self.df.columns]
            return fake_df




