import torch
import os
from torch import nn, optim
from copy import deepcopy
import numpy as np


class Linear_2048Qnet(nn.Module):
    def __init__(self, input_size: int, hidden_layers=[]):
        super(Linear_2048Qnet, self).__init__()

        layer_sizes = [input_size] + hidden_layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if(i > 0):
                layers.append(nn.ReLU())

        layers.append(nn.Linear(layer_sizes[-1], 4))
        layers.append(nn.Softmax(dim=0))

        self.layers = nn.Sequential(*layers)

    def forward(self, data):
        logits = self.layers(data)
        return logits

    def save(self, file_name="2048_Qnet"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(
        self,
        model: Linear_2048Qnet,
        learning_rate: float,
        gamma: float,
        optimizer=optim.SGD,
        criterion=nn.MSELoss,
    ) -> None:
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optimizer(model.parameters(), lr=self.learning_rate)
        self.criterion = criterion()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        game_over = torch.tensor(game_over, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            game_over = torch.unsqueeze(game_over, 0)

        # 1: predicted Q values with current state
        pred = self.model(state)

        # 2: Q_new = reward + y * max(next_predicted Q) value -> only do if not done
        target = pred.clone()
        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()
