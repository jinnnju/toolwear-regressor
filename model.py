import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM output
        out, _ = self.lstm(x)
        # Fully connected output using the last time step
        out = self.fc(out[:, -1, :])
        return out



class SVRRegressor:
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def forward(self, x):
        # SVR expects 2D input, reshape if needed
        x = x.squeeze(1).numpy()  # Reshape to (batch_size, features)
        return self.model.predict(x)

    def fit(self, x, y):
        x = x.squeeze(1).numpy()
        y = y.numpy()
        self.model.fit(x, y)



class RFRRegressor:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    def forward(self, x):
        x = x.squeeze(1).numpy()  # Reshape to (batch_size, features)
        return self.model.predict(x)

    def fit(self, x, y):
        x = x.squeeze(1).numpy()
        y = y.numpy()
        self.model.fit(x, y)


class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRURegressor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # GRU output
        out, _ = self.gru(x)
        # Fully connected output using the last time step
        out = self.fc(out[:, -1, :])
        return out

