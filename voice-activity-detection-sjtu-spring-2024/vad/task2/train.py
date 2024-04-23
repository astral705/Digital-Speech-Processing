import torch
from torch import nn 




class DNN(nn.Module):

    def __init__(self, num_inputs = 10, num_outputs = 1, num_hiddens =256, num_layers = 3):
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.first_layer = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.BatchNorm1d(num_hiddens),
        )
        self.layer = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.ReLU(),
            nn.BatchNorm1d(num_hiddens),
        )
        self.last_layer = nn.Sequential(
            nn.Linear(num_hiddens, num_outputs),
            nn.ReLU(),
        )
        


    def forward(self, inputs):
        x = self.first_layer(inputs)
        for i in range(self.num_layers - 2):
            x = self.layer(x)
        outputs = self.last_layer(x)
        return outputs

        

