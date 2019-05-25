# Step2
# This file is to construct model

import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self, input_size=[10, 3], hidden_size=[128, 256], num_classes=1, num_layer=3):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layer = num_layer
        self.para_size = input_size[0]
        self.ques_size = input_size[1]

        self.conv1 = nn.Conv1d(1, self.hidden_size[0], kernel_size=3) #in_channels, out_channels, kernel_size
        self.linear1 = nn.Linear(self.ques_size, self.hidden_size[1], bias=True)
        self.rnn = nn.LSTM(
            input_size = self.hidden_size[0], # dimVectors  128
            hidden_size = self.hidden_size[1], # hidden size 256
            num_layers = self.num_layer, # num of hidden layer 3
            batch_first = True, # input & output will has batch size as 1s dimension.
        )  # input_size, hidden_dim, num_layers
        self.out = nn.Linear(self.hidden_size[1], self.num_classes)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        # [32, 10]
        # [32, 3]
        x = x.unsqueeze(1) # [32, 1, 10]
        h = self.linear1(h) # [32, 256]
        h = h.unsqueeze(0) # [1, 32, 256]  --> # num_layers, batch, output_size
        h = h.expand(self.num_layer, -1, -1).contiguous() # [3, 32, 256]
        x = self.conv1(x) #[32, 128, 8]
        x = x.permute(0, 2, 1) #[32, 8, 128]
        r_out, (h_n, h_c) = self.rnn(x, (h, h)) # r_out [32, 8, 256]
        out = self.out(r_out[:, -1, :]) # [32, 256]
        out = self.Sigmoid(out)
        return out


