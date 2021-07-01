import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, dropout_p):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, input_sent):
        embedding = self.dropout(self.embed(input_sent))
        ops, (hidden_state, cell_state) = self.lstm(embedding)
        return hidden_state, cell_state
