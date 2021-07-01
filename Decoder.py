import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_layers, dropout_p):
        super(Decoder, self).__init__()
        # print("Input_size", input_size)
        self.embed = nn.Embedding(input_size, embed_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state_enc, cell_state_enc):
        x = x.unsqueeze(0)
        # print(x)
        embedding = self.dropout(self.embed(x))
        ops, (hidden, cell) = self.lstm(
            embedding, (hidden_state_enc, cell_state_enc))
        pred = self.fc_out(ops)
        preds = pred.squeeze(0)
        return preds, hidden, cell
