import torch
import torch.nn as nn
from torch.nn import functional as F


class LSTMEncoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int,
                 embedding_size: int, num_layers: int = 1):

        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.lin = nn.Linear(hidden_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out, (h, _) = self.rnn(x)
        out = self.lin(out)
        return out, h.squeeze(0)


class LSTMDecoder(nn.Module):

    def __init__(self, output_size: int, hidden_size: int,
                 embedding_size: int, num_layers: int = 1):

        super().__init__()
        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out, (h, _) = self.rnn(x)
        out = self.lin(out)
        return out, h.squeeze(0)


class LSTMAutoencoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int,
                 embedding_size: int, num_layers: int = 1):

        super().__init__()
        self.enc = LSTMEncoder(input_size, hidden_size,
                               embedding_size, num_layers)
        self.dec = LSTMDecoder(input_size, hidden_size,
                               embedding_size, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_enc, _ = self.enc(x)
        x_dec, _ = self.dec(x_enc)
        return x_dec
