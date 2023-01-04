import torch
import torch.nn as nn
import itertools


class Model(nn.Module):
    def __init__(self, n_word_embed, d_word_embed, n_pos_embed, d_pos_embed, d_hidden, n_layers, dropout=0.5,
                 ignore_pos=True, d_pretrained_embed=300):
        super().__init__()

        self.n_word_embed = n_word_embed
        self.d_embed = d_word_embed
        self.n_pos_embed = n_pos_embed
        self.d_pos_embed = d_pos_embed
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.ignore_pos = ignore_pos
        self.d_pretrained_embed = d_pretrained_embed

        self.d_lstm_input = (d_word_embed if ignore_pos else d_word_embed + d_pos_embed) + d_pretrained_embed

        self.word_embed = nn.Embedding(num_embeddings=n_word_embed, embedding_dim=d_word_embed)
        self.pos_embed = nn.Embedding(num_embeddings=n_pos_embed, embedding_dim=d_pos_embed)
        self.lstm = nn.LSTM(input_size=self.d_lstm_input, hidden_size=d_hidden, num_layers=n_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.fc_right = nn.Sequential(
            nn.Linear(2 * d_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        self.fc_left = nn.Sequential(
            nn.Linear(2 * d_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X, P, w2v):
        if self.ignore_pos:
            out = torch.concat((self.word_embed(X), w2v), dim=-1)
        else:
            out = torch.concat((self.word_embed(X), self.pos_embed(P), w2v), dim=-1)
        out = out.type(torch.float32)
        out, _ = self.lstm(out)
        out = out[0]
        left = self.fc_left(out)
        right = self.fc_right(out)
        out = torch.matmul(right, left.T)
        # return out.squeeze() if self.training else self.softmax(out).squeeze()
        return out.squeeze()
