import torch
import torch.nn as nn
import itertools


class Model(nn.Module):
    def __init__(self, n_word_embed, d_word_embed, n_pos_embed, d_pos_embed, d_hidden, n_layers, dropout=0.5,
                 ignore_pos=True):
        super().__init__()

        self.n_word_embed = n_word_embed
        self.d_embed = d_word_embed
        self.n_pos_embed = n_pos_embed
        self.d_pos_embed = d_pos_embed
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.ignore_pos = ignore_pos

        self.d_lstm_input = d_word_embed if ignore_pos else d_word_embed + d_pos_embed

        self.word_embed = nn.Embedding(num_embeddings=n_word_embed, embedding_dim=d_word_embed)
        self.pos_embed = nn.Embedding(num_embeddings=n_pos_embed, embedding_dim=d_pos_embed)
        self.lstm = nn.LSTM(input_size=self.d_lstm_input, hidden_size=d_hidden, num_layers=n_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * 2 * d_hidden, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X, P):
        sen_len = X.shape[1]
        if self.ignore_pos:
            out = self.word_embed(X)
        else:
            out = torch.concat((self.word_embed(X), self.pos_embed(P)), dim=-1)
        out, _ = self.lstm(out)
        out = out[0]
        # pairs_with_root = list(itertools.combinations(out, r=2))
        # pairs_without_root = list(itertools.combinations(reversed(out[1:]), r=2))
        # concat_with_root = [torch.cat(p, dim=-1) for p in pairs_with_root]
        # concat_without_root = [torch.cat(p, dim=-1) for p in pairs_without_root]
        # out = torch.stack(list(torch.cat(p, dim=-1) for p in itertools.combinations(out, r=2))
        #                   + list(torch.cat(p, dim=-1) for p in itertools.combinations(reversed(out[1:]), r=2)))
        out = torch.stack(list(torch.cat(p, dim=-1) for p in itertools.product(out, repeat=2)))
        out = out.reshape((sen_len, sen_len, 2 * 2 * self.d_hidden))
        out = self.fc(out)
        return out.squeeze() if self.training else self.softmax(out).squeeze()
