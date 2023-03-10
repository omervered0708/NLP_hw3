import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_word_embed, d_word_embed, n_pos_embed, d_pos_embed, d_hidden, n_layers, dropout1=0.5,
                 dropout2=0.5, ignore_pos=True, d_pretrained_embed=300, use_w2v=True):
        super().__init__()

        self.n_word_embed = n_word_embed
        self.d_embed = d_word_embed
        self.n_pos_embed = n_pos_embed
        self.d_pos_embed = d_pos_embed
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.ignore_pos = ignore_pos
        self.d_pretrained_embed = d_pretrained_embed

        self.d_lstm_input = (d_word_embed if ignore_pos else d_word_embed + d_pos_embed) + d_pretrained_embed

        self.word_embed = nn.Embedding(num_embeddings=n_word_embed, embedding_dim=d_word_embed)
        self.pos_embed = nn.Embedding(num_embeddings=n_pos_embed, embedding_dim=d_pos_embed)
        self.lstm = nn.LSTM(input_size=self.d_lstm_input, hidden_size=d_hidden, num_layers=n_layers, batch_first=True,
                            dropout=dropout1, bidirectional=True)
        self.fc_right = nn.Sequential(
            nn.Linear(2 * d_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(256, 128)
        )
        self.fc_left = nn.Sequential(
            nn.Linear(2 * d_hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(256, 128)
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * 2 * d_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(64, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X, P, w2v):
        sen_len = X.shape[1]
        if self.ignore_pos:
            out = torch.concat((self.word_embed(X), w2v), dim=-1)
        else:
            out = torch.concat((self.word_embed(X), self.pos_embed(P), w2v), dim=-1)
        out = out.type(torch.float32)
        out, _ = self.lstm(out)
        out = out[0]
        left = self.fc_left(out)
        right = self.fc_right(out)
        out = torch.flatten(out[torch.cartesian_prod(torch.arange(sen_len), torch.arange(sen_len))], start_dim=1)
        out = self.fc(out).squeeze()
        out = out.reshape(sen_len, -1) + torch.matmul(left, right.T)
        return out
