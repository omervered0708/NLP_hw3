import torch
import torch.nn as nn
import preprocess
import model1

from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def main():
    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load data
    preprocessor = preprocess.Preprocessor(path='./data/train.labeled')

    train_set = preprocessor.preprocess(path='./data/train.labeled', labeled=True)
    test_set = preprocess.preprocess(path='./data/test.labeled', labeled=True)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # init model
    d_word_embed = 64
    d_pos_embed = 8
    d_hidden = 128
    n_layers = 2
    dropout = 0.5

    model = model1.Model(n_word_embed=preprocessor.vocab_size, d_word_embed=d_word_embed,
                         n_pos_embed=preprocessor.pos_count, d_pos_embed=d_pos_embed, d_hidden=d_hidden,
                         n_layers=n_layers, dropout=dropout)
    model.to(device)

    # init training procedure
    lr = 1e-3
    weight_decay = 1e-5
    n_epochs = 8

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training loop
    for epoch in tqdm(range(n_epochs), leave=False, desc='main loop'):
        for words, pos, labels in tqdm(train_loader, leave=False, desc=f'epoch = {epoch}'):

            if torch.cuda.is_available():
                words, pos, labels = words.cuda(), pos.cuda(), labels.cuda()

            outputs = model(words, pos)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()