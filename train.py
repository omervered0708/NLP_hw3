import random
import os
import torch
import torch.nn as nn
import preprocess
import model1
import model2
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from chu_liu_edmonds import decode_mst
from sklearn.metrics import accuracy_score


def epoch_loop(model, dataloader, criterion, to_train=True, optimizer=None, loop_desc='eval', sample_score=0.05,
               batch_size=64):
    total_loss = 0
    hits = 0
    total_dep_loss = 0
    total_dep_pred = 0
    true_list = []
    pred_list = []
    loss = torch.zeros(1)

    for i, (words, pos, labels, w2v) in enumerate(tqdm(dataloader, leave=False, desc=loop_desc)):

        eye = torch.eye(len(labels[0]), dtype=torch.bool)
        labels = labels.type(torch.LongTensor)

        if torch.cuda.is_available():
            words, pos, labels, w2v, eye, loss = words.cuda(), pos.cuda(), labels.cuda(), w2v.cuda(), eye.cuda(), loss.cuda()

        outputs = model(words, pos, w2v).masked_fill_(eye, value=-1 * torch.inf)
        outputs_to_loss, labels_to_loss = outputs[1:], labels[0][1:]
        example_loss = criterion(outputs_to_loss, labels_to_loss)
        loss += example_loss
        total_loss += example_loss.item()
        total_dep_loss += len(labels[0]) - 1

        if to_train and (i + 1) % batch_size == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = torch.zeros(1)

        pred_dep, _ = decode_mst(outputs.cpu().detach().numpy().T, len(outputs), has_labels=False)
        labels = labels[0][1:]
        pred_dep = pred_dep[1:]
        true_list += list(labels.cpu())
        pred_list += list(pred_dep)
    return total_loss / total_dep_loss, accuracy_score(true_list, pred_list)


def train_eval_epoch(model, dataloader, criterion, to_train=True, optimizer=None, loop_desc='eval', to_print=True,
                     sample_score=1.0, batch_size=64):
    if to_train:
        epoch_loss, epoch_uas = epoch_loop(model, dataloader, criterion, to_train=True, optimizer=optimizer,
                                           loop_desc=loop_desc, sample_score=sample_score, batch_size=batch_size)
        if to_print:
            print(f'{loop_desc} train loss = {epoch_loss}, train uas = {epoch_uas}')
    else:
        model.eval()
        with torch.no_grad():
            epoch_loss, epoch_uas = epoch_loop(model, dataloader, criterion, to_train=False, optimizer=None,
                                               loop_desc=loop_desc, sample_score=sample_score, batch_size=1)
        model.train()
        if to_print:
            print(f'{loop_desc} test loss = {epoch_loss}, test uas = {epoch_uas}')

    return epoch_loss, epoch_uas


def plot(y_train, y_test, y_label):
    x = list(range(1, len(y_train) + 1))
    plt.plot(x, y_train, label=f"train {y_label}")
    plt.plot(x, y_test, label=f"test {y_label}")
    plt.xlabel("epoch")
    plt.ylabel(y_label)
    plt.title(f"{y_label} of train and test")
    plt.legend()
    plt.savefig(f"./plots/{y_label}.png")
    plt.clf()


def main():
    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load and preprocess
    preprocessor = preprocess.Preprocessor(path=None, dictionary=torch.load('./preprocessed_data/preprocessor.pkl'),
                                           load_w2v=False) \
        if os.path.isfile(f'./preprocessed_data/train_set.pkl') else preprocess.Preprocessor(
        path='./data/train.labeled', load_w2v=True)
    train_set = torch.load('./preprocessed_data/train_set.pkl') \
        if os.path.isfile('./preprocessed_data/train_set.pkl') else preprocessor.preprocess(path='./data/train.labeled',
                                                                                            labeled=True)
    test_set = torch.load('./preprocessed_data/test_set.pkl') \
        if os.path.isfile('./preprocessed_data/test_set.pkl') else preprocessor.preprocess(path='./data/test.labeled',
                                                                                           labeled=True)

    # save processed data
    if not os.path.isfile(f'./preprocessed_data/preprocessor.pkl'):
        torch.save(preprocessor.as_dict, './preprocessed_data/preprocessor.pkl')
    if not os.path.isfile(f'./preprocessed_data/train_set.pkl'):
        torch.save(train_set, './preprocessed_data/train_set.pkl')
    if not os.path.isfile(f'./preprocessed_data/test_set.pkl'):
        torch.save(test_set, './preprocessed_data/test_set.pkl')

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # init model
    d_word_embed = 256
    d_pos_embed = 48
    d_hidden = 256
    n_layers = 2
    dropout = 0.1

    model = model2.Model(n_word_embed=preprocessor.vocab_size, d_word_embed=d_word_embed,
                         n_pos_embed=preprocessor.pos_count, d_pos_embed=d_pos_embed, d_hidden=d_hidden,
                         n_layers=n_layers, dropout=dropout)
    print(f'num param: {sum([param.numel() for param in model.parameters()])}')
    model.to(device)

    # init training procedure
    lr = 1e-3
    weight_decay = 1e-5
    n_epochs = 16
    batch_size = 10

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_uas = []
    train_loss = []
    test_uas = []
    test_loss = []

    # training loop
    for epoch in range(n_epochs):
        epoch_train_loss, epoch_train_uas = train_eval_epoch(model, train_loader, criterion, to_train=True,
                                                             optimizer=optimizer, loop_desc=f'[{epoch}/{n_epochs}]',
                                                             to_print=True, sample_score=1.0, batch_size=batch_size)
        train_loss.append(epoch_train_loss)
        train_uas.append(epoch_train_uas)

        epoch_test_loss, epoch_test_uas = train_eval_epoch(model, test_loader, criterion, to_train=False,
                                                           optimizer=None, loop_desc=f'eval', to_print=True,
                                                           sample_score=1.0)
        test_loss.append(epoch_test_loss)
        test_uas.append(epoch_test_uas)

    # save
    torch.save(model.state_dict(), './models/trained_model.pkl')

    # plot
    plot(train_loss, test_loss, 'loss')
    plot(train_uas, test_uas, 'UAS')


if __name__ == '__main__':
    main()
