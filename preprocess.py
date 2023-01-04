import numpy as np
import torch
import gensim.downloader

from collections import OrderedDict
from tqdm.auto import tqdm


class Preprocessor:

    def __init__(self, path=None, dictionary=None, load_w2v=True):

        if load_w2v:
            print('loading w2v')
            self.w2v_google_news = gensim.downloader.load(f'word2vec-google-news-300')
            self.w2v_google_news_len = 300
            print('loaded w2v')

        if path and not dictionary:
            sentences = read_data(path, labeled=True)
            self.index2token, self.token2index = get_vocab(sentences)
            self.index2pos, self.pos2index = get_pos_index(sentences)
            self.vocab_size = len(self.index2token)
            self.pos_count = len(self.index2pos)
        elif dictionary and not path:
            self.index2token = dictionary['index_to_token']
            self.index2pos = dictionary['index_to_pos']
            self.vocab_size = dictionary['vocab_size']
            self.pos_count = dictionary['pos_count']

    @property
    def as_dict(self):
        return {'index_to_token': self.index2token, 'index_to_pos': self.index2pos, 'vocab_size': self.vocab_size,
                'pos_count': self.pos_count}

    def preprocess(self, path, labeled=True, to_tensor=True):
        sentences = read_data(path, labeled=labeled)
        represented_sentences = self.get_reps(sentences)
        indexed_sentences = self.get_indices(represented_sentences)
        if labeled:
            if to_tensor:
                new_indexed_sentences = [[torch.tensor([w[1] for w in s], dtype=torch.int),
                                          torch.tensor([w[2] for w in s], dtype=torch.int),
                                          torch.tensor([w[3] for w in s], dtype=torch.int),
                                          torch.cat([torch.tensor(w[4]).unsqueeze(dim=0) for w in s], dim=0)]
                                         for s in indexed_sentences]
            else:
                new_indexed_sentences = [[[w[1] for w in s], [w[2] for w in s],
                                          [w[3] for w in s], [w[4] for w in s]] for s in indexed_sentences]
        else:
            if to_tensor:
                new_indexed_sentences = [[torch.tensor([w[1] for w in s], dtype=torch.int),
                                          torch.tensor([w[2] for w in s], dtype=torch.int),
                                          torch.cat([w[-1] for w in s])] for s in indexed_sentences]
            else:
                new_indexed_sentences = [[[w[1] for w in s], [w[2] for w in s], [w[-1] for w in s]]
                                         for s in indexed_sentences]

        return new_indexed_sentences

    def get_indices(self, sentences):
        for s in tqdm(sentences, leave=False, desc='indexing'):
            for w in s:
                if w[1] in self.token2index.keys():
                    w[1] = self.token2index[w[1]]
                else:
                    w[1] = 0
                if w[2] in self.pos2index.keys():
                    w[2] = self.pos2index[w[2]]
                else:
                    w[2] = 0
        return sentences

    def get_reps(self, sentences):
        """
        append to the word representation its representation in w2v
        """
        for s in tqdm(sentences, leave=False, desc='representing'):
            for w in s:
                token = w[1]
                if token in self.w2v_google_news.index_to_key:
                    w.append(self.w2v_google_news[token])
                else:
                    if token.isnumeric():
                        w.append(self.w2v_google_news['integer'])
                    elif token.replace('.', '', 1).isdigit():
                        w.append(self.w2v_google_news['number'])
                    else:
                        w.append(np.zeros(self.w2v_google_news_len))
        return sentences


def read_data(path, labeled=True):
    with open(path, 'r') as f:
        sentences = []
        # read all sentences as without padding
        curr_sentence = [[0, 'ROOT', 'ROOT', -1]]
        for line in f:
            # check if line is blank
            split_line = line.split('\t')
            if line != '\n':
                num = int(split_line[0])
                token = split_line[1].lower()
                pos = split_line[3]
                if labeled:
                    head = int(split_line[6])
                    curr_sentence.append([num, token, pos, head])
                else:
                    curr_sentence.append([num, token, pos])
            else:
                sentences.append(curr_sentence)
                curr_sentence = [[0, 'ROOT', 'ROOT', -1]]

        # append the last sentence
        # sentences.append(curr_sentence)
    return sentences


def get_vocab(sentences):
    tokens = list(OrderedDict.fromkeys([t[1] for s in sentences for t in s]))
    index2token = ["[UNK]"] + tokens
    token2index = {t: i for i, t in enumerate(index2token)}
    return index2token, token2index


def get_pos_index(sentences):
    pos = list(OrderedDict.fromkeys([t[2] for s in sentences for t in s]))
    index2pos = ["[UNK]"] + pos
    pos2index = {t: i for i, t in enumerate(index2pos)}
    return index2pos, pos2index
