from collections import OrderedDict

import torch


class Preprocessor:

    def __init__(self, path):
        sentences = read_data(path, labeled=True)
        self.index2token, self.token2index = get_vocab(sentences)
        self.index2pos, self.pos2index = get_pos_index(sentences)
        self.vocab_size = len(self.index2token)
        self.pos_count = len(self.index2pos)

    def preprocess(self, path, labeled=True, to_tensor=True):
        sentences = read_data(path, labeled=labeled)
        indexed_sentences = self.get_indices(sentences)
        if labeled:
            if to_tensor:
                new_indexed_sentences = [[torch.tensor([w[1] for w in s], dtype=torch.int),
                                          torch.tensor([w[2] for w in s], dtype=torch.int),
                                          torch.tensor([w[3] for w in s], dtype=torch.int)] for s in indexed_sentences]
            else:
                new_indexed_sentences = [[[w[1] for w in s], [w[2] for w in s],
                                          [w[3] for w in s]] for s in indexed_sentences]
        else:
            if to_tensor:
                new_indexed_sentences = [[torch.tensor([w[1] for w in s], dtype=torch.int),
                                          torch.tensor([w[2] for w in s], dtype=torch.int)]
                                         for s in indexed_sentences]
            else:
                new_indexed_sentences = [[[w[1] for w in s], [w[2] for w in s]] for s in indexed_sentences]

        return new_indexed_sentences

    def get_indices(self, sentences):
        for s in sentences:
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


def read_data(path, labeled=True):
    with open(path, 'r') as f:
        sentences = []
        # read all sentences as without padding
        curr_sentence = []
        curr_sentence.append([0, 'ROOT', 'ROOT', -1])
        for line in f:
            # check if line is blank
            split_line = line.split('\t')
            if line != '\n':
                num = int(split_line[0])
                token = split_line[1]
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
