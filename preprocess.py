from collections import OrderedDict
class Preprocessor():

    def __init__(self,path):
        sentences =read_data(path,labeled=True)
        self.index2token,self.token2index = get_vocab(sentences)
        self.index2pos, self.pos2index = get_pos_index(sentences)

    def preprocess(self,path,labeled=True):
        sentences = read_data(path, labeled=labeled)
        indexed_sentences= self.get_indexes(sentences)
        if labeled:
            new_indexed_sentences = [[[w[1] for w in s ],[w[2] for w in s],[w[3] for w in s]] for s in indexed_sentences]
        else:
            new_indexed_sentences = [[[w[1] for w in s], [w[2] for w in s]] for s in indexed_sentences]
        return new_indexed_sentences

    def get_indexes(self,sentences):
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
def read_data(path, labeled = True):
    with open(path, 'r') as f:
        sentences = []
        # read all sentences as without padding
        curr_sentence = []
        curr_sentence.append([0,'ROOT','ROOT','_'])
        for line in f:
            # check if line is blank
            split_line = line.split('\t')
            if line != '\n':
                num = split_line[0]
                token = line[1]
                pos = line[3]
                if labeled:
                    head = line[6]
                    curr_sentence.append([num,token,pos,head])
                else:
                    curr_sentence.append([num,token,pos])
            else:
                sentences.append(curr_sentence)
                curr_sentence = []

        # append the last sentence
        sentences.append(curr_sentence)
    return sentences


def get_vocab(sentences):
    tokens = OrderedDict.fromkeys([t[1] for s in sentences for t in s])
    index2token = ["[UNK]","ROOT"] + list(tokens)
    token2index = {t:i for i,t in enumerate(index2token)}
    return index2token,token2index

def get_pos_index(sentences):
    pos = OrderedDict.fromkeys([t[1] for s in sentences for t in s])
    index2pos = ["[UNK]","ROOT"] + list(pos)
    pos2index = {t: i for i, t in enumerate(index2pos)}
    return index2pos, pos2index


