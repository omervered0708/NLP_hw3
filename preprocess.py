from collections import OrderedDict

def read_data(path, labeled = True):
    with open(path, 'r') as f:
        sentences = []
        # read all sentences as without padding
        curr_sentence = []
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
    index2token = ["[PAD]","[UNK]"] + list(tokens)
    token2index = {t:i for i,t in enumerate(index2token)}
    return index2token,token2index

def get_pos_index(sentences):
    pos = OrderedDict.fromkeys([t[1] for s in sentences for t in s])
    index2pos = ["[PAD]", "[UNK]"] + list(pos)
    pos2index = {t: i for i, t in enumerate(index2pos)}
    return index2pos, pos2index
def get_indexes(sentences):
    index2token , token2index = get_vocab(sentences)
    index2pos, pos2index = get_pos_index(sentences)
    for s in sentences:
        for w in s:
            if w[1] in token2index.keys():
                w[1] = token2index[w[1]]
            else:
                w[1] = 1
            if w[2] in pos2index.keys():
                w[2] = pos2index[w[2]]
            else:
                w[2] = 1

    return sentences