import itertools
import torch
import combined_model
import preprocess
import os

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from chu_liu_edmonds import decode_mst


def predict(dataset, model):
    """
    return predictions on the given dataset
    :param dataset: dataset to predict. assume not labeled
    :param model: trained model
    :return: list of np.arrays representing the predicted labels for each sentence in the dataset
    """
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    sen_pred = []
    dataloader = DataLoader(dataset, batch_size=1)
    with torch.no_grad():
        for i, (words, pos, w2v) in enumerate(tqdm(dataloader, leave=False, desc='predicting')):
            eye = torch.eye(len(pos[0]), dtype=torch.bool)
            if torch.cuda.is_available():
                words, pos, w2v, eye = words.cuda(), pos.cuda(), w2v.cuda(), eye.cuda()
            outputs = model(words, pos, w2v).masked_fill_(eye, value=-1 * torch.inf)
            pred_dep, _ = decode_mst(outputs.cpu().detach().numpy().T, len(outputs), has_labels=False)
            sen_pred.append(pred_dep)

    return sen_pred


def numpy_to_tensor_float32(a):
    return torch.from_numpy(a).to(torch.float32)


def preds_to_file(source_path, dest_path, preds):
    """
    create labeled file from the predictions
    :param source_path: path to unlabeled data which was used to create 'preds'
    :param dest_path: path for the labeled data file
    :param preds: predictions from the model
    """
    # read
    with open(source_path, 'r') as f:
        lines = f.readlines()

    # create linear ordering of labels
    line_labels = [[str(d) for d in line_pred[1:]] for line_pred in preds]
    line_labels = [line_pred + ['\n'] for line_pred in line_labels]
    line_labels = [c for c in itertools.chain(*line_labels)]

    # write
    with open(dest_path, 'w') as f:
        for line, label in zip(lines, line_labels):
            if line == '\n':
                f.write('\n')
            else:
                line_split = line.split('\t')
                labeled_line = '\t'.join(line_split[:5] + [label] + line_split[6:])
                f.write(labeled_line)


def generate_tagged(source_path, dest_path, preprocessed_path, save_processed_data_as='comp_set',
                    save_preprocessed=False):
    # read data
    if os.path.isfile(f'{preprocessed_path}/{save_processed_data_as}.pkl'):
        preprocessor = preprocess.Preprocessor(path=None, load_w2v=False,
                                               dictionary=torch.load(f'{preprocessed_path}/preprocessor.pkl'))
        dataset = torch.load(f'{preprocessed_path}/{save_processed_data_as}.pkl')
    else:
        preprocessor = preprocess.Preprocessor(path=None, load_w2v=True,
                                               dictionary=torch.load(f'{preprocessed_path}/preprocessor.pkl'))
        dataset = preprocessor.preprocess(source_path, labeled=False, to_tensor=True)
        if save_preprocessed:
            torch.save(dataset, f'{preprocessed_path}/{save_processed_data_as}.pkl')

    # load model
    d_word_embed = 256
    d_pos_embed = 16
    d_hidden = 512
    n_layers = 2
    dropout1 = 0.55
    dropout2 = 0.15

    model = combined_model.Model(n_word_embed=preprocessor.vocab_size, d_word_embed=d_word_embed,
                                 n_pos_embed=preprocessor.pos_count, d_pos_embed=d_pos_embed, d_hidden=d_hidden,
                                 n_layers=n_layers, dropout1=dropout1, dropout2=dropout2, ignore_pos=False,
                                 use_w2v=True, d_pretrained_embed=preprocessor.pretrained_embed_len)
    model.load_state_dict(torch.load('./models/final_trained_model.pkl'))

    # predict
    preds = predict(dataset=dataset, model=model)

    # tag
    preds_to_file(source_path, dest_path, preds)


if __name__ == '__main__':
    generate_tagged('./data/comp.unlabeled', './data/comp_213336753_212362024.labeled', '.',
                    save_processed_data_as='unlabeled_test_set')
