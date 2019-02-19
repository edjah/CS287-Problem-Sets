import math
import torch
import torchtext
from tqdm import tqdm

from namedtensor.text import NamedField
from torchtext.data.iterator import BPTTIterator
from torchtext.data import Batch, Dataset

from ngram_model import NGramModel
from nnlm_cnn import NNLMcnn

def text_to_idx(dataset):
    train_txt = next(iter(dataset)).text
    return [TEXT.vocab.stoi[w] for w in train_txt]


def generate_predictions(model):
    with open('predictions.txt', 'w') as fout:
        print('id,word', file=fout)
        sentences = list(open('data/input.txt'))
        for i, line in tqdm(enumerate(sentences, 1), total=len(sentences)):
            words = line.strip().split(' ')[:-1]
            idx = [TEXT.vocab.stoi[w] for w in words]
            probs, pred_idx = model.predict_last_word(idx).topk(20)
            predictions = [TEXT.vocab.itos[i] for i in pred_idx]
            print('%d,%s' % (i, ' '.join(predictions)), file=fout)


def train_model(model, num_iter=300, learning_rate=0.001,
                weight_decay=0, log_freq=1, batch_size=32):

    xtrain, ytrain, num_train = model.get_data(train_iter)
    xval, yval, num_val = model.get_data(val_iter)

    model.train()
    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = torch.nn.NLLLoss()

    for i in range(num_iter):
        try:
            for idx in range(len(xtrain)):
                opt.zero_grad()

                xbatch = xtrain[idx]
                ybatch = ytrain[idx]

                probs = model.forward(xbatch)
                loss = loss_fn(probs.log(), ybatch)

                # compute gradients and update weights
                loss.backward()
                opt.step()

                if (idx % 100 == 0):
                    print (i, idx)

            # evaluate performance on entire sets
            model.eval()
            train_acc, train_loss = evaluate(model, xtrain, ytrain, num_train)
            val_acc, val_loss = evaluate(model, xval, yval, num_val)

            model.train()

            if i == 0 or i == num_iter - 1 or (i + 1) % log_freq == 0:
                print(f'Epoch {i + 1}\n{"=" * len("Epoch {}".format(i + 1))}')
                print(f'Train Loss: {train_loss:.5f}\t Train Accuracy: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.5f}\t Val Accuracy: {val_acc:.2f}%\n')

        except KeyboardInterrupt:
            print(f'\nStopped training after {i} epochs...')
            break

    model.eval()


def evaluate(model, x, y, num_words):
    with torch.no_grad():
        loss_fn = torch.nn.NLLLoss(reduction='sum')
        loss = 0
        num_correct = 0

        for i in range(len(x)):
            probs = model.forward(x[i])
            loss += loss_fn(probs.log(), y[i])
            num_correct += (probs.argmax(dim=1) == y[i]).sum().float()

        return 100.0 * num_correct / num_words, loss / num_words


if __name__ == '__main__':

    if True:
        conv_nnlm = NNLMconv(5, hidden_size=100, num_layers=3,
                            dropout_rate=0.5, num_filters=50)
        train_model(conv_nnlm)

        rnn_nnlm = NNLMrec(5, hidden_size=100, num_layers=3,
                            dropout_rate=0.5)
        train_model(rnn_nnlm)

    train_idx = text_to_idx(train)
    orig_val_idx = text_to_idx(val)

    val_len = int(0.8 * len(orig_val_idx))
    val_idx, test_idx = orig_val_idx[:val_len], orig_val_idx[val_len:]

    ngram_model = NGramModel(train_idx, max_ngram=3, min_count=1)
    ngram_model.train_interpolation(val_idx, test_idx)

    sent = ['we', 'are']
    idx = [TEXT.vocab.stoi[w] for w in sent]
    print(TEXT.vocab.itos[ngram_model.predict_last_word(idx).argmax()])

    print('N-Gram: Train Perplexity:', ngram_model.perplexity(train_idx))
    print('N-Gram: Validation Perplexity:', ngram_model.perplexity(val_idx))
    print('N-Gram: Test Perplexity:', ngram_model.perplexity(test_idx))

    generate_predictions(ngram_model)
