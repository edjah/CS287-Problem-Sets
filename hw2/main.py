import time
import torch
from tqdm import tqdm

from models.ngram_model import NGramModel
from models.nnlm_cnn import NNLM_CNN
from models.basic_lstm import BasicLSTM


from data_setup import TEXT, train, test, val, train_iter, val_iter


def text_to_idx(dataset):
    train_txt = next(iter(dataset)).text
    return [TEXT.vocab.stoi[w] for w in train_txt]


def generate_predictions(model):
    exclude = {'<eos>'}
    with open('predictions.txt', 'w') as fout:
        print('id,word', file=fout)
        sentences = list(open('data/input.txt'))
        for i, line in tqdm(enumerate(sentences, 1), total=len(sentences)):
            words = line.strip().split(' ')[:-1]
            idx = [TEXT.vocab.stoi[w] for w in words]
            probs, pred_idx = model.predict_last_word(idx).topk(22)
            predictions = [TEXT.vocab.itos[i] for i in pred_idx]
            predictions = [w for w in predictions if w not in exclude][:20]
            print('%d,%s' % (i, ' '.join(predictions)), file=fout)


def evaluate(model, x, y, num_words):
    with torch.no_grad():
        loss_fn = torch.nn.NLLLoss(reduction='sum')
        loss = 0
        num_correct = 0
        total_words = 0

        for i in range(len(x)):
            log_probs = model.forward(x[i])
            loss += loss_fn(log_probs, y[i])
            num_correct += (log_probs.argmax(dim=1) == y[i]).sum().float()
            total_words = len(y[i])

        return 100.0 * num_correct / total_words, torch.exp(loss / total_words)


def train_model(model, num_iter=300, learning_rate=0.001,
                weight_decay=0, log_freq=1):

    xtrain, ytrain = model.get_data(train_iter)
    xval, yval = model.get_data(val_iter)

    model.train()
    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = torch.nn.NLLLoss()

    start_time = time.time()
    for i in range(num_iter):
        try:
            for batch in range(len(xtrain)):
                opt.zero_grad()

                log_probs = model.forward(xtrain[batch])
                loss = loss_fn(log_probs, ytrain[batch])

                # compute gradients and update weights
                loss.backward()
                opt.step()

                if (batch % 100 == 0):
                    print(i, batch)

            # evaluate performance on entire sets
            model.eval()
            train_acc, train_loss = evaluate(model, xtrain, ytrain)
            val_acc, val_loss = evaluate(model, xval, yval)
            model.train()

            if i == 0 or i == num_iter - 1 or (i + 1) % log_freq == 0:
                msg = f"{round(time.time() - start_time)} sec: Epoch {i + 1}"
                print(f'{msg}\n{"=" * len(msg)}')
                print(f'Train Perplexity: {train_loss:.5f}\t Train Accuracy: {train_acc:.2f}%')
                print(f'Val Perplexity: {val_loss:.5f}\t Val Accuracy: {val_acc:.2f}%\n')

        except KeyboardInterrupt:
            print(f'\nStopped training after {i} epochs...')
            break

    model.eval()
    print('Final results\n=============')
    train_acc, train_loss = evaluate(model, xtrain, ytrain)
    val_acc, val_loss = evaluate(model, xval, yval)
    print(f'Train Perplexity: {train_loss:.5f}\t Train Accuracy: {train_acc:.2f}%')
    print(f'Val Perplexity: {val_loss:.5f}\t Val Accuracy: {val_acc:.2f}%\n')


if __name__ == '__main__':

    # conv_nnlm = NNLM_CNN(5, hidden_size=100, num_layers=3,
    #                      dropout_rate=0.5, num_filters=50)
    # train_model(conv_nnlm)

    train_idx = text_to_idx(train)
    val_idx = text_to_idx(val)
    test_idx = text_to_idx(test)

    ngram_model = NGramModel(train_idx, max_ngram=3, min_count=1)
    ngram_model.train_interpolation(val_idx, val_idx)

    print('N-Gram: Train Perplexity:', ngram_model.perplexity(train_idx))
    print('N-Gram: Validation Perplexity:', ngram_model.perplexity(val_idx))
    print('N-Gram: Test Perplexity:', ngram_model.perplexity(test_idx))

    generate_predictions(ngram_model)
