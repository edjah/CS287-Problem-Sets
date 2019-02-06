from data_setup import torch, torchtext, test, train_iter, val_iter, test_iter
from models.naive_bayes import NaiveBayes
from models.logistic_regression import LogisticRegression
from models.cbow_nn import CbowNN
from models.cnn import CNN
from models.rnn import RNN
from utils import TimingContext, chunks


def generate_predictions(model):
    """All models should be able to be run with following command."""
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.text)
        # here we assume that the name for dimension classes is `classes`
        # _, argmax = probs.max('classes')
        argmax = (probs.flatten() > 0.5).long()
        upload += argmax.tolist()

    with open("predictions.txt", "w") as f:
        f.write("Id,Category\n")
        for i, u in enumerate(upload):
            f.write(str(i) + "," + str(u) + "\n")


def test_model(model, data_set=val_iter, description=None):
    total_correct = 0
    total = 0
    for batch in data_set:
        batch_result = model(batch.text).flatten() > 0.5
        total_correct += (batch.label.values == batch_result.long()).sum()
        total += len(batch)

    if description:
        print(description)
        print('=' * len(description))
    print(f'Num correct: {total_correct}\nTotal: {total}')
    print(f'Accuracy: {100.0 * total_correct.float() / total:.2f}%\n')


def evaluate(model, x, y):
    with torch.no_grad():
        loss_fn = torch.nn.BCELoss()
        loss = 0
        num_correct = 0
        for batch in chunks(torch.arange(len(x)), 128):
            probs = model.forward(x[batch]).flatten()
            num_correct += int(((probs > 0.5).byte() == y[batch].byte()).sum())
            loss += loss_fn(probs, y[batch].float()) * len(batch)

        return 100.0 * num_correct / len(x), loss / len(x)


# Works for all NN-based models (including LR)
def train_model(model, reg_param=0, num_iter=300, learning_rate=0.001,
                weight_decay=0, batch_size=100, log_freq=10):
    # getting properly formatted training/validation data
    xtrain, ytrain = model.get_data((train_iter, val_iter))
    xval, yval = model.get_data((test_iter,))

    # training
    model.train()
    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = torch.nn.BCELoss()

    best_params = {k: p.detach().clone() for k, p in model.named_parameters()}
    best_val_acc = 0

    for i in range(num_iter):
        # iterate through the data in a random order each time
        s1 = torch.utils.data.RandomSampler(torch.arange(len(xtrain)))
        s2 = torch.utils.data.BatchSampler(s1, batch_size=batch_size,
                                           drop_last=False)
        try:
            for idx in s2:
                opt.zero_grad()

                # computing the loss
                xbatch = xtrain[idx]
                labels_batch = ytrain[idx]
                probs = model.forward(xbatch).flatten()
                loss = loss_fn(probs, labels_batch.float())

                # computing gradients and updating the weights
                loss.backward()
                opt.step()

                # evaluating performance on the validation set
                model.eval()
                train_acc, train_loss = evaluate(model, xtrain, ytrain)
                val_acc, val_loss = evaluate(model, xval, yval)
                new_best = False
                if val_acc > best_val_acc:
                    best_params = {
                        k: p.detach().clone() for k, p in model.named_parameters()
                    }
                    best_val_acc = val_acc
                    new_best = True
                model.train()

            # logging
            if new_best or i == 0 or i == num_iter - 1 or (i + 1) % log_freq == 0:
                print(f'Epoch {i + 1}\n{"=" * len("Epoch {}".format(i + 1))}')
                print(f'Train Loss: {train_loss:.5f}\t Train Accuracy: {train_acc:.2f}%')
                print(f'Val Loss: {val_loss:.5f}\t Val Accuracy: {val_acc:.2f}%\n')
        except KeyboardInterrupt:
            print(f'\nStopped training after {i} epochs...')
            break

    model.load_state_dict(best_params)
    model.eval()


if __name__ == '__main__':
    # naive bayes
    with TimingContext('Training Naive Bayes', suffix='\n'):
        nb_model = NaiveBayes(
            alpha=1,
            do_set_of_words=False,
            do_bigrams=False
        )
    test_model(nb_model, train_iter, description='Naive Bayes: Training Set')
    test_model(nb_model, val_iter, description='Naive Bayes: Validation Set')
    test_model(nb_model, test_iter, description='Naive Bayes: Test Set')

    # logistic regression
    with TimingContext('Training Logistic Regression', suffix='\n'):
        lr_model = LogisticRegression(
            do_set_of_words=True, do_bigrams=False, do_embeddings=True
        )
        train_model(
            lr_model, reg_param=0.001, num_iter=200, learning_rate=0.01,
            log_freq=10, batch_size=1024
        )
    test_model(lr_model, train_iter, description='Logistic Reg: Training Set')
    test_model(lr_model, val_iter, description='Logistic Reg: Validation Set')
    test_model(lr_model, test_iter, description='Logistic Reg: Test Set')

    # CBOW neural net regression
    with TimingContext('Training CBOW Neural Net', suffix='\n'):
        cbow_nn = CbowNN(do_embedding=True, second_layer_size=20)
        train_model(
            cbow_nn, num_iter=1000, learning_rate=0.001, batch_size=1000,
            log_freq=50
        )
    test_model(cbow_nn, train_iter, description='CBOW NN: Training Set')
    test_model(cbow_nn, val_iter, description='CBOW NN: Validation Set')
    test_model(cbow_nn, test_iter, description='CBOW NN: Test Set')

    # CNN
    with TimingContext('Training CNN', suffix='\n'):
        cnn = CNN(
            num_filters=300, kernel_sizes=(3, 4, 5), second_layer_size=100,
            dropout_rate=0.5
        )
        train_model(
            cnn, num_iter=200, learning_rate=0.001, batch_size=256, log_freq=1,
            weight_decay=0
        )

    test_model(cnn, train_iter, description='CNN: Training Set')
    test_model(cnn, val_iter, description='CNN: Validation Set')
    test_model(cnn, test_iter, description='CNN: Test Set')
    # generate_predictions(cnn)

    # RNN
    with TimingContext('Training RNN', suffix='\n'):
        rnn = RNN(
            hidden_size=50, model_class=torch.nn.LSTM, dropout_rate=0.5,
            num_layers=2
        )
        train_model(
            rnn, num_iter=1000, learning_rate=0.001, batch_size=2048,
            log_freq=5
        )

    test_model(rnn, train_iter, description='RNN: Training Set')
    test_model(rnn, val_iter, description='RNN: Validation Set')
    test_model(rnn, test_iter, description='RNN: Test Set')
