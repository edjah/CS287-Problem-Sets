from data_setup import torch, torchtext, test, train_iter, val_iter
from models.naive_bayes import NaiveBayes
from models.logistic_regression import LogisticRegression
from models.cbow_nn import CbowNN
from models.cnn import CNN
from models.rnn import RNN
from utils import TimingContext


def test_code(model):
    """All models should be able to be run with following command."""
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.text)
        # here we assume that the name for dimension classes is `classes`
        _, argmax = probs.max('classes')
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

# Works for all NN-based models (including LR)
def train_model(model, xtrain, ytrain,
                num_iter=300, learning_rate=0.001, batch_size=100, log_freq=10,
                weight_decay=0):

    # training
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.BCELoss()

    for i in range(num_iter):
        total_loss = 0
        num_correct = 0

        s1 = torch.utils.data.RandomSampler(torch.arange(len(xtrain)))
        s2 = torch.utils.data.BatchSampler(s1, batch_size=batch_size,
                                           drop_last=False)
        try:
            for idx in s2:
                xbatch = xtrain.values[idx]
                labels_batch = ytrain[idx]

                # print(xbatch.shape)

                opt.zero_grad()
                probs = model.forward(xbatch)
                probs = probs.flatten()
                preds = probs.detach() > 0.5
                num_correct += int((preds == labels_batch.byte()).sum())

                loss = loss_fn(probs, labels_batch.float())
                loss.backward()
                total_loss += loss.detach()
                opt.step()
        except KeyboardInterrupt:
            print(f'\nStopped training after {i} epochs...')
            break

        if i == 0 or i == num_iter - 1 or (i + 1) % log_freq == 0:
            accuracy = 100.0 * num_correct / len(xtrain)
            print(f'Loss at epoch {i}: {total_loss}')
            print(f'Accuracy at epoch {i}: {accuracy:.2f}%')

    model.eval()


if __name__ == '__main__':

    # current main: runs using modified (refactored) testing function

    lr = LRNN()
    # NOTE: if you want set of words for LR, change it in this command AND the data generator
    x, y = trainingdata_lr()
    train_model(lr, x, y, num_iter=100)
    test_model(lr, train_iter, description="LR training set")
    test_model(lr, val_iter, description="LR validation set")

    cb = CbowNN()
    # NOTE: if you want set of words for LR, change it in this command AND the data generator
    x, y = trainingdata_cbownn()
    train_model(cb, x, y, num_iter=100)
    test_model(cb, train_iter, description="CbowNN training set")
    test_model(cb, val_iter, description="CbowNN validation set")

    cnn = CNN()
    # NOTE: if you want set of words for LR, change it in this command AND the data generator
    x, y = trainingdata_cnn()
    train_model(cnn, x, y, num_iter=100)
    test_model(cnn, train_iter, description="CNN training set")
    test_model(cnn, val_iter, description="CNN validation set")

else:

    # naive bayes
    with TimingContext('Training Naive Bayes', suffix='\n'):
        nb_model = NaiveBayes(
            alpha=1,
            do_set_of_words=False
        )
    test_model(nb_model, train_iter, description='Naive Bayes: Training Set')
    test_model(nb_model, val_iter, description='Naive Bayes: Validation Set')

    # logistic regression
    with TimingContext('Training Logistic Regression', suffix='\n'):
        lr_model = LogisticRegression(
            num_iter=200,
            learning_rate=0.2,
            reg_param=0.001,
            do_set_of_words=True
        )
    test_model(lr_model, train_iter, description='Logistic Reg: Training Set')
    test_model(lr_model, val_iter, description='Logistic Reg: Validation Set')

    # CBOW neural net regression
    with TimingContext('Training CBOW Neural Net', suffix='\n'):
        cbow_nn = CbowNN(
            num_iter=100,
            learning_rate=0.001,
            second_layer_size=10,
            batch_size=1000,
            log_freq=50
        )
    test_model(cbow_nn, train_iter, description='CBOW NN: Training Set')
    test_model(cbow_nn, val_iter, description='CBOW NN: Validation Set')

    # CNN
    with TimingContext('Training CNN', suffix='\n'):
        cnn = CNN(
            num_filters=20,
            kernel_sizes=(3, 6, 9),
            second_layer_size=20,
            dropout_rate=0.8,
            num_iter=200,
            learning_rate=0.001,
            batch_size=500,
            log_freq=50
        )

    test_model(cnn, train_iter, description='CNN: Training Set')
    test_model(cnn, val_iter, description='CNN: Validation Set')

    # RNN
    with TimingContext('Training RNN', suffix='\n'):
        rnn = RNN(
            hidden_size=50,
            model_class=torch.nn.RNN,
            inp_size=20,
            num_layers=2,
            num_iter=1000,
            learning_rate=0.001,
            batch_size=2048,
            log_freq=50
        )

    test_model(rnn, train_iter, description='RNN: Training Set')
    test_model(rnn, val_iter, description='RNN: Validation Set')
