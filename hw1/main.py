from data_setup import torchtext, test, train_iter, val_iter
from models.naive_bayes import NaiveBayes
from models.logistic_regression import LogisticRegression
from models.cbow_nn import CbowNN
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
        batch_result = model(batch.text) > 0.5
        total_correct += (batch.label.values == batch_result.long()).sum()
        total += len(batch)

    if description:
        print(description)
        print('=' * len(description))
    print(f'Num correct: {total_correct}\nTotal: {total}')
    print(f'Accuracy: {100.0 * total_correct.float() / total:.2f}%\n')


if __name__ == '__main__':

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
            num_iter=500,
            learning_rate=0.2,
            reg_param=0.001,
            do_set_of_words=True
        )
    test_model(lr_model, train_iter, description='Logistic Reg: Training Set')
    test_model(lr_model, val_iter, description='Logistic Reg: Validation Set')

    # CBOW neural net regression
    with TimingContext('Training CBOW Neural Net', suffix='\n'):
        lr_model = CbowNN(
            num_iter=100,
            learning_rate=0.001,
            second_layer_size=10,
            batch_size=1000,
            log_freq=50
        )
    test_model(lr_model, train_iter, description='CBOW NN: Training Set')
    test_model(lr_model, val_iter, description='CBOW NN: Validation Set')
