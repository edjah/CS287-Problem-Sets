import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

from models.naive_bayes import NaiveBayes
from models.logistic_regression import LogisticRegression

from utils import TimingContext


# setting the default tensor type to `torch.cuda.FloatTensor`
# change this to `torch.FloatTensor` if you don't have a gpu
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# Our input $x$
TEXT = NamedField(names=('seqlen',))


# Our labels $y$
LABEL = NamedField(sequential=False, names=(), unk_token=None)


train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')


TEXT.build_vocab(train)
LABEL.build_vocab(train)

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10)


# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))


def test_code(model):
    "All models should be able to be run with following command."
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
            train_iter=train_iter,
            vocab_len=len(TEXT.vocab),
            alpha=1
        )
    test_model(nb_model, train_iter, description='Naive Bayes: Training Set')
    test_model(nb_model, val_iter, description='Naive Bayes: Validation Set')

    # logistic regression
    with TimingContext('Training Logistic Regression', suffix='\n'):
        lr_model = LogisticRegression(
            train_iter=train_iter,
            vocab_len=len(TEXT.vocab),
            num_iter=500,
            learning_rate=0.05,
            reg_param=0.001,
            do_set_of_words=False
        )
    test_model(lr_model, train_iter, description='Logistic Reg: Training Set')
    test_model(lr_model, val_iter, description='Logistic Reg: Validation Set')
