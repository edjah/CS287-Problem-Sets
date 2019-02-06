import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField


# setting the default tensor type to `torch.cuda.FloatTensor`
# change this to `torch.FloatTensor` if you don't have a gpu
torch.set_default_tensor_type(torch.FloatTensor)
# torch.set_default_tensor_type(torch.FloatTensor)


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

# all functions to generate data for training/testing

def trainingdata_lr():
    # setting up
    x_lst = []
    y_lst = []

    for batch in train_iter:
        sentences = batch.text.transpose('batch', 'seqlen').values.clone()
        y_lst.append(batch.label.values.float())
        for sent in sentences:
            x_lst.append(transform_x_lr(sent))

    xtrain = ntorch.stack(x_lst, 'sentence')
    # ytrain = ntorch.tensor(torch.cat(y_lst), names=('sentence',))
    ytrain = torch.cat(y_lst)

    return (xtrain, ytrain)

def trainingdata_cbownn():
    # setting up data FOR CBOWNN
    x_lst = []
    y_lst = []
    for batch in train_iter:
        sentences = batch.text.transpose('batch', 'seqlen').values.clone()
        labels = batch.label.values
        xbatch = ntorch.tensor(TEXT.vocab.vectors[sentences].sum(dim=1),
                               names=('sentence', 'embed'))
#         xbatch = TEXT.vocab.vectors[sentences].sum(dim=1)
        x_lst.append(xbatch)
        y_lst.append(labels)

    xtrain = ntorch.cat(x_lst, dim='sentence')
#     xtrain = torch.cat(x_lst)
    ytrain = torch.cat(y_lst)

    return (xtrain, ytrain)

def trainingdata_cnn():

    # setting up data FOR CNN
    x_lst = []
    y_lst = []
    for batch in train_iter:
        sentences = batch.text.transpose('batch', 'seqlen').values.clone()
        pad_amt = (0, 60 - sentences.shape[1])
        sent_padded = torch.nn.functional.pad(sentences, pad_amt, value=1)
        x_lst.append(TEXT.vocab.vectors[sent_padded])
        y_lst.append(batch.label.values)

    xtrain = ntorch.tensor(torch.cat(x_lst).transpose(1, 2), names=('sentence', 'channel', 'seqlen'))
#     xtrain = torch.cat(x_lst).transpose(1, 2)

    print(xtrain.shape)
    ytrain = torch.cat(y_lst)

    return (xtrain, ytrain)

def transform_x_lr(sent, do_set_of_words=False, vocab_len=16284):
    # regular bag of words / set of words
    words = torch.bincount(sent, minlength=vocab_len)
    # words[vocab_len] = 1
    if do_set_of_words:
        words = (words > 0)
    return ntorch.tensor(words.float(), names=('vocab',))
