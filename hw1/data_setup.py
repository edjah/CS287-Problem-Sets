import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField


# setting the default tensor type to `torch.cuda.FloatTensor`
# change this to `torch.FloatTensor` if you don't have a gpu
torch.set_default_tensor_type(torch.cuda.FloatTensor)
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
