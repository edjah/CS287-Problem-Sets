import torch
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

from collections import defaultdict


# setting the default tensor type to `torch.cuda.FloatTensor`
# change this to `torch.FloatTensor` if you don't have a gpu
# torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# Our input $x$
TEXT = NamedField(names=('seqlen',))

# Our labels $y$
LABEL = NamedField(sequential=False, names=(), unk_token=None)

train, val, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

for dataset in (train, val, test):
    for example in dataset:
        example.text = [word.lower() for word in example.text]

TEXT.build_vocab(train, val, test)
LABEL.build_vocab(train, val, test)

train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=10)


# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

glove = GloVe(name="6B", dim=300)
glove.vectors = glove.vectors[torch.arange(len(TEXT.vocab) + 10)]


# genrating a mapping from bigrams to indexes
all_bigrams = set()
for dataset in (train, val, test):
    for example in dataset:
        idx = [TEXT.vocab.stoi[word] for word in example.text]
        all_bigrams |= set((i,) for i in idx)
        for i in range(len(idx) - 1):
            all_bigrams.add((idx[i], idx[i + 1]))

bigram_map = defaultdict(int)
for i, bigram in enumerate(sorted(all_bigrams)):
    bigram_map[bigram] = i


def make_bigrams(sent, do_unigrams=True):
    if isinstance(sent, torch.Tensor):
        sent = sent.tolist()

    if do_unigrams:
        bigrams = [(i,) for i in sent]
    else:
        bigrams = []
    for i in range(len(sent) - 1):
        bigrams.append((sent[i], sent[i + 1]))
    return torch.tensor([bigram_map[b] for b in bigrams])
