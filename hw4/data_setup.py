import random

import torch
import torchtext

# Named Tensor wrappers
from namedtensor import ntorch
from namedtensor.text import NamedField

from utils import chunks

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Our input $x$
TEXT = NamedField(names=('seqlen',))

# Our labels $y$
LABEL = NamedField(sequential=False, names=())

# loading up dataset
train, val, test = torchtext.datasets.SNLI.splits(TEXT, LABEL)

# building up the vocabulary an initial time without cleaning
TEXT.build_vocab(train)
LABEL.build_vocab(train)


def clean(sentence):
    return ['<s>'] + [w.strip('\'".!?,').lower() for w in sentence] + ['</s>']


def ordered_test_stage_1(sent):
    return clean([TEXT.vocab.itos[w] for w in sent])


def ordered_test_stage_2(sent):
    idx = torch.tensor([[TEXT.vocab.stoi[w]] for w in sent])
    return ntorch.tensor(idx, names=('seqlen', 'batch'))


# generating an ordered test set (without cleaning). this is critical for kaggle
FIRST_ORDERED_TEST = []

correct_order_test_iter = torchtext.data.BucketIterator(
    test, train=False, batch_size=10, device=torch.device('cuda')
)
for batch in correct_order_test_iter:
    for i in range(len(batch)):
        FIRST_ORDERED_TEST.append([
            ordered_test_stage_1(batch.premise.values[:, i]),
            ordered_test_stage_1(batch.hypothesis.values[:, i]),
            ntorch.stack([batch.label.get('batch', i)], 'batch')
        ])


# critical for performance: removing punctuantion and lowercasing
for dataset in train, val, test:
    for ex in dataset:
        ex.premise = clean(ex.premise)
        ex.hypothesis = clean(ex.hypothesis)

# rebuilding the vocabulary after cleaning
TEXT.build_vocab(train)
LABEL.build_vocab(train)


SECOND_ORDERED_TEST = []
for chunk in chunks(FIRST_ORDERED_TEST, 10):
    premise_batch, hypothesis_batch, label_batch = [], [], []
    for premise, hypothesis, label in chunk:
        premise_batch.append(ordered_test_stage_2(premise))
        hypothesis_batch.append(ordered_test_stage_2(hypothesis))
        label_batch.append(label)

    SECOND_ORDERED_TEST.append([
        ntorch.cat(premise_batch, 'batch'),
        ntorch.cat(hypothesis_batch, 'batch'),
        ntorch.cat(label_batch, 'batch'),
    ])

# bucketing the dataset into iterators
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=128, device=torch.device('cuda'),
    repeat=False
)

# Build the vocabulary with word embeddings
# Out-of-vocabulary (OOV) words are hashed to one of 100 random embeddings each
# initialized to mean 0 and standarad deviation 1 (Sec 5.1)
unk_vectors = [torch.randn(300) for _ in range(100)]

# uncomment this when GloVe becomes available again
TEXT.vocab.load_vectors(vectors='glove.840B.300d',
                        unk_init=lambda x: random.choice(unk_vectors))

# normalized to have l_2 norm of 1
vectors = TEXT.vocab.vectors
vectors = vectors / vectors.norm(dim=1, keepdim=True)
assert not (vectors != vectors).any(), 'NaNs exist in the embeddings!'

# set the padding embedding to 0
vectors[1] = torch.zeros(vectors.shape[1])

# update TEXT.vocab.vectors
TEXT.vocab.vectors = ntorch.tensor(vectors, ('word', 'embedding'))
WORD_VECS = TEXT.vocab.vectors
embed_size = WORD_VECS.shape['embedding']
