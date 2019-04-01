import random

import torch
import torchtext

# Named Tensor wrappers
from namedtensor import NamedTensor
from namedtensor.text import NamedField

torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Our input $x$
TEXT = NamedField(names=('seqlen',))

# Our labels $y$
LABEL = NamedField(sequential=False, names=())

# loading up dataset
train, val, test = torchtext.datasets.SNLI.splits(TEXT, LABEL)
TEXT.build_vocab(train)
LABEL.build_vocab(train)

# bucketing the dataset into iterators
train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, val, test), batch_size=64, device=torch.device('cuda'),
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
TEXT.vocab.vectors = NamedTensor(vectors, ('word', 'embedding'))
WORD_VECS = TEXT.vocab.vectors
embed_size = WORD_VECS.shape['embedding']
