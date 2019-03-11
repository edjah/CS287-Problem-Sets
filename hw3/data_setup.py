# Torch
import torch

# Text processing library and methods for pretrained word embeddings
from torchtext import data, datasets

# Named Tensor wrappers
from namedtensor import ntorch
from namedtensor.text import NamedField

# Word vectors
from torchtext.vocab import GloVe, FastText

# utilities
import time
import random
from tqdm import tqdm

import spacy

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


BOS_WORD = '<s>'
EOS_WORD = '</s>'

DE = NamedField(names=('srcSeqlen',), tokenize=tokenize_de)

# only target needs BOS/EOS
EN = NamedField(
    names=('trgSeqlen',), tokenize=tokenize_en,
    init_token=BOS_WORD, eos_token=EOS_WORD
)


MAX_LEN = 20
MIN_FREQ = 5


def filter_pred(x):
    return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN


train, val, test = datasets.IWSLT.splits(
    exts=('.de', '.en'), fields=(DE, EN), filter_pred=filter_pred,
)


DE.build_vocab(train.src, min_freq=MIN_FREQ)
EN.build_vocab(train.trg, min_freq=MIN_FREQ)

BOS_IND = EN.vocab.stoi[BOS_WORD]
EOS_IND = EN.vocab.stoi[EOS_WORD]


# Loading word vectors
EN.vocab.load_vectors(vectors=GloVe("840B"))
DE.vocab.load_vectors(vectors=FastText(language="de"))

EN_VECS = EN.vocab.vectors
DE_VECS = DE.vocab.vectors


def batcher(data, batch_size):
    # sort first by src len, then by trg len
    data = sorted(data, key=lambda x: (len(x.src), len(x.trg)))
    batch = []
    cur_len = None

    # all batches have the same src_len and trg_len
    for ex in data:
        lengths = (len(ex.src), len(ex.trg))
        if (lengths != cur_len and batch) or len(batch) == batch_size:
            yield batch
            batch = []
        cur_len = lengths
        batch.append(ex)

    if batch:
        yield batch


class GoodBucketIterator(data.Iterator):
    """
    Defines an iterator that batches examples of similar lengths together.
    Minimizes amount of padding needed while producing freshly shuffled
    batches for each new epoch. See pool for the bucketing procedure used.
    """
    def create_batches(self):
        self.batches = list(batcher(self.data(), self.batch_size))
        random.shuffle(self.batches)

    def __len__(self):
        if hasattr(self, 'batches'):
            return len(self.batches)
        return super().__len__()


BATCH_SIZE = 128
device = torch.device('cuda:0')
train_iter, val_iter = GoodBucketIterator.splits(
    (train, val), batch_size=BATCH_SIZE, device=device, repeat=False
)


def escape(l):
    return l.replace("\"", "<quote>").replace(",", "<comma>")
