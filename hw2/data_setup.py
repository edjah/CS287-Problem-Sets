import math
import torch
import torchtext
import torch.nn as nn
from torchtext.vocab import GloVe

from torchtext.data.iterator import BPTTIterator
from torchtext.data import Batch, Dataset

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField


# setting the default tensor type to `torch.cuda.FloatTensor`
# change this to `torch.FloatTensor` if you don't have a gpu
# torch.set_default_tensor_type(torch.FloatTensor)
torch.set_default_tensor_type(torch.FloatTensor)


DEBUG_MODE = False

# Our input $x$
TEXT = NamedField(names=('seqlen',))

# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path='.', train='data/train.txt', validation='data/valid.txt',
    test='data/test.txt', text_field=TEXT
)

# use a smaller vocab size when debugging
if not DEBUG_MODE:
    TEXT.build_vocab(train)
else:
    TEXT.build_vocab(train, max_size=1000)


class NamedBpttIterator(BPTTIterator):
    def __iter__(self):
        text = self.dataset[0].text
        TEXT = self.dataset.fields['text']
        TEXT.eos_token = None

        num_batches = math.ceil(len(text) / self.batch_size)
        pad_amount = int(num_batches * self.batch_size - len(text))
        text += [TEXT.pad_token] * pad_amount

        data = TEXT.numericalize([text], device=self.device)
        data = data.stack(('seqlen', 'batch'), 'flat') \
                   .split('flat', ('batch', 'seqlen'), batch=self.batch_size) \
                   .transpose('seqlen', 'batch')

        fields = [('text', TEXT), ('target', TEXT)]
        dataset = Dataset(examples=self.dataset.examples, fields=fields)

        while True:
            for i in range(0, len(self) * self.bptt_len, self.bptt_len):
                self.iterations += 1
                seq_len = min(self.bptt_len, len(data) - i - 1)
                yield Batch.fromvars(
                    dataset, self.batch_size,
                    text=data.narrow('seqlen', i, seq_len),
                    target=data.narrow('seqlen', i + 1, seq_len)
                )

            if not self.repeat:
                return


train_iter, val_iter, test_iter = NamedBpttIterator.splits(
    (train, val, test), batch_size=10, device=torch.device('cuda'),
    bptt_len=32, repeat=False
)

TEXT.vocab.load_vectors(vectors=GloVe("840B"))
WORD_VECS = TEXT.vocab.vectors
