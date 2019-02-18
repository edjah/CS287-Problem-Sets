import math
import torch
import torchtext
from tqdm import tqdm

from namedtensor.text import NamedField
from torchtext.data.iterator import BPTTIterator
from torchtext.data import Batch, Dataset

from ngram_model import NGramModel

DEBUG_MODE = False

# Our input $x$
TEXT = NamedField(names=('seqlen',))

# Data distributed with the assignment
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
    path='.', train='data/train.txt', validation='data/valid.txt',
    test='data/valid.txt', text_field=TEXT
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


def text_to_idx(dataset):
    train_txt = next(iter(dataset)).text
    return [TEXT.vocab.stoi[w] for w in train_txt]


def generate_predictions(model):
    with open('predictions.txt', 'w') as fout:
        print('id,word', file=fout)
        sentences = list(open('data/input.txt'))
        for i, line in tqdm(enumerate(sentences, 1), total=len(sentences)):
            words = line.strip().split(' ')[:-1]
            idx = [TEXT.vocab.stoi[w] for w in words]
            probs, pred_idx = model.predict_last_word(idx).topk(20)
            predictions = [TEXT.vocab.itos[i] for i in pred_idx]
            print('%d,%s' % (i, ' '.join(predictions)), file=fout)


if __name__ == '__main__':
    train_idx = text_to_idx(train)
    orig_val_idx = text_to_idx(val)

    val_len = int(0.8 * len(orig_val_idx))
    val_idx, test_idx = orig_val_idx[:val_len], orig_val_idx[val_len:]

    ngram_model = NGramModel(train_idx, max_ngram=3, min_count=1)
    ngram_model.train_interpolation(val_idx, test_idx)

    sent = ['we', 'are']
    idx = [TEXT.vocab.stoi[w] for w in sent]
    print(TEXT.vocab.itos[ngram_model.predict_last_word(idx).argmax()])

    print('N-Gram: Train Perplexity:', ngram_model.perplexity(train_idx))
    print('N-Gram: Validation Perplexity:', ngram_model.perplexity(val_idx))
    print('N-Gram: Test Perplexity:', ngram_model.perplexity(test_idx))

    generate_predictions(ngram_model)
