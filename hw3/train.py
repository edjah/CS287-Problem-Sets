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
def filter_pred(x):
    return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

train, val, test = datasets.IWSLT.splits(
    exts=('.de', '.en'), fields=(DE, EN), filter_pred=filter_pred,
)

src = open("valid.src", "w")
trg = open("valid.trg", "w")
for example in val:
    print(" ".join(example.src), file=src)
    print(" ".join(example.trg), file=trg)
src.close()
trg.close()


MIN_FREQ = 5
DE.build_vocab(train.src, min_freq=MIN_FREQ)
EN.build_vocab(train.trg, min_freq=MIN_FREQ)

print("Size of German vocab", len(DE.vocab))
print("Size of English vocab", len(EN.vocab))


# Loading word vectors
EN.vocab.load_vectors(vectors=GloVe("840B"))
DE.vocab.load_vectors(vectors=FastText(language="de"))


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
    """Defines an iterator that batches examples of similar lengths together.
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


EN_VECS = EN.vocab.vectors
DE_VECS = DE.vocab.vectors

EN_embed_size = EN_VECS.shape[1]
DE_embed_size = DE_VECS.shape[1]

EN_VOCAB_LEN = len(EN.vocab)


class EncoderRNN(ntorch.nn.Module):
    def __init__(self, num_layers, hidden_size, emb_dropout=0.1, lstm_dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.emb_dropout = ntorch.nn.Dropout(p=emb_dropout)
        self.embeddings = ntorch.nn.Embedding.from_pretrained(DE_VECS.clone(), freeze=False)
        self.lstm = ntorch.nn.LSTM(DE_embed_size, hidden_size, num_layers, dropout=lstm_dropout) \
                             .spec("embedding", "srcSeqlen", "hidden")

    def forward(self, x, hidden=None):
        emb = self.emb_dropout(self.embeddings(x))
        output, hidden = self.lstm(emb, hidden)
        return output, hidden


# TODO: remove duplicated code
class DecoderRNN(ntorch.nn.Module):
    def __init__(self, num_layers, hidden_size, emb_dropout=0.1, lstm_dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.emb_dropout = ntorch.nn.Dropout(p=emb_dropout)
        self.embeddings = ntorch.nn.Embedding.from_pretrained(EN_VECS.clone(), freeze=False)
        self.lstm = ntorch.nn.LSTM(DE_embed_size, hidden_size, num_layers, dropout=lstm_dropout) \
                             .spec("embedding", "trgSeqlen", "hidden")

    def forward(self, x, hidden):
        emb = self.emb_dropout(self.embeddings(x))
        output, hidden = self.lstm(emb, hidden)
        return output, hidden


def flip(ntensor, dim):
    ntensor = ntensor.clone()
    idx = ntensor._schema._names.index(dim)
    ntensor._tensor = ntensor._tensor.flip(idx)
    return ntensor


class Seq2Seq(ntorch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.out = ntorch.nn.Linear(decoder.hidden_size, EN_VOCAB_LEN) \
                            .spec("hidden", "vocab")

    def _shift_tgt(self, tgt):
        start_of_sent = [[EN.vocab.stoi[BOS_WORD]] * tgt.shape['batch']]
        start_of_sent = ntorch.tensor(start_of_sent, names=('trgSeqlen', 'batch'))
        end_of_sent =  tgt[{'trgSeqlen': slice(0, tgt.shape['trgSeqlen'] - 1)}]
        shifted = ntorch.cat((start_of_sent, end_of_sent), 'trgSeqlen')
        return shifted

    # this function should only be used in training/evaluation
    def forward(self, src, tgt):
        # TODO: reverse src before encoding
        src = flip(src, 'srcSeqlen')
        _, enc_hidden = self.encoder(src)
        dec_output, _ = self.decoder(self._shift_tgt(tgt), enc_hidden)
        out = self.out(dec_output).log_softmax('vocab')
        return out

    # this function should implement beam search to translate the src
    # src should be (seqLen,) NamedTensor
    def translate(self, src, max_len=30):
        model.eval()
        with torch.no_grad():
            # TODO: reverse src before encoding
            src = ntorch.tensor(src.values.unsqueeze(0), ('batch', 'srcSeqlen'))
            src = flip(src, 'srcSeqlen')
            _, enc_hidden = encoder.forward(src)

            dec_input = ntorch.tensor([[EN.vocab.stoi[BOS_WORD]]], ('batch', 'trgSeqlen'))
            dec_hidden = enc_hidden

            translated_sent = []
            for i in range(max_len):
                dec_output, dec_hidden = decoder.forward(dec_input, dec_hidden)
                prediction = self.out(dec_output).argmax('vocab')
                translated_sent.append(prediction.item())
                if prediction.item() == EN.vocab.stoi[EOS_WORD]:
                    break
                else:
                    dec_input = prediction

            return torch.tensor(translated_sent)


def evaluate(model, batches):
    model.eval()
    with torch.no_grad():
        loss_fn = ntorch.nn.NLLLoss(reduction="sum").spec("vocab")
        tot_loss = 0
        num_ex = 0
        for batch in batches:
            log_probs = model.forward(batch.src, batch.trg)
            tot_loss += loss_fn(log_probs, batch.trg).values
            num_ex += batch.trg.shape['batch'] * batch.trg.shape['trgSeqlen']

        # TODO: compute bleu
        return torch.exp(tot_loss / num_ex), 0

def train_model(model, num_epochs=300, learning_rate=0.001, weight_decay=0, log_freq=1):
    model.train()
    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    loss_fn = ntorch.nn.NLLLoss().spec("vocab")
    start_time = time.time()

    best_params = {k: p.detach().clone() for k, p in model.named_parameters()}
    best_val_loss = float('inf')

    for i in range(num_epochs):
        try:
            for batch in tqdm(train_iter, total=len(train_iter)):
                opt.zero_grad()

                log_probs = model.forward(batch.src, batch.trg)
                loss = loss_fn(log_probs, batch.trg)

                # compute gradients and update weights
                loss.backward()
                opt.step()

            # evaluate performance on entire sets
            model.eval()
            train_loss, train_bleu = evaluate(model, train_iter)
            val_loss, val_bleu = evaluate(model, val_iter)
            model.train()

            # saving the parameters with the best validation loss
            if val_loss < best_val_loss:
                best_params = {k: p.detach().clone() for k, p in model.named_parameters()}
                best_val_loss = val_loss

            # logging
            if i == 0 or i == num_epochs - 1 or (i + 1) % log_freq == 0:
                msg = f"{round(time.time() - start_time)} sec: Epoch {i + 1}"
                print(f'{msg}\n{"=" * len(msg)}')
                print(f'Train Perplexity: {train_loss:.5f}\t Train BLEU: {train_bleu:.2f}%')
                print(f'Val Perplexity: {val_loss:.5f}\t Val BLEU: {val_bleu:.2f}%\n')

        except KeyboardInterrupt:
            print(f'\nStopped training after {i} epochs...')
            break

    model.eval()
    model.load_state_dict(best_params)

    msg = f"{round(time.time() - start_time)} sec: Final Results"
    print(f'{msg}\n{"=" * len(msg)}')

    train_loss, train_bleu = evaluate(model, train_iter)
    val_loss, val_bleu = evaluate(model, val_iter)
    print(f'Train Perplexity: {train_loss:.5f}\t Train BLEU: {train_bleu:.2f}%')
    print(f'Val Perplexity: {val_loss:.5f}\t Val BLEU: {val_bleu:.2f}%\n')


# constructing the model
encoder = EncoderRNN(num_layers=3, hidden_size=300, emb_dropout=0, lstm_dropout=0)
decoder = DecoderRNN(num_layers=3, hidden_size=300, emb_dropout=0, lstm_dropout=0)
model = Seq2Seq(encoder, decoder)
# model.load_state_dict(torch.load("basic_seq2seq_weights"))

# training
train_model(model, num_epochs=40, learning_rate=0.001, weight_decay=0, log_freq=1)
torch.save(model.state_dict(), "basic_seq2seq_weights")


# generating some sample predictions
batch = next(iter(train_iter))
for i in range(len(batch.src)):
    src = batch.src[{'batch': i}]
    trg = batch.trg[{'batch': i}]

    translated = model.translate(src)

    german = ' '.join([DE.vocab.itos[i] for i in src.values])
    english_translation = ' '.join([EN.vocab.itos[i] for i in translated])
    english_actual = ' '.join([EN.vocab.itos[i] for i in trg.values])

    print('German:', german)
    print('English Translated:', english_translation)
    print('English Actual:', english_actual, end='\n\n')
