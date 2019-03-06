from namedtensor import ntorch
from data_setup import EN, DE, DE_VECS, EN_VECS, BOS_WORD, EOS_WORD

BOS_IND = EN.vocab.stoi[BOS_WORD]
EOS_IND = EN.vocab.stoi[EOS_WORD]


class EncoderRNN(ntorch.nn.Module):
    def __init__(self, num_layers, hidden_size, emb_dropout=0.1, lstm_dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.emb_dropout = ntorch.nn.Dropout(p=emb_dropout)
        self.embeddings = ntorch.nn.Embedding.from_pretrained(DE_VECS.clone(), freeze=False)
        self.lstm = ntorch.nn.LSTM(DE_VECS.shape[1], hidden_size, num_layers, dropout=lstm_dropout) \
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
        self.lstm = ntorch.nn.LSTM(EN_VECS.shape[1], hidden_size, num_layers, dropout=lstm_dropout) \
                             .spec("embedding", "trgSeqlen", "hidden")

    def forward(self, x, hidden, enc_outputs=None):
        emb = self.emb_dropout(self.embeddings(x))
        output, hidden = self.lstm(emb, hidden)
        return output, hidden


class VanillaSeq2Seq(ntorch.nn.Module):
    def __init__(self, encoder, decoder, dropout=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dropout = ntorch.nn.Dropout(p=dropout)
        self.out = ntorch.nn.Linear(decoder.hidden_size, len(EN.vocab)).spec("hidden", "vocab")

    def _shift_trg(self, trg):
        start_of_sent = [[BOS_IND] * trg.shape['batch']]
        start_of_sent = ntorch.tensor(start_of_sent, names=('trgSeqlen', 'batch'))
        end_of_sent = trg[{'trgSeqlen': slice(0, trg.shape['trgSeqlen'] - 1)}]
        shifted = ntorch.cat((start_of_sent, end_of_sent), 'trgSeqlen')
        return shifted

    def _flip(self, ntensor, dim):
        ntensor = ntensor.clone()
        idx = ntensor._schema._names.index(dim)
        ntensor._tensor = ntensor._tensor.flip(idx)
        return ntensor

    # this function should only be used in training/evaluation
    def forward(self, src, trg=None):
        src = self._flip(src, 'srcSeqlen')
        _, enc_hidden = self.encoder(src)

        dec_outputs, _ = self.decoder(self._shift_trg(trg), enc_hidden)
        all_log_probs = self.out(self.dropout(dec_outputs)).log_softmax('vocab')
        return all_log_probs
