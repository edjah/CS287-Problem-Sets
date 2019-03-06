import torch
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


class DecoderRNN(ntorch.nn.Module):
    def __init__(self, num_layers, hidden_size, emb_dropout=0.1, lstm_dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.emb_dropout = ntorch.nn.Dropout(p=emb_dropout)
        self.embeddings = ntorch.nn.Embedding.from_pretrained(EN_VECS.clone(), freeze=False)

        self.attn = ntorch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.v = ntorch.nn.Linear(self.hidden_size, 1)

        self.lstm = ntorch.nn.LSTM(
            DE_VECS.shape[1] + hidden_size, hidden_size, num_layers,
            dropout=lstm_dropout
        ).spec("embedding", "trgSeqlen", "hidden")

    def forward(self, x, hidden, encoder_outputs):
        emb = self.emb_dropout(self.embeddings(x))
        srcLength = encoder_outputs.shape['srcSeqlen']

        actual_hidden = hidden[0].get('layers', -1)
        # encoder outputs shape: batch x srcSeqlen x hidden
        # actual_hidden shape: batch x hidden
        actual_hidden = ntorch.stack([actual_hidden for i in range(srcLength)], 'srcSeqlen')

        energy = self.attn(ntorch.cat([encoder_outputs, actual_hidden], 'hidden'))
        energy = energy.tanh()

        attn_weights = self.v(energy)
        attn_weights = attn_weights.get('hidden', 0).softmax('srcSeqlen')

        context = attn_weights.dot("srcSeqlen", encoder_outputs)
        context = context.rename('hidden', 'embedding')
        context = ntorch.stack([context], 'trgSeqlen')  # hack for unsqueeze

        lstm_input = ntorch.cat((context, emb), "embedding")
        output, hidden = self.lstm(lstm_input, hidden)
        return output, hidden


class AttnSeq2Seq(ntorch.nn.Module):
    def __init__(self, encoder, decoder, dropout=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dropout = ntorch.nn.Dropout(p=dropout)
        self.out = ntorch.nn.Linear(decoder.hidden_size, len(EN.vocab)).spec("hidden", "vocab")

    def _flip(self, ntensor, dim):
        ntensor = ntensor.clone()
        idx = ntensor._schema._names.index(dim)
        ntensor._tensor = ntensor._tensor.flip(idx)
        return ntensor

    def forward(self, src, trg, teacher_forcing_ratio=1.0):
        src = self.flip(src, 'srcSeqlen')
        enc_outputs, enc_hidden = self.encoder(src)

        dec_hidden = enc_hidden
        dec_input = ntorch.tensor([[BOS_IND] * trg.shape['batch']], ('trgSeqlen', 'batch'))
        all_log_probs = []
        for i in range(trg.shape['trgSeqlen']):
            dec_output, dec_hidden = self.decoder(dec_input, dec_hidden, enc_outputs)
            dec_output = self.dropout(dec_output).get('trgSeqlen', 0)
            log_probs = self.out(dec_output).log_softmax('vocab')
            all_log_probs.append(log_probs)

            if torch.rand(1) < teacher_forcing_ratio:
                dec_input = trg[{'trgSeqlen': slice(i, i + 1)}]
            else:
                dec_input = [log_probs.argmax(dim='vocab').values.tolist()]
                dec_input = ntorch.tensor(dec_input, ('trgSeqlen', 'batch'))

        return ntorch.stack(all_log_probs, 'trgSeqlen')
