from data_setup import torch, nn, TEXT, WORD_VECS


class BasicLSTM(torch.nn.Module):
    def __init__(self, hidden_size, num_layers, emb_dropout=0.1, lstm_dropout=0.1, out_dropout=0.2):
        super().__init__()

        self.vocab_size = len(TEXT.vocab)
        self.emb_dropout = torch.nn.Dropout(p=emb_dropout)
        self.out_dropout = torch.nn.Dropout(p=out_dropout)
        self.embeddings = nn.Embedding.from_pretrained(WORD_VECS.clone(), freeze=False)
        self.rnn = torch.nn.LSTM(WORD_VECS.shape[1], hidden_size, num_layers, dropout=lstm_dropout)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, self.vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, hidden=None):
        emb = self.emb_dropout(self.embeddings(x))
        output, hidden = self.rnn(emb)
        hidden = tuple(h.detach() for h in hidden)
        result = output.view(output.size(0) * output.size(1), output.size(2))
        return self.out(self.out_dropout(result))

    def get_data(self, batch_iter):
        sent_list = []
        results_list = []

        first = None
        for batch in batch_iter:
            if first is not None and batch.text.shape['seqlen'] != first:
                pass
            else:
                first = len(batch.text.values)
                sent_list.append(batch.text.values)
                results_list.append(batch.target.values)

        return sent_list, results_list.flatten()

    def predict_last_word(self, x):
        out = self.forward(torch.tensor([x]))
        return out[-0]
