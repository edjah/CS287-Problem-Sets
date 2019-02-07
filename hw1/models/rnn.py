from data_setup import torch, TEXT


WORD_VECS = TEXT.vocab.vectors


class RNN(torch.nn.Module):
    def __init__(self, hidden_size=100, model_class=torch.nn.RNN,
                 num_layers=3, dropout_rate=0.5):
        super().__init__()

        self.max_words = 60

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.embedding = torch.nn.Embedding.from_pretrained(WORD_VECS, freeze=False)
        self.rnn = model_class(
            WORD_VECS.shape[1], hidden_size, num_layers=num_layers,
            bidirectional=True
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, 1),
            torch.nn.Sigmoid()
        )

    def __call__(self, text):
        return self.forward(self.transform(text)).flatten()

    def forward(self, batch):
        batch = batch.transpose(0, 1)

        embeddings = self.dropout(self.embedding(batch))
        _, (hidden, _) = self.rnn(embeddings)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.out(self.dropout(hidden).squeeze(0))

    def get_data(self, datasets):
        x_lst = []
        y_lst = []
        for dataset in datasets:
            for batch in dataset:
                x_lst.append(self.transform(batch.text))
                y_lst.append(batch.label.values)

        X = torch.cat(x_lst)
        Y = torch.cat(y_lst)
        return X, Y

    def transform(self, text):
        sentences = text.transpose('batch', 'seqlen').values.clone()
        pad_amt = (0, self.max_words - sentences.shape[1])
        result = torch.nn.functional.pad(sentences, pad_amt, value=1)
        return result
