from data_setup import torch, TEXT

WORD_VECS = TEXT.vocab.vectors


class NNLM_RNN(torch.nn.Module):
    def __init__(self, n, hidden_size=100, num_layers=3, dropout_rate=0.3):
        super().__init__()

        self.vocab_length = len(TEXT.vocab)
        self.n = n

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.embedding = torch.nn.Embedding.from_pretrained(WORD_VECS.clone(), freeze=False)
        self.rnn = torch.nn.LSTM(300, hidden_size, num_layers=num_layers, bidirectional=True)
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, self.vocab_length),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, xvals):
        xvals = xvals.transpose(0, 1)
        embeddings = self.dropout(self.embedding(xvals))
        _, (hidden, _) = self.rnn(embeddings)

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.out(self.dropout(hidden).squeeze(0))

    def formatBatch(self, batch):

        samples_list = []
        results_list = []
        num_words = 0
        for i in range(len(batch)):
            sent = batch.text[{"batch": i}].values.data
            targets = batch.target.get("batch", i).values.data

            num_words += len(sent)
            padded_sent = torch.nn.functional.pad(sent, (self.n - 1, 0), value=0)
            samples = torch.stack([padded_sent[i:i + self.n - 1] for i in range(0, len(sent))])
            nextwords = sent

            samples_list.append(samples)
            results_list.append(nextwords)

        allsamples = torch.cat(samples_list)
        allresults = torch.cat(results_list)

        return (allsamples, allresults)

    def get_data(self, batchiter):
        x_batches = []
        y_batches = []
        total_words = 0
        for batch in batchiter:
            x, y, num_words = self.formatBatch(batch)
            x_batches.append(x)
            y_batches.append(y)
            total_words += num_words

        return x_batches, y_batches
