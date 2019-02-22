from data_setup import torch, TEXT

WORD_VECS = TEXT.vocab.vectors

BATCHSIZE = 32


class NNLM_CNN(torch.nn.Module):
    def __init__(self, n, hidden_size=100, num_layers=3, dropout_rate=0.3, num_filters=50):
        super().__init__()

        self.n = n
        self.vocab_length = len(TEXT.vocab)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.embedding = torch.nn.Embedding.from_pretrained(WORD_VECS.clone(), freeze=False)
        self.conv = torch.nn.Conv1d(300, num_filters, n - 1, stride=1)

        # basic linear NN
        linear_blocks = [
            torch.nn.Linear(num_filters, hidden_size),
            torch.nn.Tanh(),
        ]
        for l in range(num_layers - 2):
            linear_blocks.append(torch.nn.Linear(hidden_size, hidden_size))
            linear_blocks.append(torch.nn.Tanh())

        linear_blocks += [
            torch.nn.Linear(hidden_size, self.vocab_length),
            torch.nn.Softmax(dim=-1)
        ]

        self.out = torch.nn.Sequential(*linear_blocks)

    def forward(self, xvals):
        xvals = xvals.transpose(0, 1)
        embeddings = self.dropout(self.embedding(xvals)).permute(1, 2, 0)
        convs = self.conv(embeddings).transpose(1, 2)

        out = self.out(convs).flatten(0, 1)
        return out

    def formatBatch(self, batch):
        samples_list = []
        results_list = []
        num_words = 0
        for i in range(len(batch)):
            sent = batch.text[{"batch": i}].values.data
            targets = batch.target.get("batch", i).values.data

            num_words += len(sent)
            padded_sent = torch.nn.functional.pad(sent, (self.n - 2, BATCHSIZE - len(sent)), value=0)
            nextwords = torch.nn.functional.pad(targets, (0, BATCHSIZE - len(targets)), value=0)

            samples_list.append(padded_sent)
            results_list.append(nextwords)

        allsamples = torch.stack(samples_list)
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
