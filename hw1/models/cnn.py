import torch.nn.functional as F
from data_setup import torch, TEXT

WORD_VECS = TEXT.vocab.vectors


class CNN(torch.nn.Module):
    def __init__(self, num_filters=10, kernel_sizes=(3, 4, 5),
                 second_layer_size=50, dropout_rate=0.5):
        super().__init__()

        self.max_words = 60

        self.embedding_dynamic = torch.nn.Embedding.from_pretrained(WORD_VECS.clone(), freeze=False)
        self.embedding_static = torch.nn.Embedding.from_pretrained(WORD_VECS.clone(), freeze=True)

        conv_blocks = []
        for kernel_size in kernel_sizes:
            # verify correct parameters
            if kernel_size > self.max_words:
                raise Exception("window_num must be no greater than max_words")

            conv_blocks.append(torch.nn.Conv2d(
                1, num_filters, (kernel_size, WORD_VECS.shape[1])
            ))

        self.convs = torch.nn.ModuleList(conv_blocks)

        # constructing the neural network model
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(num_filters * len(kernel_sizes), second_layer_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(second_layer_size, 1),
            torch.nn.Sigmoid()
        )

    def __call__(self, text):
        return self.forward(self.transform(text))

    def forward(self, batch):
        e1 = self.embedding_static(batch).unsqueeze(1)
        e2 = self.embedding_dynamic(batch).unsqueeze(1)

        conved = [torch.cat((conv(e1), conv(e2)), dim=2) for conv in self.convs]
        relud = [F.relu(c).squeeze(3) for c in conved]
        pooled = [F.max_pool1d(r, r.shape[2]).squeeze(2) for r in relud]

        # conved = [F.relu(conv(embeddings)).squeeze(3) for conv in self.convs]
        # pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        return self.fc(torch.cat(pooled, dim=1))

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
        return F.pad(sentences, pad_amt, value=1)
