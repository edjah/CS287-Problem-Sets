from data_setup import torch, TEXT, train_iter

class CNN(torch.nn.Module):
    def __init__(self, num_filters=10, kernel_sizes=(3, 4, 5),
                 second_layer_size=50, dropout_rate=0.5):
        super().__init__()

        self.embed_len = 300
        self.max_words = 60

        conv_blocks = []
        for kernel_size in kernel_sizes:
            # verify correct parameters
            if kernel_size > self.max_words:
                raise Exception("window_num must be no greater than max_words")

            conv_layer = torch.nn.Conv1d(self.embed_len, num_filters,
                                         kernel_size=kernel_size, stride=1)
            block = torch.nn.Sequential(
                conv_layer,
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(self.max_words - kernel_size + 1)
            )
            conv_blocks.append(block)

        self.conv_blocks = torch.nn.ModuleList(conv_blocks)

        # constructing the neural network model
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(num_filters * len(kernel_sizes), second_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(second_layer_size, 1),
            torch.nn.Sigmoid()
        )

    def __call__(self, text):
        sentences = text.transpose('batch', 'seqlen').values.clone()
        pad_amt = (0, self.max_words - sentences.shape[1])
        sent_padded = torch.nn.functional.pad(sentences, pad_amt, value=1)
        batch_embeds = TEXT.vocab.vectors[sent_padded]

        x = batch_embeds.transpose(1, 2)

        return self.forward(x)

    def forward(self, batch):

        x = torch.cat([conv_block(batch) for conv_block in self.conv_blocks], 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)
