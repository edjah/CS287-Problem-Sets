class CNN(torch.nn.Module):
    def __init__(self,
                 num_filters=10, kernel_sizes=(3, 4, 5), second_layer_size=50, dropout_rate=0.5):
        super().__init__()

        self.embed_len = 300
        self.max_words = 60

        conv_blocks = []
        for kernel_size in kernel_sizes:
            # verify correct parameters
            if kernel_size > self.max_words:
                raise Exception("window_num must be no greater than max_words")

            block = torch.nn.Sequential(
                torch.nn.Conv1d(self.embed_len, num_filters,
                                kernel_size = kernel_size,
                                stride = 1),
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

    def forward(self, batch, isTrain=False):
        if not isTrain:
            sentences = batch.transpose('batch', 'seqlen').values.clone()
            sent_padded = torch.nn.functional.pad(sentences, (0, self.max_words - sentences.shape[1]), value=1)
            batch_embeds = TEXT.vocab.vectors[sent_padded]

            # print("Batch embed dims: ", batch_embeds.shape)
            x = batch_embeds.transpose(1, 2)

        else:
            x = batch

        x = torch.cat([conv_block(x) for conv_block in self.conv_blocks], 2)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def doTrain(self, num_iter=300, learning_rate=0.001,
                 batch_size=100, log_freq=10,):
        self.train()
        x_lst = []
        y_lst = []
        for batch in train_iter:
            sentences = batch.text.transpose('batch', 'seqlen').values.clone()
            labels = batch.label.values
            sent_padded = torch.nn.functional.pad(sentences, (0, self.max_words - sentences.shape[1]), value=1)
            x_lst.append(TEXT.vocab.vectors[sent_padded])
            y_lst.append(labels)

        xtrain = torch.cat(x_lst)
        xtrain = xtrain.transpose(1, 2)
        ytrain = torch.cat(y_lst)

        # print(xtrain.shape, ytrain.shape, xtrain[0])

        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = torch.nn.BCELoss()

        for i in range(num_iter):
            total_loss = 0
            num_correct = 0

            s1 = torch.utils.data.RandomSampler(torch.arange(len(xtrain)))
            s2 = torch.utils.data.BatchSampler(s1, batch_size=batch_size,
                                               drop_last=False)
            try:
                for idx in s2:
                    xbatch = xtrain[idx]
                    labels_batch = ytrain[idx]

                    # print(xbatch.device)
                    opt.zero_grad()
                    probs = self.forward(xbatch, isTrain=True).flatten()
                    preds = probs.detach() > 0.5
                    # print(probs.device, preds.device, labels_batch.device)
                    num_correct += int((preds == labels_batch.byte()).sum())


                    loss = loss_fn(probs, labels_batch.float())
                    loss.backward()
                    total_loss += loss.detach()
                    opt.step()
            except KeyboardInterrupt:
                print(f'\nStopped training after {i} epochs...')
                break

            if i == 0 or i == num_iter - 1 or (i + 1) % log_freq == 0:
                accuracy = 100.0 * num_correct / len(xtrain)
                print(f'Loss at epoch {i}: {total_loss}')
                print(f'Accuracy at epoch {i}: {accuracy:.2f}%')

        self.eval()
