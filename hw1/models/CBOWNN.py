class CBOWNN:
    def __init__(self, num_iter=300, learning_rate=0.01, secondlayer_size=50, batch_size=100):

        first_layer = torch.nn.Linear(300, secondlayer_size)
        self.model = torch.nn.Sequential(
                first_layer,
                torch.nn.ReLU(),
                torch.nn.Linear(secondlayer_size, 1),
                torch.nn.Sigmoid()
        )

        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # REPLACE THIS!
        self.embeddings = TEXT.vocab.vectors

        embed_len = self.embeddings.shape[1]
        loss_fn = torch.nn.BCELoss()

        x_lst = []
        y_lst = []
        for batch in train_iter:

            sentences = batch.text.transpose('batch', 'seqlen').values
            labels = batch.label.values

            xbatch = ntorch.tensor(self.embeddings[sentences].mean(dim=1).cuda(), names=('sentence', 'embed'))
            x_lst.append(xbatch)
            y_lst.append(labels)

        xtrain = ntorch.cat(x_lst, dim='sentence')
        ytrain = torch.cat(y_lst)

        for i in range(num_iter):
            total_loss = 0
            num_correct = 0

            s1 = torch.utils.data.RandomSampler(torch.arange(len(xtrain)))
            s2 = torch.utils.data.BatchSampler(s1, batch_size=batch_size, drop_last=False)
            try:
                for idx in s2:

                    xbatch = xtrain.values[idx]
                    labels_batch = ytrain[idx]

                    opt.zero_grad()
                    predictions = self.model(xbatch)
                    # print(predictions.shape, labels_batch.shape)
                    num_correct += ((predictions.detach().squeeze(1) > 0.5) == labels_batch.byte()).sum()

                    loss = loss_fn(predictions, labels_batch.float())
                    loss.backward()
                    total_loss += loss.detach()
                    opt.step()

            except KeyboardInterrupt:
                print(f'\nStopped training after {i} epochs...')
                break

            if i == 0 or i == num_iter - 1 or (i + 1) % 5 == 0:
                print(f'Loss at epoch {i}: {total_loss}')
                print(f'Accuracy at epoch {i}: { num_correct.float() / len(xtrain) }')

    def __call__(self, text):

        sentences = text.transpose('batch', 'seqlen').values

        xbatch = ntorch.tensor(self.embeddings[sentences].mean(dim=1).cuda(), names=('sentence', 'embed'))
        results = torch.zeros(text.shape['batch'])

        y = self.model(xbatch.values)
        # print(y.shape)
        return y.squeeze(1)
