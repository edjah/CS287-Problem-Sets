from data_setup import torch, ntorch, train_iter, TEXT


class CbowNN:
    def __init__(self, num_iter=300, learning_rate=0.01, second_layer_size=50,
                 batch_size=100, log_freq=10):

        # constructing the neural network model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(300, second_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(second_layer_size, 1),
            torch.nn.Sigmoid()
        )

        # setting up data
        x_lst = []
        y_lst = []
        for batch in train_iter:
            sentences = batch.text.transpose('batch', 'seqlen').values.clone()
            labels = batch.label.values
            xbatch = ntorch.tensor(TEXT.vocab.vectors[sentences].sum(dim=1),
                                   names=('sentence', 'embed'))
            x_lst.append(xbatch)
            y_lst.append(labels)

        xtrain = ntorch.cat(x_lst, dim='sentence')
        ytrain = torch.cat(y_lst)

        # training
        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.BCELoss()

        for i in range(num_iter):
            total_loss = 0
            num_correct = 0

            s1 = torch.utils.data.RandomSampler(torch.arange(len(xtrain)))
            s2 = torch.utils.data.BatchSampler(s1, batch_size=batch_size,
                                               drop_last=False)
            try:
                for idx in s2:
                    xbatch = xtrain.values[idx]
                    labels_batch = ytrain[idx]

                    opt.zero_grad()
                    probs = self.model(xbatch).squeeze(1)
                    preds = probs.detach() > 0.5
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

    def __call__(self, text):
        sentences = text.transpose('batch', 'seqlen').values.clone()
        xbatch = TEXT.vocab.vectors[sentences].sum(dim=1)
        y = self.model(xbatch)
        return y.squeeze(1)
