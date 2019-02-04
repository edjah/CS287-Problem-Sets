from data_setup import torch, train_iter


class RNN(torch.nn.Module):
    def __init__(self, hidden_size=100, model_class=torch.nn.RNN, inp_size=10,
                 num_layers=3, num_iter=300, learning_rate=0.001,
                 batch_size=100, log_freq=10):
        super().__init__()

        # setting up the model
        self.max_words = 60
        self.inp = torch.nn.Linear(self.max_words, inp_size)
        self.rnn = model_class(
            inp_size, hidden_size, batch_first=True, num_layers=num_layers
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )

        # setting up data
        x_lst = []
        y_lst = []
        for batch in train_iter:
            sentences = batch.text.transpose('batch', 'seqlen').values.clone()
            pad_amt = (0, self.max_words - sentences.shape[1])
            x_lst.append(torch.nn.functional.pad(sentences, pad_amt, value=1))
            y_lst.append(batch.label.values)

        xtrain = torch.cat(x_lst).float().unsqueeze(1)
        ytrain = torch.cat(y_lst).byte()

        # training
        self.train()
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

                    opt.zero_grad()
                    probs = self.forward(xbatch, is_train=True).flatten()
                    preds = probs.detach() > 0.5
                    num_correct += int((preds == labels_batch).sum())

                    loss = loss_fn(probs, labels_batch.float())
                    loss.backward()
                    total_loss += loss.detach()
                    opt.step()
            except KeyboardInterrupt:
                print(f'\nStopped training after {i + 1} epochs...')
                break

            if i == 0 or i == num_iter - 1 or (i + 1) % log_freq == 0:
                accuracy = 100.0 * num_correct / len(xtrain)
                print(f'Loss at epoch {i + 1}: {total_loss}')
                print(f'Accuracy at epoch {i + 1}: {accuracy:.2f}%\n')

        self.eval()

    def forward(self, batch, hidden=None, is_train=False):
        if not is_train:
            sentences = batch.transpose('batch', 'seqlen').values.clone()
            pad_amt = (0, self.max_words - sentences.shape[1])
            x = torch.nn.functional.pad(sentences, pad_amt, value=1)
            x = x.unsqueeze(1).float()
        else:
            x = batch

        l1 = self.inp(x)
        l2, hidden = self.rnn(l1, hidden)
        l3 = self.out(l2)

        # if isinstance(hidden, tuple):
        #     hidden = tuple(x.detach() for x in hidden)
        # else:
        #     hidden = hidden.detach()

        return l3
