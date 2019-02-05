from data_setup import torch, TEXT, train_iter, val_iter
from utils import chunks


class CNN(torch.nn.Module):
    def __init__(self, num_filters=10, kernel_sizes=(3, 4, 5),
                 second_layer_size=50, dropout_rate=0.5, num_iter=300,
                 learning_rate=0.001, batch_size=100, log_freq=10):
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
                torch.nn.Tanh(),
                torch.nn.MaxPool1d(self.max_words - kernel_size + 1)
            )
            conv_blocks.append(block)

        self.conv_blocks = torch.nn.ModuleList(conv_blocks)

        # constructing the neural network model
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(num_filters * len(kernel_sizes), second_layer_size),
            torch.nn.Tanh(),
            torch.nn.Linear(second_layer_size, 1),
            torch.nn.Sigmoid()
        )

        # setting up data
        xtrain, ytrain = self.preprocess(train_iter)
        xval, yval = self.preprocess(val_iter)

        # training
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = torch.nn.BCELoss()

        best_params = {k: p.detach().clone() for k, p in self.named_parameters()}
        best_val_acc = 0

        for i in range(num_iter):
            total_loss = 0

            s1 = torch.utils.data.RandomSampler(torch.arange(len(xtrain)))
            s2 = torch.utils.data.BatchSampler(s1, batch_size=batch_size,
                                               drop_last=False)
            try:
                for idx in s2:
                    xbatch = xtrain[idx]
                    labels_batch = ytrain[idx]

                    opt.zero_grad()
                    probs = self.forward(xbatch, is_train=True).flatten()
                    loss = loss_fn(probs, labels_batch.float())
                    loss.backward()
                    total_loss += loss.detach()
                    opt.step()

                self.eval()
                train_acc, train_loss = self.evaluate(xtrain, ytrain)
                val_acc, val_loss = self.evaluate(xval, yval)
                new_best = False
                if val_acc > best_val_acc:
                    best_params = {k: p.detach().clone() for k, p in self.named_parameters()}
                    best_val_acc = val_acc
                    new_best = True
                self.train()

                if new_best or i == 0 or i == num_iter - 1 or (i + 1) % log_freq == 0:
                    print(f'Epoch {i + 1}\n{"=" * len("Epoch {}".format(i + 1))}')
                    print(f'Train Loss: {train_loss:.5f}\t Train Accuracy: {train_acc:.2f}%')
                    print(f'Val Loss: {val_loss:.5f}\t Val Accuracy: {val_acc:.2f}%\n')
            except KeyboardInterrupt:
                print(f'\nStopped training after {i + 1} epochs...')
                break

        self.eval()
        self.load_state_dict(best_params)

    def forward(self, batch, is_train=False):
        if not is_train:
            sentences = batch.transpose('batch', 'seqlen').values.clone()
            batch = self.transform(sentences)
            x = batch.transpose(1, 2)
        else:
            x = batch

        x = torch.cat([conv_block(x) for conv_block in self.conv_blocks], 2)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def preprocess(self, dataset):
        x_lst = []
        y_lst = []
        for batch in dataset:
            sentences = batch.text.transpose('batch', 'seqlen').values.clone()
            x_lst.append(self.transform(sentences))
            y_lst.append(batch.label.values)
        return torch.cat(x_lst).transpose(1, 2), torch.cat(y_lst)

    def evaluate(self, x, y):
        loss_fn = torch.nn.BCELoss()
        loss = 0
        num_correct = 0
        for batch in chunks(torch.arange(len(x)), 128):
            probs = self.forward(x[batch], is_train=True).flatten()
            num_correct += int(((probs > 0.5).byte() == y[batch].byte()).sum())
            loss += loss_fn(probs, y[batch].float()) * len(batch)

        return 100.0 * num_correct / len(x), loss / len(x)

    def transform(self, sentences):
        pad_amt = (0, self.max_words - sentences.shape[1])
        sent_padded = torch.nn.functional.pad(sentences, pad_amt, value=1)
        return TEXT.vocab.vectors[sent_padded]
        # return sent_padded.unsqueeze(2).float()
