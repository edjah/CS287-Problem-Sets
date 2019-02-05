from data_setup import torch, ntorch, train, val, TEXT, bert_tokenizer, bert_model
from utils import chunks
from tqdm import tqdm


class BertCbowNN:
    def __init__(self, num_iter=300, learning_rate=0.01, second_layer_size=50,
                 batch_size=100, log_freq=10, dropout_rate=0.1):

        # self.vocab_len = len(TEXT.vocab)
        self.vocab_len = 768

        # constructing the neural network model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.vocab_len, second_layer_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(second_layer_size, second_layer_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(second_layer_size, second_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(second_layer_size, 1),
            torch.nn.Sigmoid()
        )

        # setting up data
        xtrain, ytrain = self.preprocess(train)
        xval, yval = self.preprocess(val)

        # training
        opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.BCELoss()

        best_params = {k: p.detach().clone() for k, p in self.model.named_parameters()}
        best_val_acc = 0

        self.model.train()
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
                    probs = self.model(xbatch).squeeze(1)
                    preds = probs.detach() > 0.5
                    num_correct += int((preds == labels_batch.byte()).sum())

                    loss = loss_fn(probs, labels_batch.float())
                    loss.backward()
                    total_loss += loss.detach()
                    opt.step()

                self.model.eval()
                train_acc, train_loss = self.evaluate(xtrain, ytrain)
                val_acc, val_loss = self.evaluate(xval, yval)
                new_best = False
                if val_acc > best_val_acc:
                    best_params = {k: p.detach().clone() for k, p in self.model.named_parameters()}
                    best_val_acc = val_acc
                    new_best = True
                self.model.train()

                if new_best or i == 0 or i == num_iter - 1 or (i + 1) % log_freq == 0:
                    print(f'Epoch {i + 1}\n{"=" * len("Epoch {}".format(i + 1))}')
                    print(f'Train Loss: {train_loss:.5f}\t Train Accuracy: {train_acc:.2f}%')
                    print(f'Val Loss: {val_loss:.5f}\t Val Accuracy: {val_acc:.2f}%\n')

            except KeyboardInterrupt:
                print(f'\nStopped training after {i} epochs...')
                break

        self.model.eval()
        self.model.load_state_dict(best_params)

    def __call__(self, text):
        sentences = text.transpose('batch', 'seqlen').values.clone()
        # xbatch = TEXT.vocab.vectors[sentences].sum(dim=1)
        results = torch.zeros(text.shape['batch'])
        for i, sent in enumerate(sentences):
            results[i] = self.model(self.transform(sent)).item()

        return results

    def evaluate(self, x, y):
        loss_fn = torch.nn.BCELoss()
        loss = 0
        num_correct = 0
        for batch in chunks(torch.arange(len(x)), 100):
            probs = self.model(x[batch]).flatten()
            num_correct += int(((probs > 0.5).byte() == y[batch].byte()).sum())
            loss += loss_fn(probs, y[batch].float()) * len(batch)

        return 100.0 * num_correct / len(x), loss / len(x)

    def preprocess(self, dataset):
        x_lst = []
        y_lst = []
        for example in tqdm(dataset, total=len(dataset)):
            # xbatch = ntorch.tensor(TEXT.vocab.vectors[sentences].sum(dim=1),
            #                        names=('sentence', 'embed'))
            x_lst.append(self.transform(example.text))
            y_lst.append(int(example.label == 'positive'))

        x = torch.stack(x_lst)
        y = torch.tensor(y_lst)
        return x, y

    def transform(self, sent):
        sent = list(sent)
        if not isinstance(sent[0], str):
            # removing padding
            while sent[-1] <= 1:
                sent.pop()

            # converting back to text
            sent = [TEXT.vocab.itos[i] for i in sent]

        with torch.no_grad():
            indexed_tokens = bert_tokenizer.convert_tokens_to_ids(sent)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensor = torch.zeros(tokens_tensor.shape).long()
            _, embedding = bert_model(tokens_tensor, segments_tensor)
            return embedding.squeeze(0)

        # sent = torch.tensor([TEXT.vocab.stoi[s] for s in sent])
        # return TEXT.vocab.vectors[sent].sum(dim=0)
        # return torch.bincount(sent, minlength=self.vocab_len).float()
