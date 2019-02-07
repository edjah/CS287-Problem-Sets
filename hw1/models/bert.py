from data_setup import torch, TEXT, train_iter, val_iter, test_iter
from pytorch_pretrained_bert import (
    BertForSequenceClassification, BertTokenizer, BertAdam
)
from collections import defaultdict
from utils import chunks
import torch.nn.functional as F

UNKNOWN = 100


class BertModel:
    def __init__(self, num_iter=300, learning_rate=0.001,
                 weight_decay=0, batch_size=100, log_freq=10):
        self.max_words = 60
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.vocab = defaultdict(lambda: UNKNOWN, self.tokenizer.vocab)

        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=2, cache_dir='.data/bert_cache'
        )

        # training
        self.train_model(num_iter, learning_rate,
                         weight_decay, batch_size, log_freq)

    def __call__(self, text):
        with torch.no_grad():
            x = self.transform(text)
            input_ids, input_mask = x[:, 0, :], x[:, 1, :]
            segment_ids = torch.zeros(input_ids.shape).long()

            raw_res = self.model(input_ids, segment_ids, input_mask).softmax(dim=1)

            probs = torch.zeros(raw_res.shape[0])
            for i, (a, b) in enumerate(raw_res.tolist()):
                if a > b:
                    probs[i] = a
                else:
                    probs[i] = 1 - b
            # print(probs)
            return probs

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
        if isinstance(text, torch.Tensor) and text.shape[1] == 2 and text.shape[2] == self.max_words:
            return text

        sentences = text.transpose('batch', 'seqlen').values.clone()
        pad_amt = (0, self.max_words - sentences.shape[1])
        sentences = F.pad(sentences, pad_amt, value=1).tolist()

        results = []
        masks = []
        for sent in sentences:
            input_mask = torch.ones(len(sent)).long()
            for i in reversed(range(len(sent))):
                if sent[i] > 1:
                    break
                input_mask[i] = 0

            tokens = [TEXT.vocab.itos[w] for w in sent]
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            results.append(torch.tensor(ids))
            masks.append(input_mask)

        return torch.stack((torch.stack(results), torch.stack(masks)), dim=1)

    def evaluate(self, x, y):
        with torch.no_grad():
            loss_fn = torch.nn.BCELoss()
            loss = 0
            num_correct = 0
            for batch in chunks(torch.arange(len(x)), 128):
                probs = self.__call__(x[batch])
                num_correct += int(((probs > 0.5).byte() == y[batch].byte()).sum())
                loss += loss_fn(probs, y[batch].float()) * len(batch)

            return 100.0 * num_correct / len(x), loss / len(x)

    def train_model(self, num_iter=300, learning_rate=0.001,
                    weight_decay=0, batch_size=100, log_freq=10):

        model = self.model

        # getting properly formatted training/validation data
        xtrain, ytrain = self.get_data((train_iter, val_iter))
        xval, yval = self.get_data((test_iter,))

        # training
        model.train()

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        opt = BertAdam(optimizer_grouped_parameters,
                       lr=learning_rate,
                       warmup=0.1,
                       t_total=10)

        best_params = {k: p.detach().clone() for k, p in model.named_parameters()}
        best_val_acc = 0

        for i in range(num_iter):
            # iterate through the data in a random order each time
            s1 = torch.utils.data.RandomSampler(torch.arange(len(xtrain)))
            s2 = torch.utils.data.BatchSampler(s1, batch_size=batch_size,
                                               drop_last=False)
            try:
                for idx in s2:
                    opt.zero_grad()

                    # computing the loss
                    xbatch, input_mask = xtrain[idx, 0, :], xtrain[idx, 1, :]
                    segment_ids = torch.zeros(xbatch.shape).long()
                    labels_batch = ytrain[idx]

                    loss = model(xbatch, segment_ids, input_mask, labels_batch)

                    # computing gradients and updating the weights
                    loss.backward()
                    opt.step()

                    # evaluating performance on the validation set
                    model.eval()
                    train_acc, train_loss = self.evaluate(xtrain, ytrain)
                    val_acc, val_loss = self.evaluate(xval, yval)
                    new_best = False
                    if val_acc > best_val_acc:
                        best_params = {
                            k: p.detach().clone() for k, p in model.named_parameters()
                        }
                        best_val_acc = val_acc
                        new_best = True
                    model.train()

                # logging
                if new_best or i == 0 or i == num_iter - 1 or (i + 1) % log_freq == 0:
                    print(f'Epoch {i + 1}\n{"=" * len("Epoch {}".format(i + 1))}')
                    print(f'Train Loss: {train_loss:.5f}\t Train Accuracy: {train_acc:.2f}%')
                    print(f'Val Loss: {val_loss:.5f}\t Val Accuracy: {val_acc:.2f}%\n')
            except KeyboardInterrupt:
                print(f'\nStopped training after {i} epochs...')
                break

        model.load_state_dict(best_params)
        model.eval()
