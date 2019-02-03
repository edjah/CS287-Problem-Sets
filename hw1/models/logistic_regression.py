from data_setup import torch, ntorch, train_iter, TEXT


class LogisticRegression:
    def __init__(self, num_iter=100, learning_rate=0.3, reg_param=0.0,
                 do_set_of_words=True):

        self.vocab_len = len(TEXT.vocab)
        self.do_set_of_words = do_set_of_words
        self.weight = ntorch.randn(self.vocab_len + 1, requires_grad=True,
                                   names=('vocab',))

        # setting up
        x_lst = []
        y_lst = []

        for batch in train_iter:
            sentences = batch.text.transpose('batch', 'seqlen').values.clone()
            y_lst.append(batch.label.values.float())
            for sent in sentences:
                x_lst.append(self.transform_x(sent))

        x_train = ntorch.stack(x_lst, 'sentence')
        y_train = ntorch.tensor(torch.cat(y_lst), names=('sentence',))

        # training
        opt = torch.optim.Adam([self.weight.values], lr=learning_rate)

        for i in range(num_iter):
            try:
                opt.zero_grad()
                y = self.weight.dot('vocab', x_train)
                prod = (2 * y_train.values - 1) * y.values.float()
                loss = -prod.sigmoid().max(torch.tensor(1e-323)).log().mean()
                loss += self.weight.values.norm() * reg_param
                loss.backward()
                opt.step()

                if i == 0 or i == num_iter - 1 or (i + 1) % 100 == 0:
                    print(f'Loss at epoch {i}: {loss}')
            except KeyboardInterrupt:
                print(f'\nStopped training after {i} epochs...')
                break
        print()

    def __call__(self, text):
        results = torch.zeros(text.shape['batch'])
        sentences = text.transpose('batch', 'seqlen').values.clone()
        for i, sent in enumerate(sentences):
            named_x = self.transform_x(sent)
            prob = self.weight.dot('vocab', named_x).sigmoid()
            results[i] = prob.values

        return results

    def transform_x(self, sent):
        # regular bag of words / set of words
        words = torch.bincount(sent, minlength=self.vocab_len + 1)
        words[self.vocab_len] = 1
        if self.do_set_of_words:
            words = (words > 0)
        return ntorch.tensor(words.float(), names=('vocab',))
