from data_setup import torch, TEXT, bigram_map, make_bigrams


class LogisticRegression(torch.nn.Module):
    def __init__(self, do_bigrams=False, do_set_of_words=False):
        super().__init__()

        self.do_bigrams = do_bigrams
        self.do_set_of_words = do_set_of_words

        self.vocab_len = len(bigram_map) if self.do_bigrams else len(TEXT.vocab)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.vocab_len + 1, 1),
            torch.nn.Sigmoid()
        )

    def __call__(self, text):
        sentences = text.transpose('batch', 'seqlen').values.clone()
        vals = [self.transform(sent) for sent in sentences]
        return self.model(torch.stack(vals))

    def forward(self, text):
        return self.model(text)

    def get_data(self, dataset):
        x_lst = []
        y_lst = []

        for batch in dataset:
            sentences = batch.text.transpose('batch', 'seqlen').values.clone()
            y_lst.append(batch.label.values.float())
            for sent in sentences:
                x_lst.append(self.transform(sent))

        X = torch.stack(x_lst)
        Y = torch.cat(y_lst)

        return (X, Y)

    def transform(self, sent):
        # regular bag of words / set of words
        if self.do_bigrams:
            sent = make_bigrams(sent)

        words = torch.bincount(sent, minlength=self.vocab_len + 1)
        words[self.vocab_len] = 1
        if self.do_set_of_words:
            words = (words > 0)
        return words.float()
