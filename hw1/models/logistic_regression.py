from data_setup import torch, TEXT, bigram_map, make_bigrams


WORD_VECS = TEXT.vocab.vectors


class LogisticRegression(torch.nn.Module):
    def __init__(self, do_embeddings=False, do_bigrams=False, do_set_of_words=False):
        super().__init__()

        self.do_embeddings = do_embeddings
        self.do_bigrams = do_bigrams
        self.do_set_of_words = do_set_of_words

        self.vocab_len = len(bigram_map) if self.do_bigrams else len(TEXT.vocab)
        if self.do_embeddings:
            self.vocab_len = WORD_VECS.shape[1]

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.vocab_len, 1),
            torch.nn.Sigmoid()
        )

    def __call__(self, text):
        sentences = text.transpose('batch', 'seqlen').values.clone()
        vals = [self.transform(sent) for sent in sentences]
        return self.model(torch.stack(vals))

    def forward(self, text):
        return self.model(text)

    def get_data(self, datasets):
        x_lst = []
        y_lst = []

        for dataset in datasets:
            for batch in dataset:
                sentences = batch.text.transpose('batch', 'seqlen').values.clone()
                y_lst.append(batch.label.values.float())
                for sent in sentences:
                    x_lst.append(self.transform(sent))

        X = torch.stack(x_lst)
        Y = torch.cat(y_lst)

        return (X, Y)

    def transform(self, sent):
        if self.do_embeddings:
            return WORD_VECS[sent].sum(dim=0)

        # regular bag of words / set of words
        if self.do_bigrams:
            sent = make_bigrams(sent)

        words = torch.bincount(sent, minlength=self.vocab_len)
        if self.do_set_of_words:
            words = (words > 0)
        return words.float()
