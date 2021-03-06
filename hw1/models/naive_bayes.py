from data_setup import torch, train_iter, val_iter, TEXT, bigram_map, make_bigrams


class NaiveBayes:
    def __init__(self, alpha=1, do_set_of_words=False, do_bigrams=False,
                 do_unigrams=True):
        self.do_unigrams = do_unigrams
        self.do_bigrams = do_bigrams
        self.do_set_of_words = do_set_of_words

        assert self.do_unigrams or self.do_bigrams, \
            'must enable at least one unigram and bigram'

        if self.do_bigrams:
            self.vocab_len = len(bigram_map)
        else:
            self.vocab_len = len(TEXT.vocab)

        p = alpha * torch.ones(self.vocab_len)
        q = alpha * torch.ones(self.vocab_len)

        pos_counts = 0
        neg_counts = 0

        for dataset in (train_iter, val_iter):
            for batch in dataset:
                sentences = batch.text.transpose('batch', 'seqlen').values.clone()
                labels = batch.label.values
                pos_counts += labels.sum()
                neg_counts += len(labels) - labels.sum()

                for sent, label in zip(sentences, labels):
                    words = self.transform(sent)
                    p += words * float(label == 1)
                    q += words * float(label == 0)

        # computing the weight vector feature vector
        self.r = (p / p.sum()).log() - (q / q.sum()).log()
        self.b = pos_counts.float().log() - neg_counts.float().log()

    def __call__(self, text):
        results = torch.zeros(text.shape['batch'])
        sentences = text.transpose('batch', 'seqlen').values.clone()
        for i, sent in enumerate(sentences):
            words = self.transform(sent)
            results[i] = (self.r.dot(words) + self.b).sigmoid()

        return results

    def transform(self, sent):
        if self.do_bigrams:
            words = make_bigrams(sent.tolist(), self.do_unigrams)

        words = torch.bincount(sent, minlength=self.vocab_len).float()
        if self.do_set_of_words:
            words = (words > 0).float()
        return words
