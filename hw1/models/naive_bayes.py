from data_setup import torch, train_iter, TEXT


class NaiveBayes:
    def __init__(self, alpha=1, do_set_of_words=False):
        self.vocab_len = len(TEXT.vocab)
        self.do_set_of_words = do_set_of_words

        p = alpha * torch.ones(self.vocab_len)
        q = alpha * torch.ones(self.vocab_len)

        pos_counts = 0
        neg_counts = 0

        for batch in train_iter:
            sentences = batch.text.transpose('batch', 'seqlen').values
            labels = batch.label.values
            pos_counts += labels.sum()
            neg_counts += len(labels) - labels.sum()

            for sent, label in zip(sentences, labels):
                words = torch.bincount(sent, minlength=self.vocab_len).float()
                if self.do_set_of_words:
                    words = (words > 0).float()
                p += words * float(label == 1)
                q += words * float(label == 0)

        # computing the weight vector feature vector
        self.r = (p / p.sum()).log() - (q / q.sum()).log()
        self.b = pos_counts.float().log() - neg_counts.float().log()

    def __call__(self, text):
        results = torch.zeros(text.shape['batch'])
        sentences = text.transpose('batch', 'seqlen').values
        for i, sent in enumerate(sentences):
            words = torch.bincount(sent, minlength=self.vocab_len).float()
            if self.do_set_of_words:
                words = (words > 0).float()
            y = self.r.dot(words) + self.b
            results[i] = (y.sign().long() > 0)

        return results
