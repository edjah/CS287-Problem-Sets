import torch


class NaiveBayes:
    def __init__(self, alpha, train_iter, vocab_len):
        self.vocab_len = vocab_len

        p = alpha * torch.ones(self.vocab_len, device='cuda')
        q = alpha * torch.ones(self.vocab_len, device='cuda')

        pos_counts = 0
        neg_counts = 0

        for batch in train_iter:
            sentences = batch.text.transpose('batch', 'seqlen').values
            labels = batch.label.values

            pos_counts += labels.sum()
            neg_counts += len(labels) - labels.sum()

            for sent, label in zip(sentences, labels):
                bagofwords = torch.bincount(sent, minlength=self.vocab_len).float()
                p += bagofwords * (label == 1)
                q += bagofwords * (label == 0)

        # obtaining feature vector
        p, q = p.float(), q.float()
        self.r = (p / torch.norm(p, 1)).log() - (q / torch.norm(q, 1)).log()
        self.b = (pos_counts.float() / neg_counts).log()

    def __call__(self, text):
        results = torch.zeros(text.shape['batch'])
        sentences = text.transpose('batch', 'seqlen').values
        for i, sent in enumerate(sentences):
            bagofwords = torch.bincount(sent, minlength=self.vocab_len).float()
            y = self.r.dot(bagofwords.float()) + self.b
            results[i] = (y.sign().long() > 0)

        return results
