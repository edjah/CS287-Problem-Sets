import torch

class NaiveBayes:
  def __init__(self, alpha, train_iter):

    p = alpha * torch.ones(len(TEXT.vocab), device='cuda')
    q = p + 0

    pos_counts = 0
    neg_counts = 0

    for batch in train_iter:
      sentences = batch.text.transpose('batch', 'seqlen').values
      labels = batch.label.values

      pos_counts += labels.sum()
      neg_counts += len(labels) - labels.sum()

      for sent, label in zip(sentences, labels):
          bagofwords = (torch.bincount(sent, minlength=len(TEXT.vocab))).float()
          p += (bagofwords) * (label == 1)
          q += (bagofwords) * (label == 0)

    # obtaining feature vector
    p = p.float()
    q = q.float()

    # print("Counts: ", p, q, pos_counts, neg_counts)

    rtop = (p / torch.norm(p, 1))
    rbot = (q / torch.norm(q, 1))
    self.r = (rtop / rbot).log()

    # print("R: ", rtop, rbot, self.r)

    self.b = (pos_counts.float() / neg_counts).log()

    # print("b: ", self.b)

  def __call__(self, text):

    total_correct = 0
    total_wrong = 0

    results = torch.zeros(text.shape['batch'])

    sentences = text.transpose('batch', 'seqlen').values

    for i, sent in enumerate(sentences):
        bagofwords = (torch.bincount(sent, minlength=len(TEXT.vocab))).float()
        y = (self.r.dot(bagofwords.float()) + self.b).sign().long()
        results[i] = y > 0

    return results
