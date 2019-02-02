class LogisticRegression:
  def __init__(self, vocab_len, num_iter=100, learning_rate=0.3, reg_param=0.0):

    self.vocab_len = vocab_len

    self.weight = ntorch.randn(self.vocab_len + 1, requires_grad=True, names=('vocab',))

    x_lst = []
    y_lst = []

    for batch in train_iter:
      sentences = batch.text.transpose('batch', 'seqlen').values
      labels = batch.label.values

      # OPT 1: BOW.
      for sent in sentences:
        bagofwords = torch.cat((torch.bincount(sent, minlength=len(TEXT.vocab)).float().cpu(), torch.ones(1,)))
        x_lst.append(bagofwords)

      # OPT 2: wordtypes (set of words)
#       x_batch = torch.zeros(len(batch), self.vocab_len + 1)
#       for i, x in enumerate(sentences):
#         x_batch[i][x.unique()] = 1
#         x_batch[i][self.vocab_len] = 1

      # x_lst.append(x_batch)
      y_lst.append(labels)

    print(x_lst[0])

    x_train = ntorch.tensor(torch.stack(x_lst), names=('sentence', 'vocab'))
    y_train = ntorch.tensor(torch.cat(y_lst).cpu(), names=('sentence',))

    # print(x_batch.shape, labels.shape, x_train.shape, y_train.shape)
    print(x_train.get('sentence', 0))

    # forward operations

    opt = torch.optim.SGD([self.weight.values], lr=learning_rate)

    for i in range(num_iter):
      opt.zero_grad
      y = self.weight.dot('vocab', x_train)
      prod = (2 * y_train.values.float() - 1.0) * y.values.float()
      loss = -prod.sigmoid().log().mean() + self.weight.values.norm() * reg_param
      loss.backward()
      opt.step()

      if (i == 0 or (i + 1) % 50 == 0):
        print(i, loss)

  def __call__(self, text):

    results = torch.zeros(text.shape['batch'])

    sentences = text.transpose('batch', 'seqlen').values

    for i, sent in enumerate(sentences):
        # OPT 2: set of words
#         setofwords = torch.zeros(self.vocab_len + 1)
#         setofwords[sent.unique()] = 1
#         setofwords[self.vocab_len] = 1

        # OPT 1: bag of words
        bagofwords = torch.cat((torch.bincount(sent, minlength=len(TEXT.vocab)).float().cpu(), torch.ones(1,)))

        named_bagofwords = ntorch.tensor(bagofwords, names=('vocab',))
        prob = self.weight.dot('vocab', named_bagofwords).sigmoid()
        results[i] = prob.values

    return results

    pass
