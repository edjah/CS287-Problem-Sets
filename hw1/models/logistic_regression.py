from data_setup import torch, ntorch, train_iter, TEXT


class LogisticRegression(torch.nn.Module):
    def __init__(self, vocab_len=16284):
        super().__init__()

        # constructing the neural network model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(vocab_len + 1, 1),
            torch.nn.Sigmoid()
        )

    def __call__(self, text):
        results = torch.zeros(text.shape['batch'])
        sentences = text.transpose('batch', 'seqlen').values.clone()
        testx_lst = []
        for i, sent in enumerate(sentences):
            testx_lst.append(transform_x_lr(sent).values)
#             prob = self.weight.dot('vocab', named_x).sigmoid()
#             results[i] = prob.values
        return self.model(torch.stack(testx_lst))

    def forward(self, text):
        return self.model(text)
