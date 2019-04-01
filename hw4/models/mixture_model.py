from namedtensor import ntorch


class MixtureModel(ntorch.nn.Module):
    def __init__(self, K, model_cls, *model_args, **model_kwargs):
        super().__init__()
        self.K = K
        self.models = [model_cls(*model_args, **model_kwargs) for _ in range(K)]

    def p(self, c):
        return 1 / self.K

    def enumerate(self, premise, hypothesis):
        preds = [
            self.p(c) * model(premise, hypothesis)
            for c, model in enumerate(self.models)
        ]
        return sum(preds)

    def forward(self, premise, hypothesis):
        return self.enumerate(premise, hypothesis)
