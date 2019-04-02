import torch
from namedtensor import ntorch


class SimpleMixture(ntorch.nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = ntorch.nn.ModuleList(models)

    def p(self, c):
        return 1 / len(self.models)

    def enumerate(self, premise, hypothesis):
        preds = [
            self.p(c) * model(premise, hypothesis)
            for c, model in enumerate(self.models)
        ]
        return sum(preds)

    def forward(self, premise, hypothesis):
        return self.enumerate(premise, hypothesis)


class LearnedEnsemble(ntorch.nn.Module):
    def __init__(self, *models, fine_tune=False):
        super().__init__()

        self.fine_tune = fine_tune
        if fine_tune:
            self.models = ntorch.nn.ModuleList(models)
        else:
            self.models = [m.eval() for m in models]
            self.cache = {}
        self.weights = torch.nn.Parameter(torch.rand(len(self.models)))

    def f(self, model, premise, hypothesis):
        key = (model, premise.values, hypothesis.values)
        if key in self.cache:
            return self.cache[key]
        self.cache[key] = model(premise, hypothesis)
        return self.cache[key]

    def forward(self, premise, hypothesis):
        if self.fine_tune:
            preds = [model(premise, hypothesis) for model in self.models]
        else:
            with torch.no_grad():
                preds = [self.f(model, premise, hypothesis) for model in self.models]

        softmax = self.weights.softmax(dim=0)
        return sum(pred * softmax[c] for c, pred in enumerate(preds))
