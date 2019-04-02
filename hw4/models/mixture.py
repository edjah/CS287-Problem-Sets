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
    def __init__(self, *models):
        super().__init__()
        self.models = ntorch.nn.ModuleList(models)
        self.weights = torch.nn.Parameter(torch.rand(len(self.models)))

    def forward(self, premise, hypothesis):
        softmax = self.weights.softmax()
        preds = [
            softmax[c] * model(premise, hypothesis)
            for c, model in enumerate(self.models)
        ]
        return sum(preds)
