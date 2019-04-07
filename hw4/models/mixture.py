import torch
from namedtensor import ntorch
from .decomposable import AttendNN
from collections import defaultdict

ds = ntorch.distributions


class SimpleMixture(ntorch.nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = ntorch.nn.ModuleList(models)

    def forward(self, premise, hypothesis):
        preds = [model(premise, hypothesis) for model in self.models]
        return sum(preds) / len(self.models)


class VariationalAutoencoder(ntorch.nn.Module):
    def __init__(self, q_model, *models, sample_size=3, kl_importance=0.5,
                 elbo_type='exact'):
        super().__init__()
        self.q = q_model
        self.sample_size = sample_size
        self.models = ntorch.nn.ModuleList(models)
        self.K = len(self.models)
        self.kl_importance = kl_importance
        self.elbo_type = elbo_type

    def elbo_reinforce(self, premise, hypothesis, label):
        # computing the q distribution: p(c | a, b, y)
        q = self.q(premise, hypothesis, label).rename('label', 'latent')
        latent_dist = ds.Categorical(logits=q, dim_logit='latent')

        # generating some samples
        samples = latent_dist.sample([self.sample_size], names=('samples',))

        # bucketing samples by the sampled model to maximize efficiency
        buckets = defaultdict(list)
        premise_lst = premise.unbind('batch')
        hypothesis_lst = hypothesis.unbind('batch')

        samples_list = samples.transpose('batch', 'samples').tolist()
        for i, batch in enumerate(samples_list):
            p, h = premise_lst[i], hypothesis_lst[i]
            for sample in batch:
                buckets[sample].append((i, p, h))

        # evaluating the sampled models efficiently using batching
        orig_batch_size = premise.shape['batch']
        counts = [0] * orig_batch_size
        res = [None] * (self.sample_size * orig_batch_size)

        correct = label.tolist()
        for c, items in buckets.items():
            # stacking data points into batches
            batch_premise = ntorch.stack([p for _, p, _ in items], 'batch')
            batch_hypothesis = ntorch.stack([h for _, _, h in items], 'batch')
            ids = [i for i, _, _ in items]

            # evaluating the model on that batch
            predictions = self.models[c](batch_premise, batch_hypothesis)

            # updating the result at the appropriate index
            for i, log_probs in zip(ids, predictions.unbind('batch')):
                res[self.sample_size * i + counts[i]] = log_probs.values[correct[i]]
                counts[i] += 1

        # reforming and averaging the results for each sample
        res = torch.stack(res, dim=0).reshape(orig_batch_size, self.sample_size)
        res = ntorch.tensor(res, names=('batch', 'sample'))

        # computing a surrogate objective for REINFORCE
        # https://pyro.ai/examples/svi_part_iii.html
        q_log_prob = latent_dist.log_prob(samples)
        surrogate_objective = (q_log_prob * res.detach() + res).mean('sample')

        # adding on the KL regularizing term
        ones = ntorch.ones(self.K, names='latent').log_softmax(dim='latent')
        uniform_dist = ds.Categorical(logits=ones, dim_logit='latent')
        kl = ds.kl_divergence(latent_dist, uniform_dist) * self.kl_importance

        # reporting the surrogate objective as well as the actual elbo
        loss = -(surrogate_objective - kl).mean()
        elbo = -(res.detach().mean('sample') - kl.detach()).mean()
        return loss, elbo

    def elbo_exact(self, premise, hypothesis, label):
        # computing the q distribution: p(c | a, b, y)
        q = self.q(premise, hypothesis, label).rename('label', 'latent')
        latent_dist = ds.Categorical(logits=q, dim_logit='latent')

        one_hot_label = torch.eye(4).index_select(0, label.values)
        one_hot_label = ntorch.tensor(one_hot_label, names=('batch', 'label'))

        # computing p(y | a, b, c) for every c
        objective = 0
        q = q.exp()
        for c in range(len(self.models)):
            log_probs = self.models[c](premise, hypothesis)
            model_probs = q.get('latent', c)
            objective += (log_probs * one_hot_label).sum('label') * model_probs

        # adding on the KL regularizing term
        ones = ntorch.ones(self.K, names='latent').log_softmax(dim='latent')
        uniform_dist = ds.Categorical(logits=ones, dim_logit='latent')

        kl = ds.kl_divergence(latent_dist, uniform_dist) * self.kl_importance
        loss = -(objective.mean() - kl.mean())
        return loss, loss.detach()

    def infer(self, premise, hypothesis):
        label = ntorch.ones(premise.shape['batch'], names=('batch',)).long()
        predictions = 0
        for i in range(1, 4):
            q = self.q(premise, hypothesis, label * i).rename('label', 'latent').exp()
            for c in range(len(self.models)):
                log_probs = self.models[c](premise, hypothesis)
                predictions += log_probs * q.get('latent', c) / len(self.models)
        return predictions / 3

    def get_loss(self, premise, hypothesis, label):
        if self.elbo_type == 'exact':
            return self.elbo_exact(premise, hypothesis, label)
        return self.elbo_reinforce(premise, hypothesis, label)

    def forward(self, premise, hypothesis, label=None):
        return self.infer(premise, hypothesis)
