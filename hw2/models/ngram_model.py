import torch
from collections import defaultdict

VOCAB_LEN = 10001


class NGramModel:
    def __init__(self, train, max_ngram, min_count=1):
        self.max_ngram = max_ngram
        self._prior = None

        counts = {
            n: defaultdict(lambda: defaultdict(int))
            for n in range(self.max_ngram)
        }

        # making sure that the training data is left padded appropriately
        train = (0,) * (self.max_ngram - 1) + tuple(train)

        # counting
        for n in range(self.max_ngram):
            for i in range(len(train) - n):
                prefix = train[i:i+n]
                target = train[i + n]
                counts[n][prefix][target] += 1

        # converting counts to probabilities
        self.p = {
            n: defaultdict(lambda: defaultdict(lambda: 0))
            for n in range(self.max_ngram)
        }
        for n in range(self.max_ngram):
            for prefix, target_counts in counts[n].items():
                total = sum(c for c in target_counts.values() if c >= min_count)
                for target, count in target_counts.items():
                    if count >= min_count:
                        self.p[n][prefix][target] = count / total

        # verifying that the probabilities sum to 1
        for n in range(self.max_ngram):
            for prefix, target_probs in self.p[n].items():
                tot = sum(target_probs.values())
                assert abs(tot - 1) < 0.00001

    def get_raw_probs(self, sentence):
        sentence = (0,) * (self.max_ngram - 1) + tuple(sentence)

        probs = []
        for i in range(self.max_ngram - 1, len(sentence)):
            total_prob = [0] * self.max_ngram
            for n in range(self.max_ngram):
                prefix = sentence[i - n:i]
                target = sentence[i]
                total_prob[n] = self.p[n][prefix][target]
            if total_prob == [0] * self.max_ngram:
                total_prob = [1e-10] * self.max_ngram
            probs.append(total_prob)
        return torch.tensor(probs)

    def train_interpolation(self, train, val, num_epochs=1000, quiet=False):
        train_probs = self.get_raw_probs(train)
        val_probs = self.get_raw_probs(val)

        unnormed_weights = torch.randn(self.max_ngram, requires_grad=True)
        opt = torch.optim.Adam([unnormed_weights], lr=0.01)

        best_val_params = None
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            opt.zero_grad()
            normed_weights = unnormed_weights.softmax(dim=0)

            train_loss = -(train_probs @ normed_weights).log().mean()
            val_loss = -(val_probs @ normed_weights.detach()).log().mean()
            train_loss.backward()
            opt.step()

            train_loss = train_loss.detach()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_params = normed_weights.detach()

            if not quiet and (epoch % 200 == 0 or epoch == num_epochs - 1):
                print(f'Epoch: {epoch + 1} | '
                      f'Train Perplexity: {train_loss.exp():.3f} | '
                      f'Val Perplexity: {val_loss.exp():.3f}')

        self.interp_weights = best_val_params.tolist()
        print('Interpolation weights:', self.interp_weights)

    def prior(self):
        if self._prior is not None:
            return self._prior.copy()

        probabilities = [0] * VOCAB_LEN
        weight = self.interp_weights[0]
        for target, value in self.p[0][()].items():
            probabilities[target] = float(weight * value)

        self._prior = probabilities
        return probabilities.copy()

    def predict_last_word(self, x):
        # making sure that x is appropriately left padded and is a tuple
        x = ((0,) * (self.max_ngram - 1) + tuple(x))
        x = x[-self.max_ngram + 1:]

        # computing the probability of each word
        probabilities = self.prior()
        for n in range(1, self.max_ngram):
            prefix = x[self.max_ngram - n - 1:]
            weight = self.interp_weights[n]
            for target, value in self.p[n][prefix].items():
                probabilities[target] += weight * value

        return torch.tensor(probabilities)

    def perplexity(self, x):
        # making sure that x is appropriately left padded
        x = (0,) * (self.max_ngram - 1) + tuple(x)
        w = torch.tensor(self.interp_weights)
        nll = -(self.get_raw_probs(x) @ w).log().mean()
        return nll.exp()
