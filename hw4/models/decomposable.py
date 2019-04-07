import torch
from namedtensor import ntorch
from data_setup import WORD_VECS, embed_size


class ReLU(ntorch.nn.Module):
    def forward(self, x):
        return x.relu()


class LogSoftmax(ntorch.nn.Module):
    def __init__(self, dim):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.log_softmax(self.dim)


class FeedforwardNN(ntorch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2,
                 input_dim='embedding'):
        super().__init__()

        nn_layers = [
            ntorch.nn.Dropout(dropout),
            ntorch.nn.Linear(input_size, hidden_size).spec(input_dim, 'hidden'),
            ReLU(),
        ]
        for i in range(num_layers - 1):
            nn_layers.append(ntorch.nn.Dropout(dropout))
            nn_layers.append(ntorch.nn.Linear(hidden_size, hidden_size)
                                      .spec('hidden', 'hidden'))
            nn_layers.append(ReLU())
        self.nn = torch.nn.Sequential(*nn_layers)

    def forward(self, x):
        return self.nn(x)


class InputNN(ntorch.nn.Module):
    def __init__(self, num_layers, hidden_size, dropout=0.2, intra_attn=False):
        super().__init__()

        # note: noninclusive, 11 => below -10, -10 to 10, above 10
        self.d_cap = 11
        self.intra_attn = intra_attn

        self.embeddings = ntorch.nn.Embedding.from_pretrained(
            WORD_VECS.values.clone(), freeze=True
        )

        if self.intra_attn:
            self.bias = torch.nn.Parameter(torch.ones(2 * self.d_cap + 1))
            self.f_intra = FeedforwardNN(
                embed_size, hidden_size, num_layers, dropout=dropout
            )

    def forward(self, a, b):
        a = self.embeddings(a)
        b = self.embeddings(b)

        a_bar = self.self_align(a) if self.intra_attn else a
        b_bar = self.self_align(b) if self.intra_attn else b

        a_bar = a_bar.rename('seqlen', 'aSeqlen')
        b_bar = b_bar.rename('seqlen', 'bSeqlen')
        return a_bar, b_bar

    def self_align(self, a):
        # generate a' from a
        fintra_a = self.f_intra(a)

        unnamed_fintra = fintra_a.values.transpose(0, 1)
        fmat = torch.bmm(unnamed_fintra, unnamed_fintra.transpose(1, 2))
        # fmat = batch x seqlen x seqlen

        batches = fmat.shape[0]
        seqlen = fmat.shape[1]
        index_mat = torch.tensor([
            [(i - j) for j in range(seqlen)]
            for i in range(seqlen)
        ])
        index_mat = torch.clamp(index_mat, -self.d_cap, self.d_cap)
        index_mat = index_mat.unsqueeze(0).expand(batches, seqlen, seqlen)

        dmat = self.bias[index_mat + self.d_cap]
        weights = torch.softmax(fmat + dmat, dim=2)
        aprime = torch.matmul(weights, a.values.transpose(0, 1))
        aprime = ntorch.tensor(aprime, ('batch', 'seqlen', 'embedding'))

        abar = ntorch.cat([a, aprime], 'embedding')
        return abar


class AttendNN(ntorch.nn.Module):
    def __init__(self, num_layers, hidden_size, dropout=0.2, intra_attn=False,
                 use_labels=False, num_labels=4):
        super().__init__()
        self.input = InputNN(num_layers, hidden_size, dropout=dropout,
                             intra_attn=intra_attn)

        embed_size = 300
        embed_size = embed_size if (not intra_attn) else (2 * embed_size)

        self.f = FeedforwardNN(embed_size, hidden_size, num_layers, dropout)
        self.g = FeedforwardNN(2 * embed_size, hidden_size, num_layers, dropout)
        self.h = FeedforwardNN(2 * hidden_size, hidden_size, num_layers,
                               dropout, input_dim='hidden')

        self.use_labels = use_labels
        if use_labels:
            self.y_combine = FeedforwardNN(
                hidden_size + 1, hidden_size, num_layers, dropout,
                input_dim='hidden'
            )
        self.final = torch.nn.Sequential(
            ntorch.nn.Linear(hidden_size, num_labels).spec('hidden', 'label'),
            LogSoftmax(dim='label')
        )

    def forward(self, a, b, y=None):
        a_bar, b_bar = self.input(a, b)

        # ATTEND
        F_a = self.f(a_bar)
        F_b = self.f(b_bar)

        e_mat = F_a.dot('hidden', F_b)
        alpha = e_mat.softmax(dim='aSeqlen').dot('aSeqlen', a_bar)
        beta = e_mat.softmax(dim='bSeqlen').dot('bSeqlen', b_bar)

        # COMPARE AND AGGREGATE
        v1 = self.g(ntorch.cat([a_bar, beta], 'embedding')).sum('aSeqlen')
        v2 = self.g(ntorch.cat([b_bar, alpha], 'embedding')).sum('bSeqlen')

        # NOTE: currently adds log softmax layer after linear, use nllloss
        out = self.h(ntorch.cat([v1, v2], 'hidden'))
        if self.use_labels:
            y = ntorch.tensor(y.values.unsqueeze(1), names=('batch', 'hidden'))
            out = self.y_combine(ntorch.cat([out, y.float()], 'hidden'))
        yhat = self.final(out)
        return yhat
