from data_setup import torch, TEXT

WORD_VEC = TEXT.vocab.vectors


class Ensemble(torch.nn.Module):
    def __init__(self, *pretrained_models, second_layer_size=10,
                 include_cbow=False, include_bag_of_words=False):
        super().__init__()

        self.include_cbow = include_cbow
        self.include_bag_of_words = include_bag_of_words
        self.pretrained_models = pretrained_models

        self.inp_size = len(self.pretrained_models)
        if self.include_cbow:
            self.inp_size += WORD_VEC.shape[1]
        if self.include_bag_of_words:
            self.inp_size += len(TEXT.vocab)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.inp_size, second_layer_size),
            torch.nn.Tanh(),
            torch.nn.Linear(second_layer_size, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or len(x.shape) != 2 or x.shape[1] != self.inp_size:
            x = self.transform(x)
        y = self.fc(x)
        return y

    def get_data(self, datasets):
        x_lst = []
        y_lst = []

        for dataset in datasets:
            for batch in dataset:
                x_lst.append(self.transform(batch.text))
                y_lst.append(batch.label.values)

        X = torch.cat(x_lst)
        Y = torch.cat(y_lst)

        return (X, Y)

    def transform(self, batch):
        with torch.no_grad():
            individual_results = [model(batch) for model in self.pretrained_models]
            x = torch.stack(individual_results, dim=1)

            # text.transpose('batch', 'seqlen').values.clone()
            batch = batch.values.transpose(0, 1)

            if self.include_bag_of_words:
                bags_of_words = []
                for sent in batch:
                    bags_of_words.append(torch.bincount(sent, minlength=len(TEXT.vocab)))

                bags_of_words = torch.stack(bags_of_words).float()
                x = torch.cat((x, bags_of_words), dim=1)

            if self.include_cbow:
                cbows = []
                for sent in batch:
                    cbows.append(WORD_VEC[sent].sum(dim=0))
                cbows = torch.stack(cbows)
                x = torch.cat((x, cbows), dim=1)

            return x
