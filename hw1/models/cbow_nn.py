from data_setup import torch, TEXT


class CbowNN(torch.nn.Module):
    def __init__(self, do_embedding=False, second_layer_size=50):
        super().__init__()

        self.do_embedding = do_embedding

        embed_len = TEXT.vocab.vectors.shape[1]
        input_size = embed_len if self.do_embedding else len(TEXT.vocab)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, second_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(second_layer_size, second_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(second_layer_size, second_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(second_layer_size, 1),
            torch.nn.Sigmoid()
        )

    def __call__(self, text):
        return self.model(self.transform(text))

    def forward(self, text):
        return self.model(text)

    def get_data(self, dataset):
        x_lst = []
        y_lst = []
        for batch in dataset:
            x_lst.append(self.transform(batch.text))
            y_lst.append(batch.label.values)

        X = torch.cat(x_lst)
        Y = torch.cat(y_lst)
        return X, Y

    def transform(self, text):
        sentences = text.transpose('batch', 'seqlen').values.clone()
        if self.do_embedding:
            return TEXT.vocab.vectors[sentences].sum(dim=1)
        else:
            res = [s.bincount(minlength=len(TEXT.vocab)) for s in sentences]
            return torch.stack(res).float()
