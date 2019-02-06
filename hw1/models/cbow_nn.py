from data_setup import torch, ntorch, train_iter, TEXT


class CbowNN(torch.nn.Module):
    def __init__(self, second_layer_size=50):
        super().__init__()

        # constructing the neural network model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(300, second_layer_size),
            torch.nn.ReLU(),
            torch.nn.Linear(second_layer_size, 1),
            torch.nn.Sigmoid()
        )

    def __call__(self, text):
        sentences = text.transpose('batch', 'seqlen').values.clone()
        xbatch = TEXT.vocab.vectors[sentences].sum(dim=1)
        y = self.model(xbatch)
        return y

    def forward(self, text):
        return self.model(text)
