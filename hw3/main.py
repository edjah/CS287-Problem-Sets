# Torch
import time
import torch
import subprocess
from namedtensor import ntorch
from tqdm import tqdm

from data_setup import DE, EN, train_iter, val_iter, tokenize_de, BOS_IND, escape
from beam_search import beam_search, beam_search2, kaggle_search, kaggle_search2
import vanilla_seq2seq as vanilla
import attn_seq2seq as attn


def evaluate(model, batches):
    model.eval()
    with torch.no_grad():
        loss_fn = ntorch.nn.NLLLoss(reduction='sum').spec('vocab')
        tot_loss = 0
        num_ex = 0
        for batch in batches:
            log_probs = model.forward(batch.src, batch.trg)
            tot_loss += loss_fn(log_probs, batch.trg).values
            num_ex += batch.trg.shape['batch'] * batch.trg.shape['trgSeqlen']

        return torch.exp(tot_loss / num_ex)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()


def train_model(model, num_epochs=300, learning_rate=0.001, weight_decay=0, log_freq=1, self_attn_hid_dim=None):
    model.train()

    if self_attn_hid_dim:
        opt = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        opt = NoamOpt(self_attn_hid_dim, 1, 2000, opt)
    else:
        opt = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    loss_fn = ntorch.nn.NLLLoss().spec('vocab')
    start_time = time.time()

    best_params = {k: p.detach().clone() for k, p in model.named_parameters()}
    best_val_ppl = float('inf')

    for i in range(num_epochs):
        try:
            for batch in tqdm(train_iter, total=len(train_iter)):
                opt.zero_grad()

                log_probs = model.forward(batch.src, batch.trg)
                loss = loss_fn(log_probs, batch.trg)

                # compute gradients and update weights
                loss.backward()
                opt.step()

            # evaluate performance on entire sets
            model.eval()
            train_ppl = evaluate(model, train_iter)
            val_ppl = evaluate(model, val_iter)
            model.train()

            # saving the parameters with the best validation loss
            if val_ppl < best_val_ppl:
                best_params = {k: p.detach().clone() for k, p in model.named_parameters()}
                best_val_ppl = val_ppl
                torch.save(model.state_dict(), "weights/bigger_self_attn_weights")

            # logging
            if i == 0 or i == num_epochs - 1 or (i + 1) % log_freq == 0:
                msg = f"{round(time.time() - start_time)} sec: Epoch {i + 1}"
                print(f'{msg}\n{"=" * len(msg)}')
                print(f'Train Perplexity: {train_ppl:.5f}')
                print(f'Val Perplexity: {val_ppl:.5f}\n')

        except KeyboardInterrupt:
            print(f'\nStopped training after {i} epochs...')
            break

    model.eval()
    model.load_state_dict(best_params)

    msg = f"{round(time.time() - start_time)} sec: Final Results"
    print(f'{msg}\n{"=" * len(msg)}')

    train_ppl = evaluate(model, train_iter)
    val_ppl = evaluate(model, val_iter)
    print(f'Train Perplexity: {train_ppl:.5f}')
    print(f'Val Perplexity: {val_ppl:.5f}\n')


def bleu(target_file, predictions_file):
    cmd = f"./multi-bleu.perl {target_file} < {predictions_file} " \
          "-h 2> /dev/null | cut -d' ' -f3 | cut -d',' -f1"
    return float(subprocess.check_output(cmd, shell=True))


def make_translation_predictions(model, use_bs2=False):
    print('Generating translations')
    with open('test_predictions.txt', 'w') as outfile:
        with open('source_test.txt', 'r') as infile:
            for line in tqdm(list(infile)):
                tokens = [DE.vocab.stoi[w] for w in tokenize_de(line.strip())]
                src = ntorch.tensor(tokens, names="srcSeqlen")
                if use_bs2:
                    translation = beam_search2(model, src, beam_size=5, num_results=10)[0]
                else:
                    translation = beam_search(model, src, beam_size=5, num_results=10)[0]

                assert translation[0] == BOS_IND
                sent = ' '.join(EN.vocab.itos[i] for i in translation[1:])
                outfile.write(sent + '\n')


def make_kaggle_predictions(model, use_ks2=False):
    print('Generating Kaggle predictions')
    with open('kaggle_predictions.txt', 'w') as outfile:
        outfile.write('Id,Predicted\n')
        with open('source_test.txt', 'r') as infile:
            for i, line in enumerate(tqdm(list(infile))):
                tokens = [DE.vocab.stoi[w] for w in tokenize_de(line.strip())]
                src = ntorch.tensor(tokens, names="srcSeqlen")

                if use_ks2:
                    preds = kaggle_search2(model, src)
                else:
                    preds = kaggle_search(model, src)

                trigrams = []
                for trigram in preds:
                    assert len(trigram) == 3
                    trigram = escape('|'.join(EN.vocab.itos[i] for i in trigram))
                    trigrams.append(trigram)

                assert len(trigrams) == 100
                outfile.write(str(i) + ',' + ' '.join(trigrams) + '\n')


def setup_vanilla_model():
    # constructing the model
    encoder = vanilla.EncoderRNN(num_layers=2, hidden_size=512, emb_dropout=0.5, lstm_dropout=0.5)
    decoder = vanilla.DecoderRNN(num_layers=2, hidden_size=512, emb_dropout=0.5, lstm_dropout=0.5)
    model = vanilla.VanillaSeq2Seq(encoder, decoder, dropout=0.5)
    model.load_state_dict(torch.load("weights/basic_seq2seq_weights_2_layers_512"))

    # training
    train_model(model, num_epochs=100, learning_rate=0.001, weight_decay=0, log_freq=1)
    torch.save(model.state_dict(), "weights/2_basic_seq2seq_weights_2_layers_512")
    return model


def setup_attn_model():
    # constructing the model
    encoder = attn.EncoderRNN(num_layers=2, hidden_size=1024, emb_dropout=0.5, lstm_dropout=0.5)
    decoder = attn.DecoderRNN(num_layers=2, hidden_size=1024, emb_dropout=0.5, lstm_dropout=0.5)
    model = attn.AttnSeq2Seq(encoder, decoder, dropout=0.5)
    # model.load_state_dict(torch.load("weights/attention_seq2seq_weights_2_layer_1024_cell"))

    # training
    train_model(model, num_epochs=100, learning_rate=0.001, weight_decay=0, log_freq=1)
    torch.save(model.state_dict(), "weights/attention_seq2seq_weights_2_layer_1024_cell")
    return model


def setup_self_attn_model():
    import torch.nn as nn
    from transformer import Encoder, Decoder, Transformer, EncoderLayer, DecoderLayer, SelfAttention, PositionwiseFeedforward

    device = torch.device('cuda:0')
    pad_idx = DE.vocab.stoi["<pad>"]

    hid_dim = 300
    n_layers = 3
    n_heads = 4
    pf_dim = 512  # 2048
    dropout = 0.1

    input_dim = len(DE.vocab)
    enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

    output_dim = len(EN.vocab)
    dec = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

    model = Transformer(enc, dec, pad_idx, device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # model.load_state_dict(torch.load("weights/bigger_self_attn_weights"))

    train_model(model, num_epochs=100, learning_rate=0.001, weight_decay=0, log_freq=1, self_attn_hid_dim=hid_dim)
    torch.save(model.state_dict(), "weights/bigger_self_attn_weights")

    return model


if __name__ == '__main__':
    using_self_attn = True

    # model = setup_vanilla_model()
    # model = setup_attn_model()
    model = setup_self_attn_model()

    make_translation_predictions(model, using_self_attn)
    print('BLEU score:', bleu('test_predictions.txt', 'target_test.txt'))
    make_kaggle_predictions(model, using_self_attn)
