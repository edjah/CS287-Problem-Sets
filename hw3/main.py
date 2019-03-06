# Torch
import time
import torch
import subprocess
from namedtensor import ntorch
from tqdm import tqdm

from data_setup import DE, EN, train_iter, val_iter, tokenize_de, BOS_WORD, escape
from beam_search import beam_search
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


def train_model(model, num_epochs=300, learning_rate=0.001, weight_decay=0, log_freq=1):
    model.train()
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


def make_translation_predictions(model, file='test_predictions.txt'):
    print('Generating translations')
    with open(file, 'w') as outfile:
        with open('source_test.txt', 'r') as infile:
            for line in tqdm(list(infile)):
                tokens = [DE.vocab.stoi[w] for w in tokenize_de(line.strip())]
                src = ntorch.tensor(tokens, names="srcSeqlen")
                translation = beam_search(model, src, beam_size=5, num_results=10)[0]
                assert translation[0] == EN.vocab.stoi[BOS_WORD]
                sent = ' '.join(EN.vocab.itos[i] for i in translation[1:])
                outfile.write(sent + '\n')


def make_kaggle_predictions(model, file='kaggle_predictions.txt'):
    print('Generating Kaggle predictions')
    with open(file, 'w') as outfile:
        outfile.write('Id,Predicted\n')
        with open('source_test.txt', 'r') as infile:
            for i, line in enumerate(tqdm(list(infile))):
                tokens = [DE.vocab.stoi[w] for w in tokenize_de(line.strip())]
                src = ntorch.tensor(tokens, names="srcSeqlen")
                translations = beam_search(model, src, beam_size=5, num_results=200)
                trigrams = []
                for translation in translations:
                    assert translation[0] == EN.vocab.stoi[BOS_WORD]
                    trigram = escape('|'.join(EN.vocab.itos[i] for i in translation[1:4]))
                    if trigram not in trigrams:
                        trigrams.append(trigram)
                    if len(trigrams) == 100:
                        break

                padding = 100 - len(trigrams)
                trigrams += [trigrams[-1]] * padding
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
    # torch.save(model.state_dict(), "weights/basic_seq2seq_weights_2_layers_512")
    return model


def setup_attn_model():
    # constructing the model
    encoder = attn.EncoderRNN(num_layers=2, hidden_size=512, emb_dropout=0.5, lstm_dropout=0.5)
    decoder = attn.DecoderRNN(num_layers=2, hidden_size=512, emb_dropout=0.5, lstm_dropout=0.5)
    model = attn.AttnSeq2Seq(encoder, decoder, dropout=0.5)
    model.load_state_dict(torch.load("weights/attention_seq2seq_weights"))

    # training
    train_model(model, num_epochs=100, learning_rate=0.001, weight_decay=0, log_freq=1)
    # torch.save(model.state_dict(), "weights/attention_seq2seq_weights")
    return model


if __name__ == '__main__':
    model = setup_vanilla_model()
    # model = setup_attn_model()

    make_translation_predictions(model, 'test_predictions.txt')
    print('BLEU score:', bleu('test_predictions.txt', 'target_test.txt'))
    make_kaggle_predictions(model)
