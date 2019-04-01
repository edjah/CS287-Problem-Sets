import time
import torch
import os
from tqdm import tqdm
from namedtensor import ntorch
from data_setup import train_iter, val_iter

from models.decomposable import AttendNN

ATTEND_NN_WEIGHTS_FILE = 'weights/decomposable_attention_2_layers_200_hidden'


def evaluate(model, batches):
    model.eval()
    with torch.no_grad():
        loss_fn = ntorch.nn.NLLLoss(reduction='sum').spec('label')
        total_loss = 0
        num_correct = 0
        total_num = 0

        for batch in batches:
            log_probs = model.forward(batch.premise, batch.hypothesis)
            predictions = log_probs.argmax('label')
            total_loss += loss_fn(log_probs, batch.label).item()
            num_correct += (predictions == batch.label).sum().item()
            total_num += len(batch)

        return total_loss / total_num, 100.0 * num_correct / total_num


def train_model(model, num_epochs=100, learning_rate=0.0001, log_freq=100):
    # opt = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = ntorch.nn.NLLLoss().spec('label')
    start_time = time.time()

    best_params = {k: p.detach().clone() for k, p in model.named_parameters()}
    best_val_loss = float('inf')

    for epoch in range(num_epochs):

        curr_loss = 0
        num_correct = 0
        total_num = 0

        try:
            model.train()
            for i, batch in enumerate(tqdm(train_iter), 1):
                opt.zero_grad()

                log_probs = model.forward(batch.premise, batch.hypothesis)
                predictions = log_probs.detach().argmax('label')
                loss = loss_fn(log_probs, batch.label)

                curr_loss += loss.detach().item()
                num_correct += (predictions == batch.label).sum().item()
                total_num += len(batch)

                # print out intermediate training accuracy and loss
                if i % log_freq == 0:
                    msg = f'Batch: {i} / {len(train_iter)}'
                    print('\n\n' + msg + '\n' + '=' * len(msg))
                    print(f'training loss: {(curr_loss / log_freq):.5f}')
                    print(f'training acc: {(num_correct / total_num):.5f}\n')
                    curr_loss = 0
                    num_correct = 0
                    total_num = 0

                # compute gradients and update weights
                loss.backward()
                opt.step()

            # evaluate performance on the validation set
            model.eval()
            val_loss, val_acc = evaluate(model, val_iter)
            model.train()

            # saving the parameters with the best validation loss
            if val_loss < best_val_loss:
                best_params = {
                    k: p.detach().clone() for k, p in model.named_parameters()
                }
                best_val_loss = val_loss
                torch.save(model.state_dict(), ATTEND_NN_WEIGHTS_FILE)

            # logging
            msg = f'{round(time.time() - start_time)} sec: Epoch {epoch + 1}'
            print(f'{msg}\n{"=" * len(msg)}')
            print(f'Full Validation Loss: {val_loss:.5f}')
            print(f'Full Validation Acc: {val_acc:.5f}\n')

        except Exception as e:
            if not isinstance(e, KeyboardInterrupt):
                print(f'Got unexpected interrupt: {e!r}')
            print(f'\nStopped training after {epoch} epochs...')
            break

    model.load_state_dict(best_params)
    msg = f'{round(time.time() - start_time)} sec: Final Results'
    print(f'{msg}\n{"=" * len(msg)}')

    val_loss, val_acc = evaluate(model, val_iter)
    print(f'Final Val Loss: {val_loss:.5f}')
    print(f'Final Val Acc: {val_acc:.5f}\n')


if __name__ == '__main__':
    model = AttendNN(num_layers=2, hidden_size=200, intra_attn=False)
    if os.path.exists(ATTEND_NN_WEIGHTS_FILE):
        model.load_state_dict(torch.load(ATTEND_NN_WEIGHTS_FILE))
    train_model(model, learning_rate=0.001, log_freq=200)
