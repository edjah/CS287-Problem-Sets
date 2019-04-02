import time
import torch
import os
import torchtext
import traceback

from tqdm import tqdm
from namedtensor import ntorch
from data_setup import train_iter, val_iter, test

from models.decomposable import AttendNN
from models.mixture import LearnedEnsemble


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


def train_model(model, num_epochs=100, learning_rate=0.001, weight_decay=0,
                grad_clip=5, log_freq=100, save_file=None):

    if os.path.exists(save_file):
        model.load_state_dict(torch.load(save_file))

    val_loss, val_acc = evaluate(model, val_iter)
    print(f'Initial Val Loss: {val_loss:.5f}')
    print(f'Initial Val Acc: {val_acc:.5f}\n')

    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    loss_fn = ntorch.nn.NLLLoss().spec('label')
    start_time = time.time()

    best_params = {k: p.detach().clone() for k, p in model.named_parameters()}
    best_val_acc = val_acc

    for epoch in range(num_epochs):
        curr_loss = 0
        num_correct = 0
        total_num = 0

        # decay learning rate by 4% every epoch. goes to 1.6% after 100 epochs
        learning_rate *= 0.96
        state_dict = opt.state_dict()
        for param_group in state_dict['param_groups']:
            param_group['lr'] = learning_rate
        opt.load_state_dict(state_dict)

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

                # compute gradients, clipping them, and updating weights
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

                # print out intermediate training accuracy and loss
                if i % log_freq == 0:
                    msg = f'Epoch {epoch} | Batch: {i} / {len(train_iter)}'
                    print('\n\n' + msg + '\n' + '=' * len(msg))
                    print(f'training loss: {(curr_loss / log_freq):.5f}')
                    print(f'training acc: {(num_correct / total_num):.5f}\n')
                    curr_loss = 0
                    num_correct = 0
                    total_num = 0

            # evaluate performance on the validation set
            model.eval()
            val_loss, val_acc = evaluate(model, val_iter)
            model.train()

            # saving the parameters with the best validation loss
            if val_acc > best_val_acc:
                best_params = {
                    k: p.detach().clone() for k, p in model.named_parameters()
                }
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_file)

            # logging
            msg = f'Epoch {epoch + 1} | {round(time.time() - start_time)} sec'
            print(f'{msg}\n{"=" * len(msg)}')
            print(f'Full Validation Loss: {val_loss:.5f}')
            print(f'Full Validation Acc: {val_acc:.5f}\n')

        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):
                print(f'Got unexpected interrupt: {e!r}')
                traceback.print_exc()

            print(f'\nStopped training after {epoch} epochs...')
            break

    model.load_state_dict(best_params)
    msg = f'{round(time.time() - start_time)} sec: Final Results'
    print(f'{msg}\n{"=" * len(msg)}')

    val_loss, val_acc = evaluate(model, val_iter)
    print(f'Final Val Loss: {val_loss:.5f}')
    print(f'Final Val Acc: {val_acc:.5f}\n')


def generate_predictions(model):
    print('Generating predictions...')
    test_iter = torchtext.data.BucketIterator(
        test, train=False, batch_size=10, device=torch.device('cuda')
    )

    predictions = []
    num_correct = 0

    tot_count = 0

    with torch.no_grad():
        model.eval()
        for batch in test_iter:
            batch_preds = model(batch.premise, batch.hypothesis).argmax('label')
            predictions += batch_preds.tolist()
            num_correct += (batch.label == batch_preds).sum().item()
            tot_count += len(batch_preds)

    print(f'Test Accuracy: {100.0 * num_correct / tot_count:.2f}%')

    with open('predictions.txt', 'w') as fp:
        fp.write('Id,Category\n')
        for i, pred in enumerate(predictions):
            fp.write(str(i) + ',' + str(pred) + '\n')


# where weights are stored for use in ensemble models
M1_WEIGHTS = 'weights/colab_86.25_val_acc_decomposable_attention_2_layers_300_hidden_v3'
M2_WEIGHTS = 'weights/azure_86.14_val_acc_decomposable_attention_2_layers_200_hidden_v3'
M3_WEIGHTS = 'weights/local_85.64_val_acc_decomposable_attention_2_layers_200_hidden_v3_intra_attn'
M4_WEIGHTS = 'weights/azure_85.95_val_acc_decomposable_attention_2_layers_300_hidden_v3_intra_attn'


def build_ensemble(ensemble_cls, **kwargs):
    m1 = AttendNN(num_layers=2, hidden_size=300, dropout=0.2, intra_attn=False)
    m2 = AttendNN(num_layers=2, hidden_size=200, dropout=0.2, intra_attn=False)
    m3 = AttendNN(num_layers=2, hidden_size=200, dropout=0.2, intra_attn=True)
    m4 = AttendNN(num_layers=2, hidden_size=300, dropout=0.2, intra_attn=True)

    m1.load_state_dict(torch.load(M1_WEIGHTS))
    m2.load_state_dict(torch.load(M2_WEIGHTS))
    m3.load_state_dict(torch.load(M3_WEIGHTS))
    m4.load_state_dict(torch.load(M4_WEIGHTS))

    return ensemble_cls(m1, m2, m3, m4, **kwargs)


if __name__ == '__main__':
    # model = AttendNN(
    #     num_layers=2, hidden_size=200, dropout=0.2, intra_attn=True
    # )
    # save_file = 'weights/decomposable_attention_2_layers_200_hidden_v3'

    model = build_ensemble(LearnedEnsemble, fine_tune=True)
    save_file = 'weights/ensemble_of_4'

    train_model(
        model, learning_rate=0.0005, weight_decay=0, grad_clip=5,
        log_freq=1000, save_file=save_file
    )
    print('Ensemble distribution:', model.weights.detach().softmax(dim=0))
    generate_predictions(model)
