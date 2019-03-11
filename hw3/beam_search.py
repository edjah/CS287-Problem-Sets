import torch
from namedtensor import ntorch
from data_setup import BOS_IND, EOS_IND


def beam_search(model, src, beam_size=5, max_len=30, num_results=10, alpha=1.0):
    model.eval()
    with torch.no_grad():
        src = ntorch.tensor(src.values.unsqueeze(0), ('batch', 'srcSeqlen'))
        src = model._flip(src, 'srcSeqlen')
        enc_outputs, enc_hidden = model.encoder.forward(src)

        translated_sents = []
        beams = [(0.0, [BOS_IND], enc_hidden)]

        for i in range(max_len):
            old_beams = beams
            beams = []
            for old_score, beam, hidden in old_beams:
                dec_input = ntorch.tensor([beam[-1:]], ('batch', 'trgSeqlen'))
                dec_output, dec_hidden = model.decoder.forward(dec_input, hidden, enc_outputs)
                last_hidden = dec_hidden[0].get('layers', -1)
                scores = model.out(last_hidden).log_softmax('vocab')
                scores, indexes = scores.topk('vocab', beam_size)
                for score, ind in zip(scores.values.flatten(), indexes.values.flatten()):
                    beams.append((old_score + score.item(), beam + [ind.item()], dec_hidden))

            beams.sort(key=lambda x: x[0], reverse=True)
            beams = beams[:beam_size]

            new_beams = []
            for score, beam, hidden in beams:
                if beam[0:2] == [BOS_IND, BOS_IND] and (len(beam) == max_len or beam[-1] == EOS_IND):
                    if beam[-1] == EOS_IND:
                        beam.pop()
                    translated_sents.append((score, beam[1:], hidden))
                else:
                    new_beams.append((score, beam, hidden))
            beams = new_beams

            if len(translated_sents) >= num_results:
                break

        if len(translated_sents) == 0:
            translated_sents += beams

        translated_sents.sort(key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
        return [t[1] for t in translated_sents][:num_results]


def beam_search2(model, src, beam_size=5, max_len=30, num_results=10, alpha=1.0):
    model.eval()
    with torch.no_grad():
        src = ntorch.tensor(src.values.unsqueeze(0), ('batch', 'srcSeqlen'))

        translated_sents = []
        beams = [(0.0, [BOS_IND])]

        for i in range(max_len):
            old_beams = beams
            beams = []
            for old_score, beam in old_beams:
                dec_input = ntorch.tensor([beam], ('batch', 'trgSeqlen'))
                scores = model.forward(src, dec_input, shift_trg=False)
                scores = scores.get('trgSeqlen', -1)
                scores, indexes = scores.topk('vocab', beam_size)
                for score, ind in zip(scores.values.flatten(), indexes.values.flatten()):
                    beams.append((old_score + score.item(), beam + [ind.item()]))

            beams.sort(key=lambda x: x[0], reverse=True)
            beams = beams[:beam_size]

            new_beams = []
            for score, beam in beams:
                if beam[0:2] == [BOS_IND, BOS_IND] and (len(beam) == max_len or beam[-1] == EOS_IND):
                    if beam[-1] == EOS_IND:
                        beam.pop()
                    translated_sents.append((score, beam[1:]))
                else:
                    new_beams.append((score, beam))
            beams = new_beams

            if len(translated_sents) >= num_results:
                break

        if len(translated_sents) == 0:
            translated_sents += beams

        translated_sents.sort(key=lambda x: x[0] / (len(x[1]) ** alpha), reverse=True)
        return [t[1] for t in translated_sents][:num_results]


def kaggle_search(model, src):
    model.eval()
    with torch.no_grad():
        src = ntorch.tensor(src.values.unsqueeze(0), ('batch', 'srcSeqlen'))
        src = model._flip(src, 'srcSeqlen')
        enc_outputs, enc_hidden = model.encoder.forward(src)

        beams = [(0.0, [BOS_IND], enc_hidden)]

        for i in range(4):
            old_beams = beams
            beams = []
            for old_score, beam, old_hidden in old_beams:
                dec_input = ntorch.tensor([beam[-1:]], ('batch', 'trgSeqlen'))
                dec_output, dec_hidden = model.decoder.forward(dec_input, old_hidden, enc_outputs)
                last_hidden = dec_hidden[0].get('layers', -1)
                scores = model.out(last_hidden).log_softmax('vocab')
                scores, indexes = scores.topk('vocab', 10)
                for score, ind in zip(scores.values.flatten(), indexes.values.flatten()):
                    beams.append((old_score + score.item(), beam + [ind.item()], dec_hidden))
            if i == 0:
                assert beams[0][1][1] == 2
                beams = beams[:1]

        beams.sort(key=lambda x: x[0], reverse=True)
        return [t[1][2:] for t in beams][:100]


def kaggle_search2(model, src):
    model.eval()
    with torch.no_grad():
        src = ntorch.tensor(src.values.unsqueeze(0), ('batch', 'srcSeqlen'))

        beams = [(0.0, [BOS_IND])]

        for i in range(4):
            old_beams = beams
            beams = []
            for old_score, beam in old_beams:
                dec_input = ntorch.tensor([beam], ('batch', 'trgSeqlen'))
                scores = model.forward(src, dec_input, shift_trg=False)
                scores = scores.get('trgSeqlen', -1)

                scores, indexes = scores.topk('vocab', 10)
                for score, ind in zip(scores.values.flatten(), indexes.values.flatten()):
                    beams.append((old_score + score.item(), beam + [ind.item()]))
            if i == 0:
                assert beams[0][1][1] == 2
                beams = beams[:1]

        beams.sort(key=lambda x: x[0], reverse=True)
        return [t[1][2:] for t in beams][:100]
