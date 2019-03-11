from data_setup import tokenize_de, EN, DE

DE_unk = DE.vocab.stoi["<unk>"]

with open('kaggle_predictions.txt', 'r') as fp:
    next(fp)  # skip header
    predictions = [line.strip().split(',')[1] for line in fp]

with open('source_test.txt', 'r') as infile:
    with open('kaggle_predictions.txt', 'w') as outfile:
        outfile.write('Id,Predicted\n')
        for i, (preds, src) in enumerate(zip(predictions, infile), 1):
            tokens = tokenize_de(src.strip())
            token_idx = [EN.vocab.stoi[w] for w in tokens]

            # generating the unknown map
            unk_map = {}
            for idx, token in zip(token_idx, tokens):
                if idx == DE_unk:
                    unk_map[len(unk_map)] = token

            # using the unkown map to correct the predictions
            new_preds = []
            for pred in preds.split(' '):
                words = pred.split('|')
                new_words = []
                unk_count = 0
                for w in words:
                    if w == "<unk>":
                        new_words.append(unk_map.get(unk_count, "<unk>"))
                        unk_count += 1
                    else:
                        new_words.append(w)

                new_preds.append('|'.join(new_words))

            outfile.write(str(i) + ',' + ' '.join(new_preds) + '\n')
