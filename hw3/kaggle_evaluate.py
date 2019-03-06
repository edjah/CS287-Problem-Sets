score = 0

with open('target_test.txt', 'r') as source:
    with open('kaggle_predictions.txt', 'r') as predictions:
        next(predictions)  # skip kaggle predictions header
        predictions = list(predictions)
        source = list(source)
        assert len(source) == len(predictions)

        for actual, preds in zip(source, predictions):
            actual = '|'.join(actual.strip().split()[:3])
            preds = preds.strip().split(',')[1].split()
            for i in range(len(preds)):
                if preds[i] == actual:
                    score += 1 / (i + 1)
                    break

print(score / len(source))
