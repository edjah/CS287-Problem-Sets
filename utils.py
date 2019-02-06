import time
import torch


class TimingContext:
    def __init__(self, msg=None, suffix=''):
        self.msg = msg
        self.suffix = suffix

    def __enter__(self):
        if self.msg is not None:
            print(self.msg)
            print('-' * len(self.msg))
        self.start = time.time()

    def __exit__(self, typ, val, traceback):
        dur = time.time() - self.start
        prefix = self.msg + ': ' if self.msg else ''

        if dur < 1:
            print(f'{prefix}{1000 * dur:.3f} ms')
        elif dur < 60:
            print(f'{prefix}{dur:.3f} sec')
        else:
            print(f'{prefix}{int(dur) // 60} min {int(dur) % 60} sec')

        print(self.suffix, end='')


def chunks(iterator, n):
    """Yield successive n-sized chunks from an iterator."""
    assert isinstance(n, int) and n >= 1, '`n` must be a positive integer'

    if isinstance(iterator, (list, tuple, str, bytes, torch.Tensor)):
        # use a faster chunking algorithm for things that can be sliced
        for i in range(0, len(iterator), n):
            yield iterator[i:i + n]
    else:
        # use an optimized general chunking algorithm for things that can't
        count = 0
        chunk = []
        for item in iterator:
            chunk.append(item)
            count += 1
            if count >= n:
                yield chunk
                chunk = []
                count = 0
        if chunk:
            yield chunk
