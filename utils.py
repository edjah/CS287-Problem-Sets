import time


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
