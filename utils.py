import time
import torch


def humanize_duration(duration):
    """Converts a duration (in seconds) to a human-readable string"""
    if duration < 1:
        return f'{1000 * duration:.3f} ms'
    elif duration < 60:
        return f'{duration:.3f} sec'

    duration = round(duration)
    if duration < 3600:
        return f'{duration // 60} min {duration % 60} sec'
    elif duration < 86400:
        return f'{duration // 3600} hrs {(duration // 60) % 60} min'
    else:
        return f'{duration // 86400} days {(duration // 3600) % 24} hrs'


class TimingContext:
    """
    Motivating Example:
        code
        ----
        with TimingContext('expensive command'):
            with TimingContext('subroutine1'):
                subroutine1()
            with TimingContext('subroutine2'):
                subroutine2()
            other_stuff()

        stdout
        ------
            < subroutine1: 123.456 ms
            < subroutine2: 234.567 ms
        > expensive command: 427.023 ms

    Automatic indentation for nested Timing Context is not thread safe
    """

    indent_level = 0
    quiet = False

    def __init__(self, description='', quiet=False):
        self.description = description
        self.prev_quiet = TimingContext.quiet
        self.quiet = quiet

        if self.quiet:
            TimingContext.quiet = True

    def __enter__(self):
        self.start = time.time()
        TimingContext.indent_level += 1

    def __exit__(self, type, value, traceback):
        TimingContext.indent_level -= 1

        duration = humanize_duration(time.time() - self.start)
        indent = ' ' * (4 * TimingContext.indent_level)
        indent += '>' if TimingContext.indent_level == 0 else '<'
        msg = f'{indent} {self.description}: {duration}'

        if not TimingContext.quiet:
            print(msg)

        if self.quiet:
            TimingContext.quiet = self.prev_quiet


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
