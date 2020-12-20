import random
from typing import Callable


def next_gen(seq: list) -> Callable:
    def generator():
        i = 0
        while True:
            yield seq[i % len(seq)]
            i += 1
    gen = generator()

    def gen_next():
        return next(gen)

    return gen_next


def random_gen(seq: list) -> Callable:
    return lambda: random.choice(seq)
