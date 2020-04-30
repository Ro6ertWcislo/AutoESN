from typing import Callable

from torch import Tensor as Tensor

Initializer = Callable[[Tensor], Tensor]
ActivationFunction = Callable[[Tensor], Tensor]

# subreservoiry ogarnac inicjalizacja -mozna podac wczesniejszy sub
