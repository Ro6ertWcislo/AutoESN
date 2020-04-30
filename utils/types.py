from typing import Callable
from torch import Tensor as Tensor

Activation = Callable[[Tensor], Tensor]
Initializer = Callable[[Tensor, int], None]

# subreservoiry ogarnac inicjalizacja -mozna podac wczesniejszy sub
