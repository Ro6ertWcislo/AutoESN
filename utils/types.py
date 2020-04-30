from typing import Callable
from torch import Tensor as Tensor

Initializer = Callable[[Tensor, int], None]
ActivationFunction = Callable[[Tensor],Tensor]

# subreservoiry ogarnac inicjalizacja -mozna podac wczesniejszy sub
