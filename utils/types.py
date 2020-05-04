from typing import Callable, Optional

from torch import Tensor as Tensor

Initializer = Callable[[Tensor, Optional[Tensor]], Tensor]
ActivationFunction = Callable[[Tensor], Tensor]

# subreservoiry ogarnac inicjalizacja -mozna podac wczesniejszy sub
