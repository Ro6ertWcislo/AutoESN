from typing import Callable, Optional

from torch import Tensor as Tensor

Initializer = Callable[[Tensor, Optional[Tensor]], Tensor]
Metric = Callable[[Tensor, Tensor], Tensor]
ActivationFunction = Callable[[Tensor], Tensor]
SpectralCentroid =float
SpectralSpread =float

# subreservoiry ogarnac inicjalizacja -mozna podac wczesniejszy sub
