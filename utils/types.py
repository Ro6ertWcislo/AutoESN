from typing import Callable, Optional, Tuple

import torch
from torch import Tensor as Tensor

Initializer = Callable[[Tensor, Optional[Tensor]], Tensor]
Metric = Callable[[Tensor, Tensor], Tensor]
ActivationFunction = Callable[[Tensor], Tensor]
SpectralCentroid = float
SpectralSpread = float

FloatGen = Callable[[], float]
BoolGen = Callable[[], bool]
IntGen = Callable[[], int]

Loader = Callable[[], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]

# subreservoiry ogarnac inicjalizacja -mozna podac wczesniejszy sub
