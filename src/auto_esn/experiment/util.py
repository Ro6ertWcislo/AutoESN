from dataclasses import dataclass
from typing import Optional
from torch import Tensor


@dataclass
class Dataset:
    name: str
    x_train: Tensor
    y_train: Tensor
    x_val: Tensor
    y_val: Tensor
    x_test: Optional[Tensor] = None
    y_test: Optional[Tensor] = None
    baseline: Optional[float] = None
    spread: Optional[float] = None