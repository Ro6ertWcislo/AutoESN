from enum import Enum
from typing import Union

from auto_esn.Experiment import Dataset
from auto_esn.utils.dataset_loader import norm_loader_val_test_, loader_val_test, loader_val_test_shift


class DatasetType(Enum):
    SUNSPOT = "sunspot"
    MackeyGlass = "mg"
    MultipleSuperimposedOscillators = "sin3"
    Temperature = "temp"
    MemoryCapacity = "memoryCapacity"

dataset_type_to_path = {
    "sunspot": "datasets/sunspot.csv",
    "sin3": "datasets/3sins.csv",
    "mg": "datasets/mg10.csv",
    "temp": "datasets/temperature_day.csv",
    "memoryCapacity": "datasets/memoryCapacity.csv"
}

class PredefinedDataset:
    def __init__(self,dtype: DatasetType, **kwargs):
        self.dtype = dtype
        self.kwargs = kwargs

    def load(self, val_size:Union[int,float], test_size:Union[int,float]):
        X_sun, X_val_sun, X_test_sun, y_sun, y_val_sun, y_test_sun, centr_sun, spread_sun = norm_loader_val_test_(
            loader_val_test(dataset_type_to_path[self.dtype.value], val_size=val_size, test_size=test_size))
        return Dataset(self.dtype.value, x_train=X_sun, y_train=y_sun, x_val=X_val_sun, y_val=y_val_sun, x_test=X_test_sun,
                y_test=y_test_sun, baseline=centr_sun, spread=spread_sun)

    def load_with_shift(self, val_size: Union[int,float], test_size: Union[int,float], shift):
        X_sun, X_val_sun, X_test_sun, y_sun, y_val_sun, y_test_sun, centr_sun, spread_sun = norm_loader_val_test_(
            loader_val_test_shift(dataset_type_to_path[self.dtype.value], val_size=val_size, test_size=test_size,shift=shift))
        return Dataset(self.dtype.value, x_train=X_sun, y_train=y_sun, x_val=X_val_sun, y_val=y_val_sun,
                       x_test=X_test_sun,
                       y_test=y_test_sun, baseline=centr_sun, spread=spread_sun)