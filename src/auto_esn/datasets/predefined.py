from enum import Enum
from typing import Union

from auto_esn.experiment.util import Dataset
from auto_esn.utils.dataset_loader import norm_loader_val_test_, loader_val_test


class DatasetType(Enum):
    SUNSPOT = "sunspot"
    MackeyGlass = "mg"
    MultipleSuperimposedOscillators = "sin3"
    Temperature = "temp"

dataset_type_to_path = {
    "sunspot": "auto_esn/datasets/sunspot.csv",
    "sin3": "auto_esn/datasets/3sins.csv",
    "mg": "auto_esn/datasets/mg10.csv",
    "temp": "auto_esn/datasets/temperature_day.csv"
}

class PredefinedDataset:
    def __init__(self,dtype: DatasetType):
        self.dtype = dtype

    def load(self, val_size:Union[int,float], test_size:Union[int, float]):
        X_sun, X_val_sun, X_test_sun, y_sun, y_val_sun, y_test_sun, centr_sun, spread_sun = norm_loader_val_test_(
            loader_val_test(dataset_type_to_path[self.dtype.value], val_size=val_size, test_size=test_size))
        return Dataset(self.dtype.value, x_train=X_sun, y_train=y_sun, x_val=X_val_sun, y_val=y_val_sun, x_test=X_test_sun,
                y_test=y_test_sun, baseline=centr_sun, spread=spread_sun)