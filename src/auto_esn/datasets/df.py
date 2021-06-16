import os

import pandas as pd

this_dir, this_filename = os.path.split(__file__)
MackeyGlass = pd.read_csv(os.path.join(this_dir, "mg10.csv"))
Sunspot = pd.read_csv(os.path.join(this_dir, "sunspot.csv"))
SunspotSmooth = pd.read_csv(os.path.join(this_dir, "sunspotSmooth.csv"))
