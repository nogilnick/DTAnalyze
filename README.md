# DTAnalyze
Functions to estimate importance of features in determining predictions for individual samples (aka "feature activations"). Fast `nogil` implementation in Cython.


## Example Usage
```
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from DTAnalyze.Activation import GetActivations

A = np.random.rand(256, 3)
Y = (2 * (A[:, 0] > 0.5) - (A[:, 1] < 0.5) - 
     (A[:, 2] > 0.5) + np.random.normal(0, 0.1, size=256))

rfr = RandomForestRegressor(n_jobs=4).fit(A, Y)

L1 = GetActivations(rfr, A)
```

## Install

Install using pip:

`pip install DTAnalyze`

Or you can build manually from this repository:

`python setup.py build_ext`

Then copy build artifact into `DTAnalyze` (sub) folder and put that folder somewhere in your path.
