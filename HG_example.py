import numpy as np
import matplotlib.pyplot as plt
from HG import HG

hg = HG()
x = np.linspace(-10, 10, 1000)
y = hg.HG1D(x , [3, 0, 1])