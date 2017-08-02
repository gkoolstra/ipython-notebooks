import numpy as np

offsets = np.arange(0.00, 0.26, 0.01)
for offset in offsets:
    exec(open("./20170312 - Multiple electrode sweep for M018.py").read())