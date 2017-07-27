import numpy as np

offsets = np.arange(0.21, 0.30, 0.01)
for offset in offsets:
    exec(open("./20170312 - Multiple electrode sweep for M018.py").read())