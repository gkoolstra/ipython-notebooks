import numpy as np
trapVs = np.arange(-0.80, -2.00, 0.05)

# Sweep the trap for each resonator voltage point instead of the other way around.
for trapV in trapVs:
    Vrg = np.arange(0.40, -0.70, -0.005)
    exec(open("20170312 - Multiple electrode sweep for M018.py").read())