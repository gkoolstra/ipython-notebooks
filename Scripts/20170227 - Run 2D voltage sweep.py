import numpy as np
resVs = np.arange(0.975, 0.025, -0.025)

# Sweep the trap for each resonator voltage point instead of the other way around.
for Vres in resVs:
    Vtrap = np.arange(Vres-0.2, Vres+0.2, +0.01)
    exec(open("20170119 - Greater trap area voltage sweep for M018.py").read())