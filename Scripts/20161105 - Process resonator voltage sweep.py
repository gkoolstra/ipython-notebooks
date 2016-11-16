from matplotlib import pyplot as plt
import platform, os, sys, time
import numpy as np
from scipy.optimize import minimize, approx_fprime
import h5py
from tqdm import tqdm
from termcolor import colored, cprint

if 'Windows' in platform.system():
    sys.path.append(r'C:\Users\slab\Documents\Code')
    sys.path.append(r'D:\BEMPP_shared\Modules')
    import interpolate_slow
else:
    sys.path.append("/Users/gkoolstra/Documents/Code")
    from BEMHelper import interpolate_slow

from Common import common
from TrapAnalysis import artificial_anneal as anneal

save_path = r"/Users/gkoolstra/Desktop/Electron optimization/Realistic potential/Resonator"
sub_dir = r"161107_153311_resonator_sweep_50_electrons_with_perturbing"

dbin = 0.010E-6
bins = np.arange(-2.00E-6, 2.00E-6+dbin, dbin)
Vres = list()
converged = list()
save = True

with h5py.File(os.path.join(os.path.join(save_path, sub_dir), "Results.h5"), "r") as f:
    for k, step in enumerate(f.keys()):
        electron_ri = f[step + "/electron_final_coordinates"][()]
        xi, yi = anneal.r2xy(electron_ri)

        electron_hist, bin_edges = np.histogram(xi, bins=bins)

        if k == 0:
            electron_histogram = electron_hist
        else:
            electron_histogram = np.vstack((electron_histogram, electron_hist))

        Vres.append(f[step + "/potential_coefficients"][()])
        converged.append(f[step + "/solution_valid"][()])

fig=plt.figure(figsize=(7.,4.))
common.configure_axes(12)
plt.pcolormesh(Vres, bins[:-1]*1E6, np.log10(electron_histogram.T), vmin=0, vmax=1, cmap=plt.cm.hot_r)
plt.colorbar()
plt.xlim(Vres[0], Vres[-1])
plt.ylim(np.min(bins)*1E6, np.max(bins)*1E6)
plt.xlabel("Resonator voltage (V)")
plt.ylabel("Position across the channel ($\mu$m)")

if save:
    common.save_figure(fig, os.path.join(save_path, sub_dir))

plt.show()
