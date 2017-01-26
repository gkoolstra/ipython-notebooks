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

save_path = r"/Volumes/slab/Gerwin/Electron on helium/Electron optimization/Realistic potential/Resonator"
sub_dir = r"170118_145233_M018V2_resonator_Vsweep_200_electrons"

dbin = 0.006E-6
bins = np.arange(-0.50E-6, 0.50E-6+dbin, dbin)
converged = list()
save = True

with h5py.File(os.path.join(os.path.join(save_path, sub_dir), "Results.h5"), "r") as f:
    k = 0
    for step in f.keys():
        if "step" in step:
            #print(step)
            electron_ri = f[step + "/electron_final_coordinates"][()]
            xi, yi = anneal.r2xy(electron_ri)

            #electron_hist, bin_edges = np.histogram(xi, bins=bins)

            valid_solution = f[step + "/solution_converged"][()]
            converged.append(valid_solution)

            if valid_solution:
                electron_hist, bin_edges = np.histogram(xi, bins=bins)
            else:
                electron_hist = np.zeros(len(electron_hist))

            if k == 0:
                electron_histogram = electron_hist
            else:
                electron_histogram = np.vstack((electron_histogram, electron_hist))

            k += 1

    #print("Ja!")
    Vres = f["Vres"][()]

print("Out of %d simulations, %d did not converge..." % (len(Vres), len(Vres) - np.sum(converged)))

fig = plt.figure(figsize=(7.,4.))
common.configure_axes(12)
plt.pcolormesh(Vres, bins[:-1]*1E6, np.log10(electron_histogram.T), vmin=0, vmax=2.0, cmap=plt.cm.hot_r)
plt.colorbar()
plt.xlim(Vres[0], Vres[-1])
plt.ylim(np.min(bins)*1E6, np.max(bins)*1E6)
plt.xlabel("Resonator voltage (V)")
plt.ylabel("Position across the channel ($\mu$m)")
plt.title(r"$^{10}\log (n_\ell)$ for bin size = %.1f nm"%(dbin*1E9))

if save:
    common.save_figure(fig, os.path.join(save_path, sub_dir))

print("Done!")
plt.show()