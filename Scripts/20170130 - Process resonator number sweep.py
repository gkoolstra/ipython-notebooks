from matplotlib import pyplot as plt
import platform, os, sys, h5py
import numpy as np
from tqdm import tqdm
from mpltools import color

if 'Windows' in platform.system():
    sys.path.append(r'C:\Users\slab\Documents\Code')
    sys.path.append(r'D:\BEMPP_shared\Modules')
    import interpolate_slow
else:
    sys.path.append("/Users/gkoolstra/Documents/Code")
    from BEMHelper import interpolate_slow

from Common import common
from TrapAnalysis import import_data, artificial_anneal as anneal
from resonator_eom import *

save = True
save_path = r"/Volumes/slab/Gerwin/Electron on helium/Electron optimization/Realistic potential/Resonator"
sub_dirs = [r"170118_143444_M018V2_resonator_Vsweep_50_electrons",
            r"170118_140803_M018V2_resonator_Vsweep_75_electrons",
            #r"170125_230605_M018V2_resonator_Vsweep_80_electrons",
            #r"170126_062249_M018V2_resonator_Vsweep_85_electrons",
            #r"170126_064112_M018V2_resonator_Vsweep_90_electrons",
            #r"170126_070212_M018V2_resonator_Vsweep_95_electrons",
            r"170118_132418_M018V2_resonator_Vsweep_100_electrons",
            #r"170126_072531_M018V2_resonator_Vsweep_105_electrons",
            #r"170126_075343_M018V2_resonator_Vsweep_110_electrons",
            #r"170126_082410_M018V2_resonator_Vsweep_115_electrons",
            #r"170126_085811_M018V2_resonator_Vsweep_120_electrons",
            r"170124_101926_M018V2_resonator_Vsweep_125_electrons",
            #r"170124_112414_M018V2_resonator_Vsweep_130_electrons",
            #r"170124_120809_M018V2_resonator_Vsweep_135_electrons",
            #r"170124_131448_M018V2_resonator_Vsweep_140_electrons",
            #r"170124_152925_M018V2_resonator_Vsweep_145_electrons"]
             r"170124_163706_M018V2_resonator_Vsweep_150_electrons",
            #r"170124_175432_M018V2_resonator_Vsweep_155_electrons",
            #r"170125_071720_M018V2_resonator_Vsweep_160_electrons",
            #r"170126_093412_M018V2_resonator_Vsweep_170_electrons",
             r"170126_110149_M018V2_resonator_Vsweep_180_electrons",
            # r"170126_130205_M018V2_resonator_Vsweep_190_electrons",
             r"170118_145233_M018V2_resonator_Vsweep_200_electrons",
            # r"170126_211735_M018V2_resonator_Vsweep_225_electrons",
             r"170127_005927_M018V2_resonator_Vsweep_250_electrons",
            # r"170127_054958_M018V2_resonator_Vsweep_275_electrons",
             r"170127_123421_M018V2_resonator_Vsweep_300_electrons"]

h = 0.74
screening_length = 2 * h * 1E-6
box_y_length = 40E-6
noof_electrons = list()

# Load the data from the dsp file:
path = r'/Volumes/slab/Gerwin/Electron on helium/Maxwell/M018 Yggdrasil/Greater Trap Area/V1big'
#path = r"/Users/gkoolstra/Desktop/Temporary electron optimization"
potential_file = r"DCBiasPotential.dsp"
efield_file = r"DifferentialModeEx.dsp"
elements, nodes, elem_solution, bounding_box = import_data.load_dsp(os.path.join(path, potential_file))

xdata, ydata, Udata = interpolate_slow.prepare_for_interpolation(elements, nodes, elem_solution)
x0 = -2.0  # Starting point for y
k = 151  # This defines the sampling
xeval = anneal.construct_symmetric_y(x0, k)

xinterp, yinterp, Uinterp = interpolate_slow.evaluate_on_grid(xdata, ydata, Udata, xeval=xeval,
                                                              yeval=h, clim=(0.00, 1.00), plot_axes='xy',
                                                              linestyle='None',
                                                              cmap=plt.cm.viridis, plot_data=False,
                                                              **common.plot_opt("darkorange", msize=6))

# Mirror around the y-axis
xsize = len(Uinterp[0])
Uinterp_symmetric = np.zeros(2 * xsize)
Uinterp_symmetric[:xsize] = Uinterp[0]
Uinterp_symmetric[xsize:] = Uinterp[0][::-1]

x_symmetric = np.zeros(2 * xsize)
x_symmetric[:xsize] = xinterp[0]
x_symmetric[xsize:] = -xinterp[0][::-1]

elements, nodes, elem_solution, bounding_box = import_data.load_dsp(os.path.join(path, efield_file))

xdata, ydata, Udata = interpolate_slow.prepare_for_interpolation(elements, nodes, elem_solution)

xinterp, yinterp, Uinterp = interpolate_slow.evaluate_on_grid(xdata, ydata, Udata, xeval=x_symmetric,
                                                              yeval=h, clim=(-4E5, 4E5), plot_axes='xy',
                                                              linestyle='None',
                                                              cmap=plt.cm.viridis, plot_data=False,
                                                              **common.plot_opt("darkorange", msize=6))

plt.close('all')

for s, sub_dir in enumerate(sub_dirs):
    print("Loading from %s..."%sub_dir)

    with h5py.File(os.path.join(os.path.join(save_path, sub_dir), "Results.h5"), "r") as f:
        k = 0
        converged = list()

        for step in tqdm(f.keys()):
            if "step" in step:
                electron_ri = f[step + "/electron_final_coordinates"][()]

                if k == 0:
                    electron_positions = electron_ri
                else:
                    electron_positions = np.vstack((electron_positions, electron_ri))

                converged.append(f[step + "/solution_converged"][()])
                k += 1

        Vres = f["Vres"][()]

    N_electrons = np.int(np.shape(electron_positions)[1] / 2)
    cavity_f = list()
    electron_f = list()
    ten_strongest_mode_numbers = np.zeros((len(Vres), 10), dtype=np.int)
    Eigenvalues = np.zeros((len(Vres), N_electrons + 1))
    Eigenvectors = np.zeros((N_electrons + 1, N_electrons + 1, len(Vres)))

    for k, resV in tqdm(enumerate(Vres)):
        LHS = setup_eom(electron_positions=electron_positions[k, :], Vres=resV, x_symmetric=x_symmetric,
                        Uinterp_symmetric=Uinterp_symmetric, Uinterp=Uinterp,
                        screening_length=screening_length)
        evals, evecs = solve_eom(LHS)

        Eigenvalues[k, :] = evals.flatten()
        Eigenvectors[:, :, k] = evecs

        cavity_contributions = evecs[0, :]
        ranked_cavity_contributions = np.argsort(np.abs(cavity_contributions))

        cavity_mode_idx = ranked_cavity_contributions[-1]
        electron_evs = np.delete(evals, cavity_mode_idx)

        ten_strongest_mode_numbers[k, :] = ranked_cavity_contributions[-10:]

        # Now save the electron mode frequency of the mode that couples second strongest to the cavity
        electron_f.append(np.sqrt(evals[ranked_cavity_contributions[-2]]) / (2 * np.pi * 1E9))
        # And save the cavity mode frequency
        cavity_f.append(np.sqrt(evals[cavity_mode_idx]) / (2 * np.pi))

    r = get_resonator_constants()
    delta_f = np.array(cavity_f) - r['f0']
    scaled_cavity_frequency = r['f0'] + delta_f * r['l'] / box_y_length

    print("The resonator is %.0f um long, which is %.1f box lengths" % (r['l'] * 1E6, r['l'] / box_y_length))
    print("On the resonator there are %d electrons" % (N_electrons * r['l'] / box_y_length))

    if s == 0:
        expt_frequency = scaled_cavity_frequency
    else:
        expt_frequency = np.vstack((expt_frequency, scaled_cavity_frequency))

    noof_electrons.append(N_electrons)

    # Here we plot the scaled cavity frequency shift vs. the resonator voltage.
    # fig6 = plt.figure(figsize=(6., 4.))
    # plt.plot(Vres, scaled_cavity_frequency / 1E9, 'o', **common.plot_opt('red', msize=4))
    # plt.ylabel("Scaled cavity frequency $\omega_0/2\pi$ (GHz)")
    # plt.xlabel("Resonator voltage (V)")

    # if save:
    #     common.save_figure(fig6, save_path=os.path.join(save_path, sub_dir))


color.cycle_cmap(len(sub_dirs), cmap=plt.cm.Spectral)
fig=plt.figure(figsize=(8.,4.))
common.configure_axes(12)
for r in range(len(sub_dirs)):
    plt.plot(Vres, expt_frequency[r, :] / 1E9, '.-', label="N = %d" % noof_electrons[r])

plt.ylabel("Scaled cavity frequency $\omega_0/2\pi$ (GHz)")
plt.xlabel("Resonator voltage (V)")
plt.ylim(7.95, 8.0)
plt.legend(loc=0, prop={'size' : 10})

if save:
    common.save_figure(fig, save_path=os.path.join(save_path, sub_dirs[-1]))

plt.show()

