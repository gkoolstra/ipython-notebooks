import platform, os, sys, time, multiprocessing
import numpy as np
import h5py
from tqdm import tqdm

if 'Windows' in platform.system():
    sys.path.append(r'C:\Users\slab\Documents\Code')
    sys.path.append(r'D:\BEMPP_shared\Modules')
else:
    sys.path.append("/Users/gkoolstra/Documents/Code")

from Common import common
from TrapAnalysis import artificial_anneal as anneal

data_path = "/Users/gkoolstra/Desktop/Temporary electron optimization/Data/170206_213316_M018V3_resonator_Vsweep_50_electrons"
box_length = 40E-6
create_movie = True
create_images = True

# Load file
with h5py.File(os.path.join(data_path, "Results.h5"), "r") as f:
    x = f["xpoints"][()]
    y = f["ypoints"][()]
    resV = f["Vres"][()]
    V = f["electrostatic_potential"][()]

    k = 0
    for step in f.keys():
        if "step" in step:
            valid_solution = f[step + "/solution_converged"][()]
            electron_ri = f[step + "/electron_final_coordinates"][()]
            if not(valid_solution):
                electron_ri = np.zeros(len(electron_ri))

            if k == 0:
                Ri = electron_ri
            else:
                Ri = np.vstack((Ri, electron_ri))

            k += 1

post_process = anneal.PostProcess(save_path=os.path.join(data_path, "Figures"),
                                  res_pinw=0.9, edge_gapw=0.5, center_gapw=0.7, box_length=box_length * 1E6)

convergence_monitor = anneal.ConvergenceMonitor(None, None, None, Uext=None, xext=None, yext=None, verbose=True,
                                                eps=1E-12, save_path=os.path.join(data_path, "Figures"),
                                                figsize=(12.,3.), coordinate_transformation=None)

def single_thread(index, xpoints, ypoints, V, Vres):
    resonator_helper = anneal.ResonatorSolver(xpoints, V * Vres, box_length=box_length, smoothing=0.0)

    post_process.save_snapshot(Ri[index,:], xext=xpoints, yext=ypoints, Uext=resonator_helper.V,
                     figsize=(4, 6), common=common, title="Vres = %.2f V" % Vres,
                     clim=(-0.75 * max(resV), 0))

if create_images:
    for index in tqdm(range(np.shape(Ri)[0])):
        single_thread(index, x, y, V, resV[index])

N_electrons = np.shape(Ri)[1]/2

if create_movie:
    # Create a movie
    convergence_monitor.create_movie(fps=10,
                                     filenames_in=time.strftime("%Y%m%d") + "_figure_%05d.png",
                                     filename_out="resonator_%d_electrons.mp4" % (N_electrons))

    # Move the file from the Figures folder to the sub folder
    os.rename(os.path.join(data_path, "Figures/resonator_%d_electrons.mp4" % (N_electrons)),
              os.path.join(data_path, "resonator_%d_electrons.mp4" % (N_electrons)))

# pool = multiprocessing.Pool()
# tasks = [(index, x, y, V, resV[index],) for index in range(np.shape(Ri)[0])]
# results = [pool.apply_async(single_thread, t) for t in tasks]
# for result in results:
#     r = result.get()