import platform, os, sys
import numpy as np
import h5py
from termcolor import cprint
from tqdm import tqdm

if platform.system() == 'Windows':
    sys.path.append(r'C:\Users\slab\Documents\Code')
    sys.path.append(r'D:\BEMPP_shared\Modules')
else:
    sys.path.append("/Users/gkoolstra/Documents/Code")

from Common import common
from TrapAnalysis import artificial_anneal as anneal

data_path = r"/Volumes/slab/Gerwin/Electron on helium/Electron optimization/Realistic potential/Single electron loading/170307_085415_M018V6_Greater_Trap_Area_res_guard_sweep"
create_movie = False
create_images = True

# Load file
with h5py.File(os.path.join(data_path, "Results.h5"), "r") as f:
    x = f["xpoints"][()]
    y = f["ypoints"][()]
    Vres = f["Vres"][()]
    Vtrap = f["Vtrap"][()]
    Vresguard = f["Vrg"][()]
    Vtrapguard = f["Vtg"][()]

    dx = np.diff(x)[0]
    dy = np.diff(y)[0]

    electrode_names = ['Resonator', 'Trap', 'Resonator guard', 'Trap guard']
    for idx, electrode in enumerate([Vres, Vtrap, Vresguard, Vtrapguard]):
        if isinstance(electrode, np.ndarray):
            swept_electrode_idx = idx
            swept_electrode_name = electrode_names[idx]
            swept_electrode = electrode
            break

    cprint("Swept electrode: %s from %.3f to %.3f V"%(swept_electrode_name, swept_electrode[0], swept_electrode[-1]), "green")

    potential_trap = f["trap"][()]
    potential_res = f["res"][()]
    potential_trapguard = f["trap_guard"][()]
    potential_resguard = f["res_guard"][()]

    include_screening = f["include_screening"][()]
    screening_length = 2 * 0.75E-6#f["screening_length"][()]

    k = 0
    for step in tqdm(f.keys()):
        if "step" in step:
            coefficients = [Vres, Vtrap, Vresguard, Vtrapguard]
            potentials = [potential_res, potential_trap, potential_resguard, potential_trapguard]
            coefficients[swept_electrode_idx] = swept_electrode

            for i, c in enumerate(coefficients):
                if i == 0:
                    potential_data = c * potentials[i]
                else:
                    potential_data += c * potentials[i]

            PP = anneal.PostProcess(save_path=os.path.join(data_path, "Figures"))

            valid_solution = f[step + "/solution_converged"][()]
            electron_ri = f[step + "/electron_final_coordinates"][()]

            CMS = anneal.TrapAreaSolver(x[0,:], y[:,0], potential_data, spline_order_x=3, spline_order_y=3,
                                        smoothing=0.00,
                                        include_screening=include_screening, screening_length=screening_length)

            PP.save_snapshot(electron_ri, xext=x, yext=y, Uext=CMS.V,
                             figsize=(6.5, 3.), common=common,
                             title="%s = %.2f V" % (electrode_names[swept_electrode_idx], coefficients[swept_electrode_idx]),
                             clim=(-0.75 * Vres, 0),
                             # clim=(-0.75 * max(sweep_points), 0),
                             draw_resonator_pins=False,
                             draw_from_dxf={'filename': os.path.join(data_path, 'all_electrodes.dxf'),
                                            'plot_options': {'color': 'black', 'alpha': 0.6, 'lw': 0.5}})