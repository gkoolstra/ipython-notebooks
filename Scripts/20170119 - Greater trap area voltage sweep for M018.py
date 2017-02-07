from matplotlib import pyplot as plt
import os, time, h5py
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from termcolor import cprint
from Common import common
from TrapAnalysis import trap_analysis, artificial_anneal as anneal

use_gradient = True
gradient_tolerance = 1E1
epsilon = 1E-10
trap_annealing_steps = [1.0] * 10

Vres = np.arange(1.00, 0.04, -0.01)
Vtrap = -0.25
Vrg = 0.00
Vcg = 0.00
Vtg = -0.05

N_electrons = 50
N_rows = 50
row_spacing = 0.20E-6
N_cols = 1
col_spacing = 0.20E-6

inserted_res_length = 15 # units are in um
electrode_names = ['resonator', 'trap', 'res_guard', 'ctr_guard', 'trap_guard']

#electron_initial_positions = anneal.setup_initial_condition(N_electrons, (3.5E-6, 10.5E-6), (-1E-6, 1E-6), 7E-6, 0E-6, dx=None, dy=None)
electron_initial_positions = anneal.get_rectangular_initial_condition(N_electrons, N_rows=N_rows, N_cols=N_cols,
                                                                      x0=inserted_res_length*1E-6, y0=0.0E-6, dx=0.40E-6)
x_trap_init, y_trap_init = anneal.r2xy(electron_initial_positions)

simulation_name = "Greater_Trap_Area"
save_path = r"/Users/gkoolstra/Desktop/Temporary electron optimization/Data"
#save_path = r"/Volumes/slab/Gerwin/Electron on helium/Electron optimization/Realistic potential/Greater Trap Area"
sub_dir = time.strftime("%y%m%d_%H%M%S_{}".format(simulation_name))
save = True

# Evaluate all files in the following range.
xeval = np.linspace(-4, 12.5, 401)
yeval = anneal.construct_symmetric_y(-4.5, 151)

master_path = r"/Users/gkoolstra/Desktop/Temporary electron optimization/M018V2"
#master_path = r"/Volumes/slab/Gerwin/Electron on helium/Maxwell/M018 Yggdrasil/Greater Trap Area/V1"
x_eval, y_eval, output = anneal.load_data(master_path, xeval=xeval, yeval=yeval, mirror_y=True,
                                          extend_resonator=False, insert_resonator=True, do_plot=False,
                                          inserted_res_length=inserted_res_length)
# Note: x_eval and y_eval are 2D arrays that contain the x and y coordinates at which the potentials are evaluated

conv_mon_save_path = os.path.join(save_path, sub_dir, "Figures")

if save:
    # Create the directories
    os.mkdir(os.path.join(save_path, sub_dir))
    time.sleep(1)
    os.mkdir(os.path.join(save_path, sub_dir, "Figures"))

    # Save the data to a single file
    f = h5py.File(os.path.join(os.path.join(save_path, sub_dir), "Results.h5"), "w")
    f.create_dataset("use_gradient", data=use_gradient)
    f.create_dataset("gradient_tolerance", data=gradient_tolerance)
    for el_number, el_name in enumerate(electrode_names):
        f.create_dataset(el_name, data=-output[el_number]['V'].T)
    f.create_dataset("xpoints", data=x_eval)
    f.create_dataset("ypoints", data=y_eval)
    f.create_dataset("Vres", data=Vres)
    f.create_dataset("Vtrap", data=Vtrap)
    f.create_dataset("Vrg", data=Vrg)
    f.create_dataset("Vcg", data=Vcg)
    f.create_dataset("Vtg", data=Vtg)

# Take a slice through the middle, at y = 0 to check if the insertion went well and doesn't produce weird gradients.
if 0:
    U = output[0]['V'].T
    plt.figure(figsize=(10.,3.))
    common.configure_axes(13)
    plt.plot(x_eval[np.int(len(y_eval)/2), :] * 1E6, -U[np.int(len(y_eval)/2), :], '-k')
    plt.xlabel("$x$ ($\mu$m)")
    plt.ylabel("Potential energy (eV)")

# This is where the actual solving starts...
t = trap_analysis.TrapSolver()
c = trap_analysis.get_constants()

x_eval, y_eval, cropped_potentials = t.crop_potentials(output, ydomain=None, xdomain=None)

# Determine what electrode will be swept:
sweep_points = np.inf
for SweepIdx, Vpoint in enumerate([Vres, Vtrap, Vrg, Vcg, Vtg]):
    if isinstance(Vpoint, np.ndarray):
        sweep_points = Vpoint
        break

if not(isinstance(sweep_points, np.ndarray)):
    sweep_points = [Vres]
    SweepIdx = 0

cprint("Sweeping %s from %.2f V to %.2f V" % (electrode_names[SweepIdx], sweep_points[0], sweep_points[-1]),
       "green")

#cprint("Sweeping %s from %.2f V to %.2f V" % (electrode_names[SweepIdx], sweep_points[0], sweep_points[-1]), "green")

for k, s in tqdm(enumerate(sweep_points)):
    coefficients = np.array([Vres, Vtrap, Vrg, Vcg, Vtg])
    coefficients[SweepIdx] = s

    combined_potential = t.get_combined_potential(cropped_potentials, coefficients)
    # Note: x_eval and y_eval are 1D arrays that contain the x and y coordinates at which the potentials are evaluated
    # Units of x_eval and y_eval are um

    CMS = anneal.TrapAreaSolver(x_eval * 1E-6, y_eval * 1E-6, -combined_potential.T,
                                spline_order_x=3, spline_order_y=3, smoothing=0.01,
                                include_screening=True, screening_length=2*0.74E-6)

    X_eval, Y_eval = np.meshgrid(x_eval * 1E-6, y_eval * 1E-6)

    # Solve for the electron positions in the trap area!
    ConvMon = anneal.ConvergenceMonitor(Uopt=CMS.Vtotal, grad_Uopt=CMS.grad_total, N=10,
                                        Uext=CMS.V,
                                        xext=xeval * 1E-6, yext=yeval * 1E-6, verbose=False, eps=epsilon,
                                        save_path=conv_mon_save_path)
    ConvMon.figsize = (8., 2.)

    trap_minimizer_options = {'jac': CMS.grad_total,
                              'options': {'disp': False, 'gtol': gradient_tolerance, 'eps': epsilon},
                              'callback': ConvMon.monitor_convergence}

    res = minimize(CMS.Vtotal, electron_initial_positions, method='CG', **trap_minimizer_options)

    if res['status'] > 0:
        cprint("WARNING: Initial minimization for Trap did not converge!", "red")
        print("Final L-inf norm of gradient = %.2f eV/m" % (np.amax(res['jac'])))
        best_res = res
    else:
        cprint("SUCCESS: Initial minimization for Trap converged!", "green")
        # This maps the electron positions within the simulation domain
        cprint("Perturbing solution %d times at %.2f K..." \
               % (len(trap_annealing_steps), trap_annealing_steps[0]), "green")
        best_res = CMS.parallel_perturb_and_solve(CMS.Vtotal, len(trap_annealing_steps),
                                                  trap_annealing_steps[0], res, trap_minimizer_options)

        if (best_res['x'] == res['x']).all():
            cprint("Solution data unchanged after perturbing", "green")

    #t0 = time.time()
    PP = anneal.PostProcess(save_path=conv_mon_save_path)
    PP.save_snapshot(best_res['x'], xext=x_eval*1E-6, yext=y_eval*1E-6, Uext=CMS.V,
                     figsize=(12., 3.), common=common, title="Vres = %.2f V" % coefficients[0],
                     clim=(-0.75 * max(sweep_points), 0), draw_resonator_pins=False)

    f.create_dataset("step_%04d/electron_final_coordinates" % k, data=best_res['x'])
    f.create_dataset("step_%04d/electron_initial_coordinates" % k, data=electron_initial_positions)
    f.create_dataset("step_%04d/solution_converged" % k, data=True if best_res['status'] == 0 else False)
    f.create_dataset("step_%04d/energy" % k, data=best_res['fun'])
    f.create_dataset("step_%04d/jacobian" % k, data=best_res['jac'])
    f.create_dataset("step_%04d/electrons_in_trap" % k, data=PP.get_trapped_electrons(best_res['x']))
    #print("Saved data for V = %.2f V in %.2f seconds" % (Vres, time.time() - t0))

    # Use the solution from the current time step as the initial condition for the next timestep!
    electron_initial_positions = best_res['x']

trap_electrons_x, trap_electrons_y = anneal.r2xy(best_res['x'])

# Plot the resonator and trap electron configuration
fig2 = plt.figure(figsize=(12, 3))
common.configure_axes(12)
plt.pcolormesh(x_eval, y_eval, CMS.V(X_eval, Y_eval), cmap=plt.cm.Spectral_r, vmax=0.0, vmin=-0.75 * Vres)
plt.plot(x_trap_init * 1E6, y_trap_init * 1E6, 'o', color='mediumpurple', alpha=0.5)
plt.plot(trap_electrons_x * 1E6, trap_electrons_y * 1E6, 'o', color='violet', alpha=1.0)
for k, name in enumerate(["$V_\mathrm{res}$", "$V_\mathrm{trap}$", "$V_\mathrm{rg}$", "$V_\mathrm{cg}$", "$V_\mathrm{tg}$"]):
    Vs = [Vres, Vtrap, Vrg, Vcg, Vtg]
    plt.text(8, 4.15 - 0.6 * k, name+" = %.2f V" % Vs[k], fontdict={'size': 10, 'color': 'black', 'ha' : 'right'})

if best_res['status'] > 0:
    plt.text(2, -2, "Minimization did not converge", fontdict={"size" : 10})

plt.xlabel("$x$ ($\mu$m)")
plt.ylabel("$y$ ($\mu$m)")
plt.title("%d electrons" % (N_electrons))
plt.colorbar()
plt.xlim(np.min(x_eval), np.max(x_eval))
plt.ylim(np.min(y_eval), np.max(y_eval))

#common.save_figure(fig2, save_path=save_path)

num_unbounded_electrons = anneal.check_unbounded_electrons(best_res['x'],
                                                           xdomain=(np.min(x_eval) * 1E-6, np.max(x_eval) * 1E-6),
                                                           ydomain=(np.min(y_eval) * 1E-6, np.max(y_eval) * 1E-6))

print("Number of unbounded electrons = %d" % num_unbounded_electrons)

plt.show()