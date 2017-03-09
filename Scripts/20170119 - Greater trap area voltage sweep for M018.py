import os, time, h5py, platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from termcolor import cprint
from Common import common
from TrapAnalysis import trap_analysis, artificial_anneal as anneal

simulation_name = "M018V6_Greater_Trap_Area_V6.3"
include_screening = True
helium_thickness = 0.75E-6
screening_length = 2 * helium_thickness
inspect_potentials = False
use_gradient = True
gradient_tolerance = 1E1
epsilon = 1E-10
trap_annealing_steps = []
max_x_displacement = 0.05E-6
max_y_displacement = 0.05E-6
remove_unbound_electrons = False
remove_bounds = (-np.inf, -3E-6)
show_final_result = False
create_movie = False

# Set any of these to None if you can only apply GND
# Only the first electrode in this list that is set to an array instead of float will be swept.
Vres = 1.00 #np.arange(2.00, 0.04, -0.01)
Vtrap = 1.00 #np.arange(1.00, 1.50, +0.01)
Vrg = np.arange(0.34, -0.50, -0.01) #0.10 * Vres
Vtg = -0.70
Vcg = None

N_electrons = 160
N_rows = N_electrons
row_spacing = 0.10E-6
N_cols = 1
col_spacing = 0.10E-6
box_length = 20E-6 #This is the box length from the simulation in Maxwell.

inserted_res_length = 80 # This is the length to be inserted in the Maxwell potential; units are in um
electrode_names = ['resonator', 'trap', 'res_guard', 'trap_guard']
if Vcg is not None:
    electrode_names.insert(3, 'ctr_guard')

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

simulation_name += "_%s_sweep" % (electrode_names[SweepIdx])

electron_initial_positions = anneal.get_rectangular_initial_condition(N_electrons, N_rows=N_rows, N_cols=N_cols,
                                                                      x0=(inserted_res_length*1E-6 + box_length)/2., y0=0.0E-6, dx=0.40E-6)
x_trap_init, y_trap_init = anneal.r2xy(electron_initial_positions)

if platform.system() == 'Windows':
    save_path = r"S:\Gerwin\Electron on helium\Electron optimization\Realistic potential\Single electron loading"
elif platform.system() == 'Linux':
    save_path = r"/mnt/slab/Gerwin/Electron on helium/Electron optimization/Realistic potential/Single electron loading"
else:
    save_path = r"/Volumes/slab/Gerwin/Electron on helium/Electron optimization/Realistic potential/Single electron loading"

sub_dir = time.strftime("%y%m%d_%H%M%S_{}".format(simulation_name))
save = True

# Evaluate all files in the following range.
xeval = np.linspace(-4.0, box_length*1E6, 751)
yeval = anneal.construct_symmetric_y(-4.0, 201)

dx = np.diff(xeval)[0]*1E-6
dy = np.diff(yeval)[0]*1E-6

if platform.system() == 'Windows':
    master_path = r"S:\Gerwin\Electron on helium\Maxwell\M018 Yggdrasil\M018V6\V6.3"
elif platform.system() == 'Linux':
    master_path = r"/mnt/slab/Gerwin/Electron on helium/Maxwell/M018 Yggdrasil/M018V6/V6.3"
else:
    master_path = r"/Volumes/slab/Gerwin/Electron on helium/Maxwell/M018 Yggdrasil/M018V6/V6.3"

x_eval, y_eval, output = anneal.load_data(master_path, xeval=xeval, yeval=yeval, mirror_y=True,
                                          extend_resonator=False, insert_resonator=True, do_plot=inspect_potentials,
                                          inserted_res_length=inserted_res_length, smoothen_xy=(0.40E-6, 2*dy))

if inspect_potentials:
    plt.show()

# Note: x_eval and y_eval are 2D arrays that contain the x and y coordinates at which the potentials are evaluated
conv_mon_save_path = os.path.join(save_path, sub_dir, "Figures")

if save:
    # Create the directories
    os.mkdir(os.path.join(save_path, sub_dir))
    time.sleep(1)
    os.mkdir(os.path.join(save_path, sub_dir, "Figures"))
    time.sleep(1)
    os.mkdir(os.path.join(save_path, sub_dir, "2D slice"))
    time.sleep(1)

    f = h5py.File(os.path.join(os.path.join(save_path, sub_dir), "Results.h5"), "w")
    # Save the data to a single file
    f.create_dataset("use_gradient", data=use_gradient)
    f.create_dataset("gradient_tolerance", data=gradient_tolerance)
    f.create_dataset("include_screening", data=include_screening)
    f.create_dataset("screening_length", data=screening_length)
    f.create_dataset("inserted_res_length", data=inserted_res_length*1E-6)
    for el_number, el_name in enumerate(electrode_names):
        f.create_dataset(el_name, data=-output[el_number]['V'].T)
    f.create_dataset("xpoints", data=x_eval)
    f.create_dataset("ypoints", data=y_eval)
    f.create_dataset("Vres", data=Vres)
    f.create_dataset("Vtrap", data=Vtrap)
    f.create_dataset("Vrg", data=Vrg)
    if Vcg is not None:
        f.create_dataset("Vcg", data=Vcg)
    f.create_dataset("Vtg", data=Vtg)
    f.close()
    time.sleep(1)

# Take a slice through the middle, at y = 0 to check if the insertion went well and doesn't produce weird gradients.
if 1:
    U = output[0]['V'].T
    fig = plt.figure(figsize=(10.,3.))
    common.configure_axes(13)
    plt.plot(x_eval[np.int(len(y_eval)/2), :] * 1E6, -U[np.int(len(y_eval)/2), :], '-k')
    plt.plot(x_trap_init * 1E6, np.zeros(len(x_trap_init)), 'o', color='mediumpurple', alpha=0.5)
    plt.xlabel("$x$ ($\mu$m)")
    plt.ylabel("Potential energy (eV)")

    if save:
        common.save_figure(fig, save_path=os.path.join(save_path, sub_dir))

    plt.show()

# This is where the actual solving starts...
t = trap_analysis.TrapSolver()
c = trap_analysis.get_constants()

x_eval, y_eval, cropped_potentials = t.crop_potentials(output, ydomain=None, xdomain=None)

for k, s in tqdm(enumerate(sweep_points)):

    coefficients = np.array([Vres, Vtrap, Vrg, Vcg, Vtg])
    coefficients[SweepIdx] = s
    if Vcg is None:
        coefficients = np.delete(coefficients, 3)

    combined_potential = t.get_combined_potential(cropped_potentials, coefficients)
    # Note: x_eval and y_eval are 1D arrays that contain the x and y coordinates at which the potentials are evaluated
    # Units of x_eval and y_eval are um

    CMS = anneal.TrapAreaSolver(x_eval * 1E-6, y_eval * 1E-6, -combined_potential.T,
                                spline_order_x=3, spline_order_y=3, smoothing=0.01,
                                include_screening=include_screening, screening_length=screening_length)

    X_eval, Y_eval = np.meshgrid(x_eval * 1E-6, y_eval * 1E-6)

    # Solve for the electron positions in the trap area!
    ConvMon = anneal.ConvergenceMonitor(Uopt=CMS.Vtotal, grad_Uopt=CMS.grad_total, N=1,
                                        Uext=CMS.V,
                                        xext=xeval * 1E-6, yext=yeval * 1E-6, verbose=False, eps=epsilon,
                                        save_path=conv_mon_save_path)

    ConvMon.figsize = (8., 2.)

    trap_minimizer_options = {'method' : 'L-BFGS-B',
                              'jac': CMS.grad_total,
                              'options': {'disp': False, 'gtol': gradient_tolerance, 'eps': epsilon},
                              'callback': None}

    # We save the initial Jacobian of the system for purposes.
    initial_jacobian = CMS.grad_total(electron_initial_positions)
    res = minimize(CMS.Vtotal, electron_initial_positions, **trap_minimizer_options)

    while res['status'] > 0:
        # Try removing unbounded electrons and restart the minimization
        if remove_unbound_electrons:
            # Remove any electrons that are to the left of the trap
            best_x, best_y = anneal.r2xy(res['x'])
            idxs = np.where(np.logical_and(best_x > remove_bounds[0], best_x < remove_bounds[1]))[0]
            best_x = np.delete(best_x, idxs)
            best_y = np.delete(best_y, idxs)
            # Use the solution from the current time step as the initial condition for the next timestep!
            electron_initial_positions = anneal.xy2r(best_x, best_y)
            if len(best_x) < len(res['x'][::2]):
                print("%d/%d unbounded electrons removed. %d electrons remain." % (np.int(len(res['x'][::2]) - len(best_x)), len(res['x'][::2]), len(best_x)))
            res = minimize(CMS.Vtotal, electron_initial_positions, **trap_minimizer_options)
        else:
            best_x, best_y = anneal.r2xy(res['x'])
            idxs = np.union1d(np.where(best_x < -2E-6)[0], np.where(np.abs(best_y) > 2E-6)[0])
            if len(idxs) > 0:
                print("Following electrons are outside the simulation domain")
                for i in idxs:
                    print("(x,y) = (%.3f, %.3f) um"%(best_x[i]*1E6, best_y[i]*1E6))
            # To skip the infinite while loop.
            break

    if res['status'] > 0:
        cprint("WARNING: Initial minimization for Trap did not converge!", "red")
        print("Final L-inf norm of gradient = %.2f eV/m" % (np.amax(res['jac'])))
        if k == 0:
            cprint("Please check your initial condition, are all electrons confined in the simulation area?", "red")
            break
        best_res = res
    if len(trap_annealing_steps) > 0:
        # cprint("SUCCESS: Initial minimization for Trap converged!", "green")
        # This maps the electron positions within the simulation domain
        cprint("Perturbing solution %d times at %.2f K. (dx,dy) ~ (%.3f, %.3f) um..." \
              % (len(trap_annealing_steps), trap_annealing_steps[0],
                 np.mean(CMS.thermal_kick_x(res['x'][::2], res['x'][1::2], trap_annealing_steps[0], maximum_dx=max_x_displacement))*1E6,
                 np.mean(CMS.thermal_kick_y(res['x'][::2], res['x'][1::2], trap_annealing_steps[0], maximum_dy=max_y_displacement))*1E6),
               "white")
        best_res = CMS.perturb_and_solve(CMS.Vtotal, len(trap_annealing_steps), trap_annealing_steps[0],
                                         res, maximum_dx=max_x_displacement, maximum_dy=max_y_displacement, **trap_minimizer_options)
        #best_res = CMS.parallel_perturb_and_solve(CMS.Vtotal, len(trap_annealing_steps),
        #                                          trap_annealing_steps[0], res, trap_minimizer_options,
        #                                          maximum_dx=max_x_displacement, maximum_dy=max_y_displacement)
    else:
        best_res = res

    if 0:
        electron_pos = best_res['x'][::2] * 1E6

        fig = plt.figure(figsize=(7., 3.))
        common.configure_axes(13)
        plt.plot(x_eval[x_eval < 6], CMS.V(x_eval[x_eval < 6]*1E-6, 0), '-', color='orange')
        plt.plot(electron_pos[electron_pos < 6], CMS.V(electron_pos[electron_pos < 6] * 1E-6, 0) + 0.005, 'o', color='cornflowerblue')
        plt.xlabel("$x$ ($\mu$m)")
        plt.ylabel("Potential energy (eV)")
        plt.title("%s = %.2f V" % (electrode_names[SweepIdx], coefficients[SweepIdx]))
        plt.ylim(-0.7, -0.55)

        if save:
            common.save_figure(fig, save_path=os.path.join(save_path, sub_dir, "2D slice"))

        plt.close(fig)

    def magE(x, y):
        return np.log10(np.sqrt(CMS.dVdx(x, y)**2 + CMS.dVdy(x, y)**2))

    PP = anneal.PostProcess(save_path=conv_mon_save_path)
    x_plot = np.arange(-1E-6, +1E-6, dx)
    y_plot = y_eval[np.logical_and(y_eval > -1, y_eval < 1)]*1E-6
    PP.save_snapshot(best_res['x'], xext=x_plot, yext=y_plot, Uext=magE,
                     figsize=(6.5, 3.), common=common, title="Gradient at %s = %.3f V" % (electrode_names[SweepIdx], coefficients[SweepIdx]),
                     clim=(1.5, 6.0),
                     draw_resonator_pins=False,
                     draw_from_dxf={'filename' : os.path.join(master_path, 'all_electrodes.dxf'),
                                    'plot_options' : {'color' : 'black', 'alpha' : 0.6, 'lw' : 0.5}})

    f = h5py.File(os.path.join(os.path.join(save_path, sub_dir), "Results.h5"), "a")
    #f.create_dataset("step_%04d/x_during_simulation" % k, data=ConvMon.curr_xk)
    #|f.create_dataset("step_%04d/jac_during_simulation" % k, data=ConvMon.jac)
    f.create_dataset("step_%04d/initial_jacobian" % k, data=initial_jacobian)
    f.create_dataset("step_%04d/final_jacobian" % k, data=best_res['jac'])
    f.create_dataset("step_%04d/electron_initial_coordinates" % k, data=electron_initial_positions)
    f.create_dataset("step_%04d/electron_final_coordinates" % k, data=best_res['x'])
    f.create_dataset("step_%04d/final_energy" % k, data=best_res['fun'])
    f.create_dataset("step_%04d/solution_converged" % k, data=True if best_res['status'] == 0 else False)
    f.create_dataset("step_%04d/electrons_in_trap" % k, data=PP.get_trapped_electrons(best_res['x'], trap_area_x=(-1.2E-6, +1.2E-6)))
    f.close()

    # Use the solution from the current time step as the initial condition for the next timestep!
    electron_initial_positions = best_res['x']

trap_electrons_x, trap_electrons_y = anneal.r2xy(electron_initial_positions)

if create_movie:
    try:
        # Create a movie
        ConvMon.create_movie(fps=10,
                             filenames_in=time.strftime("%Y%m%d")+"_figure_%05d.png",
                             filename_out=simulation_name + ".mp4")

        # Move the file from the Figures folder to the sub folder
        os.rename(os.path.join(save_path, sub_dir, "Figures", simulation_name + ".mp4"),
                  os.path.join(save_path, sub_dir, simulation_name + ".mp4"))
    except:
        print("Failed making a movie!")

if show_final_result:
    # Plot the resonator and trap electron configuration
    fig2 = plt.figure(figsize=(12, 3))
    common.configure_axes(12)
    plt.pcolormesh(x_eval, y_eval, CMS.V(X_eval, Y_eval), cmap=plt.cm.Spectral_r, vmax=0.0, vmin=-0.75 * np.max(Vres))
    plt.plot(x_trap_init * 1E6, y_trap_init * 1E6, 'o', color='mediumpurple', alpha=0.5)
    plt.plot(trap_electrons_x * 1E6, trap_electrons_y * 1E6, 'o', color='violet', alpha=1.0)

    # for k, name in enumerate(["$V_\mathrm{res}$", "$V_\mathrm{trap}$", "$V_\mathrm{rg}$", "$V_\mathrm{cg}$", "$V_\mathrm{tg}$"]):
    #     Vs = [Vres, Vtrap, Vrg, Vcg, Vtg]
    #     if Vcg is not None:
    #         plt.text(8, 4.15 - 0.6 * k, name+" = %.2f V" % Vs[k], fontdict={'size': 10, 'color': 'black', 'ha' : 'right'})

    if best_res['status'] > 0:
       plt.text(2, -2, "Minimization did not converge", fontdict={"size" : 10})

    plt.xlabel("$x$ ($\mu$m)")
    plt.ylabel("$y$ ($\mu$m)")
    plt.title("%d electrons" % (N_electrons))
    plt.xlim(np.min(x_eval), np.max(x_eval))
    plt.ylim(np.min(y_eval), np.max(y_eval))

    if save:
        common.save_figure(fig2, save_path=os.path.join(save_path, sub_dir))

    num_unbounded_electrons = anneal.check_unbounded_electrons(best_res['x'],
                                                               xdomain=(np.min(x_eval) * 1E-6, np.max(x_eval) * 1E-6),
                                                               ydomain=(np.min(y_eval) * 1E-6, np.max(y_eval) * 1E-6))

    print("Number of unbounded electrons = %d" % num_unbounded_electrons)

    plt.show()