from matplotlib import pyplot as plt
import platform, os, sys, time, scipy, h5py
import numpy as np
from scipy.optimize import minimize, approx_fprime
from termcolor import cprint
from Common import common

from TrapAnalysis import trap_analysis, artificial_anneal as anneal

use_gradient = True
gradient_tolerance = 1E1
epsilon = 1E-12

N_resonator_electrons = 150
N_trap_electrons = np.int(11 / 50. * N_resonator_electrons)
resonator_box_length = 40E-6
resonator_annealing_steps = [1.0] * 10
trap_annealing_steps = [1.0] * 10

Vres = 0.15
Vtrap = 0.15
Vrg = 0.00
Vcg = 0.00
Vtg = -0.20

simulation_name = "Combined_Model"
save_path = r"/Users/gkoolstra/Desktop/Electron optimization/Realistic potential/Combined Model"
sub_dir = time.strftime("%y%m%d_%H%M%S_{}".format(simulation_name))
save = False

# Evaluate all files in the following range.
xeval = np.linspace(-4, 9.5, 501)
yeval = anneal.construct_symmetric_y(-3.0, 121)

master_path = r"/Users/gkoolstra/Desktop/Electron optimization/Realistic potential/Potentials/V6 - Greater Trap Area"
x_eval, y_eval, output = anneal.load_data(master_path, xeval=xeval, yeval=yeval, mirror_y=True,
                                          extend_resonator=True, do_plot=False)
# Note: x_eval and y_eval are 2D arrays that contain the x and y coordinates at which the potentials are evaluated

# Use resonator potential to solve for equilibrium position of resonator electrons
U_resonator = -Vres * output[0]['V'][-1, :]

# Solve resonator potential
RS = anneal.ResonatorSolver(y_eval[:, -1], U_resonator, efield_data=None,
                            box_length=resonator_box_length, spline_order_x=3, smoothing=0)

res_initial_condition = anneal.setup_initial_condition(N_resonator_electrons,
                                                       (-2.5E-6, 2.5E-6),
                                                       (-0.75 * resonator_box_length / 2.,
                                                        +0.75 * resonator_box_length / 2.),
                                                       0E-6, 0E-6)

x_init, y_init = anneal.r2xy(res_initial_condition)

if use_gradient:
    jac = RS.grad_total
    grad_Uopt = RS.grad_total
else:
    jac = None
    grad_Uopt = lambda x: approx_fprime(x, RS.Vtotal, epsilon)

x_box = np.linspace(-3E-6, 3E-6, 501)
y_box = np.linspace(-resonator_box_length / 2, resonator_box_length / 2, 11)
X_box, Y_box = np.meshgrid(x_box, y_box)

conv_mon_save_path = os.path.join(save_path, sub_dir, "Figures")

ConvMon = anneal.ConvergenceMonitor(Uopt=RS.Vtotal, grad_Uopt=RS.grad_total, N=10,
                                    Uext=RS.V,
                                    xext=x_box, yext=y_box, verbose=False, eps=epsilon,
                                    save_path=conv_mon_save_path, figsize=(4, 6),
                                    coordinate_transformation=RS.coordinate_transformation)

res_minimizer_options = {'jac': jac,
                         'options': {'disp': False, 'gtol': gradient_tolerance, 'eps': epsilon},
                         'callback': ConvMon.monitor_convergence}

res = minimize(RS.Vtotal, res_initial_condition, method='CG', **res_minimizer_options)

if res['status'] > 0:
    cprint("WARNING: Initial minimization for Resonator did not converge!", "red")
    print("Final L-inf norm of gradient = %.2f eV/m" % (np.amax(res['jac'])))
    best_res = res
    best_res['x'] = RS.coordinate_transformation(best_res['x'])
if 1:
    cprint("SUCCESS: Initial minimization for Resonator converged!", "green")
    # This maps the electron positions within the simulation domain
    res['x'] = RS.coordinate_transformation(res['x'])

    cprint("Perturbing solution %d times at %.2f K" \
           % (len(resonator_annealing_steps), resonator_annealing_steps[0]), "green")
    best_res = RS.perturb_and_solve(RS.Vtotal, len(resonator_annealing_steps),
                                    resonator_annealing_steps[0], res, **res_minimizer_options)
    best_res['x'] = RS.coordinate_transformation(best_res['x'])

    if (best_res['x'] == res['x']).all():
        cprint("Solution data unchanged after perturbing!", "green")

res_electrons_y, res_electrons_x = anneal.r2xy(best_res['x'])

# For the next part we will need to transform the coordinates of the resonator electrons
# to the boundary of the resonator. We add half of the box length and move it to the right side of the trap area box.
res_right_x = np.max(x_eval)
res_electrons_x += resonator_box_length / 2. + res_right_x - 1.5E-6

print("Switching to greater trap area...")
t = trap_analysis.TrapSolver()
c = trap_analysis.get_constants()

x_eval, y_eval, cropped_potentials = t.crop_potentials(output, ydomain=None, xdomain=None)
coefficients = np.array([Vres, Vtrap, Vrg, Vcg, Vtg])
combined_potential = t.get_combined_potential(cropped_potentials, coefficients)
# Note: x_eval and y_eval are 1D arrays that contain the x and y coordinates at which the potentials are evaluated
# Units of x_eval and y_eval are um

CMS = anneal.CombinedModelSolver(x_eval * 1E-6, y_eval * 1E-6, -combined_potential.T,
                                 anneal.xy2r(res_electrons_x, res_electrons_y),
                                 spline_order_x=3, spline_order_y=3, smoothing=0.01)

X_eval, Y_eval = np.meshgrid(x_eval * 1E-6, y_eval * 1E-6)

if 0:
    # Plot the resonator electron configuration
    fig1 = plt.figure(figsize=(9.5, 2.5))
    common.configure_axes(12)
    plt.pcolormesh(x_eval, y_eval, CMS.V(X_eval, Y_eval), cmap=plt.cm.Spectral_r, vmax=0.0, vmin=-0.75 * Vres)
    plt.pcolormesh((y_box + res_right_x + resonator_box_length / 2.) * 1E6, x_box * 1E6, RS.V(X_box, Y_box).T,
                   cmap=plt.cm.Spectral_r, vmax=0.0, vmin=-0.75 * Vres)
    plt.plot((y_init + res_right_x + resonator_box_length / 2.) * 1E6, x_init * 1E6, 'o', color='palegreen', alpha=0.5)
    plt.plot(res_electrons_x * 1E6, res_electrons_y * 1E6, 'o', color='deepskyblue', alpha=1.0)
    plt.xlabel("$x$ ($\mu$m)")
    plt.ylabel("$y$ ($\mu$m)")
    plt.title("Resonator configuration")
    plt.colorbar()
    plt.xlim(np.min(x_eval), np.max(x_eval) + resonator_box_length * 1E6)

VBG_mat = np.zeros((len(y_eval), len(x_eval)))
for j, X in enumerate(x_eval * 1E-6):
    VBG_mat[:, j] = CMS.Vbg(X * np.ones(len(y_eval)), y_eval * 1E-6)

# Plot the back ground matrix
plt.figure(figsize=(9.5, 2.5))
common.configure_axes(12)
plt.pcolormesh(x_eval, y_eval, VBG_mat / CMS.qe, cmap=plt.cm.Spectral_r)
plt.xlabel("$x$ ($\mu$m)")
plt.ylabel("$y$ ($\mu$m)")
plt.xlim(np.min(x_eval), np.max(x_eval))
plt.title("Background energy (eV)")
plt.colorbar()

# Solve for the electron positions in the trap area!
ConvMon = anneal.ConvergenceMonitor(Uopt=CMS.Vtotal, grad_Uopt=CMS.grad_total, N=10,
                                    Uext=CMS.V,
                                    xext=xeval * 1E-6, yext=yeval * 1E-6, verbose=False, eps=epsilon,
                                    save_path=conv_mon_save_path)
ConvMon.figsize = (8., 2.)

trap_initial_condition = anneal.setup_initial_condition(N_trap_electrons,
                                                        (np.min(x_eval) * 1E-6, np.max(x_eval) * 1E-6),
                                                        (-2.5E-6, 2.5E-6),
                                                        4.0E-6, 0E-6)

x_trap_init, y_trap_init = anneal.r2xy(trap_initial_condition)

trap_minimizer_options = {'jac': CMS.grad_total,
                          'options': {'disp': False, 'gtol': gradient_tolerance, 'eps': epsilon},
                          'callback': ConvMon.monitor_convergence}

trap = minimize(CMS.Vtotal, trap_initial_condition, method='CG', **trap_minimizer_options)

if trap['status'] > 0:
    cprint("WARNING: Initial minimization for Trap did not converge!", "red")
    print("Final L-inf norm of gradient = %.2f eV/m" % (np.amax(trap['jac'])))
    best_trap = trap
else:
    cprint("SUCCESS: Initial minimization for Trap converged!", "green")
    # This maps the electron positions within the simulation domain
    trap['x'] = RS.coordinate_transformation(trap['x'])

    cprint("Perturbing solution %d times at %.2f K" \
           % (len(trap_annealing_steps), trap_annealing_steps[0]), "green")
    best_trap = CMS.perturb_and_solve(CMS.Vtotal, len(trap_annealing_steps),
                                      trap_annealing_steps[0], trap, **trap_minimizer_options)

    if (best_trap['x'] == trap['x']).all():
        cprint("Solution data unchanged after perturbing!", "green")

trap_electrons_x, trap_electrons_y = anneal.r2xy(best_trap['x'])

# Plot the resonator and trap electron configuration
fig2 = plt.figure(figsize=(12, 3))
common.configure_axes(12)
plt.pcolormesh(x_eval, y_eval, CMS.V(X_eval, Y_eval), cmap=plt.cm.Spectral_r, vmax=0.0, vmin=-0.75 * Vres)
plt.pcolormesh((y_box + res_right_x + resonator_box_length / 2.) * 1E6, x_box * 1E6, RS.V(X_box, Y_box).T,
               cmap=plt.cm.Spectral_r, vmax=0.0, vmin=-0.75 * Vres)
plt.plot((y_init + res_right_x + resonator_box_length / 2.) * 1E6, x_init * 1E6, 'o', color='paleturquoise', alpha=0.5)
plt.plot(x_trap_init * 1E6, y_trap_init * 1E6, 'o', color='mediumpurple', alpha=0.5)
plt.plot(trap_electrons_x * 1E6, trap_electrons_y * 1E6, 'o', color='violet', alpha=1.0)
plt.plot(res_electrons_x * 1E6, res_electrons_y * 1E6, 'o', color='deepskyblue', alpha=1.0)
for k, name in enumerate(["$V_\mathrm{res}$", "$V_\mathrm{trap}$", "$V_\mathrm{rg}$", "$V_\mathrm{cg}$", "$V_\mathrm{tg}$"]):
    Vs = [Vres, Vtrap, Vrg, Vcg, Vtg]
    plt.text(-0.0, 2.6 - 0.35 * k, name+" = %.2f V" % Vs[k], fontdict={'size': 10, 'color': 'black', 'ha' : 'right'})

if best_res['status'] > 0:
    plt.text(8, -2, "Minimization did not converge", fontdict={"size" : 10})
if best_trap['status'] > 0:
    plt.text(2, -2, "Minimization did not converge", fontdict={"size" : 10})

plt.xlabel("$x$ ($\mu$m)")
plt.ylabel("$y$ ($\mu$m)")
plt.title("Combined model: res, trap = (%d, %d) electrons" % (N_resonator_electrons, N_trap_electrons))
plt.colorbar()
plt.xlim(np.min(x_eval), 14)  # np.max(x_eval) + resonator_box_length * 1E6)
plt.ylim(np.min(y_eval), np.max(y_eval))

common.save_figure(fig2, save_path=save_path)

num_unbounded_electrons = anneal.check_unbounded_electrons(best_trap['x'],
                                                           xdomain=(np.min(x_eval) * 1E-6, np.max(x_eval) * 1E-6),
                                                           ydomain=(np.min(y_eval) * 1E-6, np.max(y_eval) * 1E-6))

print("Number of unbounded electrons = %d" % num_unbounded_electrons)
print("Electron density on resonator = %.2e" % \
      anneal.get_electron_density_by_area(anneal.xy2r(res_electrons_x, res_electrons_y)))

plt.show()
