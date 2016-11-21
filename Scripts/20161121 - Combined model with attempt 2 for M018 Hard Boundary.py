from matplotlib import pyplot as plt
import platform, os, sys, time, scipy, h5py
import numpy as np
from scipy.optimize import minimize, approx_fprime
from termcolor import cprint
from Common import common

from TrapAnalysis import trap_analysis, artificial_anneal as anneal

use_gradient = True
gradient_tolerance = 1E-1
epsilon = 1E-12

N_trap_electrons = 11
trap_annealing_steps = [0.5] * 200

Vres = 0.01
Vtrap = 0.01
Vrg = 0.00
Vcg = 0.00
Vtg = -0.05

trap_initial_condition = anneal.single_row_init(N_trap_electrons, (-12.5E-6, 12.5E-6), (-1.0E-6, 1.0E-6))#anneal.setup_initial_condition(N_trap_electrons,
                                                        #(-12.5E-6, 12.5E-6),
                                                        #(-1.0E-6, 1.0E-6),
                                                        #6.0E-6, 0E-6,
                                                        #dx=2E-6, dy=0.20E-6)


simulation_name = "Combined_Model"
save_path = r"/Users/gkoolstra/Desktop/Electron optimization/Realistic potential/Combined Model"
sub_dir = time.strftime("%y%m%d_%H%M%S_{}".format(simulation_name))
save = False

# Evaluate all files in the following range.
xeval = np.linspace(-12.5, 12.5, 501)
yeval = anneal.construct_symmetric_y(-5.0, 151)

master_path = r"/Volumes/slab/Gerwin/Electron on helium/Maxwell/M018 Yggdrasil/Greater Trap Area/V1"
#master_path = r"/Users/gkoolstra/Desktop/Electron optimization/Realistic potential/Potentials/V6 - Greater Trap Area"
x_eval, y_eval, output = anneal.load_data(master_path, xeval=xeval, yeval=yeval, mirror_y=True,
                                          extend_resonator=False, do_plot=False)
# Note: x_eval and y_eval are 2D arrays that contain the x and y coordinates at which the potentials are evaluated

conv_mon_save_path = os.path.join(save_path, sub_dir, "Figures")

print("Switching to greater trap area...")
t = trap_analysis.TrapSolver()
c = trap_analysis.get_constants()

x_eval, y_eval, cropped_potentials = t.crop_potentials(output, ydomain=None, xdomain=None)
coefficients = np.array([Vres, Vtrap, Vrg, Vcg, Vtg])
combined_potential = t.get_combined_potential(cropped_potentials, coefficients)
# Note: x_eval and y_eval are 1D arrays that contain the x and y coordinates at which the potentials are evaluated
# Units of x_eval and y_eval are um

CMS = anneal.CombinedModelSolver(x_eval * 1E-6, y_eval * 1E-6, -combined_potential.T,
                                 anneal.xy2r([], []),
                                 spline_order_x=3, spline_order_y=3, smoothing=0.01)

X_eval, Y_eval = np.meshgrid(x_eval * 1E-6, y_eval * 1E-6)


VBG_mat = np.zeros((len(y_eval), len(x_eval)))
for j, X in enumerate(x_eval * 1E-6):
    VBG_mat[:, j] = CMS.Vbg(X * np.ones(len(y_eval)), y_eval * 1E-6)


# Solve for the electron positions in the trap area!
ConvMon = anneal.ConvergenceMonitor(Uopt=CMS.Vtotal, grad_Uopt=CMS.grad_total, N=10,
                                    Uext=CMS.V,
                                    xext=xeval * 1E-6, yext=yeval * 1E-6, verbose=False, eps=epsilon,
                                    save_path=conv_mon_save_path)
ConvMon.figsize = (8., 2.)

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
# plt.plot(x_trap_init * 1E6, y_trap_init * 1E6, 'o', color='mediumpurple', alpha=0.5)
plt.plot(trap_electrons_x * 1E6, trap_electrons_y * 1E6, 'o', color='violet', alpha=1.0)
for k, name in enumerate(["$V_\mathrm{res}$", "$V_\mathrm{trap}$", "$V_\mathrm{rg}$", "$V_\mathrm{cg}$", "$V_\mathrm{tg}$"]):
    Vs = [Vres, Vtrap, Vrg, Vcg, Vtg]
    plt.text(-2.0, 4.15 - 0.6 * k, name+" = %.2f V" % Vs[k], fontdict={'size': 10, 'color': 'black', 'ha' : 'right'})

if best_trap['status'] > 0:
    plt.text(2, -2, "Minimization did not converge", fontdict={"size" : 10})

plt.xlabel("$x$ ($\mu$m)")
plt.ylabel("$y$ ($\mu$m)")
plt.title("%d electrons" % (N_trap_electrons))
plt.colorbar()
plt.xlim(np.min(x_eval), 12.5)  # np.max(x_eval) + resonator_box_length * 1E6)
plt.ylim(np.min(y_eval), np.max(y_eval))

#common.save_figure(fig2, save_path=save_path)

num_unbounded_electrons = anneal.check_unbounded_electrons(best_trap['x'],
                                                           xdomain=(np.min(x_eval) * 1E-6, np.max(x_eval) * 1E-6),
                                                           ydomain=(np.min(y_eval) * 1E-6, np.max(y_eval) * 1E-6))

print("Number of unbounded electrons = %d" % num_unbounded_electrons)

plt.show()
