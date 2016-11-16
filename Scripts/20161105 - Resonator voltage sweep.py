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

from Common import common, kfit
from TrapAnalysis import trap_analysis, import_data, artificial_anneal as anneal

# Parameters:
box_length = 40E-6
N_electrons = 50
N_cols = 25
N_rows = 2
resVs = np.arange(1.0, 0.05, -0.01)

h = 0.74E-3
fitdomain=(-2E-3, 2E-3)

epsilon = 1e-10
use_gradient = True
gradient_tolerance = 1E1

annealing_steps = [1.0]*10
simulation_name = "resonator_sweep_%d_electrons_with_perturbing" % N_electrons
save_path = r"/Users/gkoolstra/Desktop/Electron optimization/Realistic potential/Resonator"
sub_dir = time.strftime("%y%m%d_%H%M%S_{}".format(simulation_name))
save = True

# Load the data from the dsp file:
path = r'/Users/gkoolstra/Desktop/Electron optimization/Realistic potential/Potentials/2D Resonator Potentials/DCBiasPotential.dsp'
elements, nodes, elem_solution, bounding_box = import_data.load_dsp(path)

xdata, ydata, Udata = interpolate_slow.prepare_for_interpolation(elements, nodes, elem_solution)

# Do not evaluate the files on the simulation boundary, this gives errors.
# Here I construct an array from y0 to -dy/2, with spacing dy. This limits
# our choice for y-points because, the spacing must be such that there's
# an integer number of points in yeval. This can only be the case if
# dy = 2 * ymin / (2*k+1) and Ny = ymin / dy - 0.5 + 1
# yeval = y0, y0 - dy, ... , -3dy/2, -dy/2

x0 = -3.0E-3 # Starting point for y
k = 501 # This defines the sampling
dx = 2*np.abs(x0)/np.float(2*k+1)
xeval = np.linspace(x0, -dx/2., (np.abs(x0)-0.5*dx)/dx + 1)

xinterp, yinterp, Uinterp = interpolate_slow.evaluate_on_grid(xdata, ydata, Udata, xeval=xeval,
                                                     yeval=h, clim=(0.00, 1.00), plot_axes='xy', linestyle='None',
                                                     cmap=plt.cm.viridis, plot_data=False,
                                                     **common.plot_opt("darkorange", msize=6))

# Mirror around the y-axis
xsize = len(Uinterp[0])
Uinterp_symmetric = np.zeros(2*xsize)
Uinterp_symmetric[:xsize] = Uinterp[0]
Uinterp_symmetric[xsize:] = Uinterp[0][::-1]

x_symmetric = np.zeros(2 * xsize)
x_symmetric[:xsize] = xinterp[0]
x_symmetric[xsize:] = -xinterp[0][::-1]

plt.figure(figsize=(5.,3.))
common.configure_axes(12)
plt.plot(x_symmetric, -Uinterp_symmetric, '.k')
plt.ylim(-0.8, 0.1)
plt.xlim(-3E-3, 3E-3)
plt.ylabel("$U_{\mathrm{ext}}$ (eV)")
plt.xlabel("$x$ (mm)")

ax = plt.gca()
ax.set_axis_bgcolor('none')
fr, ferr = kfit.fit_poly(x_symmetric, -Uinterp_symmetric, mode='even', fitparams=[0, 1E4, 1E4], domain=fitdomain)
plt.plot(x_symmetric, kfit.polyfunc_even(x_symmetric, *fr), color='r', lw=2.0)

t = trap_analysis.TrapSolver()
t.get_electron_frequency([fr[0], -fr[1]/1E6], [ferr[0], -ferr[1]/1E6]);

# Rectangle
if N_cols*N_rows != N_electrons:
    raise ValueError("N_cols and N_rows are not compatible with N_electrons")
else:
    y_separation = 1.25E-6
    x_separation = 1.0E-6
    y0 = -15E-6
    ys = np.linspace(y0, y0+N_cols*y_separation, N_cols)
    yinit = np.tile(np.array(ys), N_rows)
    xs = np.linspace(-(N_rows-1)/2.*x_separation, +(N_rows-1)/2.*x_separation, N_rows)
    xinit = np.repeat(xs, N_cols)

electron_initial_positions = anneal.xy2r(xinit, yinit)

x_box = np.linspace(-3E-6, 3E-6, 501)
y_box = np.linspace(-box_length/2, box_length/2, 11)
X_box, Y_box = np.meshgrid(x_box, y_box)

def coordinate_transformation(r):
    x, y = anneal.r2xy(r)
    y_new = EP.map_y_into_domain(y)
    r_new = anneal.xy2r(x, y_new)
    return r_new

# This is where the actual solving starts...
os.mkdir(os.path.join(save_path, sub_dir))
time.sleep(1)
os.mkdir(os.path.join(save_path, sub_dir, "Figures"))

# Save the data to a single file
f = h5py.File(os.path.join(os.path.join(save_path, sub_dir), "Results.h5"), "w")

conv_mon_save_path = os.path.join(save_path, sub_dir, "Figures")


for k, Vres in tqdm(enumerate(resVs)):
    EP = anneal.ResonatorSolver(x_symmetric*1E-3, -Vres*Uinterp_symmetric, box_length=box_length)

    if use_gradient:
        jac = EP.grad_total
        grad_Uopt = EP.grad_total
    else:
        jac = None
        grad_Uopt = lambda x: approx_fprime(x, EP.Vtotal, epsilon)

    ConvMon = anneal.ConvergenceMonitor(Uopt=EP.Vtotal, grad_Uopt=EP.grad_total, N=50,
                                        Uext=EP.V,
                                        xext=x_box, yext=y_box, verbose=False, eps=epsilon,
                                        save_path=conv_mon_save_path, figsize=(4, 6),
                                        coordinate_transformation=coordinate_transformation)

    minimizer_options = {'jac': jac,
                         'options': {'disp': False, 'gtol': gradient_tolerance, 'eps': epsilon},
                         'callback': None}#ConvMon.monitor_convergence}

    res = minimize(EP.Vtotal, electron_initial_positions, method='CG', **minimizer_options)

    if res['status'] > 0:
        cprint("WARNING: Step %d (Vres = %.2f V) did not converge!"%(k,Vres), "red")

    # This maps the electron positions within the simulation domain
    res['x'] = coordinate_transformation(res['x'])

    res = EP.perturb_and_solve(EP.Vtotal, len(annealing_steps), annealing_steps[0], res, **minimizer_options)
    res['x'] = coordinate_transformation(res['x'])

    PP = anneal.PostProcess(save_path=conv_mon_save_path)
    PP.save_snapshot(res['x'], xext=x_box, yext=y_box, Uext=EP.V,
                     figsize=(4,6), common=common, title="Vres = %.2f V"%Vres,
                     clim=(-0.75*max(resVs), 0))

    f.create_dataset("step_%04d/potential_x" % k, data=x_box)
    f.create_dataset("step_%04d/potential_y" % k, data=y_box)
    f.create_dataset("step_%04d/potential_V" % k, data=EP.V(X_box, Y_box))
    f.create_dataset("step_%04d/potential_coefficients" % k, data=Vres)
    f.create_dataset("step_%04d/use_gradient" % k, data=use_gradient)
    f.create_dataset("step_%04d/electron_final_coordinates" % k, data=res['x'])
    f.create_dataset("step_%04d/electron_initial_coordinates" % k, data=electron_initial_positions)
    f.create_dataset("step_%04d/solution_valid" % k, data=True if res['status'] == 0 else False)
    f.create_dataset("step_%04d/energy" % k, data=res['fun'])
    f.create_dataset("step_%04d/jacobian" % k, data=res['jac'])
    f.create_dataset("step_%04d/electrons_in_trap" % k, data=PP.get_trapped_electrons(res['x']))

    # Use the solution from the current time step as the initial condition for the next timestep!
    electron_initial_positions = res['x']

f.close()

# Create a movie
ConvMon.create_movie(fps=10,
                     filenames_in=time.strftime("%Y%m%d")+"_figure_%05d.png",
                     filename_out="resonator_%d_electrons.mp4" % (N_electrons))

# Move the file from the Figures folder to the sub folder
os.rename(os.path.join(save_path, sub_dir, "Figures/resonator_%d_electrons.mp4" % (N_electrons)),
          os.path.join(save_path, sub_dir, "resonator_%d_electrons.mp4" % (N_electrons)))

x, y = anneal.r2xy(res['x'])
final_func_val = res['fun']
n_iterations = res['nit']

if 0:
    if len(np.shape(ConvMon.jac)) > 1:
        figgy = plt.figure(figsize=(6, 4))
        common.configure_axes(12)
        # LInf-norm
        plt.plot(ConvMon.iter, np.amax(np.abs(ConvMon.jac[:, ::2]), axis=1),
                 '.-r', label=r'$L^\infty$-norm $\nabla_x U_\mathrm{opt}$')
        plt.plot(ConvMon.iter, np.amax(np.abs(ConvMon.jac[:, 1::2]), axis=1),
                 '.-b', label=r'$L^\infty$-norm $\nabla_y U_\mathrm{opt}$')
        # L2-norm
        plt.plot(ConvMon.iter, np.sum(np.abs(ConvMon.jac[:, ::2]) ** 2.0, axis=1) ** (1 / 2.),
                 '.-c', label=r'$L^2$-norm $\nabla_x U_\mathrm{opt}$')
        plt.plot(ConvMon.iter, np.sum(np.abs(ConvMon.jac[:, 1::2]) ** 2.0, axis=1) ** (1 / 2.),
                 '.-m', label=r'$L^2$-norm $\nabla_y U_\mathrm{opt}$')

        plt.title("%d electrons at Vres = %.2f V" % (N_electrons, Vres))

        plt.xlabel("Iterations")
        plt.ylabel(r"$\nabla_{x,y} U_\mathrm{opt}$")
        plt.yscale('log')
        plt.xlim(0, res['nit'])
        plt.grid()
        plt.legend(loc=0, ncol=2, prop={'size': 9})

        common.save_figure(figgy, save_path=os.path.join(save_path, sub_dir))

if 1:
    y_in_domain = EP.map_y_into_domain(y)

    figgy = plt.figure(figsize=(4, 6))
    common.configure_axes(12)
    plt.pcolormesh(x_box * 1E6, y_box * 1E6, EP.V(X_box, Y_box), cmap=plt.cm.Spectral_r, vmax=0.0)
    plt.plot(xinit * 1E6, yinit * 1E6, 'o', color='palegreen', alpha=0.5)
    plt.plot(x * 1E6, y_in_domain * 1E6, 'o', color='deepskyblue', alpha=1.0)
    plt.xlabel("$x$ ($\mu$m)")
    plt.ylabel("$y$ ($\mu$m)")
    plt.colorbar()

    plt.show()