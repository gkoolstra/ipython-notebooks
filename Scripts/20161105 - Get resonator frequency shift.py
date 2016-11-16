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
from TrapAnalysis import import_data, artificial_anneal as anneal

save_path = r"/Users/gkoolstra/Desktop/Electron optimization/Realistic potential/Resonator"
sub_dir = r"161107_154034_resonator_sweep_50_electrons_with_perturbing"

h = 0.74E-3
Vres = list()
converged = list()
save = True

plt.close('all')

with h5py.File(os.path.join(os.path.join(save_path, sub_dir), "Results.h5"), "r") as f:
    for k, step in enumerate(f.keys()):
        electron_ri = f[step + "/electron_final_coordinates"][()]

        if k == 0:
            electron_positions = electron_ri
        else:
            electron_positions = np.vstack((electron_positions, electron_ri))

        Vres.append(f[step + "/potential_coefficients"][()])
        converged.append(f[step + "/solution_valid"][()])

# Load the data from the dsp file:
path = r'/Users/gkoolstra/Desktop/Electron optimization/Realistic potential/Potentials/2D Resonator Potentials'
potential_file = r"DCBiasPotential.dsp"
efield_file = r"DifferentialModeEx.dsp"
elements, nodes, elem_solution, bounding_box = import_data.load_dsp(os.path.join(path, potential_file))

xdata, ydata, Udata = interpolate_slow.prepare_for_interpolation(elements, nodes, elem_solution)

# Do not evaluate the files on the simulation boundary, this gives errors.
# Here I construct an array from y0 to -dy/2, with spacing dy. This limits
# our choice for y-points because, the spacing must be such that there's
# an integer number of points in yeval. This can only be the case if
# dy = 2 * ymin / (2*k+1) and Ny = ymin / dy - 0.5 + 1
# yeval = y0, y0 - dy, ... , -3dy/2, -dy/2

x0 = -3.0E-3 # Starting point for y
k = 151 # This defines the sampling
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

elements, nodes, elem_solution, bounding_box = import_data.load_dsp(os.path.join(path, efield_file))

xdata, ydata, Udata = interpolate_slow.prepare_for_interpolation(elements, nodes, elem_solution)

xinterp, yinterp, Uinterp = interpolate_slow.evaluate_on_grid(xdata, ydata, Udata, xeval=x_symmetric,
                                                              yeval=h, clim=(0.00, 1.00), plot_axes='xy',
                                                              linestyle='None',
                                                              cmap=plt.cm.viridis, plot_data=False,
                                                              **common.plot_opt("darkorange", msize=6))

if 1:
    plt.figure(figsize=(5.,3.))
    common.configure_axes(12)
    plt.plot(x_symmetric, -Uinterp_symmetric, '.k')
    plt.ylim(-0.8, 0.1)
    plt.xlim(-3E-3, 3E-3)
    plt.ylabel("$U_{\mathrm{ext}}$ (eV)")
    plt.xlabel("$x$ (mm)")
    plt.title("DC Bias potential data")

    plt.figure(figsize=(5.,3.))
    common.configure_axes(12)
    plt.plot(xinterp[0], Uinterp[0], '.k')
    #plt.ylim(-0.8, 0.1)
    plt.xlim(-3E-3, 3E-3)
    plt.ylabel("$E_{x}$ (eV/m)")
    plt.xlabel("$x$ (mm)")
    plt.title("Differential mode Ex data")

def get_resonator_constants():
    """
    Returns a dictionary of resonator constants used in the calculations in this module.
    :return: {f0, Z0, Q, input power}
    """
    constants = {'f0' : 8E9, 'Z0' : 70.0, 'Q' : 10000, 'P' : -100, 'l' : 3000E-6}
    return constants

def get_physical_constants():
    """
    Returns a dictionary of physical constants used in the calculations in this module.
    :return: {'e' : 1.602E-19, 'm_e' : 9.11E-31, 'eps0' : 8.85E-12}
    """
    constants = {'e' : 1.602E-19, 'm_e' : 9.11E-31, 'eps0' : 8.85E-12}
    return constants

def setup_eom(electron_positions, Vres):
    """
    Set up the Matrix used for determining the electron frequency.
    You must make sure to have one of the following:
    if use_FEM_data = True, you must supply RF_efield_data
        - self.x_RF_FEM: x-axis for self.U_RF_FEM (units: m)
        - self.U_RF_FEM: RF E-field (units: V/m)
        - self.x_DC_FEM: x-axis for self.V_DC_FEM (units: m)
        - self.V_DC_FEM: Curvature a1(x), in V_DC(x) = a0 + a1(x-a2)**2 (units: V/m**2)
    if use_FEM_data = False you must supply:
        - self.dc_params: [a0, a1, a2] --> a0 + a1*(x-a2)**2
        - self.rf_params: [a0, a1] --> a0 + a1*x

    :param electron_positions: Electron positions, in the form
            np.array([[x0, x1, ...],
                      [y0, y1, ...)
    :return: M^(-1) * K
    """
    RS = anneal.ResonatorSolver(x_symmetric*1E-3, -Vres*Uinterp_symmetric, efield_data=0.01*Uinterp[0])

    c = get_physical_constants()
    r = get_resonator_constants()

    omega0 = 2 * np.pi * r['f0']
    L = r['Z0'] / omega0
    C = 1 / (omega0 ** 2 * L)

    num_electrons = np.int(len(electron_positions)/2)
    xi, yi = anneal.r2xy(electron_positions)

    # Set up the inverse of the mass matrix first
    diag_invM = 1 / c['m_e'] * np.ones(num_electrons + 1)
    diag_invM[0] = 1 / L
    invM = np.diag(diag_invM)

    # Set up the kinetic matrix next
    K = np.zeros(np.shape(invM))
    K[0, 0] = 1 / C  # Bare cavity
    # Note this should be the first derivative of the RF potential,
    # and not the DC potential
    K[1:, 0] = K[0, 1:] = c['e'] / C * RS.Ex(xi, yi)

    kij_plus = np.zeros((num_electrons, num_electrons))
    for idx in range(num_electrons):
        rij = np.sqrt((xi[idx] - xi) ** 2 + (yi[idx] - yi) ** 2)
        tij = np.arctan((yi[idx] - yi) / (xi[idx] - xi))
        kij_plus[idx, :] = 1 / 2. * c['e'] ** 2 / (4 * np.pi * c['eps0']) * (1 + 3 * np.cos(2 * tij)) / rij ** 3

    np.fill_diagonal(kij_plus, 0)

    K_block = -kij_plus + np.diag(2 * c['e'] * RS.ddVdx(xi, yi) + np.sum(kij_plus, axis=1))
    K[1:, 1:] = K_block

    return np.dot(invM, K)

def solve_eom(LHS):
    """
    Solves the eigenvalues and eigenvectors for the system of equations constructed with setup_eom()
    :param LHS: matrix product of M^(-1) K
    :return: Eigenvalues, Eigenvectors
    """
    EVals, EVecs = np.linalg.eig(LHS)
    return EVals, EVecs

N_electrons = 50
cavity_f = list()
electron_f = list()
ten_strongest_mode_numbers = np.zeros((len(Vres), 10), dtype=np.int)
Eigenvalues = np.zeros((len(Vres), N_electrons+1))
Eigenvectors = np.zeros((N_electrons+1, N_electrons+1, len(Vres)))

for k, resV in tqdm(enumerate(Vres)):
    LHS = setup_eom(electron_positions=electron_positions[k,:], Vres=resV)
    evals, evecs = solve_eom(LHS)

    Eigenvalues[k, :] = evals.flatten()
    Eigenvectors[:, :, k] = evecs

    cavity_contributions = evecs[0,:]
    ranked_cavity_contributions = np.argsort(np.abs(cavity_contributions))

    cavity_mode_idx = ranked_cavity_contributions[-1]
    electron_evs = np.delete(evals, cavity_mode_idx)

    ten_strongest_mode_numbers[k, :] = ranked_cavity_contributions[-10:]

    # Now save the electron mode frequency of the mode that couples second strongest to the cavity
    electron_f.append(np.sqrt(evals[ranked_cavity_contributions[-2]])/(2*np.pi*1E9))
    # And save the cavity mode frequency
    cavity_f.append(np.sqrt(evals[cavity_mode_idx])/(2*np.pi*1E9))

plt.figure(figsize=(6.,4.))
plt.plot(np.abs(cavity_contributions), 'o', **common.plot_opt('blueviolet'))
plt.ylim(0., cavity_contributions[ten_strongest_mode_numbers[-1, -2]])
plt.xlabel("Mode number")
plt.ylabel("Cavity contributions")
plt.title("$V_\mathrm{res}$ = %.2f V"%Vres[-1])

plt.figure(figsize=(6.,4.))
plt.plot(np.sqrt(electron_evs)/(2*np.pi*1E9), 'o', **common.plot_opt('red'))
plt.xlabel("Mode number")
plt.ylabel("$\omega_e/2\pi$ (GHz)")

# Look at the electron position of the last voltage step and single out the electrons in the left and right row
xi, yi = anneal.r2xy(electron_positions[-1,:])
right_idxs = np.where(xi > 0)[0]
sorted_right_idxs = np.argsort(right_idxs)
left_idxs = np.where(xi < 0)[0]
sorted_left_idxs = np.argsort(left_idxs)

plt.figure(figsize=(12.,10.))

Voi = 0.85
Voi_idx = common.find_nearest(np.array(Vres), Voi)
modes = ten_strongest_mode_numbers[Voi_idx, :9]

for i, m in enumerate(modes):
    plt.subplot(3,3,i+1)
    plt.plot(Eigenvectors[sorted_left_idxs, m, Voi_idx], 'o', label="Left row", **common.plot_opt("red", msize=8))
    plt.plot(Eigenvectors[sorted_right_idxs, m, Voi_idx], 'o', label="Right row", **common.plot_opt("black", msize=5))
    if not i%3:
        plt.ylabel("Amplitude")
    if i > 9:
        plt.xlabel("Sorted electron number in the row")

    plt.title("Mode %d @ %.2f V: $\omega_e/2\pi$ = %.2f GHz"%(m, Voi, np.sqrt(evals[m])/(2*np.pi*1E9)))
    plt.legend(loc=0, prop={'size' : 10})

#print(np.shape(ten_strongest_mode_numbers[:, -1]))
#print(ten_strongest_mode_numbers[:, -1])
#print(np.shape(Eigenvalues))

if 1:
    f_01 = [np.sqrt(Eigenvalues[k, ten_strongest_mode_numbers[k, -1]])/(2*np.pi*1E9) for k in range(len(Vres))]
    f_02 = [np.sqrt(Eigenvalues[k, ten_strongest_mode_numbers[k, -2]]) / (2 * np.pi * 1E9) for k in range(len(Vres))]

    plt.figure(figsize=(6.,4.))
    plt.plot(Vres, f_01, color="cyan", label="cavity")
    plt.plot(Vres, f_02, color="navy", label="2nd strongest")
    #plt.plot(Vres, np.sqrt(Eigenvalues[:, ten_strongest_mode_numbers[:, -3]])/(2*np.pi*1E9), color="maroon", label="3rd strongest")
    #plt.plot(Vres, np.sqrt(Eigenvalues[:, ten_strongest_mode_numbers[:, -4]])/(2*np.pi*1E9), color="lightgreen", label="4th strongest")
    plt.ylabel("Frequency of modes that couple strongest (GHz)")
    plt.xlabel("Resonator voltage (V)")
    plt.legend(loc=0, prop={"size": 10})

    plt.figure(figsize=(6., 4.))
    plt.plot(Vres, ten_strongest_mode_numbers[:, -2], color='navy', label='2nd strongest')
    plt.plot(Vres, ten_strongest_mode_numbers[:, -3], color='maroon', label='3rd strongest')
    plt.plot(Vres, ten_strongest_mode_numbers[:, -4], color='lightgreen', label='4th strongest')
    plt.ylabel("Electron mode indices that couple strongest")
    plt.xlabel("Resonator voltage (V)")
    plt.legend(loc=0, prop={"size" : 10})

r = get_resonator_constants()
delta_f = np.array(cavity_f)*1e9 - r['f0']
scaled_cavity_frequency = r['f0'] + delta_f * r['l']/40E-6

plt.figure(figsize=(6.,4.))
plt.plot(Vres, delta_f/1E6 * r['l']/40E-6)
plt.ylabel("Cavity frequency shift $\Delta \omega_0/2\pi$ (MHz)")
plt.xlabel("Resonator voltage (V)")

xplot = np.linspace(-3E-6, 3E-6, 501)
yplot = np.linspace(-20E-6, 20E-6, 11)
Xplot, Yplot = np.meshgrid(xplot, yplot)

RS = anneal.ResonatorSolver(x_symmetric * 1E-3, -Vres[-1] * Uinterp_symmetric, efield_data=0.01 * Uinterp[0])

plt.figure(figsize=(4.,6.))
plt.pcolormesh(xplot*1E6, yplot*1E6, RS.V(Xplot, Yplot), cmap=plt.cm.Spectral_r, vmin=-Vres[-1]*0.75, vmax=0)
plt.plot(electron_positions[-1,::2]*1E6, electron_positions[-1,1::2]*1E6, 'o', color='blueviolet')
plt.plot(electron_positions[-1, 2*right_idxs]*1E6, electron_positions[-1, 2*right_idxs+1]*1E6, 'o', color='yellow')
plt.plot(electron_positions[-1, 2*left_idxs]*1E6, electron_positions[-1, 2*left_idxs+1]*1E6, 'o', color='lightgreen')

for k in range(np.int(len(electron_positions[-1,:])/2)):
    plt.text(electron_positions[-1, 2*k]*1E6, electron_positions[-1, 2*k+1]*1E6, "%d"%k)
plt.xlabel("$x$ ($\mu$m)")
plt.ylabel("$y$ ($\mu$m)")

plt.show()