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

save = True
save_path = r"/Users/gkoolstra/Desktop"
#save_path = r"/Volumes/slab/Gerwin/Electron on helium/Electron optimization/Realistic potential/Resonator"
sub_dir = r"Temporary electron optimization"

h = 0.74
box_y_length = 40E-6
Vres = list()
save = False

plt.close('all')
print("Loading optimization data...")

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

# Load the data from the dsp file:
#path = r'/Volumes/slab/Gerwin/Electron on helium/Maxwell/M018 Yggdrasil/Greater Trap Area/V1big'
path = r"/Users/gkoolstra/Desktop/Temporary electron optimization"
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

if 1:
    fig1 = plt.figure(figsize=(6., 4.))
    common.configure_axes(12)
    plt.plot(x_symmetric, -Uinterp_symmetric, 'o', **common.plot_opt('red', msize=4))
    plt.ylim(-0.8, 0.1)
    plt.xlim(-2, 2)
    plt.ylabel("$U_{\mathrm{ext}}$ (eV)")
    plt.xlabel("$x$ (mm)")
    plt.title("DC Bias potential data")

    fig2 = plt.figure(figsize=(6., 4.))
    common.configure_axes(12)
    plt.plot(xinterp[0], Uinterp[0]/1E3, 'o', **common.plot_opt('red', msize=4))
    # plt.ylim(-0.8, 0.1)
    plt.xlim(-2, 2)
    plt.ylabel("$E_{x}$ (keV/m)")
    plt.xlabel("$x$ (mm)")
    plt.title("Differential mode Ex data")


def get_resonator_constants():
    """
    Returns a dictionary of resonator constants used in the calculations in this module.
    :return: {f0, Z0, Q, input power}
    """
    constants = {'f0': 8E9, 'Z0': 70.0, 'Q': 10000, 'P': -100, 'l': 3000E-6}
    return constants

def get_physical_constants():
    """
    Returns a dictionary of physical constants used in the calculations in this module.
    :return: {'e' : 1.602E-19, 'm_e' : 9.11E-31, 'eps0' : 8.85E-12}
    """
    constants = {'e': 1.602E-19, 'm_e': 9.11E-31, 'eps0': 8.85E-12, 'hbar': 1.055E-34}
    return constants

def get_V0():
    """
    Returns the single photon voltage
    :return: V0 in V
    """
    c = get_physical_constants()
    r = get_resonator_constants()
    return c['hbar'] * 2 * np.pi * r['f0'] / c['e']

def get_z0(electron_frequency):
    """
    Returns the electron motion quantum in m
    :param electron_frequency: frequency in Hz
    :return: z0 in m
    """
    c = get_physical_constants()
    return np.sqrt(c['hbar'] / (2 * c['m_e'] * 2 * np.pi * electron_frequency))

def get_g(electron_frequency, Efield):
    """
    This returns the coupling in units of radians: g[rad] = 2*pi*g[Hz]
    :param electron_frequency: electron frequency in Hz
    :param Efield: Electric field evaluated at a single point in space, with 1V applied to the electrodes
    :return: g[rad] = 2 * pi * g[Hz]
    """
    c = get_physical_constants()
    V0 = get_V0()
    z0 = get_z0(electron_frequency)
    return c['e'] * V0 * Efield * z0 / c['hbar']

def calculate_metrics(xi, yi, eps=1E-15):
    Xi, Yi = np.meshgrid(xi, yi)
    Xj, Yj = Xi.T, Yi.T

    Yi_shifted = Yi.copy()
    Yi_shifted[Yi_shifted > 0] -= box_y_length  # Shift entire box length
    Yj_shifted = Yi_shifted.T
    XiXj = Xi - Xj
    YiYj = Yi - Yj
    YiYj_shifted = Yi_shifted - Yj_shifted

    Rij_standard = np.sqrt((XiXj) ** 2 + (YiYj) ** 2)
    Rij_shifted = np.sqrt((XiXj) ** 2 + (YiYj_shifted) ** 2)
    R_ij = np.minimum(Rij_standard, Rij_shifted)

    # Avoid an raising a warning when dividing by Xi-Xj in theta_ij
    np.fill_diagonal(XiXj, eps)

    theta_ij = np.arctan(YiYj / XiXj)
    theta_ij_shifted = np.arctan(YiYj_shifted / XiXj)

    # Use shifted y-coordinate only in this case:
    np.copyto(theta_ij, theta_ij_shifted, where=Rij_shifted < Rij_standard)

    # Remember to set the diagonal back to 0
    np.fill_diagonal(theta_ij, 0)

    return R_ij, theta_ij

def setup_eom(electron_positions, Vres, eps=1E-15):
    """
    Sets up the inverse matrix that needs to be solved
    :param electron_positions: coordinates of the electron configuration = [x0, y0, x1, y1, ...]
    :param Vres: float, resonator voltage.
    :return: M^(-1) * K
    """
    c = get_physical_constants()
    r = get_resonator_constants()

    omega0 = 2 * np.pi * r['f0']
    L = r['Z0'] / omega0
    C = 1 / (omega0 ** 2 * L)

    RS = anneal.ResonatorSolver(x_symmetric * 1E-6, -Vres * Uinterp_symmetric, efield_data=Uinterp[0])

    num_electrons = np.int(len(electron_positions) / 2)
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

    Rij, Tij = calculate_metrics(xi, yi, eps=eps)
    np.fill_diagonal(Rij, eps)
    kij_plus_matrix = 1 / 4. * c['e'] ** 2 / (4 * np.pi * c['eps0']) * (1 + 3 * np.cos(2 * Tij)) / Rij ** 3
    np.fill_diagonal(kij_plus_matrix, 0)
    kij_plus_sum = np.sum(kij_plus_matrix, axis=1)

    # Note: added - sign for c['e']
    K_block = -kij_plus_matrix + np.diag(c['e'] * RS.ddVdx(xi, yi) + kij_plus_sum)
    K[1:, 1:] = K_block

    return np.dot(invM, K)

#RS = anneal.ResonatorSolver(x_symmetric * 1E-6, -0.50 * Uinterp_symmetric, efield_data=3E-5 * Uinterp[0])

#plt.figure()
#plt.plot(np.linspace(-2, 2)*1E-6, RS.Ex(np.linspace(-2, 2)*1E-6, 0))

def solve_eom(LHS):
    """
    Solves the eigenvalues and eigenvectors for the system of equations constructed with setup_eom()
    :param LHS: matrix product of M^(-1) K
    :return: Eigenvalues, Eigenvectors
    """
    EVals, EVecs = np.linalg.eig(LHS)
    return EVals, EVecs


N_electrons = np.int(np.shape(electron_positions)[1] / 2)
cavity_f = list()
electron_f = list()
ten_strongest_mode_numbers = np.zeros((len(Vres), 10), dtype=np.int)
Eigenvalues = np.zeros((len(Vres), N_electrons + 1))
Eigenvectors = np.zeros((N_electrons + 1, N_electrons + 1, len(Vres)))

for k, resV in tqdm(enumerate(Vres)):
    LHS = setup_eom(electron_positions=electron_positions[k, :], Vres=resV)
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

if 0:
    plt.figure(figsize=(6., 4.))
    plt.plot(np.abs(cavity_contributions), 'o', **common.plot_opt('blueviolet'))
    plt.ylim(0., cavity_contributions[ten_strongest_mode_numbers[-1, -2]])
    plt.xlabel("Mode number")
    plt.ylabel("Cavity contributions")
    plt.title("$V_\mathrm{res}$ = %.2f V" % Vres[-1])

Voi = 0.25
Voi_idx = common.find_nearest(np.array(Vres), Voi)
modes = ten_strongest_mode_numbers[Voi_idx, :9]

fig3 = plt.figure(figsize=(6., 4.))
plt.plot(np.sqrt(Eigenvalues[Voi_idx, :]) / (2 * np.pi * 1E9), 'o', **common.plot_opt('red', msize=4))
plt.xlabel("Mode number")
plt.ylabel("$\omega_e/2\pi$ (GHz)")
plt.title("Frequency of all modes at Vres = %.2f V" % Voi)
if save:
    common.save_figure(fig3, save_path=os.path.join(save_path, sub_dir))

# Look at the electron position of the last voltage step and single out the electrons in the left and right row
# Note: this only makes sense if you have two rows!
xi, yi = anneal.r2xy(electron_positions[Voi_idx, :])
right_idxs = np.where(xi > 0)[0]
sorted_right_idxs = np.argsort(right_idxs)
left_idxs = np.where(xi < 0)[0]
sorted_left_idxs = np.argsort(left_idxs)
sorted_idxs = np.argsort(yi)

if len(right_idxs) == 0 or len(left_idxs) == 0:
    print("2 row configuration not detected, assuming 1 row")
    two_rows = False
else:
    two_rows = True

# Here we create a plot with the amplitude of the electrons in the chain (from bottom to top) for the first
# 9 strongest eigenmodes. The last subplot shows the mode that couples strongest.
fig4 = plt.figure(figsize=(12., 10.))
for i, m in enumerate(modes):
    plt.subplot(3, 3, i + 1)

    if two_rows:
        plt.plot(Eigenvectors[sorted_right_idxs, m, Voi_idx], '-o', label="Right row",
                 **common.plot_opt("black", msize=5))
        plt.plot(Eigenvectors[sorted_left_idxs, m, Voi_idx], '-o', label="Left row",
                 **common.plot_opt("red", msize=4))
    else:
        plt.plot(Eigenvectors[sorted_idxs, m, Voi_idx], '-o', **common.plot_opt("red", msize=4))

    if not i % 3:
        plt.ylabel("Amplitude")
    if i > 9:
        plt.xlabel("Position along the row")

    plt.title(u"Mode {0:d} @ {1:.2f} V: $\omega_e/2\pi$ = {2:.2f} GHz" \
              .format(m, Voi, np.sqrt(Eigenvalues[Voi_idx, m]) / (2 * np.pi * 1E9)))
    if two_rows:
        plt.legend(loc=0, prop={'size': 10})

if save:
    common.save_figure(fig4, save_path=os.path.join(save_path, sub_dir))
    time.sleep(1.0)

# This plot shows the cavity mode frequency as function of resonator voltage.
# On top of that it shows the frequency of the mode that couples the strongest to the cavity
f_01 = [np.sqrt(Eigenvalues[k, ten_strongest_mode_numbers[k, -1]]) / (2 * np.pi * 1E9) for k in range(len(Vres))]
f_02 = [np.sqrt(Eigenvalues[k, ten_strongest_mode_numbers[k, -2]]) / (2 * np.pi * 1E9) for k in range(len(Vres))]

fig5 = plt.figure(figsize=(6., 4.))
plt.title("Frequency of modes that couple strongest")
plt.plot(Vres, f_01, 'o', label="cavity", **common.plot_opt('cyan', msize=4))
plt.plot(Vres, f_02, 'o', label="2nd strongest", **common.plot_opt('navy', msize=4))
plt.ylabel("Frequency of modes that couple strongest (GHz)")
plt.xlabel("Resonator voltage (V)")
plt.legend(loc=0, prop={"size": 10})

if save:
    common.save_figure(fig5, save_path=os.path.join(save_path, sub_dir))

if 0:
    plt.figure(figsize=(6., 4.))
    plt.plot(Vres, ten_strongest_mode_numbers[:, -2], color='navy', label='2nd strongest')
    plt.plot(Vres, ten_strongest_mode_numbers[:, -3], color='maroon', label='3rd strongest')
    plt.plot(Vres, ten_strongest_mode_numbers[:, -4], color='lightgreen', label='4th strongest')
    plt.ylabel("Electron mode indices that couple strongest")
    plt.xlabel("Resonator voltage (V)")
    plt.legend(loc=0, prop={"size": 10})

r = get_resonator_constants()
delta_f = np.array(cavity_f) - r['f0']
scaled_cavity_frequency = r['f0'] + delta_f * r['l'] / box_y_length

print("The resonator is %.0f um long, which is %.1f box lengths"%(r['l'] * 1E6, r['l'] / box_y_length))
print("On the resonator there are %d electrons" % (N_electrons * r['l'] / box_y_length))

# Here we plot the scaled cavity frequency shift vs. the resonator voltage.
fig6 = plt.figure(figsize=(6., 4.))
plt.plot(Vres, scaled_cavity_frequency / 1E9, 'o', **common.plot_opt('red', msize=4))
plt.ylabel("Scaled cavity frequency $\omega_0/2\pi$ (GHz)")
plt.xlabel("Resonator voltage (V)")

if save:
    common.save_figure(fig6, save_path=os.path.join(save_path, sub_dir))

xplot = np.linspace(-3E-6, 3E-6, 501)
yplot = np.linspace(-box_y_length / 2, box_y_length / 2, 11)
Xplot, Yplot = np.meshgrid(xplot, yplot)

RS = anneal.ResonatorSolver(x_symmetric * 1E-6, -Voi * Uinterp_symmetric, efield_data=Uinterp[0])

# Here we plot the potential landscape and the electron positions at the voltage point of interest
fig7 = plt.figure(figsize=(4., 6.))
plt.pcolormesh(xplot * 1E6, yplot * 1E6, RS.V(Xplot, Yplot), cmap=plt.cm.Spectral_r, vmin=-Voi * 0.75, vmax=0)
plt.plot(electron_positions[Voi_idx, ::2] * 1E6, electron_positions[Voi_idx, 1::2] * 1E6, 'o', color='blueviolet')

if two_rows:
    print("There are %d electrons in the left row and %d electrons in the right row." % \
          (len(sorted_left_idxs), len(sorted_right_idxs)))
    plt.plot(electron_positions[Voi_idx, 2 * right_idxs] * 1E6, electron_positions[Voi_idx, 2 * right_idxs + 1] * 1E6,
             'o', color='yellow')
    plt.plot(electron_positions[Voi_idx, 2 * left_idxs] * 1E6, electron_positions[Voi_idx, 2 * left_idxs + 1] * 1E6,
             'o', color='lightgreen')

ax = plt.gca()
for k in range(np.int(len(electron_positions[Voi_idx, :]) / 2)):
    #plt.text(electron_positions[Voi_idx, 2 * k] * 1E6, electron_positions[Voi_idx, 2 * k + 1] * 1E6, "%d" % k)
    mode_nr = ten_strongest_mode_numbers[Voi_idx, -4]
    amp = Eigenvectors[k, mode_nr, Voi_idx]
    x_p = electron_positions[Voi_idx, 2 * k] * 1E6
    y_p = electron_positions[Voi_idx, 2 * k + 1] * 1E6

    ax.arrow(x_p, y_p, amp * 5, 0, head_width=0.05, head_length=0.1, fc='k', ec='k')

plt.xlabel("$x$ ($\mu$m)")
plt.ylabel("$y$ ($\mu$m)")

if save:
    common.save_figure(fig7, save_path=os.path.join(save_path, sub_dir))

RS = anneal.ResonatorSolver(x_symmetric * 1E-6, -Uinterp_symmetric, efield_data=Uinterp[0])
print("At the center, the E-field is %.2e V/m" % RS.Ex(0.0E-6, 0.0E-6))
print("The curvature at the center of the channel is %.2e m^-2" % RS.ddVdx(0.0E-6, 0.0E-6))

plt.show()
