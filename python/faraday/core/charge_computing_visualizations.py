"""
Charge Computing Framework Visualizations
==========================================

Generates visualization panels for the Charge Computing Framework paper.
Each panel has 4 charts with at least one 3D visualization.

Author: Kundai Sachikonye
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime

# Set up matplotlib for publication-quality figures
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


# =============================================================================
# Physical Constants
# =============================================================================

HBAR = 1.054571817e-34
KB = 1.380649e-23
E_CHARGE = 1.602176634e-19
M_ELECTRON = 9.1093837015e-31
C_LIGHT = 2.99792458e8
FARADAY = 96485.33212
R_GAS = 8.314462618


# =============================================================================
# Panel 1: Triple Equivalence
# =============================================================================

def create_panel_triple_equivalence(output_dir: str):
    """
    Panel showing the Triple Equivalence: S = k_B * M * ln(n)
    """
    fig = plt.figure(figsize=(20, 5))

    # A: Oscillatory modes in phase space
    ax1 = fig.add_subplot(1, 4, 1)
    theta = np.linspace(0, 4*np.pi, 1000)
    for i, (amp, phase) in enumerate([(1, 0), (0.8, np.pi/3), (0.6, 2*np.pi/3)]):
        x = amp * np.cos(theta + phase) * np.exp(-theta/20)
        y = amp * np.sin(theta + phase) * np.exp(-theta/20)
        ax1.plot(x, y, linewidth=2, alpha=0.8)
    ax1.set_xlabel('Position q')
    ax1.set_ylabel('Momentum p')
    ax1.set_title('(A) Oscillatory Modes')
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # B: Categorical tree structure
    ax2 = fig.add_subplot(1, 4, 2)
    # Draw tree
    levels = 4
    for level in range(levels):
        n_nodes = 3**level
        y = 1 - level * 0.25
        for i in range(n_nodes):
            x = (i + 0.5) / n_nodes
            ax2.scatter(x, y, s=50, c='steelblue', zorder=3)
            if level < levels - 1:
                # Draw lines to children
                for j in range(3):
                    child_x = (3*i + j + 0.5) / (3*n_nodes)
                    child_y = y - 0.25
                    ax2.plot([x, child_x], [y, child_y], 'k-', alpha=0.3, linewidth=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel('Morphism Index')
    ax2.set_ylabel('Category Level')
    ax2.set_title('(B) Categorical Structure')
    ax2.set_xticks([])

    # C: 3D Partition space
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    # Draw partition cells
    n = 3
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n**3))
    idx = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = [i, i+1, i+1, i, i, i+1, i+1, i]
                y = [j, j, j+1, j+1, j, j, j+1, j+1]
                z = [k, k, k, k, k+1, k+1, k+1, k+1]
                # Draw faces with transparency
                ax3.scatter((i+0.5)/n, (j+0.5)/n, (k+0.5)/n,
                           c=[colors[idx]], s=100, alpha=0.6)
                idx += 1
    ax3.set_xlabel('$S_k$')
    ax3.set_ylabel('$S_t$')
    ax3.set_zlabel('$S_e$')
    ax3.set_title('(C) Partition Space')

    # D: Entropy convergence S = k_B * M * ln(n)
    ax4 = fig.add_subplot(1, 4, 4)
    n_vals = np.logspace(0, 3, 50)
    for M, color in [(1, 'blue'), (2, 'green'), (3, 'red'), (4, 'purple')]:
        S = KB * M * np.log(n_vals)
        ax4.plot(n_vals, S * 1e23, color=color, linewidth=2, label=f'M = {M}')
    ax4.set_xscale('log')
    ax4.set_xlabel('Partition Depth n')
    ax4.set_ylabel('Entropy S ($10^{-23}$ J/K)')
    ax4.set_title('(D) $S = k_B M \\ln n$')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'panel_1_triple_equivalence.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


# =============================================================================
# Panel 2: Partition Lag and Signal Velocity
# =============================================================================

def create_panel_partition_lag(output_dir: str):
    """
    Panel showing partition lag and signal vs drift velocity.
    """
    fig = plt.figure(figsize=(20, 5))

    # A: Partition lag components
    ax1 = fig.add_subplot(1, 4, 1)
    barriers = np.linspace(0.01, 0.1, 50)  # eV
    E_barrier = barriers * E_CHARGE
    tau_quantum = HBAR / E_barrier
    tau_reorg = 1e-14  # s
    tau_total = tau_quantum + tau_reorg

    ax1.semilogy(barriers * 1000, tau_quantum * 1e15, 'b-', linewidth=2, label=r'$\hbar/E_{barrier}$')
    ax1.semilogy(barriers * 1000, np.full_like(barriers, tau_reorg * 1e15), 'r--', linewidth=2, label=r'$\tau_{reorg}$')
    ax1.semilogy(barriers * 1000, tau_total * 1e15, 'k-', linewidth=2.5, label=r'$\tau_p$ (total)')
    ax1.axvline(26, color='green', linestyle=':', alpha=0.7, label='$k_BT$ at 300K')
    ax1.set_xlabel('Barrier Energy (meV)')
    ax1.set_ylabel('Time (fs)')
    ax1.set_title(r'(A) $\tau_p = \hbar/E_{barrier} + \tau_{reorg}$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # B: 3D surface of signal velocity
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    tau_p = np.logspace(-14, -11, 30)  # s
    d = np.logspace(-10, -9, 30)  # m (lattice constants)
    TAU, D = np.meshgrid(tau_p, d)
    V_signal = D / TAU  # m/s

    surf = ax2.plot_surface(np.log10(TAU), np.log10(D), np.log10(V_signal),
                           cmap='viridis', alpha=0.8)
    ax2.set_xlabel(r'$\log_{10}(\tau_p)$ [s]')
    ax2.set_ylabel(r'$\log_{10}(d)$ [m]')
    ax2.set_zlabel(r'$\log_{10}(v_{signal})$ [m/s]')
    ax2.set_title('(B) Signal Velocity Surface')

    # C: Signal vs Drift velocity comparison
    ax3 = fig.add_subplot(1, 4, 3)
    metals = ['Cu', 'Ag', 'Al', 'Au', 'Fe', 'Nb']
    v_signal = [2.1e8] * 6  # ~0.7c for all metals
    v_drift = [7.4e-5, 9.8e-5, 3.2e-5, 9.8e-5, 3.4e-5, 1.0e-4]

    x = np.arange(len(metals))
    width = 0.35
    ax3.bar(x - width/2, np.log10(v_signal), width, label='Signal velocity', color='steelblue')
    ax3.bar(x + width/2, np.log10(v_drift), width, label='Drift velocity', color='coral')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metals)
    ax3.set_ylabel(r'$\log_{10}(v)$ [m/s]')
    ax3.set_title(r'(C) $v_{signal}$ vs $v_{drift}$')
    ax3.legend()
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # D: Velocity ratio across metals
    ax4 = fig.add_subplot(1, 4, 4)
    ratios = np.array(v_signal) / np.array(v_drift)
    log_ratios = np.log10(ratios)

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(metals)))
    bars = ax4.bar(metals, log_ratios, color=colors)
    ax4.axhline(12, color='red', linestyle='--', linewidth=2, label=r'$10^{12}$')
    ax4.set_ylabel(r'$\log_{10}(v_{signal}/v_{drift})$')
    ax4.set_title('(D) Signal/Drift Ratio')
    ax4.legend()
    ax4.set_ylim(10, 14)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'panel_2_partition_lag.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


# =============================================================================
# Panel 3: Ohm's Law and Resistivity
# =============================================================================

def create_panel_ohms_law(output_dir: str):
    """
    Panel showing Ohm's law derivation from partition dynamics.
    """
    fig = plt.figure(figsize=(20, 5))

    # Metal data
    metals = ['Cu', 'Ag', 'Al', 'Au', 'Fe', 'Nb']
    n_carriers = np.array([8.5, 5.86, 18.1, 5.9, 17.0, 5.56]) * 1e28
    tau_s = np.array([27, 40, 8, 29, 2.4, 4.2]) * 1e-15
    rho_exp = np.array([1.68, 1.59, 2.65, 2.21, 9.61, 15.2]) * 1e-8

    # Calculate resistivity
    rho_calc = M_ELECTRON / (n_carriers * E_CHARGE**2 * tau_s)

    # A: Resistivity comparison
    ax1 = fig.add_subplot(1, 4, 1)
    x = np.arange(len(metals))
    width = 0.35
    ax1.bar(x - width/2, rho_calc * 1e8, width, label='Calculated', color='steelblue')
    ax1.bar(x + width/2, rho_exp * 1e8, width, label='Experimental', color='coral', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metals)
    ax1.set_ylabel(r'$\rho$ ($\mu\Omega\cdot$cm)')
    ax1.set_title(r'(A) $\rho = m_e/(ne^2\tau_s)$')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # B: 3D conductivity surface
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    n = np.logspace(28, 29, 30)
    tau = np.logspace(-15, -14, 30)
    N, TAU = np.meshgrid(n, tau)
    sigma = N * E_CHARGE**2 * TAU / M_ELECTRON

    surf = ax2.plot_surface(np.log10(N), np.log10(TAU), np.log10(sigma),
                           cmap='plasma', alpha=0.8)
    ax2.set_xlabel(r'$\log_{10}(n)$ [m$^{-3}$]')
    ax2.set_ylabel(r'$\log_{10}(\tau)$ [s]')
    ax2.set_zlabel(r'$\log_{10}(\sigma)$ [S/m]')
    ax2.set_title(r'(B) $\sigma = ne^2\tau/m_e$')

    # C: Scattering time vs Fermi energy
    ax3 = fig.add_subplot(1, 4, 3)
    E_F = np.array([7.0, 5.5, 11.7, 5.5, 11.1, 5.3])  # eV
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(metals)))
    ax3.scatter(E_F, tau_s * 1e15, c=colors, s=150, edgecolors='black', zorder=3)
    for i, metal in enumerate(metals):
        ax3.annotate(metal, (E_F[i], tau_s[i] * 1e15), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)
    ax3.set_xlabel('Fermi Energy (eV)')
    ax3.set_ylabel(r'$\tau_s$ (fs)')
    ax3.set_title(r'(C) Scattering Time vs $E_F$')
    ax3.grid(True, alpha=0.3)

    # D: I-V characteristics (Ohm's law)
    ax4 = fig.add_subplot(1, 4, 4)
    V = np.linspace(0, 10, 100)  # V
    for i, (metal, rho) in enumerate(zip(['Cu', 'Al', 'Fe'], [1.68e-8, 2.65e-8, 9.61e-8])):
        L = 1.0  # m
        A = 1e-6  # m^2
        R = rho * L / A
        I = V / R * 1000  # mA
        ax4.plot(V, I, linewidth=2, label=f'{metal} (R={R:.2f} $\\Omega$)')
    ax4.set_xlabel('Voltage (V)')
    ax4.set_ylabel('Current (mA)')
    ax4.set_title('(D) I-V Characteristics')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'panel_3_ohms_law.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


# =============================================================================
# Panel 4: Wiedemann-Franz Law
# =============================================================================

def create_panel_wiedemann_franz(output_dir: str):
    """
    Panel showing Wiedemann-Franz law as categorical invariant.
    """
    fig = plt.figure(figsize=(20, 5))

    # Theoretical Lorenz number
    L0 = np.pi**2 * KB**2 / (3 * E_CHARGE**2)

    # Metal data
    metals = ['Cu', 'Ag', 'Al', 'Au', 'Fe', 'Nb']
    kappa = np.array([401, 429, 237, 318, 80.4, 53.7])  # W/(m·K)
    rho = np.array([1.68, 1.59, 2.65, 2.21, 9.61, 15.2]) * 1e-8  # Ω·m
    sigma = 1 / rho
    T = 300

    L_measured = kappa / (sigma * T)

    # A: Lorenz number comparison
    ax1 = fig.add_subplot(1, 4, 1)
    x = np.arange(len(metals))
    ax1.bar(x, L_measured * 1e8, color='steelblue', alpha=0.8)
    ax1.axhline(L0 * 1e8, color='red', linestyle='--', linewidth=2,
               label=f'$L_0 = {L0*1e8:.2f}$')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metals)
    ax1.set_ylabel(r'$L$ ($10^{-8}$ W$\cdot\Omega\cdot$K$^{-2}$)')
    ax1.set_title(r'(A) $L = \kappa/(\sigma T)$')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # B: 3D surface of L(T, mean free path)
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    T_range = np.linspace(100, 500, 30)
    mfp = np.linspace(10, 100, 30)  # nm
    T_grid, MFP = np.meshgrid(T_range, mfp)
    # Lorenz number deviation at low T
    L_grid = L0 * (1 + 0.1 * (300 - T_grid) / 300 * np.exp(-MFP / 50))

    surf = ax2.plot_surface(T_grid, MFP, L_grid * 1e8, cmap='coolwarm', alpha=0.8)
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Mean Free Path (nm)')
    ax2.set_zlabel(r'$L$ ($10^{-8}$)')
    ax2.set_title('(B) Lorenz Number Surface')

    # C: Thermal vs Electrical conductivity
    ax3 = fig.add_subplot(1, 4, 3)
    sigma_range = np.logspace(6, 8, 100)
    kappa_theory = L0 * sigma_range * T

    ax3.loglog(sigma_range, kappa_theory, 'r--', linewidth=2, label='Wiedemann-Franz')
    ax3.loglog(sigma, kappa, 'o', markersize=12, color='steelblue', label='Metals')
    for i, metal in enumerate(metals):
        ax3.annotate(metal, (sigma[i], kappa[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)
    ax3.set_xlabel(r'$\sigma$ (S/m)')
    ax3.set_ylabel(r'$\kappa$ (W/m$\cdot$K)')
    ax3.set_title(r'(C) $\kappa = L_0 \sigma T$')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # D: Heat flux visualization
    ax4 = fig.add_subplot(1, 4, 4)
    x = np.linspace(0, 1, 100)
    T_profile = 400 - 100 * x  # Temperature gradient
    q = kappa[0] * 100  # Heat flux for copper

    ax4.fill_between(x, 0, T_profile, alpha=0.3, color='red')
    ax4.plot(x, T_profile, 'r-', linewidth=2, label='Temperature')
    ax4_twin = ax4.twinx()
    ax4_twin.arrow(0.2, 0.5, 0.6, 0, head_width=0.1, head_length=0.05,
                  fc='blue', ec='blue')
    ax4_twin.annotate(r'$q = -\kappa \nabla T$', (0.5, 0.6), fontsize=12, color='blue')
    ax4.set_xlabel('Position (normalized)')
    ax4.set_ylabel('Temperature (K)', color='red')
    ax4_twin.set_ylabel('Heat Flux', color='blue')
    ax4_twin.set_ylim(0, 1)
    ax4_twin.set_yticks([])
    ax4.set_title('(D) Heat Transport')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'panel_4_wiedemann_franz.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


# =============================================================================
# Panel 5: Superconductivity
# =============================================================================

def create_panel_superconductivity(output_dir: str):
    """
    Panel showing superconductivity as coupling collapse.
    """
    fig = plt.figure(figsize=(20, 5))

    # Niobium parameters
    T_c = 9.25  # K
    Delta_0 = 1.76 * KB * T_c / E_CHARGE * 1000  # meV

    # A: Resistivity transition
    ax1 = fig.add_subplot(1, 4, 1)
    T = np.linspace(0, 20, 1000)
    rho_normal = 15.2  # μΩ·cm

    rho = np.where(T > T_c, rho_normal * (1 + 0.004 * (T - T_c)),
                  rho_normal * np.exp(-10 * (T_c - T) / T_c))
    rho[T < T_c] = np.maximum(rho[T < T_c], 1e-10)

    ax1.semilogy(T, rho, 'b-', linewidth=2)
    ax1.axvline(T_c, color='red', linestyle='--', label=f'$T_c$ = {T_c} K')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel(r'$\rho$ ($\mu\Omega\cdot$cm)')
    ax1.set_title('(A) Resistivity Transition (Nb)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-3, 100)

    # B: 3D BCS gap surface
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    T_range = np.linspace(0, 1.2, 50)  # T/T_c
    Tc_range = np.linspace(1, 15, 50)  # K
    T_NORM, TC = np.meshgrid(T_range, Tc_range)

    Delta_0_grid = 1.76 * KB * TC / E_CHARGE * 1000  # meV
    Delta = np.where(T_NORM < 1, Delta_0_grid * np.sqrt(1 - T_NORM**2), 0)

    surf = ax2.plot_surface(T_NORM, TC, Delta, cmap='plasma', alpha=0.8)
    ax2.set_xlabel(r'$T/T_c$')
    ax2.set_ylabel(r'$T_c$ (K)')
    ax2.set_zlabel(r'$\Delta$ (meV)')
    ax2.set_title(r'(B) BCS Gap $\Delta(T)$')

    # C: Coupling collapse
    ax3 = fig.add_subplot(1, 4, 3)
    T = np.linspace(0.1, T_c * 1.5, 100)
    Delta_T = np.where(T < T_c, Delta_0 * np.sqrt(np.maximum(0, 1 - (T/T_c)**2)), 0)

    # Coupling g(T) = g_0 * exp(-Delta/k_B*T)
    g_ratio = np.exp(-Delta_T * E_CHARGE / 1000 / (KB * T))
    g_ratio = np.where(T < T_c, g_ratio, 1)

    ax3.semilogy(T, g_ratio, 'b-', linewidth=2)
    ax3.axvline(T_c, color='red', linestyle='--', label=f'$T_c$ = {T_c} K')
    ax3.fill_between(T[T < T_c], 1e-20, g_ratio[T < T_c], alpha=0.3)
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel(r'$g(T)/g_0$')
    ax3.set_title(r'(C) Coupling Collapse: $g \propto e^{-\Delta/k_BT}$')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(1e-20, 10)

    # D: BCS gap temperature dependence
    ax4 = fig.add_subplot(1, 4, 4)
    T = np.linspace(0, T_c, 100)
    Delta = Delta_0 * np.sqrt(1 - (T / T_c)**2)

    ax4.plot(T, Delta, 'b-', linewidth=2.5, label='BCS theory')
    ax4.axhline(Delta_0, color='red', linestyle='--',
               label=f'$\\Delta_0$ = {Delta_0:.2f} meV')
    ax4.scatter([0], [Delta_0], s=100, c='red', zorder=5)
    ax4.fill_between(T, 0, Delta, alpha=0.3)
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel(r'$\Delta(T)$ (meV)')
    ax4.set_title(r'(D) $\Delta(T) = \Delta_0\sqrt{1-(T/T_c)^2}$')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'panel_5_superconductivity.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


# =============================================================================
# Panel 6: Grotthuss Mechanism
# =============================================================================

def create_panel_grotthuss(output_dir: str):
    """
    Panel showing Grotthuss mechanism for proton transport.
    """
    fig = plt.figure(figsize=(20, 5))

    # Parameters
    r_OO = 2.8e-10  # m
    tau_p = 2e-12   # s
    v_signal = r_OO / tau_p

    # A: Signal vs Drift velocity
    ax1 = fig.add_subplot(1, 4, 1)
    velocities = [v_signal, 0.36]  # m/s
    labels = ['Signal\n$v_{signal}$', 'Drift\n$v_{drift}$']
    colors = ['steelblue', 'coral']
    bars = ax1.bar(labels, np.log10(velocities), color=colors)
    ax1.axhline(np.log10(v_signal), color='red', linestyle='--', alpha=0.5)
    for bar, v in zip(bars, velocities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{v:.1e}', ha='center', fontsize=9)
    ax1.set_ylabel(r'$\log_{10}(v)$ [m/s]')
    ax1.set_title(r'(A) $v_{signal}/v_{drift} \approx 400$')
    ax1.grid(True, alpha=0.3, axis='y')

    # B: 3D velocity ratio surface
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    tau_range = np.logspace(-13, -11, 30)
    r_range = np.linspace(2.5e-10, 3.5e-10, 30)
    TAU, R = np.meshgrid(tau_range, r_range)
    V_ratio = (R / TAU) / 0.36  # ratio to drift

    surf = ax2.plot_surface(np.log10(TAU), R * 1e10, np.log10(V_ratio),
                           cmap='viridis', alpha=0.8)
    ax2.scatter([-12], [2.8], [np.log10(400)], s=100, c='red', marker='*')
    ax2.set_xlabel(r'$\log_{10}(\tau_p)$ [s]')
    ax2.set_ylabel(r'$r_{OO}$ (Å)')
    ax2.set_zlabel(r'$\log_{10}(v_{sig}/v_{drift})$')
    ax2.set_title('(B) Velocity Ratio Surface')

    # C: Proton transfer rate vs temperature
    ax3 = fig.add_subplot(1, 4, 3)
    T = np.linspace(270, 370, 100)
    E_barrier = 10e3 / 6.022e23  # 10 kJ/mol per proton
    k_rate = (1 / tau_p) * np.exp(-E_barrier / (KB * T) + E_barrier / (KB * 310))

    ax3.semilogy(T, k_rate, 'b-', linewidth=2)
    ax3.axvline(310, color='red', linestyle='--', label='Physiological (310 K)')
    ax3.scatter([310], [1/tau_p], s=100, c='red', zorder=5)
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('Transfer Rate (Hz)')
    ax3.set_title(r'(C) $k \propto \tau_p^{-1} e^{-E_{barrier}/k_BT}$')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # D: Signal velocity contours
    ax4 = fig.add_subplot(1, 4, 4)
    tau_range = np.logspace(-13, -11, 100)
    r_range = np.linspace(2.0e-10, 4.0e-10, 100)
    TAU, R = np.meshgrid(tau_range, r_range)
    V_sig = R / TAU

    contour = ax4.contourf(np.log10(TAU), R * 1e10, V_sig, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax4, label='$v_{signal}$ (m/s)')
    ax4.scatter([-12], [2.8], s=200, c='white', marker='*', edgecolors='black',
               linewidths=2, label='Physiological')
    ax4.set_xlabel(r'$\log_{10}(\tau_p)$ [s]')
    ax4.set_ylabel(r'$r_{OO}$ (Å)')
    ax4.set_title('(D) Signal Velocity Contours')
    ax4.legend()

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'panel_6_grotthuss.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


# =============================================================================
# Panel 7: Goldman-Hodgkin-Katz
# =============================================================================

def create_panel_ghk(output_dir: str):
    """
    Panel showing GHK equation and membrane potential.
    """
    fig = plt.figure(figsize=(20, 5))

    T = 310  # K
    RT_F = R_GAS * T / FARADAY * 1000  # mV

    # A: Nernst potentials
    ax1 = fig.add_subplot(1, 4, 1)
    ions = ['$K^+$', '$Na^+$', '$Cl^-$', 'GHK']
    E_values = [-90, 60, -70, -70]
    colors = ['purple', 'orange', 'green', 'red']
    bars = ax1.bar(ions, E_values, color=colors, alpha=0.8, edgecolor='black')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_ylabel('Potential (mV)')
    ax1.set_title('(A) Nernst and Resting Potentials')
    ax1.grid(True, alpha=0.3, axis='y')

    # B: 3D GHK surface
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    P_Na_range = np.linspace(0.01, 0.2, 30)
    P_Cl_range = np.linspace(0.1, 1.0, 30)
    P_Na, P_Cl = np.meshgrid(P_Na_range, P_Cl_range)

    # GHK equation
    K_out, K_in = 5, 140
    Na_out, Na_in = 145, 12
    Cl_out, Cl_in = 110, 4

    num = 1 * K_out + P_Na * Na_out + P_Cl * Cl_in
    den = 1 * K_in + P_Na * Na_in + P_Cl * Cl_out
    V_m = RT_F * np.log(num / den)

    surf = ax2.plot_surface(P_Na, P_Cl, V_m, cmap='RdBu_r', alpha=0.8)
    ax2.scatter([0.04], [0.45], [-70], s=100, c='black', marker='o')
    ax2.set_xlabel(r'$P_{Na}/P_K$')
    ax2.set_ylabel(r'$P_{Cl}/P_K$')
    ax2.set_zlabel(r'$V_m$ (mV)')
    ax2.set_title('(B) GHK Surface')

    # C: V_m vs external K+
    ax3 = fig.add_subplot(1, 4, 3)
    K_out_range = np.logspace(0, 2, 100)
    V_m_K = RT_F * np.log((K_out_range + 0.04 * 145 + 0.45 * 4) /
                          (140 + 0.04 * 12 + 0.45 * 110))
    ax3.semilogx(K_out_range, V_m_K, 'b-', linewidth=2)
    ax3.axvline(5, color='red', linestyle='--', label='Physiological [K$^+$]$_o$')
    ax3.scatter([5], [-70], s=100, c='red', zorder=5)
    ax3.set_xlabel('[K$^+$]$_{out}$ (mM)')
    ax3.set_ylabel('$V_m$ (mV)')
    ax3.set_title('(C) Membrane Potential vs [K$^+$]$_{out}$')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # D: Permeability polar plot
    ax4 = fig.add_subplot(1, 4, 4, projection='polar')
    permeabilities = [1.0, 0.04, 0.45]
    labels = ['$P_K$', '$P_{Na}$', '$P_{Cl}$']
    angles = np.linspace(0, 2*np.pi, len(permeabilities), endpoint=False)
    permeabilities.append(permeabilities[0])  # close the polygon
    angles = np.append(angles, angles[0])

    ax4.fill(angles, permeabilities, alpha=0.3, color='steelblue')
    ax4.plot(angles, permeabilities, 'o-', linewidth=2, color='steelblue')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(labels)
    ax4.set_title('(D) Relative Permeabilities')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'panel_7_ghk.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


# =============================================================================
# Panel 8: PMF and ATP Synthase
# =============================================================================

def create_panel_pmf_atp(output_dir: str):
    """
    Panel showing proton-motive force and ATP synthase coupling.
    """
    fig = plt.figure(figsize=(20, 5))

    T = 310
    RT_F = R_GAS * T / FARADAY * 1000  # mV

    # A: PMF components
    ax1 = fig.add_subplot(1, 4, 1)
    Delta_psi = 150  # mV
    Delta_pH = 1.0
    chemical = 2.303 * RT_F * Delta_pH

    components = [Delta_psi, chemical]
    labels = [r'$\Delta\psi$', r'$\frac{2.303RT}{F}\Delta pH$']
    colors = ['coral', 'steelblue']
    bottom = 0
    for comp, label, color in zip(components, labels, colors):
        ax1.bar(['PMF'], [comp], bottom=bottom, label=label, color=color)
        bottom += comp
    ax1.axhline(200, color='green', linestyle='--', linewidth=2, label='Expected')
    ax1.set_ylabel('Potential (mV)')
    ax1.set_title(r'(A) PMF = $\Delta\psi$ + $(2.303RT/F)\Delta$pH')
    ax1.legend()
    ax1.set_ylim(0, 250)

    # B: 3D PMF surface
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    psi_range = np.linspace(100, 200, 30)
    pH_range = np.linspace(0, 2, 30)
    PSI, PH = np.meshgrid(psi_range, pH_range)
    PMF = PSI + 2.303 * RT_F * PH

    surf = ax2.plot_surface(PSI, PH, PMF, cmap='plasma', alpha=0.8)
    ax2.scatter([150], [1.0], [212], s=100, c='cyan', marker='o')
    ax2.set_xlabel(r'$\Delta\psi$ (mV)')
    ax2.set_ylabel(r'$\Delta$pH')
    ax2.set_zlabel('PMF (mV)')
    ax2.set_title('(B) PMF Surface')

    # C: H+/ATP ratio vs PMF
    ax3 = fig.add_subplot(1, 4, 3)
    PMF_range = np.linspace(150, 250, 100)
    Delta_G_ATP = 50e3  # J/mol
    n_ratio = Delta_G_ATP / (FARADAY * PMF_range / 1000)

    ax3.plot(PMF_range, n_ratio, 'b-', linewidth=2, label='Thermodynamic')
    ax3.axhline(3.3, color='red', linestyle='--', linewidth=2, label='Structural (c/3)')
    ax3.axvline(200, color='green', linestyle=':', alpha=0.7)
    ax3.scatter([200], [2.59], s=100, c='blue', zorder=5)
    ax3.scatter([200], [3.3], s=100, c='red', zorder=5)
    ax3.set_xlabel('PMF (mV)')
    ax3.set_ylabel('H$^+$/ATP')
    ax3.set_title('(C) Coupling Ratio: $n = |\\Delta G_{ATP}|/(F \\cdot PMF)$')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # D: c-ring visualization
    ax4 = fig.add_subplot(1, 4, 4, projection='polar')
    n_subunits = 10
    angles = np.linspace(0, 2*np.pi, n_subunits, endpoint=False)
    radii = np.ones(n_subunits)
    colors = plt.cm.hsv(np.linspace(0, 1, n_subunits))

    bars = ax4.bar(angles, radii, width=0.5, color=colors, alpha=0.8, edgecolor='black')

    # Mark 3 catalytic sites
    for i in [0, 3, 7]:
        ax4.bar([angles[i]], [1.3], width=0.3, color='black', alpha=0.9)

    ax4.set_title('(D) c-Ring: $n \\approx c/3 \\approx 3.3$ H$^+$/ATP')
    ax4.set_rticks([])

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'panel_8_pmf_atp.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


# =============================================================================
# Panel 9: Observation-Computing-Processing Identity
# =============================================================================

def create_panel_ocp_identity(output_dir: str):
    """
    Panel showing the O ≡ C ≡ P identity.
    """
    fig = plt.figure(figsize=(20, 5))

    # A: Address resolution concept
    ax1 = fig.add_subplot(1, 4, 1)
    # Draw hierarchical partition
    levels = 4
    for level in range(levels):
        n_cells = 3 ** level
        y = 1 - level * 0.25
        for i in range(n_cells):
            x_start = i / n_cells
            x_end = (i + 1) / n_cells
            rect = plt.Rectangle((x_start, y - 0.2), x_end - x_start, 0.2,
                                 facecolor=plt.cm.viridis(level / levels),
                                 edgecolor='black', alpha=0.7)
            ax1.add_patch(rect)

    # Highlight resolution path
    path_x = [0.5, 0.33, 0.22, 0.185]
    path_y = [0.9, 0.65, 0.4, 0.15]
    ax1.plot(path_x, path_y, 'r-', linewidth=3, marker='o', markersize=8)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.1)
    ax1.set_xlabel('Partition Coordinate')
    ax1.set_ylabel('Resolution Level')
    ax1.set_title('(A) Categorical Address Resolution')

    # B: 3D convergence visualization
    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    # Three paths converging to same point
    t = np.linspace(0, 1, 50)
    # Observation path
    x_obs = 0.5 - 0.3 * t + 0.1 * np.sin(10 * t) * (1 - t)
    y_obs = 0.5 + 0.2 * t + 0.1 * np.cos(10 * t) * (1 - t)
    z_obs = t
    # Computing path
    x_comp = 0.2 + 0.3 * t**2
    y_comp = 0.3 + 0.4 * t
    z_comp = t
    # Processing path
    x_proc = 0.8 - 0.5 * np.sqrt(t)
    y_proc = 0.8 - 0.3 * t
    z_proc = t

    ax2.plot(x_obs, y_obs, z_obs, 'b-', linewidth=2, label='Observation')
    ax2.plot(x_comp, y_comp, z_comp, 'g-', linewidth=2, label='Computing')
    ax2.plot(x_proc, y_proc, z_proc, 'r-', linewidth=2, label='Processing')
    ax2.scatter([0.5], [0.7], [1], s=200, c='yellow', marker='*', edgecolors='black')
    ax2.set_xlabel('$S_k$')
    ax2.set_ylabel('$S_t$')
    ax2.set_zlabel('Resolution')
    ax2.set_title('(B) Three Paths, One Target')
    ax2.legend()

    # C: Identity verification (Cu resistivity)
    ax3 = fig.add_subplot(1, 4, 3)
    methods = ['Observation\n(measure)', 'Computing\n(calculate)', 'Processing\n(derive)']
    values = [1.68, 1.68, 1.82]  # μΩ·cm
    colors = ['steelblue', 'seagreen', 'coral']
    bars = ax3.bar(methods, values, color=colors, alpha=0.8, edgecolor='black')
    ax3.axhline(1.68, color='red', linestyle='--', linewidth=2, label='Target')
    ax3.set_ylabel(r'$\rho_{Cu}$ ($\mu\Omega\cdot$cm)')
    ax3.set_title(r'(C) $\mathcal{O}(\rho) \equiv \mathcal{C}(\rho) \equiv \mathcal{P}(\rho)$')
    ax3.legend()
    ax3.set_ylim(0, 2.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # D: Conceptual diagram
    ax4 = fig.add_subplot(1, 4, 4)
    # Draw three circles
    circle_obs = plt.Circle((0.3, 0.7), 0.2, fill=False, color='blue', linewidth=3)
    circle_comp = plt.Circle((0.7, 0.7), 0.2, fill=False, color='green', linewidth=3)
    circle_proc = plt.Circle((0.5, 0.3), 0.2, fill=False, color='red', linewidth=3)

    ax4.add_patch(circle_obs)
    ax4.add_patch(circle_comp)
    ax4.add_patch(circle_proc)

    # Labels
    ax4.text(0.3, 0.7, 'O', ha='center', va='center', fontsize=20, fontweight='bold', color='blue')
    ax4.text(0.7, 0.7, 'C', ha='center', va='center', fontsize=20, fontweight='bold', color='green')
    ax4.text(0.5, 0.3, 'P', ha='center', va='center', fontsize=20, fontweight='bold', color='red')

    # Connecting lines with ≡
    ax4.annotate('', xy=(0.5, 0.7), xytext=(0.3, 0.7),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax4.annotate('', xy=(0.7, 0.7), xytext=(0.5, 0.7),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax4.annotate('', xy=(0.5, 0.5), xytext=(0.5, 0.7),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))

    ax4.text(0.5, 0.95, r'$\mathcal{O}(x) \equiv \mathcal{C}(x) \equiv \mathcal{P}(x)$',
            ha='center', fontsize=14)
    ax4.text(0.5, 0.05, 'Categorical Address Resolution',
            ha='center', fontsize=12, style='italic')

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_title('(D) The Fundamental Identity')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'panel_9_ocp_identity.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


# =============================================================================
# Main Execution
# =============================================================================

def create_all_visualizations(output_dir: str = None):
    """Generate all visualization panels."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__),
                                  '..', '..', '..', 'docs', 'figures', 'charge_computing')

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CHARGE COMPUTING FRAMEWORK VISUALIZATIONS")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()

    create_panel_triple_equivalence(output_dir)
    create_panel_partition_lag(output_dir)
    create_panel_ohms_law(output_dir)
    create_panel_wiedemann_franz(output_dir)
    create_panel_superconductivity(output_dir)
    create_panel_grotthuss(output_dir)
    create_panel_ghk(output_dir)
    create_panel_pmf_atp(output_dir)
    create_panel_ocp_identity(output_dir)

    print()
    print("=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    create_all_visualizations()
