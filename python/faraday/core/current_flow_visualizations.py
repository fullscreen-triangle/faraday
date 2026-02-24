"""
CURRENT FLOW VALIDATION VISUALIZATIONS
=======================================

Panel charts for visualizing electrical transport validation results.
Each section has 4 charts in a row, with at least one 3D chart.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import os

# Physical constants
k_B = 1.380649e-23
e = 1.602176634e-19
m_e = 9.1093837015e-31
hbar = 1.054571817e-34
c = 299792458
L_0 = (np.pi**2 / 3) * (k_B / e)**2

# Metal data
METALS = ['Cu', 'Al', 'Ag', 'Au', 'Fe', 'Nb']
METAL_COLORS = ['#b87333', '#848789', '#c0c0c0', '#ffd700', '#434343', '#5c6bc0']

METAL_PROPS = {
    'Cu': {'n': 8.47e28, 'rho_300': 1.68e-8, 'rho_77': 0.2e-8, 'kappa': 401, 'theta_D': 343, 'E_F': 7.0},
    'Al': {'n': 18.1e28, 'rho_300': 2.65e-8, 'rho_77': 0.3e-8, 'kappa': 237, 'theta_D': 428, 'E_F': 11.7},
    'Ag': {'n': 5.86e28, 'rho_300': 1.59e-8, 'rho_77': 0.2e-8, 'kappa': 429, 'theta_D': 225, 'E_F': 5.5},
    'Au': {'n': 5.90e28, 'rho_300': 2.44e-8, 'rho_77': 0.5e-8, 'kappa': 318, 'theta_D': 165, 'E_F': 5.5},
    'Fe': {'n': 17.0e28, 'rho_300': 9.71e-8, 'rho_77': 0.8e-8, 'kappa': 80, 'theta_D': 470, 'E_F': 11.1},
    'Nb': {'n': 5.56e28, 'rho_300': 15.2e-8, 'rho_77': 3.0e-8, 'kappa': 54, 'theta_D': 275, 'E_F': 5.3, 'T_c': 9.25},
}


def setup_style():
    """Set up matplotlib style."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def create_newton_cradle_panel(save_path=None):
    """
    Panel 1: Newton's Cradle Mechanism
    - Velocity comparison bars
    - 3D: Velocity ratio landscape
    - Log ratio across metals
    - Signal propagation schematic
    """
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle("Newton's Cradle: Signal vs Drift Velocity", fontsize=12, y=1.02)

    v_signal = 0.7 * c
    wire_area = 1e-6
    current = 1.0

    # Calculate drift velocities for all metals
    v_drifts = {}
    ratios = {}
    for metal, props in METAL_PROPS.items():
        v_drift = current / (props['n'] * e * wire_area)
        v_drifts[metal] = v_drift
        ratios[metal] = v_signal / v_drift

    # Chart 1: Velocity comparison (Cu)
    ax1 = fig.add_subplot(141)
    v_drift_cu = v_drifts['Cu']
    bars = ax1.bar(['Signal', 'Drift'], [v_signal, v_drift_cu],
                   color=['#e74c3c', '#3498db'], edgecolor='black', linewidth=1.2)
    ax1.set_yscale('log')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_ylim(1e-6, 1e9)
    ax1.set_title('Copper Wire')

    # Chart 2: 3D - Ratio vs carrier density and current
    ax2 = fig.add_subplot(142, projection='3d')
    n_range = np.linspace(5e28, 20e28, 25)
    I_range = np.linspace(0.1, 10, 25)
    N, I = np.meshgrid(n_range, I_range)
    V_DRIFT = I / (N * e * wire_area)
    RATIO = np.log10(v_signal / V_DRIFT)
    surf = ax2.plot_surface(N/1e28, I, RATIO, cmap=cm.plasma, alpha=0.9)
    ax2.set_xlabel('n (×10²⁸ m⁻³)')
    ax2.set_ylabel('I (A)')
    ax2.set_zlabel('log₁₀(ratio)')
    ax2.view_init(elev=20, azim=45)

    # Chart 3: Ratio comparison across metals
    ax3 = fig.add_subplot(143)
    log_ratios = [np.log10(ratios[m]) for m in METALS]
    bars = ax3.bar(METALS, log_ratios, color=METAL_COLORS, edgecolor='black')
    ax3.axhline(12, color='red', linestyle='--', alpha=0.7, label='Expected')
    ax3.set_ylabel('log₁₀(v_signal / v_drift)')
    ax3.set_ylim(11.5, 13)
    ax3.legend()

    # Chart 4: Wave propagation visualization
    ax4 = fig.add_subplot(144)
    x = np.linspace(0, 10, 100)
    t_values = [0, 0.25, 0.5, 0.75]
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(t_values)))
    for i, t in enumerate(t_values):
        wave = np.exp(-(x - 5*t)**2 / 2) * np.cos(4*np.pi*(x - 5*t))
        ax4.plot(x, wave + 3*i, color=colors[i], linewidth=2)
        ax4.fill_between(x, 3*i, wave + 3*i, alpha=0.3, color=colors[i])
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Time →')
    ax4.set_yticks([])
    ax4.set_xlim(0, 10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def create_ohm_law_panel(save_path=None):
    """
    Panel 2: Ohm's Law from Partition Dynamics
    - Resistivity comparison
    - 3D: Conductivity surface
    - Scattering time correlation
    - I-V characteristic
    """
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle("Ohm's Law: V = IR from Partition Dynamics", fontsize=12, y=1.02)

    # Chart 1: Resistivity at 300K
    ax1 = fig.add_subplot(141)
    rho_values = [METAL_PROPS[m]['rho_300'] * 1e8 for m in METALS]
    bars = ax1.bar(METALS, rho_values, color=METAL_COLORS, edgecolor='black')
    ax1.set_ylabel('ρ (×10⁻⁸ Ω·m)')
    ax1.set_title('Resistivity at 300K')

    # Chart 2: 3D - Conductivity vs n and tau
    ax2 = fig.add_subplot(142, projection='3d')
    n_range = np.linspace(5e28, 20e28, 25)
    tau_range = np.linspace(1e-15, 50e-15, 25)
    N, TAU = np.meshgrid(n_range, tau_range)
    SIGMA = N * e**2 * TAU / m_e
    surf = ax2.plot_surface(N/1e28, TAU*1e15, np.log10(SIGMA), cmap=cm.viridis, alpha=0.9)
    ax2.set_xlabel('n (×10²⁸ m⁻³)')
    ax2.set_ylabel('τ (fs)')
    ax2.set_zlabel('log₁₀(σ)')
    ax2.view_init(elev=25, azim=135)

    # Chart 3: Scattering time vs Fermi energy
    ax3 = fig.add_subplot(143)
    tau_values = []
    E_F_values = []
    for metal, props in METAL_PROPS.items():
        tau = m_e / (props['n'] * e**2 * props['rho_300'])
        tau_values.append(tau * 1e15)  # fs
        E_F_values.append(props['E_F'])
    ax3.scatter(E_F_values, tau_values, c=METAL_COLORS, s=150, edgecolor='black', zorder=5)
    for i, m in enumerate(METALS):
        ax3.annotate(m, (E_F_values[i], tau_values[i]), xytext=(5, 5),
                     textcoords='offset points', fontsize=9)
    ax3.set_xlabel('Fermi Energy (eV)')
    ax3.set_ylabel('Scattering Time (fs)')

    # Chart 4: I-V curves (linear - Ohmic)
    ax4 = fig.add_subplot(144)
    V = np.linspace(0, 10, 50)
    for i, metal in enumerate(['Cu', 'Al', 'Fe']):
        R = METAL_PROPS[metal]['rho_300'] * 0.1 / 1e-6  # 10cm wire, 1mm² area
        I = V / R
        ax4.plot(V, I * 1000, color=METAL_COLORS[METALS.index(metal)],
                 linewidth=2, label=metal)
    ax4.set_xlabel('Voltage (V)')
    ax4.set_ylabel('Current (mA)')
    ax4.legend()
    ax4.set_title('I-V Characteristic')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def create_temperature_panel(save_path=None):
    """
    Panel 3: Temperature Dependence
    - ρ(T) curves
    - 3D: Resistivity landscape
    - Bloch-Grüneisen behavior
    - RRR comparison
    """
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('Temperature Dependence: ρ(T) = ρ₀ + αT', fontsize=12, y=1.02)

    T_range = np.linspace(10, 400, 100)

    def bloch_gruneisen(T, props):
        theta_D = props['theta_D']
        rho_300 = props['rho_300']
        rho_77 = props['rho_77']
        rho_0 = rho_77 * 0.1

        rho = np.zeros_like(T)
        high_T = T > theta_D
        low_T = ~high_T

        rho[high_T] = rho_300 * (T[high_T] / 300)
        rho[low_T] = rho_0 + rho_300 * (T[low_T] / 300)**5 / (300 / theta_D)**4

        return rho

    # Chart 1: ρ(T) for multiple metals
    ax1 = fig.add_subplot(141)
    for i, metal in enumerate(['Cu', 'Ag', 'Al']):
        rho = bloch_gruneisen(T_range, METAL_PROPS[metal])
        ax1.plot(T_range, rho * 1e8, color=METAL_COLORS[METALS.index(metal)],
                 linewidth=2, label=metal)
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('ρ (×10⁻⁸ Ω·m)')
    ax1.legend()
    ax1.set_xlim(0, 400)

    # Chart 2: 3D - Resistivity vs T and Debye temperature
    ax2 = fig.add_subplot(142, projection='3d')
    T_grid = np.linspace(50, 400, 25)
    theta_grid = np.linspace(100, 500, 25)
    T_G, THETA_G = np.meshgrid(T_grid, theta_grid)
    RHO_G = np.where(T_G > THETA_G, T_G / 300, (T_G / 300)**5 / (300 / THETA_G)**4)
    surf = ax2.plot_surface(T_G, THETA_G, RHO_G, cmap=cm.coolwarm, alpha=0.9)
    ax2.set_xlabel('T (K)')
    ax2.set_ylabel('θ_D (K)')
    ax2.set_zlabel('ρ/ρ₃₀₀')
    ax2.view_init(elev=20, azim=225)

    # Chart 3: Low-T behavior (T^5)
    ax3 = fig.add_subplot(143)
    T_low = np.linspace(10, 100, 50)
    for i, metal in enumerate(['Cu', 'Fe']):
        rho = bloch_gruneisen(T_low, METAL_PROPS[metal])
        ax3.plot(T_low, rho * 1e9, color=METAL_COLORS[METALS.index(metal)],
                 linewidth=2, label=metal)
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('ρ (×10⁻⁹ Ω·m)')
    ax3.legend()
    ax3.set_title('Low-T: ρ ∝ T⁵')

    # Chart 4: RRR (Residual Resistivity Ratio)
    ax4 = fig.add_subplot(144)
    rrr_values = [METAL_PROPS[m]['rho_300'] / METAL_PROPS[m]['rho_77'] for m in METALS]
    bars = ax4.bar(METALS, rrr_values, color=METAL_COLORS, edgecolor='black')
    ax4.axhline(5, color='red', linestyle='--', alpha=0.7)
    ax4.set_ylabel('RRR = ρ₃₀₀/ρ₇₇')
    ax4.set_title('Residual Resistivity Ratio')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def create_matthiessen_panel(save_path=None):
    """
    Panel 4: Matthiessen's Rule
    - Phonon vs impurity contributions
    - 3D: Total resistivity surface
    - Additive scattering visualization
    - Impurity concentration effect
    """
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle("Matthiessen's Rule: ρ = ρ_phonon + ρ_impurity", fontsize=12, y=1.02)

    # Chart 1: Stacked bar - phonon vs impurity
    ax1 = fig.add_subplot(141)
    rho_phonon = []
    rho_impurity = []
    for metal in METALS:
        props = METAL_PROPS[metal]
        rho_ph = props['rho_300'] - props['rho_77'] * 0.1
        rho_imp = props['rho_300'] * 0.001 * 10  # 0.1% impurities
        rho_phonon.append(rho_ph * 1e8)
        rho_impurity.append(rho_imp * 1e8)

    x = np.arange(len(METALS))
    ax1.bar(x, rho_phonon, color='#e74c3c', label='Phonon', edgecolor='black')
    ax1.bar(x, rho_impurity, bottom=rho_phonon, color='#3498db',
            label='Impurity', edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(METALS)
    ax1.set_ylabel('ρ (×10⁻⁸ Ω·m)')
    ax1.legend()

    # Chart 2: 3D - Total resistivity vs T and impurity
    ax2 = fig.add_subplot(142, projection='3d')
    T_range = np.linspace(77, 400, 25)
    imp_range = np.linspace(0, 0.01, 25)
    T_G, IMP_G = np.meshgrid(T_range, imp_range)

    props = METAL_PROPS['Cu']
    RHO_PH = props['rho_300'] * (T_G / 300)
    RHO_IMP = props['rho_300'] * IMP_G * 10
    RHO_TOTAL = RHO_PH + RHO_IMP

    surf = ax2.plot_surface(T_G, IMP_G * 100, RHO_TOTAL * 1e8, cmap=cm.magma, alpha=0.9)
    ax2.set_xlabel('T (K)')
    ax2.set_ylabel('Impurity (%)')
    ax2.set_zlabel('ρ (×10⁻⁸)')
    ax2.view_init(elev=25, azim=45)

    # Chart 3: Scattering rate addition
    ax3 = fig.add_subplot(143)
    T = np.linspace(50, 400, 100)
    tau_ph = 25e-15 * (300 / T)  # Phonon scattering time
    tau_imp = 100e-15 * np.ones_like(T)  # Constant impurity scattering
    tau_total = 1 / (1/tau_ph + 1/tau_imp)  # Matthiessen

    ax3.plot(T, tau_ph * 1e15, 'r-', linewidth=2, label='τ_phonon')
    ax3.plot(T, tau_imp * 1e15, 'b--', linewidth=2, label='τ_impurity')
    ax3.plot(T, tau_total * 1e15, 'k-', linewidth=2.5, label='τ_total')
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('Scattering Time (fs)')
    ax3.legend()
    ax3.set_ylim(0, 150)

    # Chart 4: Impurity concentration sweep
    ax4 = fig.add_subplot(144)
    x_imp = np.linspace(0, 0.05, 50)  # 0-5% impurities
    rho_base = METAL_PROPS['Cu']['rho_300']
    rho_with_imp = rho_base * (1 + x_imp * 10)
    ax4.plot(x_imp * 100, rho_with_imp * 1e8, 'b-', linewidth=2)
    ax4.fill_between(x_imp * 100, rho_base * 1e8, rho_with_imp * 1e8,
                     alpha=0.3, color='blue')
    ax4.set_xlabel('Impurity Concentration (%)')
    ax4.set_ylabel('ρ (×10⁻⁸ Ω·m)')
    ax4.axhline(rho_base * 1e8, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def create_wiedemann_franz_panel(save_path=None):
    """
    Panel 5: Wiedemann-Franz Law
    - κ/σT comparison
    - 3D: Lorenz number surface
    - Thermal vs electrical conductivity
    - Electron heat transport
    """
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('Wiedemann-Franz Law: κ/σT = L₀', fontsize=12, y=1.02)

    # Calculate Lorenz numbers
    L_measured = {}
    for metal, props in METAL_PROPS.items():
        sigma = 1 / props['rho_300']
        kappa = props['kappa']
        L_measured[metal] = kappa / (sigma * 300)

    # Chart 1: Lorenz number comparison
    ax1 = fig.add_subplot(141)
    L_values = [L_measured[m] * 1e8 for m in METALS]
    bars = ax1.bar(METALS, L_values, color=METAL_COLORS, edgecolor='black')
    ax1.axhline(L_0 * 1e8, color='red', linestyle='--', linewidth=2, label='L₀ (theory)')
    ax1.set_ylabel('L (×10⁻⁸ W·Ω·K⁻²)')
    ax1.legend()
    ax1.set_ylim(0, 3.5)

    # Chart 2: 3D - L vs temperature and mean free path
    ax2 = fig.add_subplot(142, projection='3d')
    T_range = np.linspace(100, 500, 25)
    mfp_range = np.linspace(10, 100, 25)  # nm
    T_G, MFP_G = np.meshgrid(T_range, mfp_range)
    # At low T, phonon drag can cause deviations
    L_G = L_0 * (1 + 0.1 * np.exp(-T_G / 100) * (MFP_G / 50 - 1))
    surf = ax2.plot_surface(T_G, MFP_G, L_G * 1e8, cmap=cm.RdYlBu_r, alpha=0.9)
    ax2.set_xlabel('T (K)')
    ax2.set_ylabel('λ (nm)')
    ax2.set_zlabel('L (×10⁻⁸)')
    ax2.view_init(elev=25, azim=135)

    # Chart 3: Thermal vs electrical conductivity
    ax3 = fig.add_subplot(143)
    sigma_values = [1 / METAL_PROPS[m]['rho_300'] / 1e6 for m in METALS]
    kappa_values = [METAL_PROPS[m]['kappa'] for m in METALS]
    ax3.scatter(sigma_values, kappa_values, c=METAL_COLORS, s=150, edgecolor='black', zorder=5)
    for i, m in enumerate(METALS):
        ax3.annotate(m, (sigma_values[i], kappa_values[i]), xytext=(5, 5),
                     textcoords='offset points', fontsize=9)
    # Fit line
    sigma_fit = np.linspace(5, 70, 50)
    kappa_fit = L_0 * sigma_fit * 1e6 * 300
    ax3.plot(sigma_fit, kappa_fit, 'r--', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('σ (×10⁶ S/m)')
    ax3.set_ylabel('κ (W/m·K)')

    # Chart 4: Heat flux visualization
    ax4 = fig.add_subplot(144)
    x = np.linspace(0, 1, 100)
    # Temperature gradient
    T_profile = 400 - 100 * x
    ax4.plot(x, T_profile, 'r-', linewidth=2, label='T(x)')
    ax4.fill_between(x, 300, T_profile, alpha=0.3, color='red')
    # Heat flux arrows
    for xi in np.linspace(0.1, 0.9, 5):
        ax4.annotate('', xy=(xi + 0.08, 350), xytext=(xi, 350),
                     arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax4.set_xlabel('Position (normalized)')
    ax4.set_ylabel('Temperature (K)')
    ax4.set_ylim(280, 420)
    ax4.legend(loc='upper right')
    ax4.set_title('Heat Transport')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def create_superconductivity_panel(save_path=None):
    """
    Panel 6: Superconductivity as Coupling Collapse
    - ρ(T) transition
    - 3D: Energy gap surface
    - Coupling collapse curve
    - BCS gap temperature dependence
    """
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('Superconductivity: Coupling Collapse Below T_c', fontsize=12, y=1.02)

    T_c = 9.25  # Niobium
    delta_0 = 1.76 * k_B * T_c / e  # BCS gap at T=0 (eV)

    # Chart 1: Resistivity transition
    ax1 = fig.add_subplot(141)
    T = np.linspace(1, 20, 200)
    rho_normal = METAL_PROPS['Nb']['rho_300'] * (T / 300)

    # BCS-like transition
    g = np.ones_like(T)
    below_Tc = T < T_c
    delta_J = delta_0 * e
    g[below_Tc] = np.exp(-delta_J / (k_B * T[below_Tc]))
    rho = rho_normal * g

    ax1.plot(T, rho * 1e8, 'b-', linewidth=2)
    ax1.fill_between(T, 0, rho * 1e8, alpha=0.3)
    ax1.axvline(T_c, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('ρ (×10⁻⁸ Ω·m)')
    ax1.set_xlim(0, 20)
    ax1.annotate('T_c', (T_c, 0.5), fontsize=10, color='red')

    # Chart 2: 3D - Gap vs T and coupling strength
    ax2 = fig.add_subplot(142, projection='3d')
    T_range = np.linspace(0.1, 15, 30)
    Tc_range = np.linspace(5, 15, 30)
    T_G, TC_G = np.meshgrid(T_range, Tc_range)

    # BCS gap temperature dependence approximation
    t = T_G / TC_G
    DELTA = np.where(t < 1, 1.76 * (1 - t**2)**0.5, 0)

    surf = ax2.plot_surface(T_G, TC_G, DELTA, cmap=cm.coolwarm, alpha=0.9)
    ax2.set_xlabel('T (K)')
    ax2.set_ylabel('T_c (K)')
    ax2.set_zlabel('Δ/Δ₀')
    ax2.view_init(elev=20, azim=225)

    # Chart 3: Coupling strength collapse
    ax3 = fig.add_subplot(143)
    T_plot = np.linspace(1, 15, 100)
    g_plot = np.ones_like(T_plot)
    below = T_plot < T_c
    g_plot[below] = np.exp(-delta_0 * e / (k_B * T_plot[below]))

    ax3.plot(T_plot, g_plot, 'r-', linewidth=2.5)
    ax3.fill_between(T_plot, 0, g_plot, alpha=0.3, color='red')
    ax3.axvline(T_c, color='black', linestyle='--', alpha=0.7)
    ax3.axhline(1, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('Coupling g/g₀')
    ax3.set_xlim(0, 15)
    ax3.set_ylim(0, 1.1)
    ax3.set_title('Coupling Collapse')

    # Chart 4: BCS gap vs temperature
    ax4 = fig.add_subplot(144)
    t = np.linspace(0, 1, 100)
    delta_bcs = np.sqrt(1 - t**2)  # Approximate BCS
    delta_bcs[t > 1] = 0

    ax4.plot(t * T_c, delta_bcs * delta_0 * 1000, 'purple', linewidth=2.5)
    ax4.fill_between(t * T_c, 0, delta_bcs * delta_0 * 1000, alpha=0.3, color='purple')
    ax4.set_xlabel('Temperature (K)')
    ax4.set_ylabel('Δ (meV)')
    ax4.scatter([0], [delta_0 * 1000], color='red', s=100, zorder=5, edgecolor='black')
    ax4.annotate(f'Δ₀ = {delta_0*1000:.2f} meV', (0.5, delta_0 * 1000 * 0.9), fontsize=9)
    ax4.set_xlim(0, 12)
    ax4.set_title('BCS Energy Gap')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def create_all_panels(output_dir=None):
    """Generate all panel charts and save them."""
    setup_style()

    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "..", "..", "docs", "figures", "current_flow")

    os.makedirs(output_dir, exist_ok=True)

    panels = [
        ("panel_1_newton_cradle", create_newton_cradle_panel),
        ("panel_2_ohm_law", create_ohm_law_panel),
        ("panel_3_temperature", create_temperature_panel),
        ("panel_4_matthiessen", create_matthiessen_panel),
        ("panel_5_wiedemann_franz", create_wiedemann_franz_panel),
        ("panel_6_superconductivity", create_superconductivity_panel),
    ]

    saved_paths = []
    for name, create_func in panels:
        path = os.path.join(output_dir, f"{name}.png")
        fig = create_func(path)
        saved_paths.append(path)
        plt.close(fig)
        print(f"Saved: {path}")

    return saved_paths


if __name__ == "__main__":
    paths = create_all_panels()
    print(f"\nGenerated {len(paths)} panel charts")
