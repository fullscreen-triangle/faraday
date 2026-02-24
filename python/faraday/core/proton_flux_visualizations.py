"""
PROTON FLUX VALIDATION VISUALIZATIONS
=====================================

Panel charts for visualizing proton transport validation results.
Each section has 4 charts in a row, with at least one 3D chart.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import os
from datetime import datetime

# Physical constants
k_B = 1.380649e-23
e = 1.602176634e-19
R = 8.314462618
F = 96485.33212
N_A = 6.02214076e23
m_p = 1.67262192e-27
hbar = 1.054571817e-34


def setup_style():
    """Set up matplotlib style for publication-quality figures."""
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


def create_grotthuss_panel(save_path=None):
    """
    Panel 1: Grotthuss Mechanism Validation
    - Signal vs drift velocity comparison
    - 3D: Velocity ratio landscape
    - Proton transfer rate curve
    - Signal velocity contour
    """
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('Grotthuss Mechanism: Categorical State Propagation', fontsize=12, y=1.02)

    # Parameters
    r_OO = 2.8e-10  # m
    tau_p_base = 2.01e-12  # s
    v_signal = r_OO / tau_p_base
    mu_H = 36e-8  # m²/(V·s)
    E_field = 1e6  # V/m
    v_drift = mu_H * E_field

    # Chart 1: Velocity comparison (log scale bar)
    ax1 = fig.add_subplot(141)
    velocities = [v_signal, v_drift]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax1.bar(['Signal', 'Drift'], velocities, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_yscale('log')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_ylim(0.1, 1000)
    for bar, v in zip(bars, velocities):
        ax1.text(bar.get_x() + bar.get_width()/2, v*1.5, f'{v:.0e}', ha='center', fontsize=8)

    # Chart 2: 3D Surface - Velocity ratio vs tau_p and r_OO
    ax2 = fig.add_subplot(142, projection='3d')
    tau_range = np.linspace(0.5e-12, 5e-12, 30)
    r_range = np.linspace(2.4e-10, 3.2e-10, 30)
    TAU, R_OO = np.meshgrid(tau_range, r_range)
    V_SIGNAL = R_OO / TAU
    RATIO = np.log10(V_SIGNAL / v_drift)
    surf = ax2.plot_surface(TAU*1e12, R_OO*1e10, RATIO, cmap=cm.viridis, alpha=0.9)
    ax2.set_xlabel('τ_p (ps)')
    ax2.set_ylabel('r_OO (Å)')
    ax2.set_zlabel('log₁₀(ratio)')
    ax2.view_init(elev=25, azim=45)

    # Chart 3: Transfer rate vs temperature
    ax3 = fig.add_subplot(143)
    T_range = np.linspace(273, 350, 50)
    E_barrier = 10e3 / N_A  # J
    rates = (1/tau_p_base) * np.exp(-E_barrier / (k_B * T_range))
    ax3.plot(T_range, rates/1e11, 'b-', linewidth=2)
    ax3.fill_between(T_range, rates/1e11, alpha=0.3)
    ax3.axvline(310, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Temperature (K)')
    ax3.set_ylabel('Transfer Rate (×10¹¹ Hz)')
    ax3.scatter([310], [(1/tau_p_base) * np.exp(-E_barrier / (k_B * 310))/1e11],
                color='red', s=50, zorder=5)

    # Chart 4: Signal velocity field
    ax4 = fig.add_subplot(144)
    tau_grid = np.linspace(1e-12, 4e-12, 50)
    r_grid = np.linspace(2.5e-10, 3.1e-10, 50)
    TAU_G, R_G = np.meshgrid(tau_grid, r_grid)
    V_G = R_G / TAU_G
    contour = ax4.contourf(TAU_G*1e12, R_G*1e10, V_G, levels=20, cmap='plasma')
    ax4.plot([tau_p_base*1e12], [r_OO*1e10], 'w*', markersize=15, markeredgecolor='black')
    ax4.set_xlabel('τ_p (ps)')
    ax4.set_ylabel('r_OO (Å)')
    cbar = plt.colorbar(contour, ax=ax4)
    cbar.set_label('v_signal (m/s)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def create_gramicidin_panel(save_path=None):
    """
    Panel 2: Gramicidin Channel Conductance
    - Computed vs experimental range
    - 3D: Conductance landscape
    - Conductance vs water chain length
    - Series resistance model
    """
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('Gramicidin A: Single-File Water Channel Conductance', fontsize=12, y=1.02)

    # Parameters
    n_waters = 9
    tau_p = 2.01e-12
    g_hbond = 4.98  # J/m²
    T = 310.15

    def compute_conductance(n, tau, g):
        g_single = (e**2 / (k_B * T)) * (g * 1e-20 / tau)
        return (g_single / n) * 1e12  # pS

    G_computed = compute_conductance(n_waters, tau_p, g_hbond)
    G_exp_low, G_exp_high = 10, 100

    # Chart 1: Computed vs experimental
    ax1 = fig.add_subplot(141)
    ax1.bar([0], [G_computed], width=0.4, color='#3498db', label='Model', edgecolor='black')
    ax1.fill_between([-0.3, 0.3], [G_exp_low, G_exp_low], [G_exp_high, G_exp_high],
                     color='#2ecc71', alpha=0.3, label='Exp. range')
    ax1.axhline(G_exp_low, color='#2ecc71', linestyle='--', linewidth=1.5)
    ax1.axhline(G_exp_high, color='#2ecc71', linestyle='--', linewidth=1.5)
    ax1.set_xlim(-0.8, 0.8)
    ax1.set_xticks([0])
    ax1.set_xticklabels(['Gramicidin'])
    ax1.set_ylabel('Conductance (pS)')
    ax1.legend(loc='upper right')

    # Chart 2: 3D Surface - Conductance vs n_waters and tau_p
    ax2 = fig.add_subplot(142, projection='3d')
    n_range = np.arange(5, 15)
    tau_range = np.linspace(1e-12, 5e-12, 20)
    N, TAU = np.meshgrid(n_range, tau_range)
    G = np.zeros_like(N, dtype=float)
    for i in range(N.shape[0]):
        for j in range(N.shape[1]):
            G[i, j] = compute_conductance(N[i, j], TAU[i, j], g_hbond)
    surf = ax2.plot_surface(N, TAU*1e12, G, cmap=cm.coolwarm, alpha=0.9)
    ax2.set_xlabel('n waters')
    ax2.set_ylabel('τ_p (ps)')
    ax2.set_zlabel('G (pS)')
    ax2.view_init(elev=20, azim=135)

    # Chart 3: Conductance vs water chain length
    ax3 = fig.add_subplot(143)
    n_waters_range = np.arange(3, 20)
    G_vs_n = [compute_conductance(n, tau_p, g_hbond) for n in n_waters_range]
    ax3.plot(n_waters_range, G_vs_n, 'o-', color='#9b59b6', linewidth=2, markersize=6)
    ax3.axhspan(G_exp_low, G_exp_high, alpha=0.2, color='green')
    ax3.axvline(9, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Number of Waters')
    ax3.set_ylabel('Conductance (pS)')
    ax3.set_yscale('log')

    # Chart 4: Series resistance visualization
    ax4 = fig.add_subplot(144)
    n_bonds = np.arange(1, 16)
    R_single = 1 / compute_conductance(1, tau_p, g_hbond)
    R_total = R_single * n_bonds
    G_total = 1 / R_total
    ax4.fill_between(n_bonds, G_total, alpha=0.5, color='#e67e22')
    ax4.plot(n_bonds, G_total, 'k-', linewidth=2)
    ax4.scatter([9], [1/R_total[8]], color='red', s=100, zorder=5, edgecolor='black')
    ax4.set_xlabel('H-bonds in Series')
    ax4.set_ylabel('Channel Conductance (pS)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def create_ghk_panel(save_path=None):
    """
    Panel 3: Goldman-Hodgkin-Katz Equation
    - Nernst potentials comparison
    - 3D: GHK potential surface
    - Membrane potential vs K concentration
    - Permeability ratio effects
    """
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('Goldman-Hodgkin-Katz: Membrane Potential', fontsize=12, y=1.02)

    T = 310.15
    # Ion concentrations (mM)
    K_in, K_out = 140, 5
    Na_in, Na_out = 10, 145
    Cl_in, Cl_out = 10, 110

    def nernst(z, c_out, c_in):
        return (R * T / (z * F)) * np.log(c_out / c_in) * 1000  # mV

    def ghk(P_K, P_Na, P_Cl):
        num = P_K * K_out + P_Na * Na_out + P_Cl * Cl_in
        den = P_K * K_in + P_Na * Na_in + P_Cl * Cl_out
        return (R * T / F) * np.log(num / den) * 1000  # mV

    E_K = nernst(1, K_out, K_in)
    E_Na = nernst(1, Na_out, Na_in)
    E_Cl = nernst(-1, Cl_out, Cl_in)
    V_ghk = ghk(1.0, 0.04, 0.45)

    # Chart 1: Nernst potentials
    ax1 = fig.add_subplot(141)
    ions = ['K⁺', 'Na⁺', 'Cl⁻', 'GHK']
    potentials = [E_K, E_Na, E_Cl, V_ghk]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    bars = ax1.bar(ions, potentials, color=colors, edgecolor='black', linewidth=1.2)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axhline(-70, color='gray', linestyle='--', alpha=0.7)
    ax1.set_ylabel('Potential (mV)')

    # Chart 2: 3D Surface - GHK vs P_Na and P_Cl
    ax2 = fig.add_subplot(142, projection='3d')
    P_Na_range = np.linspace(0.01, 0.2, 25)
    P_Cl_range = np.linspace(0.1, 1.0, 25)
    P_NA, P_CL = np.meshgrid(P_Na_range, P_Cl_range)
    V_GHK = np.zeros_like(P_NA)
    for i in range(P_NA.shape[0]):
        for j in range(P_NA.shape[1]):
            V_GHK[i, j] = ghk(1.0, P_NA[i, j], P_CL[i, j])
    surf = ax2.plot_surface(P_NA, P_CL, V_GHK, cmap=cm.RdYlBu_r, alpha=0.9)
    ax2.set_xlabel('P_Na/P_K')
    ax2.set_ylabel('P_Cl/P_K')
    ax2.set_zlabel('V_m (mV)')
    ax2.view_init(elev=25, azim=225)

    # Chart 3: Membrane potential vs extracellular K+
    ax3 = fig.add_subplot(143)
    K_out_range = np.linspace(2, 50, 50)
    V_vs_K = []
    for k in K_out_range:
        num = 1.0 * k + 0.04 * Na_out + 0.45 * Cl_in
        den = 1.0 * K_in + 0.04 * Na_in + 0.45 * Cl_out
        V_vs_K.append((R * T / F) * np.log(num / den) * 1000)
    ax3.plot(K_out_range, V_vs_K, 'b-', linewidth=2)
    ax3.fill_between(K_out_range, V_vs_K, -100, alpha=0.2)
    ax3.axvline(5, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('[K⁺]_out (mM)')
    ax3.set_ylabel('V_m (mV)')

    # Chart 4: Permeability ratio wheel
    ax4 = fig.add_subplot(144, projection='polar')
    P_ratios = [1.0, 0.04, 0.45]  # K, Na, Cl
    labels = ['P_K', 'P_Na', 'P_Cl']
    angles = np.linspace(0, 2*np.pi, len(P_ratios), endpoint=False)
    # Normalize for visualization
    P_norm = np.array(P_ratios) / max(P_ratios)
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax4.bar(angles, P_norm, width=0.5, color=colors, alpha=0.8, edgecolor='black')
    ax4.set_xticks(angles)
    ax4.set_xticklabels(labels)
    ax4.set_ylim(0, 1.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def create_pmf_panel(save_path=None):
    """
    Panel 4: Proton-Motive Force
    - PMF components stacked
    - 3D: PMF landscape
    - PMF vs pH gradient
    - Energy coupling diagram
    """
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('Proton-Motive Force: Chemiosmotic Coupling', fontsize=12, y=1.02)

    T = 310.15
    delta_psi = 150  # mV
    delta_pH = 1.0

    def pmf(psi, dpH):
        chemical = (2.303 * R * T / F) * dpH * 1000  # mV
        return abs(psi) + chemical

    pmf_total = pmf(delta_psi, delta_pH)
    chemical_component = (2.303 * R * T / F) * delta_pH * 1000

    # Chart 1: Stacked components
    ax1 = fig.add_subplot(141)
    ax1.bar([0], [delta_psi], color='#e74c3c', label='Δψ', edgecolor='black')
    ax1.bar([0], [chemical_component], bottom=[delta_psi], color='#3498db',
            label='ΔpH', edgecolor='black')
    ax1.set_xlim(-0.8, 0.8)
    ax1.set_xticks([0])
    ax1.set_xticklabels(['PMF'])
    ax1.set_ylabel('Potential (mV)')
    ax1.legend(loc='upper right')
    ax1.axhline(200, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

    # Chart 2: 3D Surface - PMF vs delta_psi and delta_pH
    ax2 = fig.add_subplot(142, projection='3d')
    psi_range = np.linspace(100, 200, 25)
    pH_range = np.linspace(0, 2, 25)
    PSI, DPH = np.meshgrid(psi_range, pH_range)
    PMF = np.zeros_like(PSI)
    for i in range(PSI.shape[0]):
        for j in range(PSI.shape[1]):
            PMF[i, j] = pmf(PSI[i, j], DPH[i, j])
    surf = ax2.plot_surface(PSI, DPH, PMF, cmap=cm.magma, alpha=0.9)
    ax2.set_xlabel('Δψ (mV)')
    ax2.set_ylabel('ΔpH')
    ax2.set_zlabel('PMF (mV)')
    ax2.scatter([150], [1.0], [pmf_total], color='cyan', s=100, edgecolor='black')
    ax2.view_init(elev=20, azim=45)

    # Chart 3: PMF vs pH gradient
    ax3 = fig.add_subplot(143)
    dpH_range = np.linspace(0, 2, 50)
    pmf_vs_pH = [pmf(150, dp) for dp in dpH_range]
    ax3.plot(dpH_range, pmf_vs_pH, 'r-', linewidth=2)
    ax3.fill_between(dpH_range, pmf_vs_pH, 150, alpha=0.3, color='blue', label='ΔpH contribution')
    ax3.fill_between(dpH_range, 0, 150, alpha=0.3, color='red', label='Δψ contribution')
    ax3.axvline(1.0, color='black', linestyle='--', alpha=0.7)
    ax3.set_xlabel('ΔpH')
    ax3.set_ylabel('PMF (mV)')
    ax3.legend(loc='upper left')

    # Chart 4: Energy flux arrows (vector field)
    ax4 = fig.add_subplot(144)
    x = np.linspace(0, 2, 8)
    y = np.linspace(0, 2, 8)
    X, Y = np.meshgrid(x, y)
    # Arrows pointing toward high PMF region
    U = 0.3 * np.ones_like(X)
    V = 0.3 * np.ones_like(Y)
    magnitude = np.sqrt(X**2 + Y**2)
    ax4.quiver(X, Y, U, V, magnitude, cmap='plasma', scale=5)
    ax4.scatter([1.5], [1.5], s=200, c='red', marker='*', edgecolor='black', zorder=5)
    ax4.set_xlabel('Δψ (norm)')
    ax4.set_ylabel('ΔpH (norm)')
    ax4.set_xlim(-0.2, 2.2)
    ax4.set_ylim(-0.2, 2.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def create_atp_synthase_panel(save_path=None):
    """
    Panel 5: ATP Synthase Coupling
    - H+/ATP comparison
    - 3D: Coupling efficiency
    - Energy balance
    - Rotary mechanism phase
    """
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('ATP Synthase: Rotary Mechanism Coupling', fontsize=12, y=1.02)

    pmf_mV = 200
    delta_G_ATP = -50  # kJ/mol

    def protons_per_atp(pmf, dG):
        pmf_J = pmf * 1e-3 * F  # J/mol
        return abs(dG * 1000) / pmf_J

    n_computed = protons_per_atp(pmf_mV, delta_G_ATP)
    n_experimental = 3.3

    # Chart 1: H+/ATP comparison
    ax1 = fig.add_subplot(141)
    ax1.bar(['Computed', 'Measured'], [n_computed, n_experimental],
            color=['#3498db', '#2ecc71'], edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('H⁺ per ATP')
    ax1.set_ylim(0, 4)
    for i, v in enumerate([n_computed, n_experimental]):
        ax1.text(i, v + 0.1, f'{v:.2f}', ha='center')

    # Chart 2: 3D - H+/ATP vs PMF and dG_ATP
    ax2 = fig.add_subplot(142, projection='3d')
    pmf_range = np.linspace(150, 250, 25)
    dG_range = np.linspace(-60, -40, 25)
    PMF, DG = np.meshgrid(pmf_range, dG_range)
    N_H = np.zeros_like(PMF)
    for i in range(PMF.shape[0]):
        for j in range(PMF.shape[1]):
            N_H[i, j] = protons_per_atp(PMF[i, j], DG[i, j])
    surf = ax2.plot_surface(PMF, DG, N_H, cmap=cm.viridis, alpha=0.9)
    ax2.set_xlabel('PMF (mV)')
    ax2.set_ylabel('ΔG_ATP (kJ/mol)')
    ax2.set_zlabel('H⁺/ATP')
    ax2.scatter([200], [-50], [n_computed], color='red', s=100, edgecolor='black')
    ax2.view_init(elev=25, azim=135)

    # Chart 3: Energy efficiency curve
    ax3 = fig.add_subplot(143)
    pmf_scan = np.linspace(100, 300, 50)
    efficiency = []
    for p in pmf_scan:
        n = protons_per_atp(p, delta_G_ATP)
        # Efficiency = (energy captured) / (energy available)
        captured = n * p * 1e-3 * F / 1000  # kJ/mol
        eff = min(captured / abs(delta_G_ATP) * 100, 100)
        efficiency.append(eff)
    ax3.plot(pmf_scan, efficiency, 'g-', linewidth=2)
    ax3.fill_between(pmf_scan, efficiency, alpha=0.3, color='green')
    ax3.axvline(200, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('PMF (mV)')
    ax3.set_ylabel('Efficiency (%)')
    ax3.set_ylim(0, 110)

    # Chart 4: Rotary mechanism - c-ring subunits
    ax4 = fig.add_subplot(144, projection='polar')
    c_subunits = int(np.round(n_computed * 3))  # ~10 c-subunits
    angles = np.linspace(0, 2*np.pi, c_subunits, endpoint=False)
    radii = np.ones(c_subunits)
    colors_ring = plt.cm.hsv(np.linspace(0, 1, c_subunits))
    bars = ax4.bar(angles, radii, width=2*np.pi/c_subunits * 0.9,
                   color=colors_ring, alpha=0.8, edgecolor='black')
    ax4.set_ylim(0, 1.5)
    ax4.set_yticks([])
    # Mark 3 catalytic sites
    for i in range(3):
        angle = i * 2*np.pi/3
        ax4.plot([angle, angle], [1.1, 1.4], 'k-', linewidth=3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def create_hbond_panel(save_path=None):
    """
    Panel 6: Hydrogen Bond Parameters
    - Parameter validation bars
    - 3D: Partition lag surface
    - Coupling strength sensitivity
    - Energy landscape
    """
    fig = plt.figure(figsize=(16, 4))
    fig.suptitle('H-Bond Network Parameters: Literature Validation', fontsize=12, y=1.02)

    # Model parameters
    E_hbond = 20.0  # kJ/mol
    g_hbond = 30.0  # kJ/(mol·Å²)
    tau_reorg = 2.0  # ps
    E_barrier = 10.0  # kJ/mol

    # Literature ranges
    E_range = [15, 25]
    g_range = [20, 50]
    tau_range = [1, 10]
    barrier_range = [5, 15]

    # Chart 1: Parameter validation
    ax1 = fig.add_subplot(141)
    params = ['E_hbond', 'g_hbond', 'τ_reorg', 'E_barrier']
    values = [E_hbond, g_hbond, tau_reorg, E_barrier]
    ranges = [E_range, g_range, tau_range, barrier_range]
    x_pos = np.arange(len(params))

    # Normalize for comparison
    norm_values = [(v - r[0]) / (r[1] - r[0]) for v, r in zip(values, ranges)]
    colors = ['#2ecc71' if 0 <= nv <= 1 else '#e74c3c' for nv in norm_values]

    ax1.bar(x_pos, [1]*4, color='lightgray', edgecolor='black', alpha=0.5)
    ax1.bar(x_pos, norm_values, color=colors, edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(params, rotation=45)
    ax1.set_ylabel('Normalized Value')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axhline(1, color='black', linewidth=0.5)

    # Chart 2: 3D - Partition lag vs E_barrier and tau_reorg
    ax2 = fig.add_subplot(142, projection='3d')
    E_b_range = np.linspace(5, 20, 25)
    tau_r_range = np.linspace(0.5, 5, 25)
    E_B, TAU_R = np.meshgrid(E_b_range, tau_r_range)
    # tau_p = hbar/E + tau_reorg
    TAU_P = hbar / (E_B * 1000 / N_A) * 1e12 + TAU_R  # ps
    surf = ax2.plot_surface(E_B, TAU_R, TAU_P, cmap=cm.plasma, alpha=0.9)
    ax2.set_xlabel('E_barrier (kJ/mol)')
    ax2.set_ylabel('τ_reorg (ps)')
    ax2.set_zlabel('τ_p (ps)')
    ax2.scatter([10], [2], [2.01], color='cyan', s=100, edgecolor='black')
    ax2.view_init(elev=25, azim=225)

    # Chart 3: Coupling strength effect on conductance
    ax3 = fig.add_subplot(143)
    g_scan = np.linspace(10, 60, 50)
    T = 310.15
    tau_p = 2.01e-12
    n_bonds = 9
    conductance = []
    for g in g_scan:
        g_J_m2 = g * 1000 / N_A * 1e20
        g_single = (e**2 / (k_B * T)) * (g_J_m2 * 1e-20 / tau_p)
        G = (g_single / n_bonds) * 1e12
        conductance.append(G)
    ax3.plot(g_scan, conductance, 'b-', linewidth=2)
    ax3.fill_between(g_scan, conductance, alpha=0.3)
    ax3.axvline(30, color='red', linestyle='--')
    ax3.axvspan(20, 50, alpha=0.1, color='green')
    ax3.set_xlabel('g_hbond (kJ/mol·Å²)')
    ax3.set_ylabel('Conductance (pS)')

    # Chart 4: Energy landscape (double well)
    ax4 = fig.add_subplot(144)
    x = np.linspace(-2, 2, 100)
    # Double well potential
    V = E_barrier * (x**4 - 2*x**2) / 4 + E_hbond/2
    ax4.plot(x, V, 'k-', linewidth=2)
    ax4.fill_between(x, V, V.min()-2, alpha=0.3, color='purple')
    ax4.axhline(E_barrier, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Reaction Coordinate')
    ax4.set_ylabel('Energy (kJ/mol)')
    ax4.set_ylim(V.min()-2, V.max()+5)
    # Mark minima
    ax4.scatter([-1, 1], [V[25], V[75]], color='blue', s=80, zorder=5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def create_all_panels(output_dir=None):
    """Generate all panel charts and save them."""
    setup_style()

    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "..", "..", "..", "docs", "figures", "proton_flux")

    os.makedirs(output_dir, exist_ok=True)

    panels = [
        ("panel_1_grotthuss", create_grotthuss_panel),
        ("panel_2_gramicidin", create_gramicidin_panel),
        ("panel_3_ghk", create_ghk_panel),
        ("panel_4_pmf", create_pmf_panel),
        ("panel_5_atp_synthase", create_atp_synthase_panel),
        ("panel_6_hbond", create_hbond_panel),
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
