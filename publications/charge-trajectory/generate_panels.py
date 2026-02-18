"""
Generate visualization panels for Cellular Charge Trajectories paper.
Each panel: 1x4 layout, minimal text, at least one 3D chart.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os

# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['legend.fontsize'] = 6

# Colors
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'success': '#3A7D44',
    'dark': '#1B1B1E',
    'light': '#E8E8E8',
    'A': '#E63946',  # Adenine - red
    'T': '#457B9D',  # Thymine - blue
    'G': '#2A9D8F',  # Guanine - green
    'C': '#E9C46A',  # Cytosine - yellow
}

def create_output_dir():
    """Create figures directory if it doesn't exist."""
    os.makedirs('figures', exist_ok=True)

# =============================================================================
# PANEL 1: Partition Coordinates from Bounded Phase Space
# =============================================================================
def generate_panel1_partition():
    """Section 2: Partition coordinates visualization."""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Chart 1: 3D partition states (n, ℓ, m)
    ax1 = fig.add_subplot(gs[0], projection='3d')

    states = []
    colors = []
    sizes = []
    for n in range(1, 5):
        for l in range(n):
            for m in range(-l, l+1):
                states.append((n, l, m))
                colors.append(plt.cm.viridis(n/5))
                sizes.append(50 + 20*n)

    states = np.array(states)
    ax1.scatter(states[:,0], states[:,1], states[:,2],
                c=[plt.cm.viridis(s[0]/5) for s in states],
                s=sizes, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax1.set_xlabel('n', fontsize=8)
    ax1.set_ylabel('ℓ', fontsize=8)
    ax1.set_zlabel('m', fontsize=8)
    ax1.set_title('Partition States', fontsize=9, fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # Chart 2: Capacity formula C(n) = 2n²
    ax2 = fig.add_subplot(gs[1])
    n_vals = np.arange(1, 8)
    capacity = 2 * n_vals**2
    ax2.bar(n_vals, capacity, color=COLORS['primary'], alpha=0.8, edgecolor='white')
    ax2.plot(n_vals, capacity, 'o-', color=COLORS['quaternary'], markersize=5, linewidth=1.5)
    ax2.set_xlabel('n')
    ax2.set_ylabel('C(n)')
    ax2.set_title('C(n) = 2n²', fontsize=9, fontweight='bold')
    ax2.set_xticks(n_vals)

    # Chart 3: Subshell capacities
    ax3 = fig.add_subplot(gs[2])
    subshells = ['s', 'p', 'd', 'f', 'g']
    l_vals = [0, 1, 2, 3, 4]
    capacities = [2*(2*l+1) for l in l_vals]
    bars = ax3.bar(subshells, capacities, color=[plt.cm.plasma(l/5) for l in l_vals],
                   alpha=0.8, edgecolor='white')
    ax3.set_xlabel('Subshell')
    ax3.set_ylabel('Capacity')
    ax3.set_title('2(2ℓ+1)', fontsize=9, fontweight='bold')
    for bar, cap in zip(bars, capacities):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(cap), ha='center', va='bottom', fontsize=7)

    # Chart 4: Cumulative shell filling
    ax4 = fig.add_subplot(gs[3])
    n_range = np.arange(1, 8)
    cumulative = np.cumsum(2 * n_range**2)
    ax4.fill_between(n_range, cumulative, alpha=0.3, color=COLORS['primary'])
    ax4.plot(n_range, cumulative, 'o-', color=COLORS['primary'], markersize=6, linewidth=2)
    # Mark noble gases
    noble = {2: 'He', 10: 'Ne', 28: 'Ni*', 60: '*'}
    for n, cum in zip(n_range, cumulative):
        if cum in [2, 10]:
            ax4.annotate(noble[cum], (n, cum), textcoords="offset points",
                        xytext=(5, 5), fontsize=7)
    ax4.set_xlabel('n')
    ax4.set_ylabel('Σ C(n)')
    ax4.set_title('Cumulative', fontsize=9, fontweight='bold')
    ax4.set_xticks(n_range)

    plt.tight_layout()
    plt.savefig('figures/panel1_partition.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figures/panel1_partition.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated panel1_partition.png")

# =============================================================================
# PANEL 2: Selection Rules from Boundary Continuity
# =============================================================================
def generate_panel2_selection():
    """Section 3: Selection rules visualization."""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Chart 1: 3D transition pathways
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Plot states
    states = []
    for n in range(1, 4):
        for l in range(n):
            for m in range(-l, l+1):
                states.append((n, l, m))
    states = np.array(states)
    ax1.scatter(states[:,0], states[:,1], states[:,2],
                c='gray', s=60, alpha=0.6, edgecolors='white')

    # Plot allowed transitions (Δℓ = ±1)
    transitions = []
    for i, s1 in enumerate(states):
        for j, s2 in enumerate(states):
            if abs(s1[1] - s2[1]) == 1 and abs(s1[2] - s2[2]) <= 1:
                transitions.append([s1, s2])

    for t in transitions[:20]:  # Limit for clarity
        ax1.plot([t[0][0], t[1][0]], [t[0][1], t[1][1]], [t[0][2], t[1][2]],
                'g-', alpha=0.5, linewidth=1)

    ax1.set_xlabel('n', fontsize=8)
    ax1.set_ylabel('ℓ', fontsize=8)
    ax1.set_zlabel('m', fontsize=8)
    ax1.set_title('Allowed Paths', fontsize=9, fontweight='bold')
    ax1.view_init(elev=25, azim=45)

    # Chart 2: Selection rule matrix
    ax2 = fig.add_subplot(gs[1])
    l_vals = [0, 1, 2, 3]
    n_l = len(l_vals)
    matrix = np.zeros((n_l, n_l))
    for i in range(n_l):
        for j in range(n_l):
            if abs(l_vals[i] - l_vals[j]) == 1:
                matrix[i, j] = 1

    im = ax2.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_xticks(range(n_l))
    ax2.set_yticks(range(n_l))
    ax2.set_xticklabels(['s', 'p', 'd', 'f'])
    ax2.set_yticklabels(['s', 'p', 'd', 'f'])
    ax2.set_xlabel("ℓ'")
    ax2.set_ylabel('ℓ')
    ax2.set_title('Δℓ = ±1', fontsize=9, fontweight='bold')

    # Chart 3: Enforcement ratio
    ax3 = fig.add_subplot(gs[2])
    categories = ['Allowed', 'Forbidden']
    rates = [1e12, 1e4]
    bars = ax3.bar(categories, rates, color=[COLORS['success'], COLORS['quaternary']],
                   alpha=0.8, edgecolor='white')
    ax3.set_yscale('log')
    ax3.set_ylabel('Rate (s⁻¹)')
    ax3.set_title('Enforcement >10⁸', fontsize=9, fontweight='bold')
    ax3.axhline(y=1e8, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Chart 4: Categorical distance
    ax4 = fig.add_subplot(gs[3])
    delta_l = [0, 1, 2, 3, 4]
    d_c = delta_l  # Categorical distance = |Δℓ|
    ax4.bar(delta_l, d_c, color=COLORS['primary'], alpha=0.8, edgecolor='white')
    ax4.plot(delta_l, d_c, 'o-', color=COLORS['quaternary'], markersize=6, linewidth=2)
    ax4.set_xlabel('|Δℓ|')
    ax4.set_ylabel('dC')
    ax4.set_title('dC = |Δℓ|', fontsize=9, fontweight='bold')
    ax4.set_xticks(delta_l)

    plt.tight_layout()
    plt.savefig('figures/panel2_selection.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figures/panel2_selection.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated panel2_selection.png")

# =============================================================================
# PANEL 3: Phase-Lock Dynamics from Coupled Oscillators
# =============================================================================
def generate_panel3_phaselock():
    """Section 4: Phase-lock dynamics visualization."""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Chart 1: 3D oscillator network
    ax1 = fig.add_subplot(gs[0], projection='3d')

    np.random.seed(42)
    n_osc = 30
    pos = np.random.randn(n_osc, 3) * 2
    phases = np.random.uniform(0, 2*np.pi, n_osc)

    # Color by phase
    colors = plt.cm.hsv(phases / (2*np.pi))
    ax1.scatter(pos[:,0], pos[:,1], pos[:,2], c=colors, s=80, alpha=0.8,
                edgecolors='white', linewidth=0.5)

    # Draw some connections
    for i in range(n_osc):
        for j in range(i+1, n_osc):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < 2.5:
                ax1.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                        [pos[i,2], pos[j,2]], 'k-', alpha=0.2, linewidth=0.5)

    ax1.set_xlabel('x', fontsize=8)
    ax1.set_ylabel('y', fontsize=8)
    ax1.set_zlabel('z', fontsize=8)
    ax1.set_title('Network', fontsize=9, fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # Chart 2: Order parameter evolution
    ax2 = fig.add_subplot(gs[1])
    t = np.linspace(0, 10, 200)
    # Simulate synchronization transition
    r = 0.1 + 0.8 * (1 - np.exp(-t/2)) + 0.05*np.sin(5*t)*np.exp(-t/3)
    ax2.plot(t, r, color=COLORS['primary'], linewidth=2)
    ax2.axhline(y=0.8, color=COLORS['success'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(y=0.5, color=COLORS['quaternary'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.fill_between(t, 0.8, 1, alpha=0.2, color=COLORS['success'])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('⟨r⟩')
    ax2.set_ylim(0, 1)
    ax2.set_title('Order Parameter', fontsize=9, fontweight='bold')

    # Chart 3: Coupling strength decay
    ax3 = fig.add_subplot(gs[2])
    r = np.linspace(0, 5, 100)
    lambda_D = 0.8  # nm
    K = np.exp(-r/lambda_D) / (r + 0.1)
    K = K / K.max()
    ax3.plot(r, K, color=COLORS['secondary'], linewidth=2)
    ax3.fill_between(r, K, alpha=0.3, color=COLORS['secondary'])
    ax3.axvline(x=lambda_D, color='gray', linestyle='--', linewidth=1)
    ax3.text(lambda_D + 0.1, 0.8, 'λD', fontsize=8)
    ax3.set_xlabel('r (nm)')
    ax3.set_ylabel('K/K₀')
    ax3.set_title('Coupling Decay', fontsize=9, fontweight='bold')

    # Chart 4: Phase distribution (polar)
    ax4 = fig.add_subplot(gs[3], projection='polar')
    # Synchronized phases
    phases_sync = np.random.normal(0, 0.3, 50) % (2*np.pi)
    ax4.scatter(phases_sync, np.ones_like(phases_sync), c=COLORS['success'],
                s=30, alpha=0.7, label='Sync')
    # Disordered phases
    phases_disorder = np.random.uniform(0, 2*np.pi, 50)
    ax4.scatter(phases_disorder, 0.5*np.ones_like(phases_disorder), c=COLORS['quaternary'],
                s=30, alpha=0.5, label='Disorder')
    ax4.set_ylim(0, 1.2)
    ax4.set_title('Phases', fontsize=9, fontweight='bold', pad=10)
    ax4.set_rticks([])

    plt.tight_layout()
    plt.savefig('figures/panel3_phaselock.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figures/panel3_phaselock.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated panel3_phaselock.png")

# =============================================================================
# PANEL 4: Four-State Partition Operators
# =============================================================================
def generate_panel4_fourstate():
    """Section 5: Four-state partition operators."""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Chart 1: 3D four states
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Four states as corners of a square in 3D
    states = {
        'A': (1, 1, 1),    # High potential, electron absent
        'T': (-1, 1, -1),  # Low potential, electron absent
        'G': (1, -1, -1),  # High potential, electron present
        'C': (-1, -1, 1),  # Low potential, electron present
    }

    for base, pos in states.items():
        ax1.scatter(*pos, s=200, c=COLORS[base], edgecolors='white', linewidth=2,
                   label=base, alpha=0.9)
        ax1.text(pos[0]*1.3, pos[1]*1.3, pos[2]*1.3, base, fontsize=10, fontweight='bold')

    # Draw complementary pairs
    ax1.plot([states['A'][0], states['T'][0]],
             [states['A'][1], states['T'][1]],
             [states['A'][2], states['T'][2]], 'k--', alpha=0.5, linewidth=2)
    ax1.plot([states['G'][0], states['C'][0]],
             [states['G'][1], states['C'][1]],
             [states['G'][2], states['C'][2]], 'k--', alpha=0.5, linewidth=2)

    ax1.set_xlabel('Potential', fontsize=8)
    ax1.set_ylabel('Electron', fontsize=8)
    ax1.set_zlabel('State', fontsize=8)
    ax1.set_title('Four States', fontsize=9, fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # Chart 2: Partition diagram
    ax2 = fig.add_subplot(gs[1])
    # 2x2 grid
    data = np.array([[1, 2], [3, 4]])
    im = ax2.imshow(data, cmap='coolwarm', alpha=0.5)

    labels = [['A\n(H,A)', 'G\n(H,P)'], ['T\n(L,A)', 'C\n(L,P)']]
    colors_grid = [[COLORS['A'], COLORS['G']], [COLORS['T'], COLORS['C']]]

    for i in range(2):
        for j in range(2):
            ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                         facecolor=colors_grid[i][j], alpha=0.7))
            ax2.text(j, i, labels[i][j], ha='center', va='center',
                    fontsize=9, fontweight='bold')

    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Absent', 'Present'])
    ax2.set_yticklabels(['High', 'Low'])
    ax2.set_xlabel('Electron')
    ax2.set_ylabel('Potential')
    ax2.set_title('Partition', fontsize=9, fontweight='bold')

    # Chart 3: Complementarity
    ax3 = fig.add_subplot(gs[2])
    bases = ['A', 'T', 'G', 'C']
    comp_matrix = np.array([
        [0, 1, 0, 0],  # A pairs with T
        [1, 0, 0, 0],  # T pairs with A
        [0, 0, 0, 1],  # G pairs with C
        [0, 0, 1, 0],  # C pairs with G
    ])
    im = ax3.imshow(comp_matrix, cmap='Greens', vmin=0, vmax=1)
    ax3.set_xticks(range(4))
    ax3.set_yticks(range(4))
    ax3.set_xticklabels(bases)
    ax3.set_yticklabels(bases)
    ax3.set_title('Pairing', fontsize=9, fontweight='bold')

    for i in range(4):
        for j in range(4):
            if comp_matrix[i,j] == 1:
                ax3.text(j, i, '✓', ha='center', va='center', fontsize=12,
                        color='white', fontweight='bold')

    # Chart 4: H-bond count
    ax4 = fig.add_subplot(gs[3])
    pairs = ['A-T', 'G-C']
    hbonds = [2, 3]
    bars = ax4.bar(pairs, hbonds, color=[COLORS['A'], COLORS['G']],
                   alpha=0.8, edgecolor='white', width=0.5)
    ax4.set_ylabel('H-bonds')
    ax4.set_title('Bond Count', fontsize=9, fontweight='bold')
    ax4.set_ylim(0, 4)
    for bar, h in zip(bars, hbonds):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(h), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/panel4_fourstate.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figures/panel4_fourstate.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated panel4_fourstate.png")

# =============================================================================
# PANEL 5: Charge Stabilization Architecture
# =============================================================================
def generate_panel5_architecture():
    """Section 6: Double-strand architecture."""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Chart 1: 3D double helix
    ax1 = fig.add_subplot(gs[0], projection='3d')

    t = np.linspace(0, 4*np.pi, 200)
    r = 1
    # Strand 1
    x1 = r * np.cos(t)
    y1 = r * np.sin(t)
    z1 = t / (2*np.pi)
    # Strand 2 (offset by π)
    x2 = r * np.cos(t + np.pi)
    y2 = r * np.sin(t + np.pi)
    z2 = t / (2*np.pi)

    ax1.plot(x1, y1, z1, color=COLORS['primary'], linewidth=2, label='5\'→3\'')
    ax1.plot(x2, y2, z2, color=COLORS['secondary'], linewidth=2, label='3\'→5\'')

    # Base pairs (rungs)
    for i in range(0, len(t), 20):
        ax1.plot([x1[i], x2[i]], [y1[i], y2[i]], [z1[i], z2[i]],
                'gray', linewidth=1, alpha=0.5)

    ax1.set_xlabel('x', fontsize=8)
    ax1.set_ylabel('y', fontsize=8)
    ax1.set_zlabel('bp', fontsize=8)
    ax1.set_title('Double Helix', fontsize=9, fontweight='bold')
    ax1.view_init(elev=15, azim=45)

    # Chart 2: Stability vs strands
    ax2 = fig.add_subplot(gs[1])
    strands = [1, 2]
    stability = [1, 10]  # Relative stability
    ax2.bar(strands, stability, color=[COLORS['quaternary'], COLORS['success']],
            alpha=0.8, edgecolor='white', width=0.5)
    ax2.set_xticks(strands)
    ax2.set_xticklabels(['Single', 'Double'])
    ax2.set_ylabel('Relative Stability')
    ax2.set_title('Strand Count', fontsize=9, fontweight='bold')
    ax2.set_yscale('log')

    # Chart 3: Mismatch energy
    ax3 = fig.add_subplot(gs[2])
    pairs = ['A-T', 'G-C', 'A-C', 'G-T']
    energies = [-1.5, -2.5, 0.5, 0.3]  # kcal/mol (negative = stable)
    colors_e = [COLORS['success'], COLORS['success'], COLORS['quaternary'], COLORS['quaternary']]
    bars = ax3.bar(pairs, energies, color=colors_e, alpha=0.8, edgecolor='white')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_ylabel('ΔG (kcal/mol)')
    ax3.set_title('Pairing Energy', fontsize=9, fontweight='bold')

    # Chart 4: Error rate vs energy
    ax4 = fig.add_subplot(gs[3])
    dG = np.linspace(0, 5, 100)
    kT = 0.6  # kcal/mol at 310K
    error = np.exp(-dG/kT)
    ax4.semilogy(dG, error, color=COLORS['primary'], linewidth=2)
    ax4.fill_between(dG, error, alpha=0.3, color=COLORS['primary'])
    ax4.set_xlabel('ΔΔG (kcal/mol)')
    ax4.set_ylabel('Error Rate')
    ax4.set_title('exp(-ΔG/kT)', fontsize=9, fontweight='bold')
    ax4.axvline(x=2, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('figures/panel5_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figures/panel5_architecture.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated panel5_architecture.png")

# =============================================================================
# PANEL 6: Capacitive Properties
# =============================================================================
def generate_panel6_capacitance():
    """Section 7: DNA capacitance."""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Chart 1: 3D cylindrical capacitor
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Inner cylinder (DNA)
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(0, 5, 20)
    Theta, Z = np.meshgrid(theta, z)

    r_inner = 1
    r_outer = 3

    X_inner = r_inner * np.cos(Theta)
    Y_inner = r_inner * np.sin(Theta)
    ax1.plot_surface(X_inner, Y_inner, Z, alpha=0.7, color=COLORS['quaternary'])

    X_outer = r_outer * np.cos(Theta)
    Y_outer = r_outer * np.sin(Theta)
    ax1.plot_surface(X_outer, Y_outer, Z, alpha=0.3, color=COLORS['primary'])

    ax1.set_xlabel('x', fontsize=8)
    ax1.set_ylabel('y', fontsize=8)
    ax1.set_zlabel('z', fontsize=8)
    ax1.set_title('Capacitor', fontsize=9, fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # Chart 2: Capacitance vs length
    ax2 = fig.add_subplot(gs[1])
    N = np.logspace(3, 10, 100)  # base pairs
    # C = 2πε₀εᵣL / ln(b/a), with chromatin compaction
    eps_0 = 8.85e-12
    eps_r = 80
    L = N * 0.34e-9 / 1e4  # compacted length
    C = 2 * np.pi * eps_0 * eps_r * L / np.log(3)  # in F
    C_pF = C * 1e12

    ax2.loglog(N, C_pF, color=COLORS['primary'], linewidth=2)
    ax2.axhline(y=300, color=COLORS['success'], linestyle='--', linewidth=1, alpha=0.7)
    ax2.axvline(x=3e9, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.text(3e9, 10, 'Human', fontsize=7, rotation=90, va='bottom')
    ax2.set_xlabel('Base pairs')
    ax2.set_ylabel('C (pF)')
    ax2.set_title('Capacitance', fontsize=9, fontweight='bold')

    # Chart 3: Energy storage
    ax3 = fig.add_subplot(gs[2])
    V = np.linspace(0, 0.1, 100)  # Volts
    C_val = 300e-12  # 300 pF
    U = 0.5 * C_val * V**2 * 1e6  # in μJ
    ax3.plot(V*1000, U, color=COLORS['secondary'], linewidth=2)
    ax3.fill_between(V*1000, U, alpha=0.3, color=COLORS['secondary'])
    ax3.set_xlabel('V (mV)')
    ax3.set_ylabel('U (μJ)')
    ax3.set_title('Energy', fontsize=9, fontweight='bold')

    # Chart 4: RC time constant
    ax4 = fig.add_subplot(gs[3])
    R = np.logspace(3, 8, 100)  # Ohms
    C_val = 300e-12
    tau = R * C_val * 1e6  # in μs
    ax4.loglog(R/1e6, tau, color=COLORS['tertiary'], linewidth=2)
    ax4.axhline(y=30, color=COLORS['success'], linestyle='--', linewidth=1, alpha=0.7)
    ax4.axvline(x=0.1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('R (MΩ)')
    ax4.set_ylabel('τ (μs)')
    ax4.set_title('τ = RC', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/panel6_capacitance.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figures/panel6_capacitance.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated panel6_capacitance.png")

# =============================================================================
# PANEL 7: Polymerase Catalysis
# =============================================================================
def generate_panel7_polymerase():
    """Section 8: Polymerase catalysis."""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Chart 1: 3D active site
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Template strand
    t = np.linspace(0, 2*np.pi, 50)
    x_template = np.cos(t)
    y_template = np.sin(t)
    z_template = t / (2*np.pi) * 2
    ax1.plot(x_template, y_template, z_template, 'b-', linewidth=3, label='Template')

    # Incoming dNTP
    ax1.scatter([0], [0], [1], s=200, c=COLORS['tertiary'], marker='o',
                edgecolors='white', linewidth=2, label='dNTP')

    # Mg²⁺ ions
    ax1.scatter([0.3, -0.3], [0.3, -0.3], [0.9, 1.1], s=80, c=COLORS['success'],
                marker='^', edgecolors='white', label='Mg²⁺')

    # Arrow showing incorporation direction
    ax1.quiver(0, 0, 0.5, 0, 0, 0.4, color='red', arrow_length_ratio=0.3, linewidth=2)

    ax1.set_xlabel('x', fontsize=8)
    ax1.set_ylabel('y', fontsize=8)
    ax1.set_zlabel('z', fontsize=8)
    ax1.set_title('Active Site', fontsize=9, fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # Chart 2: Rate vs categorical distance
    ax2 = fig.add_subplot(gs[1])
    dC = [1, 2, 3, 4]
    k_cat = [1e3, 5e2, 2e2, 1e2]
    ax2.bar(dC, k_cat, color=COLORS['primary'], alpha=0.8, edgecolor='white')
    ax2.set_yscale('log')
    ax2.set_xlabel('dC')
    ax2.set_ylabel('kcat (s⁻¹)')
    ax2.set_title('Rate ∝ 1/dC', fontsize=9, fontweight='bold')
    ax2.set_xticks(dC)

    # Chart 3: Processivity
    ax3 = fig.add_subplot(gs[2])
    n_incorp = np.arange(0, 1000, 10)
    # Processivity depends on cumulative charge
    P_attached = np.exp(-n_incorp / 5000)  # Very high processivity
    ax3.plot(n_incorp, P_attached, color=COLORS['secondary'], linewidth=2)
    ax3.fill_between(n_incorp, P_attached, alpha=0.3, color=COLORS['secondary'])
    ax3.set_xlabel('Nucleotides')
    ax3.set_ylabel('P(attached)')
    ax3.set_title('Processivity', fontsize=9, fontweight='bold')
    ax3.set_ylim(0, 1.1)

    # Chart 4: Fidelity (error rate)
    ax4 = fig.add_subplot(gs[3])
    mechanisms = ['Intrinsic', '+Proofreading', '+Mismatch\nRepair']
    error_rates = [1e-4, 1e-7, 1e-10]
    bars = ax4.bar(mechanisms, error_rates, color=[COLORS['quaternary'],
                   COLORS['tertiary'], COLORS['success']], alpha=0.8, edgecolor='white')
    ax4.set_yscale('log')
    ax4.set_ylabel('Error Rate')
    ax4.set_title('Fidelity', fontsize=9, fontweight='bold')
    ax4.tick_params(axis='x', labelsize=6)

    plt.tight_layout()
    plt.savefig('figures/panel7_polymerase.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figures/panel7_polymerase.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated panel7_polymerase.png")

# =============================================================================
# PANEL 8: Cellular Charge Architecture
# =============================================================================
def generate_panel8_cellular():
    """Section 9: Three-layer cellular architecture."""
    fig = plt.figure(figsize=(14, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Chart 1: 3D three-layer cell
    ax1 = fig.add_subplot(gs[0], projection='3d')

    # Create spherical shells
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)

    # Nucleus (inner)
    r_nuc = 3
    x_nuc = r_nuc * np.outer(np.cos(u), np.sin(v))
    y_nuc = r_nuc * np.outer(np.sin(u), np.sin(v))
    z_nuc = r_nuc * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_nuc, y_nuc, z_nuc, alpha=0.8, color=COLORS['quaternary'])

    # Membrane (outer)
    r_mem = 10
    x_mem = r_mem * np.outer(np.cos(u), np.sin(v))
    y_mem = r_mem * np.outer(np.sin(u), np.sin(v))
    z_mem = r_mem * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_mem, y_mem, z_mem, alpha=0.2, color=COLORS['primary'])

    ax1.set_xlabel('x', fontsize=8)
    ax1.set_ylabel('y', fontsize=8)
    ax1.set_zlabel('z', fontsize=8)
    ax1.set_title('Three Layers', fontsize=9, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    ax1.set_box_aspect([1,1,1])

    # Chart 2: Electric field distribution
    ax2 = fig.add_subplot(gs[1])
    rho = np.linspace(3, 10, 100)  # μm
    Q_gen = 1e-9  # 1 nC
    eps_0 = 8.85e-12
    eps_r = 80
    E = Q_gen / (4 * np.pi * eps_0 * eps_r * (rho*1e-6)**2) / 1e6  # MV/m
    ax2.plot(rho, E, color=COLORS['primary'], linewidth=2)
    ax2.fill_between(rho, E, alpha=0.3, color=COLORS['primary'])
    ax2.set_xlabel('ρ (μm)')
    ax2.set_ylabel('|E| (MV/m)')
    ax2.set_title('Field', fontsize=9, fontweight='bold')
    ax2.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
    ax2.text(3.2, E.max()*0.9, 'Nucleus', fontsize=7)

    # Chart 3: Screened potential
    ax3 = fig.add_subplot(gs[2])
    r = np.linspace(0, 5, 100)  # nm from surface
    lambda_D = 0.8  # nm
    phi = np.exp(-r / lambda_D)
    ax3.plot(r, phi, color=COLORS['secondary'], linewidth=2)
    ax3.fill_between(r, phi, alpha=0.3, color=COLORS['secondary'])
    ax3.axvline(x=lambda_D, color='gray', linestyle='--', alpha=0.5)
    ax3.text(lambda_D + 0.1, 0.8, 'λD', fontsize=8)
    ax3.set_xlabel('r (nm)')
    ax3.set_ylabel('φ/φ₀')
    ax3.set_title('Screening', fontsize=9, fontweight='bold')

    # Chart 4: Chamber formation
    ax4 = fig.add_subplot(gs[3])
    z = np.linspace(0, 100, 100)  # nm from membrane
    # Potential well from superposition
    phi_gen = z / 100  # Linear from genome
    phi_patch = 0.5 * np.exp(-((z - 50)/20)**2)  # Gaussian from patch
    phi_total = phi_gen - phi_patch
    ax4.plot(z, phi_total, color=COLORS['tertiary'], linewidth=2)
    ax4.fill_between(z, phi_total.min(), phi_total, alpha=0.3, color=COLORS['tertiary'])
    ax4.axhline(y=phi_total.min(), color='gray', linestyle='--', alpha=0.5)
    ax4.scatter([50], [phi_total.min()], s=100, c=COLORS['success'], zorder=5,
               edgecolors='white', linewidth=2)
    ax4.set_xlabel('z (nm)')
    ax4.set_ylabel('φ (a.u.)')
    ax4.set_title('Chamber', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/panel8_cellular.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figures/panel8_cellular.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated panel8_cellular.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Generate all panels."""
    print("Generating panels for Cellular Charge Trajectories paper...")
    print("=" * 60)

    create_output_dir()

    # Part I: Framework
    generate_panel1_partition()
    generate_panel2_selection()
    generate_panel3_phaselock()

    # Part II: Application
    generate_panel4_fourstate()
    generate_panel5_architecture()
    generate_panel6_capacitance()
    generate_panel7_polymerase()
    generate_panel8_cellular()

    print("=" * 60)
    print("All panels generated successfully!")
    print("Output directory: figures/")

if __name__ == "__main__":
    main()
