"""
PANEL 10: TRAJECTORY RECONSTRUCTION
Generates 4 charts showing hierarchical ternary encoding and trajectory reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import Axes3D
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = plt.cm.tab10(np.linspace(0, 1, 10))

# Create figure with 2x2 grid
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# ============================================================================
# CHART 1: Hierarchical Ternary Encoding Structure
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)

# Title
ax1.text(5, 9.5, 'Hierarchical Ternary Encoding Structure', 
         ha='center', fontsize=14, fontweight='bold')

# Level 1: Temporal
y1 = 8
ax1.text(1, y1, 'Level 1: Temporal', fontsize=11, fontweight='bold')
for i, label in enumerate(['t₁', 't₂', 't₃']):
    x = 3 + i * 2
    rect = FancyBboxPatch((x-0.4, y1-0.3), 0.8, 0.6, 
                          boxstyle="round,pad=0.1", 
                          facecolor=colors[0], alpha=0.7, 
                          edgecolor='black', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(x, y1, label, ha='center', va='center', 
            fontsize=10, fontweight='bold')

# Arrows down
for i in range(3):
    x = 3 + i * 2
    arrow = FancyArrowPatch((x, y1-0.4), (x, y1-1.1),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black')
    ax1.add_patch(arrow)

# Level 2: Spatial
y2 = 6.5
ax1.text(1, y2, 'Level 2: Spatial', fontsize=11, fontweight='bold')
for i, label in enumerate(['p₁', 'p₂', 'p₃']):
    x = 3 + i * 2
    rect = FancyBboxPatch((x-0.4, y2-0.3), 0.8, 0.6, 
                          boxstyle="round,pad=0.1", 
                          facecolor=colors[1], alpha=0.7, 
                          edgecolor='black', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(x, y2, label, ha='center', va='center', 
            fontsize=10, fontweight='bold')

# Arrows down
for i in range(3):
    x = 3 + i * 2
    arrow = FancyArrowPatch((x, y2-0.4), (x, y2-1.1),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black')
    ax1.add_patch(arrow)

# Level 3: Molecular Degrees of Freedom
y3 = 5
ax1.text(1, y3, 'Level 3: Molecular', fontsize=11, fontweight='bold')
dof_labels = ['Electronic\n(n)', 'Vibrational\n(ℓ)', 'Rotational\n(m)', 'Spin\n(s)']
for i, label in enumerate(dof_labels):
    x = 2.5 + i * 1.8
    rect = FancyBboxPatch((x-0.6, y3-0.5), 1.2, 1.0, 
                          boxstyle="round,pad=0.1", 
                          facecolor=colors[2+i], alpha=0.7, 
                          edgecolor='black', linewidth=2)
    ax1.add_patch(rect)
    ax1.text(x, y3, label, ha='center', va='center', 
            fontsize=9, fontweight='bold')

# Arrows down to trits
for i in range(4):
    x = 2.5 + i * 1.8
    arrow = FancyArrowPatch((x, y3-0.6), (x, y3-1.3),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black')
    ax1.add_patch(arrow)

# Trit values
y4 = 3
ax1.text(1, y4, 'Trit Values:', fontsize=11, fontweight='bold')
for i in range(4):
    x = 2.5 + i * 1.8
    for j, trit in enumerate([0, 1, 2]):
        y = y4 - j * 0.4
        circle = Circle((x, y), 0.15, facecolor='white', 
                       edgecolor='black', linewidth=1.5)
        ax1.add_patch(circle)
        ax1.text(x, y, str(trit), ha='center', va='center', 
                fontsize=8, fontweight='bold')

# Example encoding
y5 = 1.2
ax1.text(1, y5, 'Example (H 1s→2p):', fontsize=11, fontweight='bold')
ax1.text(1, y5-0.5, 'Initial: [0][0][1][2] = 0012₃', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor=colors[0], alpha=0.3))
ax1.text(1, y5-1.0, 'Final:   [1][1][1][2] = 1112₃', fontsize=10,
        bbox=dict(boxstyle='round', facecolor=colors[3], alpha=0.3))

# ============================================================================
# CHART 2: 3D Trajectory in S-Entropy Space
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1], projection='3d')

# Generate trajectory for 1s->2p transition
n_points = 100
t = np.linspace(0, 1, n_points)

# S-entropy coordinates evolve during transition
# Initial state (1s): (0.23, 0.15, 0.08)
# Final state (2p): (0.45, 0.35, 0.25)
S_k = 0.23 + 0.22 * (1 - np.exp(-3*t))
S_t = 0.15 + 0.20 * (1 - np.exp(-3*t))
S_e = 0.08 + 0.17 * (1 - np.exp(-3*t))

# Add some oscillatory behavior (Poincaré recurrence)
S_k += 0.02 * np.sin(10 * np.pi * t) * np.exp(-2*t)
S_t += 0.02 * np.sin(12 * np.pi * t) * np.exp(-2*t)
S_e += 0.02 * np.sin(8 * np.pi * t) * np.exp(-2*t)

# Plot trajectory
ax2.plot(S_k, S_t, S_e, linewidth=3, color=colors[0], alpha=0.8, 
        label='Electron Trajectory')

# Mark initial and final states
ax2.scatter([S_k[0]], [S_t[0]], [S_e[0]], c='green', s=200, 
           marker='o', edgecolors='black', linewidths=2, 
           label='Initial (1s)', zorder=5)
ax2.scatter([S_k[-1]], [S_t[-1]], [S_e[-1]], c='red', s=200, 
           marker='s', edgecolors='black', linewidths=2, 
           label='Final (2p)', zorder=5)

# Mark intermediate points
intermediate_indices = [25, 50, 75]
ax2.scatter(S_k[intermediate_indices], S_t[intermediate_indices], 
           S_e[intermediate_indices], c='blue', s=100, 
           marker='^', edgecolors='black', linewidths=1.5, 
           alpha=0.6, label='Intermediate', zorder=4)

ax2.set_xlabel('Knowledge Entropy (Sₖ)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Temporal Entropy (Sₜ)', fontsize=11, fontweight='bold')
ax2.set_zlabel('Evolution Entropy (Sₑ)', fontsize=11, fontweight='bold')
ax2.set_title('Electron Trajectory in S-Entropy Space\n(1s→2p Transition)', 
             fontsize=13, fontweight='bold', pad=20)
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3)

# ============================================================================
# CHART 3: Trit Sequence Evolution
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])

# Time points
time_points = np.array([0, 2.5, 5.0, 7.5, 10.0])  # ns
time_labels = ['0 ns\n(Initial)', '2.5 ns', '5.0 ns', '7.5 ns', '10 ns\n(Final)']

# Trit sequences at each time point (4 trits: Elec, Vib, Rot, Spin)
trit_sequences = [
    [0, 0, 1, 2],  # Initial (1s)
    [0, 1, 1, 2],  # Early transition
    [1, 0, 1, 2],  # Mid transition
    [1, 1, 0, 2],  # Late transition
    [1, 1, 1, 2]   # Final (2p)
]

# Create heatmap
trit_array = np.array(trit_sequences).T
im = ax3.imshow(trit_array, cmap='viridis', aspect='auto', 
               vmin=0, vmax=2, interpolation='nearest')

# Add colorbar
cbar = plt.colorbar(im, ax=ax3, ticks=[0, 1, 2])
cbar.set_label('Trit Value', fontsize=11, fontweight='bold')
cbar.ax.set_yticklabels(['0', '1', '2'], fontsize=10)

# Customize axes
ax3.set_xticks(range(len(time_labels)))
ax3.set_xticklabels(time_labels, fontsize=10)
ax3.set_yticks(range(4))
ax3.set_yticklabels(['Electronic (n)', 'Vibrational (ℓ)', 
                    'Rotational (m)', 'Spin (s)'], fontsize=10)
ax3.set_xlabel('Time During Transition', fontsize=12, fontweight='bold')
ax3.set_ylabel('Molecular Degree of Freedom', fontsize=12, fontweight='bold')
ax3.set_title('Trit Sequence Evolution During 1s→2p Transition\n' +
             '(Each Cell Shows Trit Value 0, 1, or 2)', 
             fontsize=13, fontweight='bold')

# Add grid
for i in range(4):
    for j in range(5):
        text = ax3.text(j, i, str(trit_array[i, j]), 
                       ha="center", va="center", 
                       color="white", fontsize=14, fontweight='bold')

# Highlight changes
for i in range(4):
    for j in range(1, 5):
        if trit_array[i, j] != trit_array[i, j-1]:
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                               fill=False, edgecolor='red', 
                               linewidth=3)
            ax3.add_patch(rect)

# ============================================================================
# CHART 4: Measurement Modality to Trit Mapping
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)

# Title
ax4.text(5, 9.5, 'Measurement Modality → Trit Mapping', 
        ha='center', fontsize=14, fontweight='bold')

# Modalities and their mappings
modalities = [
    ('Optical\nSpectroscopy', 'Electronic\nState', 'n', colors[0]),
    ('Raman\nSpectroscopy', 'Vibrational\nMode', 'ℓ', colors[1]),
    ('Microwave\nSpectroscopy', 'Rotational\nState', 'm', colors[2]),
    ('Magnetic\nResonance', 'Spin\nProjection', 's', colors[3])
]

y_start = 8
y_step = 2

for i, (modality, dof, coord, color) in enumerate(modalities):
    y = y_start - i * y_step
    
    # Modality box
    rect1 = FancyBboxPatch((0.5, y-0.4), 2, 0.8, 
                          boxstyle="round,pad=0.1", 
                          facecolor=color, alpha=0.7, 
                          edgecolor='black', linewidth=2)
    ax4.add_patch(rect1)
    ax4.text(1.5, y, modality, ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Arrow
    arrow = FancyArrowPatch((2.6, y), (3.9, y),
                           arrowstyle='->', mutation_scale=30, 
                           linewidth=3, color='black')
    ax4.add_patch(arrow)
    
    # Degree of freedom box
    rect2 = FancyBboxPatch((4, y-0.4), 2, 0.8, 
                          boxstyle="round,pad=0.1", 
                          facecolor=color, alpha=0.5, 
                          edgecolor='black', linewidth=2)
    ax4.add_patch(rect2)
    ax4.text(5, y, dof, ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Arrow
    arrow2 = FancyArrowPatch((6.1, y), (7.4, y),
                            arrowstyle='->', mutation_scale=30, 
                            linewidth=3, color='black')
    ax4.add_patch(arrow2)
    
    # Partition coordinate box
    rect3 = FancyBboxPatch((7.5, y-0.4), 1.5, 0.8, 
                          boxstyle="round,pad=0.1", 
                          facecolor=color, alpha=0.3, 
                          edgecolor='black', linewidth=2)
    ax4.add_patch(rect3)
    ax4.text(8.25, y, coord, ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Trit values
    ax4.text(8.25, y-0.7, '{0,1,2}', ha='center', va='top', 
            fontsize=8, style='italic')

# Add S-entropy coupling at bottom
y_bottom = 0.8
ax4.text(5, y_bottom+0.5, 'S-Entropy Coupling:', 
        ha='center', fontsize=11, fontweight='bold')
ax4.text(2, y_bottom, 'Sₖ ← Electronic', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor=colors[0], alpha=0.3))
ax4.text(4, y_bottom, 'Sₜ ← Vibrational', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor=colors[1], alpha=0.3))
ax4.text(6, y_bottom, 'Sₑ ← Rotational', ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor=colors[2], alpha=0.3))

# Add overall title
fig.suptitle('PANEL 10: TRAJECTORY RECONSTRUCTION VIA HIERARCHICAL TERNARY ENCODING\n' +
             'Multi-Level Structure Maps Molecular Degrees of Freedom to Partition Coordinates',
             fontsize=16, fontweight='bold', y=0.98)

# Save
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('c:/Users/kundai/Documents/foundry/faraday/validation/panels/panel_10_trajectory_reconstruction.png', 
            dpi=300, bbox_inches='tight')
print("[OK] Panel 10 saved: panel_10_trajectory_reconstruction.png")

# Save data
data = {
    'panel': 10,
    'title': 'Trajectory Reconstruction',
    'trajectory': {
        'S_k': S_k.tolist(),
        'S_t': S_t.tolist(),
        'S_e': S_e.tolist(),
        'n_points': n_points
    },
    'trit_sequences': {
        'time_points': time_points.tolist(),
        'sequences': trit_sequences
    },
    'modality_mapping': [
        {'modality': m[0], 'dof': m[1], 'coordinate': m[2]} 
        for m in modalities
    ]
}

with open('c:/Users/kundai/Documents/foundry/faraday/validation/results/panel_10_data.json', 'w') as f:
    json.dump(data, f, indent=2)

print("[OK] Panel 10 data saved: panel_10_data.json")
print(f"\nPanel 10 Statistics:")
print(f"  Trajectory points: {n_points}")
print(f"  Time span: 10 ns")
print(f"  Trit changes: {sum(1 for i in range(4) for j in range(1, 5) if trit_sequences[j][i] != trit_sequences[j-1][i])}")
print(f"  S-entropy range: S_k=[{S_k[0]:.2f}, {S_k[-1]:.2f}], S_t=[{S_t[0]:.2f}, {S_t[-1]:.2f}], S_e=[{S_e[0]:.2f}, {S_e[-1]:.2f}]")

plt.show()
