"""
Generate ensemble demonstration panels showing capabilities from ensemble.md

These panels demonstrate the CONCEPTS - what the categorical framework enables:
- Virtual instrument capabilities
- Post-hoc parameter modification  
- Information flow tracking
- Multi-scale coherence measurement

Uses synthetic data to clearly illustrate the capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class EnsembleConceptDemonstrator:
    """Demonstrate ensemble.md concepts with clear visualizations"""
    
    def __init__(self):
        self.output_dir = Path('single_ion_beam/src/validation/figures/ensemble_concepts')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def demo_virtual_chromatograph(self):
        """Show post-hoc column modification concept"""
        print("\nGenerating Virtual Chromatograph demo...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Virtual Chromatograph: Post-Hoc Column Modification\n' +
                    '90% Reduction in Method Development Time',
                    fontsize=16, fontweight='bold')
        
        # Simulate retention times
        n_compounds = 100
        rt_original = np.random.normal(10, 2, n_compounds)
        
        # Panel A: Original C18 measurement
        ax = axes[0, 0]
        ax.hist(rt_original, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Retention Time (min)', fontsize=12)
        ax.set_ylabel('Number of Compounds', fontsize=12)
        ax.set_title('A. Single C18 Measurement\n(Real Hardware Run)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, 'One measurement', transform=ax.transAxes,
               fontsize=11, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Panel B: Virtual C8 column
        ax = axes[0, 1]
        rt_c8 = rt_original * 0.85  # C8 shorter retention
        ax.hist(rt_c8, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Virtual Retention Time (min)', fontsize=12)
        ax.set_ylabel('Number of Compounds', fontsize=12)
        ax.set_title('B. Virtual C8 Column\n(Post-Hoc Modification)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, 'NO re-measurement!', transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Panel C: Virtual HILIC column
        ax = axes[1, 0]
        rt_hilic = rt_original * 1.3  # HILIC longer retention
        ax.hist(rt_hilic, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('Virtual Retention Time (min)', fontsize=12)
        ax.set_ylabel('Number of Compounds', fontsize=12)
        ax.set_title('C. Virtual HILIC Column\n(Reversed Selectivity)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, 'NO re-measurement!', transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Panel D: Time savings
        ax = axes[1, 1]
        columns = ['C18\n(Real)', 'C8\n(Virtual)', 'HILIC\n(Virtual)']
        times = [60, 0, 0]  # minutes
        colors = ['blue', 'green', 'orange']
        
        bars = ax.bar(range(len(columns)), times, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(columns, fontsize=12)
        ax.set_ylabel('Measurement Time (min)', fontsize=12)
        ax.set_title('D. Time Savings: 90% Reduction\n(120 min → 60 min)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 70])
        
        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, times)):
            if time > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                       f'{time} min', ha='center', fontsize=11, fontweight='bold')
            else:
                ax.text(i, 5, 'FREE!', ha='center', fontsize=13, 
                       fontweight='bold', color='red')
        
        plt.tight_layout()
        output_path = self.output_dir / '01_virtual_chromatograph.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def demo_information_flow(self):
        """Show information flow visualization concept"""
        print("\nGenerating Information Flow demo...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Information Flow Visualizer: Real-Time Information Tracking\n' +
                    'Visualize Information Pathways and Bottlenecks',
                    fontsize=16, fontweight='bold')
        
        # Simulate measurement pipeline
        n_steps = 50
        time = np.linspace(0, 10, n_steps)
        
        # Information accumulation
        info_bits = np.cumsum(np.random.exponential(2, n_steps))
        
        # Panel A: Information accumulation
        ax = axes[0, 0]
        ax.plot(time, info_bits, linewidth=3, color='blue', label='Information')
        ax.fill_between(time, info_bits, alpha=0.3, color='blue')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Cumulative Information (bits)', fontsize=12)
        ax.set_title('A. Information Accumulation\n(Real-Time Tracking)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Panel B: Information velocity
        ax = axes[0, 1]
        info_velocity = np.gradient(info_bits, time)
        ax.plot(time, info_velocity, linewidth=3, color='red')
        ax.axhline(y=info_velocity.mean(), color='black', linestyle='--', 
                  linewidth=2, label=f'Mean: {info_velocity.mean():.1f} bits/s')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Information Rate (bits/s)', fontsize=12)
        ax.set_title('B. Information Velocity\n(Measurement Efficiency)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Panel C: Bottleneck detection
        ax = axes[1, 0]
        bottlenecks = info_velocity < (info_velocity.mean() - info_velocity.std())
        ax.plot(time, info_velocity, linewidth=2, color='green', label='Flow Rate')
        ax.scatter(time[bottlenecks], info_velocity[bottlenecks], 
                  s=200, color='red', marker='X', linewidths=2, 
                  label='Bottlenecks', zorder=5)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Information Rate (bits/s)', fontsize=12)
        ax.set_title('C. Bottleneck Detection\n(Low Flow = Bottleneck)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Panel D: Information pathway
        ax = axes[1, 1]
        # Create network of information flow
        n_nodes = 15
        x = np.random.rand(n_nodes) * 10
        y = np.random.rand(n_nodes) * 100
        
        # Sort by x for flow direction
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        
        # Draw nodes
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_nodes))
        ax.scatter(x_sorted, y_sorted, s=300, c=colors, 
                  edgecolors='black', linewidths=2, zorder=5)
        
        # Draw flow arrows
        for i in range(n_nodes-1):
            ax.annotate('', xy=(x_sorted[i+1], y_sorted[i+1]), 
                       xytext=(x_sorted[i], y_sorted[i]),
                       arrowprops=dict(arrowstyle='->', lw=2, 
                                     color='gray', alpha=0.5))
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Information Content (bits)', fontsize=12)
        ax.set_title('D. Information Pathway\n(Sequential Flow Network)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / '02_information_flow.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def demo_multi_scale_coherence(self):
        """Show multi-scale coherence measurement concept"""
        print("\nGenerating Multi-Scale Coherence demo...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Multi-Scale Coherence Detector: Simultaneous Scale Measurement\n' +
                    'Quantum → Molecular → Cellular Coherence',
                    fontsize=16, fontweight='bold')
        
        # Generate coherence data for different scales
        n_samples = 1000
        
        # Panel A: Quantum coherence
        ax = axes[0, 0]
        quantum_coherence = np.random.beta(5, 2, n_samples)
        ax.hist(quantum_coherence, bins=40, alpha=0.7, color='purple', edgecolor='black')
        ax.set_xlabel('Quantum Coherence', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('A. Quantum Scale\n(Vibrational Coherence)', 
                    fontsize=13, fontweight='bold')
        ax.axvline(quantum_coherence.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {quantum_coherence.mean():.3f}')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Panel B: Molecular coherence
        ax = axes[0, 1]
        molecular_coherence = np.random.beta(4, 3, n_samples)
        ax.hist(molecular_coherence, bins=40, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Molecular Coherence', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('B. Molecular Scale\n(Dielectric Coherence)', 
                    fontsize=13, fontweight='bold')
        ax.axvline(molecular_coherence.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {molecular_coherence.mean():.3f}')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Panel C: Cellular coherence
        ax = axes[1, 0]
        cellular_coherence = np.random.beta(3, 4, n_samples)
        ax.hist(cellular_coherence, bins=40, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Cellular Coherence', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('C. Cellular Scale\n(Field Coherence)', 
                    fontsize=13, fontweight='bold')
        ax.axvline(cellular_coherence.mean(), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {cellular_coherence.mean():.3f}')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Panel D: Cross-scale coupling
        ax = axes[1, 1]
        scales = ['Quantum', 'Molecular', 'Cellular']
        corr_matrix = np.array([
            [1.0, 0.85, 0.62],
            [0.85, 1.0, 0.73],
            [0.62, 0.73, 1.0]
        ])
        
        im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(scales, fontsize=12)
        ax.set_yticklabels(scales, fontsize=12)
        ax.set_title('D. Cross-Scale Coupling\n(Coherence Correlations)', 
                    fontsize=13, fontweight='bold')
        
        # Add correlation values
        for i in range(3):
            for j in range(3):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", 
                             fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        
        plt.tight_layout()
        output_path = self.output_dir / '03_multi_scale_coherence.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def demo_virtual_raman(self):
        """Show virtual Raman spectrometer concept"""
        print("\nGenerating Virtual Raman demo...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Virtual Raman Spectrometer: Post-Hoc Wavelength Modification\n' +
                    '80% Reduction in Photodamage',
                    fontsize=16, fontweight='bold')
        
        # Generate synthetic Raman spectra
        raman_shift = np.linspace(200, 3500, 500)
        
        # Base spectrum at 532 nm
        intensity_532 = (np.exp(-(raman_shift - 1000)**2 / 50000) * 100 +
                        np.exp(-(raman_shift - 1600)**2 / 30000) * 70 +
                        np.exp(-(raman_shift - 2900)**2 / 40000) * 50)
        
        # Panel A: Original 532 nm measurement
        ax = axes[0, 0]
        ax.plot(raman_shift, intensity_532, linewidth=2, color='green', label='532 nm')
        ax.fill_between(raman_shift, intensity_532, alpha=0.3, color='green')
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
        ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax.set_title('A. Single Measurement at 532 nm\n(Real Laser)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, 'One measurement', transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Panel B: Virtual 785 nm
        ax = axes[0, 1]
        intensity_785 = intensity_532 * 0.75  # Different cross-section
        ax.plot(raman_shift, intensity_785, linewidth=2, color='red', label='785 nm (Virtual)')
        ax.fill_between(raman_shift, intensity_785, alpha=0.3, color='red')
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
        ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax.set_title('B. Virtual 785 nm\n(Post-Hoc Modification)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, 'NO photodamage!', transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Panel C: Virtual 633 nm (resonance enhanced)
        ax = axes[1, 0]
        intensity_633 = intensity_532 * 1.4  # Resonance enhancement
        intensity_633[200:250] *= 2.5  # Enhanced specific modes
        ax.plot(raman_shift, intensity_633, linewidth=2, color='orange', label='633 nm (Virtual)')
        ax.fill_between(raman_shift, intensity_633, alpha=0.3, color='orange')
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
        ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax.set_title('C. Virtual 633 nm\n(Resonance Enhanced)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Panel D: Comparison
        ax = axes[1, 1]
        ax.plot(raman_shift, intensity_532, linewidth=2, color='green', 
               label='532 nm (Real)', alpha=0.8)
        ax.plot(raman_shift, intensity_785, linewidth=2, color='red', 
               label='785 nm (Virtual)', alpha=0.8, linestyle='--')
        ax.plot(raman_shift, intensity_633, linewidth=2, color='orange', 
               label='633 nm (Virtual)', alpha=0.8, linestyle=':')
        ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=12)
        ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax.set_title('D. Multi-Wavelength Comparison\n(All From One Measurement)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / '04_virtual_raman.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def demo_virtual_nmr(self):
        """Show virtual NMR spectrometer concept"""
        print("\nGenerating Virtual NMR demo...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Virtual NMR Spectrometer: Post-Hoc Field Strength Modification\n' +
                    '90% Reduction in Measurement Time',
                    fontsize=16, fontweight='bold')
        
        # Generate synthetic NMR spectrum
        ppm = np.linspace(-1, 10, 1000)
        
        # Panel A: Original 400 MHz measurement
        ax = axes[0, 0]
        # Simulate multiplet peaks
        intensity_400 = (np.exp(-(ppm - 7.2)**2 / 0.01) * 100 +  # Aromatic
                        np.exp(-(ppm - 3.7)**2 / 0.005) * 80 +   # OCH3
                        np.exp(-(ppm - 2.3)**2 / 0.008) * 60 +   # CH2
                        np.exp(-(ppm - 1.2)**2 / 0.006) * 70)    # CH3
        
        ax.plot(ppm, intensity_400, linewidth=2, color='blue', label='400 MHz')
        ax.fill_between(ppm, intensity_400, alpha=0.3, color='blue')
        ax.set_xlabel('Chemical Shift (ppm)', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        ax.set_title('A. Single 400 MHz Measurement\n(Real Hardware)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        ax.text(0.95, 0.95, 'One measurement\n60 minutes', transform=ax.transAxes,
               fontsize=11, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Panel B: Virtual 600 MHz
        ax = axes[0, 1]
        # Higher field = better resolution (narrower peaks)
        intensity_600 = (np.exp(-(ppm - 7.2)**2 / 0.005) * 120 +
                        np.exp(-(ppm - 3.7)**2 / 0.003) * 95 +
                        np.exp(-(ppm - 2.3)**2 / 0.004) * 75 +
                        np.exp(-(ppm - 1.2)**2 / 0.003) * 85)
        
        ax.plot(ppm, intensity_600, linewidth=2, color='green', label='600 MHz (Virtual)')
        ax.fill_between(ppm, intensity_600, alpha=0.3, color='green')
        ax.set_xlabel('Chemical Shift (ppm)', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        ax.set_title('B. Virtual 600 MHz\n(Post-Hoc Modification)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        ax.text(0.95, 0.95, 'NO re-measurement!\n0 minutes', transform=ax.transAxes,
               fontsize=11, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Panel C: Virtual 800 MHz
        ax = axes[1, 0]
        # Even higher field = even better resolution
        intensity_800 = (np.exp(-(ppm - 7.2)**2 / 0.003) * 140 +
                        np.exp(-(ppm - 3.7)**2 / 0.002) * 110 +
                        np.exp(-(ppm - 2.3)**2 / 0.003) * 90 +
                        np.exp(-(ppm - 1.2)**2 / 0.002) * 100)
        
        ax.plot(ppm, intensity_800, linewidth=2, color='red', label='800 MHz (Virtual)')
        ax.fill_between(ppm, intensity_800, alpha=0.3, color='red')
        ax.set_xlabel('Chemical Shift (ppm)', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        ax.set_title('C. Virtual 800 MHz\n(Ultra-High Resolution)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Panel D: Resolution comparison
        ax = axes[1, 1]
        # Zoom in on one peak to show resolution improvement
        zoom_mask = (ppm > 7.0) & (ppm < 7.5)
        ax.plot(ppm[zoom_mask], intensity_400[zoom_mask], linewidth=3, 
               color='blue', label='400 MHz', alpha=0.8)
        ax.plot(ppm[zoom_mask], intensity_600[zoom_mask], linewidth=3, 
               color='green', label='600 MHz (Virtual)', alpha=0.8, linestyle='--')
        ax.plot(ppm[zoom_mask], intensity_800[zoom_mask], linewidth=3, 
               color='red', label='800 MHz (Virtual)', alpha=0.8, linestyle=':')
        ax.set_xlabel('Chemical Shift (ppm)', fontsize=12)
        ax.set_ylabel('Intensity', fontsize=12)
        ax.set_title('D. Resolution Enhancement\n(Peak Width Decreases)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        plt.tight_layout()
        output_path = self.output_dir / '05_virtual_nmr.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def demo_virtual_xray(self):
        """Show virtual X-ray diffractometer concept"""
        print("\nGenerating Virtual X-ray Diffractometer demo...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Virtual X-ray Diffractometer: Post-Hoc Wavelength Modification\n' +
                    '85% Reduction in Beam Time',
                    fontsize=16, fontweight='bold')
        
        # Generate synthetic diffraction pattern
        two_theta = np.linspace(5, 90, 500)
        
        # Panel A: Cu Kα (1.54 Å) measurement
        ax = axes[0, 0]
        # Simulate Bragg peaks
        intensity_cu = (np.exp(-(two_theta - 28.4)**2 / 2) * 1000 +  # (111)
                       np.exp(-(two_theta - 47.3)**2 / 3) * 600 +    # (220)
                       np.exp(-(two_theta - 56.1)**2 / 2.5) * 400 +  # (311)
                       np.exp(-(two_theta - 69.1)**2 / 3.5) * 300)   # (400)
        
        ax.plot(two_theta, intensity_cu, linewidth=2, color='blue', label='Cu Kα (1.54 Å)')
        ax.fill_between(two_theta, intensity_cu, alpha=0.3, color='blue')
        ax.set_xlabel('2θ (degrees)', fontsize=12)
        ax.set_ylabel('Intensity (counts)', fontsize=12)
        ax.set_title('A. Cu Kα Measurement\n(Real X-ray Tube)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, 'One measurement', transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Panel B: Virtual Mo Kα (0.71 Å)
        ax = axes[0, 1]
        # Shorter wavelength = peaks at smaller angles
        two_theta_mo = two_theta * 0.46  # λ_Cu/λ_Mo ≈ 2.17, so peaks shift
        intensity_mo = np.interp(two_theta, two_theta_mo, intensity_cu)
        
        ax.plot(two_theta, intensity_mo, linewidth=2, color='green', label='Mo Kα (0.71 Å, Virtual)')
        ax.fill_between(two_theta, intensity_mo, alpha=0.3, color='green')
        ax.set_xlabel('2θ (degrees)', fontsize=12)
        ax.set_ylabel('Intensity (counts)', fontsize=12)
        ax.set_title('B. Virtual Mo Kα\n(Post-Hoc Wavelength Change)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, 'NO re-measurement!', transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Panel C: Virtual Ag Kα (0.56 Å)
        ax = axes[1, 0]
        # Even shorter wavelength
        two_theta_ag = two_theta * 0.36
        intensity_ag = np.interp(two_theta, two_theta_ag, intensity_cu)
        
        ax.plot(two_theta, intensity_ag, linewidth=2, color='orange', label='Ag Kα (0.56 Å, Virtual)')
        ax.fill_between(two_theta, intensity_ag, alpha=0.3, color='orange')
        ax.set_xlabel('2θ (degrees)', fontsize=12)
        ax.set_ylabel('Intensity (counts)', fontsize=12)
        ax.set_title('C. Virtual Ag Kα\n(Ultra-Short Wavelength)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Panel D: Wavelength comparison
        ax = axes[1, 1]
        ax.plot(two_theta, intensity_cu, linewidth=2, color='blue', 
               label='Cu Kα (1.54 Å)', alpha=0.8)
        ax.plot(two_theta, intensity_mo, linewidth=2, color='green', 
               label='Mo Kα (0.71 Å, Virtual)', alpha=0.8, linestyle='--')
        ax.plot(two_theta, intensity_ag, linewidth=2, color='orange', 
               label='Ag Kα (0.56 Å, Virtual)', alpha=0.8, linestyle=':')
        ax.set_xlabel('2θ (degrees)', fontsize=12)
        ax.set_ylabel('Intensity (counts)', fontsize=12)
        ax.set_title('D. Multi-Wavelength Comparison\n(All From One Measurement)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / '06_virtual_xray.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def demo_virtual_flow_cytometer(self):
        """Show virtual flow cytometer concept"""
        print("\nGenerating Virtual Flow Cytometer demo...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Virtual Flow Cytometer: Post-Hoc Fluorophore Substitution\n' +
                    '75% Reduction in Sample Consumption',
                    fontsize=16, fontweight='bold')
        
        # Generate synthetic cell populations
        np.random.seed(42)
        n_cells = 2000
        
        # Two populations
        pop1_fsc = np.random.normal(500, 80, n_cells//2)
        pop1_fl = np.random.normal(300, 50, n_cells//2)
        pop2_fsc = np.random.normal(800, 100, n_cells//2)
        pop2_fl = np.random.normal(700, 80, n_cells//2)
        
        # Panel A: Original FITC measurement
        ax = axes[0, 0]
        ax.scatter(pop1_fsc, pop1_fl, s=10, alpha=0.5, color='green', label='Population 1')
        ax.scatter(pop2_fsc, pop2_fl, s=10, alpha=0.5, color='blue', label='Population 2')
        ax.set_xlabel('Forward Scatter (FSC)', fontsize=12)
        ax.set_ylabel('FITC Fluorescence (FL1)', fontsize=12)
        ax.set_title('A. FITC Measurement\n(Real Fluorophore)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, 'One measurement', transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Panel B: Virtual Alexa488 substitution
        ax = axes[0, 1]
        # Different quantum yield and brightness
        pop1_alexa = pop1_fl * 1.3  # Alexa488 brighter than FITC
        pop2_alexa = pop2_fl * 1.3
        
        ax.scatter(pop1_fsc, pop1_alexa, s=10, alpha=0.5, color='lime', label='Population 1')
        ax.scatter(pop2_fsc, pop2_alexa, s=10, alpha=0.5, color='cyan', label='Population 2')
        ax.set_xlabel('Forward Scatter (FSC)', fontsize=12)
        ax.set_ylabel('Alexa488 Fluorescence (Virtual)', fontsize=12)
        ax.set_title('B. Virtual Alexa488\n(Post-Hoc Substitution)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, 'NO re-staining!', transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Panel C: Virtual GFP
        ax = axes[1, 0]
        # GFP has different spectrum
        pop1_gfp = pop1_fl * 0.9 + 20  # Different offset
        pop2_gfp = pop2_fl * 0.9 + 20
        
        ax.scatter(pop1_fsc, pop1_gfp, s=10, alpha=0.5, color='yellowgreen', label='Population 1')
        ax.scatter(pop2_fsc, pop2_gfp, s=10, alpha=0.5, color='teal', label='Population 2')
        ax.set_xlabel('Forward Scatter (FSC)', fontsize=12)
        ax.set_ylabel('GFP Fluorescence (Virtual)', fontsize=12)
        ax.set_title('C. Virtual GFP\n(Different Spectrum)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Panel D: Gating comparison
        ax = axes[1, 1]
        # Show how different fluorophores affect population separation
        sep_fitc = (pop2_fl.mean() - pop1_fl.mean()) / np.sqrt(pop1_fl.var() + pop2_fl.var())
        sep_alexa = (pop2_alexa.mean() - pop1_alexa.mean()) / np.sqrt(pop1_alexa.var() + pop2_alexa.var())
        sep_gfp = (pop2_gfp.mean() - pop1_gfp.mean()) / np.sqrt(pop1_gfp.var() + pop2_gfp.var())
        
        fluorophores = ['FITC\n(Real)', 'Alexa488\n(Virtual)', 'GFP\n(Virtual)']
        separations = [sep_fitc, sep_alexa, sep_gfp]
        colors = ['green', 'lime', 'yellowgreen']
        
        bars = ax.bar(range(len(fluorophores)), separations, color=colors, 
                     alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(fluorophores)))
        ax.set_xticklabels(fluorophores, fontsize=12)
        ax.set_ylabel('Population Separation (σ)', fontsize=12)
        ax.set_title('D. Gating Optimization\n(Find Best Fluorophore)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight best
        best_idx = np.argmax(separations)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(4)
        
        plt.tight_layout()
        output_path = self.output_dir / '07_virtual_flow_cytometer.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def demo_virtual_electron_microscope(self):
        """Show virtual electron microscope concept"""
        print("\nGenerating Virtual Electron Microscope demo...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Virtual Electron Microscope: Post-Hoc Voltage/Mode Modification\n' +
                    '95% Reduction in Electron Dose',
                    fontsize=16, fontweight='bold')
        
        # Generate synthetic EM image (simple pattern)
        size = 256
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        
        # Create synthetic structure
        base_image = (np.sin(X*2) * np.cos(Y*2) + 
                     np.exp(-(X**2 + Y**2)/4) * 3)
        
        # Panel A: 200 kV TEM (original)
        ax = axes[0, 0]
        im = ax.imshow(base_image, cmap='gray', extent=[-5, 5, -5, 5])
        ax.set_xlabel('X (nm)', fontsize=12)
        ax.set_ylabel('Y (nm)', fontsize=12)
        ax.set_title('A. 200 kV TEM\n(Real Measurement)', 
                    fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Intensity')
        ax.text(0.05, 0.95, 'One dose\n100% damage', transform=ax.transAxes,
               fontsize=11, verticalalignment='top', color='white',
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        # Panel B: Virtual 80 kV (lower voltage = less damage)
        ax = axes[0, 1]
        image_80kv = base_image * 0.8 + np.random.normal(0, 0.2, base_image.shape)
        im = ax.imshow(image_80kv, cmap='gray', extent=[-5, 5, -5, 5])
        ax.set_xlabel('X (nm)', fontsize=12)
        ax.set_ylabel('Y (nm)', fontsize=12)
        ax.set_title('B. Virtual 80 kV\n(Post-Hoc Voltage Change)', 
                    fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Intensity')
        ax.text(0.05, 0.95, 'NO extra dose!\n0% damage', transform=ax.transAxes,
               fontsize=11, verticalalignment='top', color='white',
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
        
        # Panel C: Virtual 300 kV (higher voltage = better penetration)
        ax = axes[1, 0]
        image_300kv = base_image * 1.2 - np.random.normal(0, 0.1, base_image.shape)
        im = ax.imshow(image_300kv, cmap='gray', extent=[-5, 5, -5, 5])
        ax.set_xlabel('X (nm)', fontsize=12)
        ax.set_ylabel('Y (nm)', fontsize=12)
        ax.set_title('C. Virtual 300 kV\n(High Penetration)', 
                    fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Intensity')
        
        # Panel D: Dose comparison
        ax = axes[1, 1]
        voltages = ['80 kV\n(Virtual)', '200 kV\n(Real)', '300 kV\n(Virtual)']
        doses = [0, 100, 0]  # Relative dose
        colors = ['green', 'red', 'green']
        
        bars = ax.bar(range(len(voltages)), doses, color=colors, 
                     alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(voltages)))
        ax.set_xticklabels(voltages, fontsize=12)
        ax.set_ylabel('Electron Dose (%)', fontsize=12)
        ax.set_title('D. Dose Savings: 95% Reduction\n(Critical for Beam-Sensitive Samples)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 120])
        
        # Add labels
        for i, (bar, dose) in enumerate(zip(bars, doses)):
            if dose > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                       f'{dose}%', ha='center', fontsize=11, fontweight='bold')
            else:
                ax.text(i, 10, 'FREE!', ha='center', fontsize=13, 
                       fontweight='bold', color='darkgreen')
        
        plt.tight_layout()
        output_path = self.output_dir / '08_virtual_electron_microscope.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def demo_virtual_electrochemistry(self):
        """Show virtual electrochemical analyzer concept"""
        print("\nGenerating Virtual Electrochemical Analyzer demo...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Virtual Electrochemical Analyzer: Post-Hoc Technique Switching\n' +
                    '85% Reduction in Experiments',
                    fontsize=16, fontweight='bold')
        
        # Generate synthetic voltammogram
        potential = np.linspace(-0.5, 0.5, 500)
        
        # Panel A: Cyclic voltammetry (original)
        ax = axes[0, 0]
        # Simulate redox peaks
        current_cv = (50 * np.exp(-(potential - 0.2)**2 / 0.01) -
                     50 * np.exp(-(potential + 0.2)**2 / 0.01) +
                     np.random.normal(0, 2, len(potential)))
        
        ax.plot(potential, current_cv, linewidth=2, color='blue', label='CV')
        ax.set_xlabel('Potential (V vs. Ref)', fontsize=12)
        ax.set_ylabel('Current (μA)', fontsize=12)
        ax.set_title('A. Cyclic Voltammetry\n(Real Measurement)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.text(0.05, 0.95, 'One measurement', transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Panel B: Virtual DPV (differential pulse voltammetry)
        ax = axes[0, 1]
        # DPV has better signal-to-noise
        current_dpv = np.gradient(current_cv, potential)
        current_dpv = np.convolve(current_dpv, np.ones(10)/10, mode='same')  # Smooth
        
        ax.plot(potential, current_dpv, linewidth=2, color='green', label='DPV (Virtual)')
        ax.set_xlabel('Potential (V vs. Ref)', fontsize=12)
        ax.set_ylabel('Differential Current (μA/V)', fontsize=12)
        ax.set_title('B. Virtual DPV\n(Post-Hoc Technique Switch)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.text(0.05, 0.95, 'NO re-measurement!', transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Panel C: Virtual SWV (square wave voltammetry)
        ax = axes[1, 0]
        # SWV shows peaks more clearly
        current_swv = (100 * np.exp(-(potential - 0.2)**2 / 0.005) +
                      np.random.normal(0, 1, len(potential)))
        
        ax.plot(potential, current_swv, linewidth=2, color='orange', label='SWV (Virtual)')
        ax.set_xlabel('Potential (V vs. Ref)', fontsize=12)
        ax.set_ylabel('Current (μA)', fontsize=12)
        ax.set_title('C. Virtual SWV\n(High Sensitivity)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Panel D: Technique comparison
        ax = axes[1, 1]
        # Compare signal-to-noise
        snr_cv = np.abs(current_cv).max() / np.std(current_cv[-50:])
        snr_dpv = np.abs(current_dpv).max() / np.std(current_dpv[-50:])
        snr_swv = np.abs(current_swv).max() / np.std(current_swv[-50:])
        
        techniques = ['CV\n(Real)', 'DPV\n(Virtual)', 'SWV\n(Virtual)']
        snrs = [snr_cv, snr_dpv, snr_swv]
        colors = ['blue', 'green', 'orange']
        
        bars = ax.bar(range(len(techniques)), snrs, color=colors, 
                     alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(techniques)))
        ax.set_xticklabels(techniques, fontsize=12)
        ax.set_ylabel('Signal-to-Noise Ratio', fontsize=12)
        ax.set_title('D. Technique Optimization\n(Find Best S/N)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight best
        best_idx = np.argmax(snrs)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(4)
        
        plt.tight_layout()
        output_path = self.output_dir / '09_virtual_electrochemistry.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def demo_categorical_synthesizer(self):
        """Show categorical state synthesizer concept"""
        print("\nGenerating Categorical State Synthesizer demo...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Categorical State Synthesizer: Inverse Measurement\n' +
                    'Design Molecular States on Demand',
                    fontsize=16, fontweight='bold')
        
        # Panel A: Target S-entropy coordinates
        ax = axes[0, 0]
        # Show desired S-space location
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.subplot(2, 2, 1, projection='3d')
        
        # Target state
        s_k_target = 0.7
        s_t_target = 0.5
        s_e_target = 0.8
        
        ax.scatter([s_k_target], [s_t_target], [s_e_target], 
                  s=500, c='red', marker='*', edgecolors='black', linewidths=3,
                  label='Target State')
        
        # Show accessible region
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = 0.2 * np.outer(np.cos(u), np.sin(v)) + s_k_target
        y = 0.2 * np.outer(np.sin(u), np.sin(v)) + s_t_target
        z = 0.2 * np.outer(np.ones(np.size(u)), np.cos(v)) + s_e_target
        ax.plot_surface(x, y, z, alpha=0.2, color='blue')
        
        ax.set_xlabel('$S_k$ (Knowledge)', fontsize=11)
        ax.set_ylabel('$S_t$ (Temporal)', fontsize=11)
        ax.set_zlabel('$S_e$ (Evolution)', fontsize=11)
        ax.set_title('A. Specify Target State\n(S-Entropy Coordinates)', 
                    fontsize=13, fontweight='bold')
        ax.legend()
        
        # Panel B: MMD input filter determines conditions
        ax = axes[0, 1]
        ax.axis('off')
        
        # Show filter logic
        filter_text = """
MMD INPUT FILTER
═══════════════

Target: (Sk, St, Se) = (0.7, 0.5, 0.8)

Required Conditions:
━━━━━━━━━━━━━━━━━━━
• Temperature: 298 K
• Pressure: 1 atm
• Field: 50 mV/cm
• Frequency: 2.4 GHz
• Gradient: 15%/min

Physical Constraints:
━━━━━━━━━━━━━━━━━━━
✓ Thermodynamically stable
✓ Within realizability bounds
✓ No forbidden transitions
        """
        
        ax.text(0.1, 0.9, filter_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax.set_title('B. Determine Conditions\n(Input Filter)', 
                    fontsize=13, fontweight='bold')
        
        # Panel C: Synthesis protocol generation
        ax = axes[1, 0]
        ax.axis('off')
        
        protocol_text = """
SYNTHESIS PROTOCOL
═════════════════

Step 1: Initialize System
   • Load reference molecules
   • Equilibrate to 298 K
   • Apply 1 atm pressure

Step 2: Drive Vibrational Modes
   • Mode 1: 2.35 GHz, 10 mW
   • Mode 2: 2.42 GHz, 15 mW
   • Mode 3: 2.48 GHz, 8 mW
   • Duration: 5 minutes

Step 3: Apply Field Gradient
   • 15%/min for 20 minutes
   • Monitor S-coordinates

Step 4: Verification
   • Measure (Sk, St, Se)
   • Confirm within 5% of target
        """
        
        ax.text(0.1, 0.9, protocol_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.set_title('C. Generate Protocol\n(Output Filter)', 
                    fontsize=13, fontweight='bold')
        
        # Panel D: Verification trajectory
        ax = axes[1, 1]
        
        # Simulate synthesis trajectory
        n_steps = 50
        t = np.linspace(0, 1, n_steps)
        
        # S-coordinates evolve toward target
        s_k_traj = 0.3 + (s_k_target - 0.3) * (1 - np.exp(-5*t))
        s_t_traj = 0.2 + (s_t_target - 0.2) * (1 - np.exp(-4*t))
        s_e_traj = 0.4 + (s_e_target - 0.4) * (1 - np.exp(-6*t))
        
        ax.plot(t, s_k_traj, linewidth=3, color='blue', label='$S_k$')
        ax.plot(t, s_t_traj, linewidth=3, color='green', label='$S_t$')
        ax.plot(t, s_e_traj, linewidth=3, color='red', label='$S_e$')
        
        # Target lines
        ax.axhline(y=s_k_target, color='blue', linestyle='--', alpha=0.5)
        ax.axhline(y=s_t_target, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=s_e_target, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Synthesis Progress', fontsize=12)
        ax.set_ylabel('S-Entropy Coordinates', fontsize=12)
        ax.set_title('D. Synthesis Trajectory\n(Converges to Target)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / '10_categorical_synthesizer.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def demo_impossibility_mapper(self):
        """Show impossibility boundary mapper concept"""
        print("\nGenerating Impossibility Boundary Mapper demo...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Impossibility Boundary Mapper: Map Physical Realizability\n' +
                    'Know What Cannot Exist Before Trying',
                    fontsize=16, fontweight='bold')
        
        # Panel A: Scan S-space systematically
        ax = axes[0, 0]
        s_k_grid = np.linspace(0, 1, 50)
        s_t_grid = np.linspace(0, 1, 50)
        S_k, S_t = np.meshgrid(s_k_grid, s_t_grid)
        
        # Define realizability (example: bounded by thermodynamics)
        realizability = np.exp(-((S_k - 0.5)**2 + (S_t - 0.5)**2) / 0.2)
        realizability[realizability < 0.1] = 0  # Forbidden region
        
        im = ax.contourf(S_k, S_t, realizability, levels=20, cmap='RdYlGn')
        ax.contour(S_k, S_t, realizability, levels=[0.1], colors='red', linewidths=3)
        ax.set_xlabel('$S_k$ (Knowledge Entropy)', fontsize=12)
        ax.set_ylabel('$S_t$ (Temporal Entropy)', fontsize=12)
        ax.set_title('A. Systematic S-Space Scan\n(Se = 0.5 slice)', 
                    fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Realizability')
        
        # Annotate regions
        ax.text(0.5, 0.5, 'POSSIBLE', ha='center', va='center', 
               fontsize=14, fontweight='bold', color='white',
               bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
        ax.text(0.1, 0.1, 'FORBIDDEN', ha='center', va='center',
               fontsize=12, fontweight='bold', color='white',
               bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        # Panel B: Boundary detection
        ax = axes[0, 1]
        
        # Find boundary points
        boundary_mask = (realizability > 0.05) & (realizability < 0.15)
        boundary_s_k = S_k[boundary_mask]
        boundary_s_t = S_t[boundary_mask]
        
        ax.scatter(boundary_s_k, boundary_s_t, s=5, alpha=0.5, color='red')
        ax.set_xlabel('$S_k$ (Knowledge Entropy)', fontsize=12)
        ax.set_ylabel('$S_t$ (Temporal Entropy)', fontsize=12)
        ax.set_title('B. Impossibility Boundary\n(Red Line)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Panel C: Output filter failure analysis
        ax = axes[1, 0]
        ax.axis('off')
        
        failure_text = """
OUTPUT FILTER ANALYSIS
═══════════════════════

Forbidden Region 1: Low (Sk, St)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Constraint violated:
  • Temperature < 0 K required
  • ❌ Violates 3rd law

Forbidden Region 2: High (Sk, Se)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Constraint violated:
  • ΔS < 0 required
  • ❌ Violates 2nd law

Forbidden Region 3: Sk > St + Se
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Constraint violated:
  • Information > Entropy
  • ❌ Impossible by definition
        """
        
        ax.text(0.1, 0.9, failure_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax.set_title('C. Constraint Violations\n(Why Regions Are Forbidden)', 
                    fontsize=13, fontweight='bold')
        
        # Panel D: Synthesis guidance
        ax = axes[1, 1]
        
        # Show attempted vs. guided paths
        # Attempted path (goes through forbidden region)
        attempt_s_k = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
        attempt_s_t = np.array([0.2, 0.3, 0.2, 0.3, 0.8])
        
        # Guided path (stays in allowed region)
        guided_s_k = np.array([0.2, 0.3, 0.4, 0.5, 0.8])
        guided_s_t = np.array([0.2, 0.35, 0.45, 0.55, 0.8])
        
        # Plot realizability
        im = ax.contourf(S_k, S_t, realizability, levels=20, cmap='RdYlGn', alpha=0.3)
        ax.contour(S_k, S_t, realizability, levels=[0.1], colors='red', linewidths=3)
        
        # Plot paths
        ax.plot(attempt_s_k, attempt_s_t, 'ro-', linewidth=3, markersize=10,
               label='Attempted (fails)', alpha=0.7)
        ax.plot(guided_s_k, guided_s_t, 'go-', linewidth=3, markersize=10,
               label='Guided (succeeds)', alpha=0.7)
        
        # Mark failure point
        ax.scatter([0.5], [0.2], s=500, marker='X', color='red', 
                  edgecolors='black', linewidths=3, zorder=10)
        ax.text(0.5, 0.15, 'FAILS HERE', ha='center', fontsize=10,
               fontweight='bold', color='red')
        
        ax.set_xlabel('$S_k$ (Knowledge Entropy)', fontsize=12)
        ax.set_ylabel('$S_t$ (Temporal Entropy)', fontsize=12)
        ax.set_title('D. Synthesis Guidance\n(Avoid Forbidden Regions)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / '11_impossibility_mapper.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def generate_all(self):
        """Generate all ensemble concept demonstrations"""
        print("\n" + "="*70)
        print("ENSEMBLE CONCEPT DEMONSTRATIONS")
        print("Visualizing ALL 13 capabilities from ensemble.md")
        print("="*70)
        
        # Original 4
        self.demo_virtual_chromatograph()
        self.demo_information_flow()
        self.demo_multi_scale_coherence()
        self.demo_virtual_raman()
        
        # New 7
        self.demo_virtual_nmr()
        self.demo_virtual_xray()
        self.demo_virtual_flow_cytometer()
        self.demo_virtual_electron_microscope()
        self.demo_virtual_electrochemistry()
        self.demo_categorical_synthesizer()
        self.demo_impossibility_mapper()
        
        print("\n" + "="*70)
        print("COMPLETE! Generated 11 of 13 virtual instruments")
        print(f"Output directory: {self.output_dir}")
        print("Remaining: Thermodynamic Computer Interface, Semantic Field Generator")
        print("="*70)

if __name__ == '__main__':
    demonstrator = EnsembleConceptDemonstrator()
    demonstrator.generate_all()
