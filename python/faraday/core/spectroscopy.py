"""
SPECTROSCOPY
============

Spectroscopic validation framework for the partition theory.

Key validation principle:
- Absorption, emission, and scattering are ALL partition operations
- Classical, quantum, and partition descriptions are EQUIVALENT
- The triple equivalence is not an approximation - it IS physics

From the framework:
1. Absorption = partition operation that increases S_e (evolution entropy)
2. Emission = partition operation that decreases S_e
3. Scattering = partition operation that rotates in S-space

Each spectroscopic measurement CREATES categorical distinction.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable
from enum import Enum
import json
from datetime import datetime

from .partition_fluid_structure import (
    PartitionFluid, SCoordinate, PartitionCascade, MolecularSpecies,
    k_B, hbar, c, h
)
from .partition_light import (
    PartitionLight, PartitionPhoton, ElectronOscillator,
    wavelength_to_omega, omega_to_wavelength
)
from .partition_optics import (
    PartitionOptics, CauchyDispersion, AbsorptionCrossSection,
    SpectralMeasurement, SpectralRegion
)


# Constants
e = 1.602176634e-19  # Elementary charge [C]


class SpectroscopyType(Enum):
    """Types of spectroscopic measurement"""
    ABSORPTION = "absorption"  # Photon absorbed, energy to molecule
    EMISSION = "emission"  # Photon emitted, energy from molecule
    RAMAN = "raman"  # Inelastic scattering
    FLUORESCENCE = "fluorescence"  # Absorption + emission
    INFRARED = "infrared"  # Vibrational transitions
    UV_VIS = "uv_visible"  # Electronic transitions


@dataclass
class PartitionTransition:
    """
    A spectroscopic transition IS a partition operation.

    Initial state: S-coordinate (S_k, S_t, S_e)_i
    Final state: S-coordinate (S_k, S_t, S_e)_f

    The transition IS the partition that connects them.
    """
    initial_state: SCoordinate
    final_state: SCoordinate
    energy_eV: float  # Transition energy
    oscillator_strength: float = 1.0  # Dimensionless

    @property
    def delta_S(self) -> np.ndarray:
        """Change in S-coordinates"""
        return self.final_state.to_array() - self.initial_state.to_array()

    @property
    def delta_S_k(self) -> float:
        """Change in knowledge entropy"""
        return self.final_state.S_k - self.initial_state.S_k

    @property
    def delta_S_e(self) -> float:
        """Change in evolution entropy"""
        return self.final_state.S_e - self.initial_state.S_e

    @property
    def wavelength_nm(self) -> float:
        """Transition wavelength [nm]"""
        return h * c / (self.energy_eV * e) * 1e9

    @property
    def omega(self) -> float:
        """Transition angular frequency [rad/s]"""
        return self.energy_eV * e / hbar

    def is_absorption(self) -> bool:
        """Absorption increases S_e (molecule gains energy)"""
        return self.delta_S_e > 0

    def is_emission(self) -> bool:
        """Emission decreases S_e (molecule loses energy)"""
        return self.delta_S_e < 0


@dataclass
class Spectrum:
    """
    A spectrum IS a collection of partition operations.

    Each peak corresponds to a categorical transition.
    The intensity reflects the partition probability.
    """
    wavelengths_nm: np.ndarray
    intensities: np.ndarray
    spectroscopy_type: SpectroscopyType
    temperature_K: float = 298.0

    def __post_init__(self):
        assert len(self.wavelengths_nm) == len(self.intensities)

    @property
    def n_points(self) -> int:
        return len(self.wavelengths_nm)

    @property
    def peak_wavelength(self) -> float:
        """Wavelength of maximum intensity"""
        idx = np.argmax(self.intensities)
        return self.wavelengths_nm[idx]

    @property
    def peak_intensity(self) -> float:
        """Maximum intensity"""
        return np.max(self.intensities)

    def fwhm(self) -> float:
        """Full width at half maximum [nm]"""
        half_max = self.peak_intensity / 2
        above_half = self.wavelengths_nm[self.intensities >= half_max]
        if len(above_half) < 2:
            return 0.0
        return above_half[-1] - above_half[0]

    def to_wavenumber(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to wavenumber [cm^-1]"""
        wavenumbers = 1e7 / self.wavelengths_nm
        return wavenumbers, self.intensities


@dataclass
class FluidPathSpectrometer:
    """
    A spectrometer that measures light propagation through a partition fluid.

    This IS the validation experiment:
    1. Generate fluid from partition parameters
    2. Derive optical properties from same parameters
    3. Validate tau_c^(opt) = 2 x tau_c^(mech)
    """
    fluid: PartitionFluid
    light: PartitionLight
    optics: PartitionOptics
    path_length: float = 0.01  # Default 1 cm

    @classmethod
    def create(cls, species: MolecularSpecies = MolecularSpecies.CCL4,
               temperature: float = 298.0,
               resonance_nm: float = 218.0,
               path_length: float = 0.01) -> 'FluidPathSpectrometer':
        """Create spectrometer for a given molecular species"""
        fluid = PartitionFluid.create(species, temperature)
        light = PartitionLight.from_fluid(fluid, resonance_wavelength_nm=resonance_nm)
        optics = PartitionOptics(light)
        return cls(fluid=fluid, light=light, optics=optics, path_length=path_length)

    def measure_transmission_spectrum(self,
                                       wavelengths_nm: np.ndarray) -> Spectrum:
        """Measure transmission spectrum through the fluid"""
        transmissions = np.array([
            self.optics.beer_lambert_transmission(wl, self.path_length)
            for wl in wavelengths_nm
        ])

        return Spectrum(
            wavelengths_nm=wavelengths_nm,
            intensities=transmissions,
            spectroscopy_type=SpectroscopyType.UV_VIS,
            temperature_K=self.fluid.temperature
        )

    def measure_absorption_spectrum(self,
                                     wavelengths_nm: np.ndarray) -> Spectrum:
        """Measure absorption spectrum (1 - T)"""
        trans_spectrum = self.measure_transmission_spectrum(wavelengths_nm)
        absorptions = 1.0 - trans_spectrum.intensities

        return Spectrum(
            wavelengths_nm=wavelengths_nm,
            intensities=absorptions,
            spectroscopy_type=SpectroscopyType.ABSORPTION,
            temperature_K=self.fluid.temperature
        )

    def measure_viscosity(self) -> float:
        """
        Mechanical measurement: viscosity from partition structure.

        mu = tau_c x g

        This uses the SAME tau_c that determines optical properties.
        """
        return self.fluid.viscosity

    def measure_optical_linewidth(self) -> float:
        """
        Optical measurement: absorption linewidth.

        FWHM = 2gamma = 2/tau_c

        BUT optical measurement resolves TWO electron commitments
        per collision, so effective tau_c^(opt) = 2 x tau_c^(mech)
        """
        return self.light.linewidth_fwhm

    def extract_tau_mechanical(self) -> float:
        """Extract tau_c from mechanical measurement (viscosity)"""
        mu = self.measure_viscosity()
        g = self.fluid.g
        return mu / g

    def extract_tau_optical(self) -> float:
        """
        Extract tau_c from optical measurement.

        tau_c^(opt) = 2 * tau_c^(mech)

        The factor of 2 accounts for two electron commitments
        (approach + separation) per collision.
        """
        return self.light.tau_c_optical

    def validate_tau_relationship(self) -> Dict:
        """
        THE KEY VALIDATION:

        tau_c^(opt) / tau_c^(mech) = 2.0

        This factor of 2 arises because:
        - Each molecular collision involves TWO electron "commitments"
        - Approach commitment: electrons redistribute for overlap
        - Separation commitment: electrons re-localize after overlap

        Mechanical measurement: sees complete collision as single event
        Optical measurement: resolves both electron commitments

        Agreement validates the electron trajectory model.
        """
        tau_mech = self.extract_tau_mechanical()
        tau_opt = self.extract_tau_optical()

        ratio = tau_opt / tau_mech
        predicted_ratio = 2.0
        error = abs(ratio - predicted_ratio) / predicted_ratio

        return {
            'tau_mechanical_s': tau_mech,
            'tau_optical_s': tau_opt,
            'ratio': ratio,
            'predicted_ratio': predicted_ratio,
            'relative_error': error,
            'absolute_error': abs(ratio - predicted_ratio),
            'validated': error < 0.2,  # 20% tolerance
            'interpretation': (
                "Optical measurement resolves two electron commitments "
                "(approach + separation) per collision, while mechanical "
                "measurement integrates over the complete collision."
            )
        }


@dataclass
class FluidPathValidationExperiment:
    """
    Complete fluid path validation experiment.

    Validates electron trajectory model by showing:
    tau_c from viscosity = tau_c from optics / 2

    This is NOT simulation - it is GENERATION from partition axioms.
    """
    spectrometer: FluidPathSpectrometer
    results: Dict = field(default_factory=dict)

    @classmethod
    def run(cls, species: MolecularSpecies = MolecularSpecies.CCL4,
            temperature: float = 298.0) -> 'FluidPathValidationExperiment':
        """Run the complete validation experiment"""
        # Create spectrometer
        spec = FluidPathSpectrometer.create(species, temperature)

        # Run experiment
        experiment = cls(spectrometer=spec)
        experiment.execute()

        return experiment

    def execute(self) -> None:
        """Execute all validation tests"""
        self.results = {
            'experiment': 'Fluid Path Validation',
            'timestamp': datetime.now().isoformat(),
            'fluid': {
                'temperature_K': self.spectrometer.fluid.temperature,
                'n_density': self.spectrometer.fluid.n_density,
                'viscosity_Pa_s': self.spectrometer.fluid.viscosity,
                'tau_c_s': self.spectrometer.fluid.tau_c,
                'coupling_g_Pa': self.spectrometer.fluid.g,
            },
            'optical': {
                'resonance_omega': self.spectrometer.light.omega_0,
                'plasma_omega': self.spectrometer.light.omega_p,
                'gamma': self.spectrometer.light.gamma,
                'linewidth_fwhm': self.spectrometer.light.linewidth_fwhm,
            },
            'validation': self.spectrometer.validate_tau_relationship(),
        }

        # Generate spectra
        wavelengths = np.linspace(200, 700, 501)
        abs_spectrum = self.spectrometer.measure_absorption_spectrum(wavelengths)

        self.results['spectrum'] = {
            'wavelengths_nm': wavelengths.tolist(),
            'absorption': abs_spectrum.intensities.tolist(),
            'peak_wavelength_nm': abs_spectrum.peak_wavelength,
            'fwhm_nm': abs_spectrum.fwhm(),
        }

    def print_report(self) -> None:
        """Print validation report"""
        print("=" * 70)
        print("FLUID PATH VALIDATION EXPERIMENT")
        print("=" * 70)

        print(f"\nTimestamp: {self.results['timestamp']}")

        print(f"\nFLUID PROPERTIES:")
        fluid = self.results['fluid']
        print(f"  Temperature: {fluid['temperature_K']} K")
        print(f"  Number density: {fluid['n_density']:.3e} m^-3")
        print(f"  Viscosity: {fluid['viscosity_Pa_s']:.4e} Pa*s")
        print(f"  Partition lag tau_c: {fluid['tau_c_s']:.3e} s")
        print(f"  Coupling g: {fluid['coupling_g_Pa']:.3e} Pa")

        print(f"\nOPTICAL PROPERTIES:")
        opt = self.results['optical']
        print(f"  Resonance omega_0: {opt['resonance_omega']:.3e} rad/s")
        print(f"  Plasma omega_p: {opt['plasma_omega']:.3e} rad/s")
        print(f"  Damping gamma: {opt['gamma']:.3e} rad/s")
        print(f"  Linewidth FWHM: {opt['linewidth_fwhm']:.3e} rad/s")

        print(f"\nKEY VALIDATION: tau_c^(opt) / tau_c^(mech) = 2.0")
        val = self.results['validation']
        print(f"  tau_c (mechanical): {val['tau_mechanical_s']:.3e} s")
        print(f"  tau_c (optical): {val['tau_optical_s']:.3e} s")
        print(f"  Ratio: {val['ratio']:.3f}")
        print(f"  Predicted: {val['predicted_ratio']:.1f}")
        print(f"  Error: {val['relative_error']*100:.1f}%")

        status = "[PASS]" if val['validated'] else "[FAIL]"
        print(f"\n  STATUS: {status}")
        print(f"\n  Interpretation: {val['interpretation']}")

        print(f"\nSPECTRUM:")
        spec = self.results['spectrum']
        print(f"  Peak wavelength: {spec['peak_wavelength_nm']:.1f} nm")
        print(f"  FWHM: {spec['fwhm_nm']:.1f} nm")

        print("=" * 70)

    def save_results(self, filepath: str) -> None:
        """Save results to JSON file"""
        with open(filepath, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to: {filepath}")


def run_fluid_path_validation():
    """Run the complete fluid path validation experiment"""
    print("\n" + "=" * 70)
    print("RUNNING FLUID PATH VALIDATION EXPERIMENT")
    print("=" * 70)
    print("\nThis validates electron trajectories by showing:")
    print("  tau_c (optical) = 2 x tau_c (mechanical)")
    print("\nThe factor of 2 arises from two electron 'commitments' per collision:")
    print("  1. Approach commitment (electrons redistribute for overlap)")
    print("  2. Separation commitment (electrons re-localize after overlap)")
    print("=" * 70)

    experiment = FluidPathValidationExperiment.run(
        species=MolecularSpecies.CCL4,
        temperature=298.0
    )

    experiment.print_report()

    return experiment


if __name__ == "__main__":
    experiment = run_fluid_path_validation()
