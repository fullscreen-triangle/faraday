"""
FARADAY CORE
============

Core modules for partition-based physics generation.

This is NOT simulation - it IS physics:
- Fluids ARE partition networks with τ_c and g
- Light IS partition completion propagation
- Spectroscopy IS categorical state measurement

Key modules:
- partition_fluid_structure: Fluid as partition network
- partition_light: Light as partition completion cascade
- partition_optics: Optical properties from partition dynamics
- spectroscopy: Spectroscopic validation framework
"""

from .partition_fluid_structure import (
    PartitionFluid,
    PartitionCascade,
    PartitionOperation,
    SCoordinate,
    MolecularSpecies,
    k_B, hbar, c, h,
)

from .partition_light import (
    PartitionLight,
    PartitionPhoton,
    ElectronOscillator,
    wavelength_to_omega,
    omega_to_wavelength,
)

from .partition_optics import (
    PartitionOptics,
    CauchyDispersion,
    AbsorptionCrossSection,
    SpectralMeasurement,
    SpectralRegion,
)

from .spectroscopy import (
    FluidPathSpectrometer,
    FluidPathValidationExperiment,
    Spectrum,
    SpectroscopyType,
    PartitionTransition,
)

__all__ = [
    # Fluid structure
    'PartitionFluid',
    'PartitionCascade',
    'PartitionOperation',
    'SCoordinate',
    'MolecularSpecies',
    # Light
    'PartitionLight',
    'PartitionPhoton',
    'ElectronOscillator',
    'wavelength_to_omega',
    'omega_to_wavelength',
    # Optics
    'PartitionOptics',
    'CauchyDispersion',
    'AbsorptionCrossSection',
    'SpectralMeasurement',
    'SpectralRegion',
    # Spectroscopy
    'FluidPathSpectrometer',
    'FluidPathValidationExperiment',
    'Spectrum',
    'SpectroscopyType',
    'PartitionTransition',
    # Constants
    'k_B', 'hbar', 'c', 'h',
]
