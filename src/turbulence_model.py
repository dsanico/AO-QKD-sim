# src/turbulence_model.py
from aotools.turbulence import ft_phase_screen
import hcipy
import yaml
import numpy as np
import re

def generate_phase_screen(config):
    """Create atmospheric phase screen using Von Karman statistics"""
    # Default values
    defaults = {
        'grid_size': 256,
        'pixel_scale': 0.01,  # meters per pixel
        'outer_scale': config.get('L0', 25),  # meters
        'inner_scale': config.get('l0', 0.01),  # meters
        'r0': config.get('r0', 0.15)  # meters
    }
    
    # Update defaults with any provided values
    for key in defaults:
        if key in config:
            defaults[key] = config[key]
    
    return ft_phase_screen(
        r0=defaults['r0'],
        N=defaults['grid_size'],
        delta=defaults['pixel_scale'],
        L0=defaults['outer_scale'],
        l0=defaults['inner_scale']
    )

def _convert_numeric_strings(d):
    if isinstance(d, dict):
        return {k: _convert_numeric_strings(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_convert_numeric_strings(x) for x in d]
    elif isinstance(d, str):
        try:
            return float(d)
        except ValueError:
            return d
    else:
        return d

def load_config(config_path):
    """Load and parse configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = _convert_numeric_strings(config)
    return config

# src/beam_propagator.py  
def propagate_beam(wavelength, phase_screen):
    """HCIPy-based beam propagation through turbulence"""
    grid = hcipy.make_pupil_grid(phase_screen.shape[0])
    aperture = hcipy.circular_aperture(1)(grid)
    
    # Create coherent beam with quantum-level photons
    field = hcipy.Field(np.sqrt(1e-3)*aperture, grid)  # Single photon approximation
    distorted_field = field * np.exp(1j * phase_screen)
    
    # Fraunhofer propagation to focal plane
    propagator = hcipy.FraunhoferPropagator(grid, wavelength)
    return propagator.forward(distorted_field)
