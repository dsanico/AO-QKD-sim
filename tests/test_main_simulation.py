import numpy as np
from src.turbulence_model import generate_phase_screen
from src.beam_propagator import propagate_beam
from numpy.testing import assert_allclose

def test_phase_screen_generation(config):
    phase_screen = generate_phase_screen(config['atmosphere'])
    
    # Basic shape validation
    assert phase_screen.shape == (config['atmosphere']['grid_size'], 
                                 config['atmosphere']['grid_size'])
    
    # Statistical validation
    theoretical_var = 1.03*(config['atmosphere']['r0']**(-5/3)) * \
                     (config['atmosphere']['L0']/config['atmosphere']['l0'])**(1/3)
    assert_allclose(np.var(phase_screen), theoretical_var, rtol=0.2)

def test_beam_propagation(config):
    phase_screen = generate_phase_screen(config['atmosphere'])
    focal_plane = propagate_beam(config['simulation']['wavelength'], phase_screen)
    
    # Validate output dimensions
    assert focal_plane.intensity.shape == (config['atmosphere']['grid_size'], 
                                          config['atmosphere']['grid_size'])
    
    # Photon conservation check
    input_energy = np.sum(np.exp(1j * phase_screen))
    output_energy = np.sum(focal_plane.intensity)
    assert_allclose(input_energy, output_energy, atol=1e-3)

def test_config_loading(config):
    # Validate critical parameters
    assert config['simulation']['wavelength'] == 1550e-9
    assert config['atmosphere']['r0'] == 0.15
    assert config['quantum']['photon_rate'] == 1e6
