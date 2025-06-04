import pytest
import numpy as np
from src.beam_propagator import propagate_beam

def test_propagation_basic():
    """Test basic Fraunhofer propagation with a simple phase screen."""
    # Create a simple phase screen (Gaussian phase)
    N = 256
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    phase_screen = np.exp(-(X**2 + Y**2))  # Gaussian phase
    
    # Propagate with visible wavelength
    result = propagate_beam(500e-9, phase_screen)
    
    # Basic checks
    assert result.intensity.shape == (N, N)
    assert result.phase.shape == (N, N)
    assert np.all(np.isfinite(result.intensity))
    assert np.all(np.isfinite(result.phase))
    
    # Check that intensity is non-negative
    assert np.all(result.intensity >= 0)
    
    # Check that phase is in [-π, π]
    assert np.all(result.phase >= -np.pi)
    assert np.all(result.phase <= np.pi)

def test_propagation_flat_phase():
    """Test propagation with a flat phase screen."""
    N = 256
    phase_screen = np.zeros((N, N))
    
    result = propagate_beam(500e-9, phase_screen)
    
    # With flat phase, should get a central peak
    center = N // 2
    assert np.argmax(result.intensity[center, :]) == center
    assert np.argmax(result.intensity[:, center]) == center
