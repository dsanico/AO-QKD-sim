import numpy as np
from dataclasses import dataclass

@dataclass
class PropagationResult:
    intensity: np.ndarray
    phase: np.ndarray

def propagate_beam(wavelength: float, phase_screen: np.ndarray) -> PropagationResult:
    """
    Simple Fraunhofer (far-field) propagation of a beam through a phase screen.
    
    Args:
        wavelength: Wavelength of the beam in meters
        phase_screen: 2D array of phase values in radians
        
    Returns:
        PropagationResult containing the intensity and phase in the focal plane
    """
    # Create input field from phase screen
    field = np.exp(1j * phase_screen)
    
    # Fraunhofer propagation: FFT of the field
    fft_field = np.fft.fftshift(np.fft.fft2(field))
    
    # Calculate intensity and phase
    intensity = np.abs(fft_field)**2
    phase = np.angle(fft_field)
    
    return PropagationResult(
        intensity=intensity,
        phase=phase
    )
