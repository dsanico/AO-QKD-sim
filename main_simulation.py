# main_simulation.py
from src.turbulence_model import generate_phase_screen
from src.beam_propagator import propagate_beam
import matplotlib.pyplot as plt

# Initialize components
phase_screen = generate_phase_screen(config['atmosphere']) 
focal_plane = propagate_beam(config['simulation']['wavelength'], phase_screen)

# Visualize results
plt.figure(figsize=(10,4))
plt.subplot(121).imshow(phase_screen, cmap='jet')
plt.title('Atmospheric Phase Screen [rad]')

plt.subplot(122).imshow(focal_plane.intensity.shaped, norm='log')
plt.title('Focal Plane Intensity (log scale)')
plt.tight_layout()
plt.savefig('data/beam_propagation/first_run.png')
