"""
Fresnel Diffraction & Monte Carlo Integration Simulator
Author: Bo Bryan-Low
Description: Simulates 1D and 2D Fresnel diffraction patterns for rectangular 
and circular apertures. Compares deterministic numerical integration (SciPy) 
with Monte Carlo integration methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# Core Physics Functions
def field_real(yp: float, xp: float, y: float, x: float, k: float, z: float) -> float:
    """Real component of the Fresnel diffraction integral."""
    return np.cos((k / (2 * z)) * ((x - xp)**2 + (y - yp)**2))

def field_imag(yp: float, xp: float, y: float, x: float, k: float, z: float) -> float:
    """Imaginary component of the Fresnel diffraction integral."""
    return np.sin((k / (2 * z)) * ((x - xp)**2 + (y - yp)**2))

# Simulation Modules
def simulate_1d_fresnel():
    """Part 1: Simulates 1D Fresnel diffraction pattern."""
    print("Running Part 1: 1D Fresnel Diffraction...")
    
    def total_field(y, x, wavelength, z, xp1, xp2, yp1, yp2):
        k = 2 * np.pi / wavelength
        E0 = 1 
        
        realpart, realerror = dblquad(field_real, xp1, xp2, yp1, yp2, args=(y, x, k, z), epsabs=1e-10, epsrel=1e-10)
        imagpart, imagerror = dblquad(field_imag, xp1, xp2, yp1, yp2, args=(y, x, k, z), epsabs=1e-10, epsrel=1e-10)
        
        error_total = np.sqrt(realerror**2 + imagerror**2)
        E = (k * E0 / (2 * np.pi * z)) * (realpart + imagpart * 1j)
        return E, error_total

    xp1 = yp1 = -2.5e-5
    xp2 = yp2 = 2.5e-5
    y = 0
    num = 200
    z = 0.1
    wavelength = 1e-6
    xvals = np.linspace(-5e-3, 5e-3, num)

    intensities, errors = [], []
    for x in xvals:
        E, err = total_field(y, x, wavelength, z, xp1, xp2, yp1, yp2)
        intensities.append(np.abs(E)**2)
        errors.append(err)

    # Plotting
    plt.figure(figsize=(7, 4))
    plt.plot(xvals, intensities)
    plt.xlabel("x (m)")
    plt.ylabel(r"Relative Intensity $|E|^2$")
    plt.title("1D Fresnel Diffraction Pattern")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(xvals, errors)
    plt.xlabel("x (m)")
    plt.ylabel("Integration Error Estimate")
    plt.title("Part 1: Integration Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def simulate_2d_rectangular():
    """Part 2: Simulates 2D rectangular aperture diffraction."""
    print("Running Part 2: 2D Rectangular Aperture...")
    
    def total_field(y, x, wavelength, z, aperture_width):
        k = 2 * np.pi / wavelength
        E0 = 1
        xp1 = yp1 = -aperture_width / 2
        xp2 = yp2 = aperture_width / 2

        realpart, realerror = dblquad(field_real, xp1, xp2, yp1, yp2, args=(y, x, k, z), epsabs=1e-10, epsrel=1e-10)
        imagpart, imagerror = dblquad(field_imag, xp1, xp2, yp1, yp2, args=(y, x, k, z), epsabs=1e-10, epsrel=1e-10)
        
        error_total = np.sqrt(realerror**2 + imagerror**2)
        E = (k * E0 / (2 * np.pi * z)) * (realpart + imagpart * 1j)
        return E, error_total

    aperture_width = 2e-5 
    z = 0.1 
    wavelength = 5e-7
    x1 = y1 = -0.01 
    x2 = y2 = -x1
    num = 50

    xvals = np.linspace(x1, x2, num)
    yvals = np.linspace(x1, x2, num)
    X, Y = np.meshgrid(xvals, yvals) 

    E_field = np.zeros_like(X, dtype=complex)
    error_total = np.zeros_like(X, dtype=float)

    for i in range(num):
        for j in range(num):
            E, err = total_field(Y[i, j], X[i, j], wavelength, z, aperture_width)
            E_field[i, j] = E
            error_total[i, j] = err
            
    intensity = np.abs(E_field)**2 
    extents = (x1, x2, y1, y2) 

    # Plot 1: Intensity
    plt.imshow(intensity, vmin=0.0, vmax=1.0 * intensity.max(), extent=extents, origin="lower", cmap="nipy_spectral_r") 
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f'Rectangular aperture diffraction\nz = {z:4.2f}m, aperture_width = {aperture_width:1.1e}m')
    plt.colorbar(label=r"Intensity $|E|^2$")
    plt.show()

    # Plot 2: Integration Error
    plt.imshow(error_total, vmin=0.0, vmax=1.0 * error_total.max(), extent=extents, origin="lower", cmap="nipy_spectral_r") 
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Part 2 Integration Error")
    plt.colorbar(label="Error Estimate")
    plt.show()


def simulate_2d_circular():
    """Part 3: Simulates 2D circular aperture diffraction using dblquad."""
    print("Running Part 3: 2D Circular Aperture (SciPy Integration)...")
    
    def total_field(y, x, wavelength, z, R):
        k = 2 * np.pi / wavelength
        E0 = 1
        xp1, xp2 = -R, R

        def yp1func(xp): return -np.sqrt(R**2 - xp**2)
        def yp2func(xp): return np.sqrt(R**2 - xp**2)

        realpart, realerror = dblquad(field_real, xp1, xp2, yp1func, yp2func, args=(y, x, k, z), epsabs=1e-10, epsrel=1e-10)
        imagpart, imagerror = dblquad(field_imag, xp1, xp2, yp1func, yp2func, args=(y, x, k, z), epsabs=1e-10, epsrel=1e-10)
        
        error_total = np.sqrt(realerror**2 + imagerror**2)
        E = (k * E0 / (2 * np.pi * z)) * (realpart + imagpart * 1j)
        return E, error_total

    aperture_width = 2e-5
    R = aperture_width / 2.0
    z = 0.1  
    wavelength = 5e-7
    x1 = y1 = -0.01
    x2 = y2 = -x1   
    num = 50  

    xvals = np.linspace(x1, x2, num)
    yvals = np.linspace(y1, y2, num)
    X, Y = np.meshgrid(xvals, yvals) 

    E_field = np.zeros_like(X, dtype=complex)
    error_total = np.zeros_like(X, dtype=float)
    
    for i in range(num):
        for j in range(num):
            E, err = total_field(Y[i, j], X[i, j], wavelength, z, R)
            E_field[i, j] = E
            error_total[i, j] = err

    intensity = np.abs(E_field)**2 
    extents = (x1, x2, y1, y2) 

    # Plot 1: Intensity
    plt.imshow(intensity, vmin=0.0, vmax=1.0 * intensity.max(), extent=extents, origin="lower", cmap="nipy_spectral_r") 
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f'Circular aperture diffraction\nz = {z:4.2f}m, R = {R:1.1e}m')
    plt.colorbar(label=r"Intensity $|E|^2$")
    plt.show()

    # Plot 2: Integration Error
    plt.imshow(error_total, vmin=0.0, vmax=1.0 * error_total.max(), extent=extents, origin="lower", cmap="nipy_spectral_r") 
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Part 3 Integration Error")
    plt.colorbar(label="Error Estimate")
    plt.show()


def simulate_mc_circular():
    """Part 4: Simulates 2D circular aperture using Monte Carlo Integration."""
    print("Running Part 4: 2D Circular Aperture (Monte Carlo Integration)...")
    
    def field_mc(y, x, wavelength, z, R, N):
        k = 2 * np.pi / wavelength
        E0 = 1

        xsamples = np.random.uniform(-R, R, N)
        ysamples = np.random.uniform(-R, R, N)

        # Filter points within the circular aperture
        values = xsamples**2 + ysamples**2 <= R*R
        xsamples = xsamples[values]
        ysamples = ysamples[values]

        real_vals = field_real(ysamples, xsamples, y, x, k, z)
        imag_vals = field_imag(ysamples, xsamples, y, x, k, z)

        real_mean = np.mean(real_vals)
        imag_mean = np.mean(imag_vals)
        
        A = np.pi * R * R 
        
        real_var = np.mean(real_vals**2) - real_mean**2
        imag_var = np.mean(imag_vals**2) - imag_mean**2

        real_err = A * np.sqrt(real_var) / np.sqrt(N)
        imag_err = A * np.sqrt(imag_var) / np.sqrt(N)
        
        error_total = np.sqrt(real_err**2 + imag_err**2)
        E = (E0 * k / (2 * np.pi * z)) * (A * real_mean + 1j * A * imag_mean)
        return E, error_total

    aperture_width = 2e-5
    R = aperture_width / 2
    wavelength = 5e-7
    z = 0.1
    N = 1500
    x1 = -0.01
    x2 = -x1
    num = 60

    x_vals = np.linspace(x1, x2, num)
    y_vals = np.linspace(x1, x2, num)
    X, Y = np.meshgrid(x_vals, y_vals)

    E_field = np.zeros_like(X, dtype=complex)
    error_total = np.zeros_like(X, dtype=float)

    for i in range(num):
        for j in range(num):
            E, err = field_mc(Y[i, j], X[i, j], wavelength, z, R, N)
            E_field[i, j] = E
            error_total[i, j] = err

    intensity = np.abs(E_field)**2
    extents = (x1, x2, x1, x2) 

    # Plot 1: Intensity
    plt.imshow(intensity, vmin=0.0, vmax=1.0 * intensity.max(), extent=extents, origin="lower", cmap="nipy_spectral_r") 
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f'Monte Carlo Integration for Circular aperture\nz = {z:4.2f}m, R = {R:1.1e}m')
    plt.colorbar(label=r"Intensity $|E|^2$")
    plt.show()

    # Plot 2: Integration Error
    plt.imshow(error_total, vmin=0.0, vmax=1.0 * error_total.max(), extent=extents, origin="lower", cmap="nipy_spectral_r") 
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Part 4 Integration Error")
    plt.colorbar(label="Error Estimate")
    plt.show()

# Execution Block
if __name__ == "__main__":
    # Uncomment the function you wish to run:
    
    simulate_1d_fresnel()
    simulate_2d_rectangular()
    simulate_2d_circular()
    simulate_mc_circular()
