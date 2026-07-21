# fresnel-diffraction-monte-carlo-simulator
modeling 1D and 2D Fresnel diffraction patterns for rectangular and circular apertures. Compares deterministic numerical integration with stochastic Monte Carlo methods.

## Features & Mathematical Methods
* **1D & 2D Integration Pipelines:** Uses SciPy's `dblquad` to evaluate the real and imaginary components of the Fresnel diffraction integral.
* **Complex Aperture Geometries:** Models spatial filtering for both rectangular constraints and custom circular bounds.
* **Monte Carlo Integration:** Replaces deterministic 2D quadrature with stochastic area sampling to estimate integrals, complete with variance-based error bounding.

## Sample Output
![Fresnel 1D Diffraction Pattern](Fresnel_1D.png)
![Fresnel 1D error](Fresnel_1D_error.png)
![Fresnel 2D Diffraction Pattern](Fresnel_2D.png)
![Fresnel 2D error](Fresnel_2D_error.png)
![Fresnel 2D Circle Diffraction Pattern](Fresnel_2D_circle.png)
![Fresnel 2D Circle error](Fresnel_2D_circle_error.png)
![Fresnel Monte Carlo Diffraction Pattern](Fresnel_MC.png)
![Fresnel Monte Carlo error](Fresnel_MC_error.png)

## How to Run
```bash
# Clone repository
git clone [https://github.com/bobryanlow-dotcom/fresnel-diffraction-monte-carlo-simulator.git](https://github.com/bobryanlow-dotcom/fresnel-diffraction-monte-carlo-simulator.git)

# Run simulation
python main.py
