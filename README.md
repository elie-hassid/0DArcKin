# Arc Discharge Simulator

This Python project simulates the time evolution of a cylindrical arc discharge using measured transport parameters and calculates key plasma properties, including current, voltage, power, electron density, reduced electric field, and Specific Energy Input (SEI). It also provides visualization of the simulation results.

---

## Features

- Interpolates transport parameters (mobility, ionization, attachment) from CSV data.
- Solves for the electric field and electron density at each time step using `scipy.optimize.brentq`.
- Computes arc current, voltage, power, arc length, reduced electric field, and electron density.
- Calculates energy dissipated per reduced field bin and SEI.
- Produces comprehensive plots with:
  - Current & voltage on dual y-axis
  - Power
  - Arc length
  - Reduced electric field
  - Electron density
  - Energy histogram with total energy and mean SEI labels

---

## Requirements

- Python 3.8+
- Packages:
  - `numpy`
  - `pandas`
  - `scipy`
  - `matplotlib`

Install with:

```bash
pip install numpy pandas scipy matplotlib
```

---

## Usage

1. **Prepare transport data**  
   Provide a CSV file (`transport_params.csv`) with transport parameters. The CSV should have columns including `E/N_Td`, mobility, ionization frequency, and attachment frequency.

2. **Define arc movement**  
   Provide a function for arc length vs time, e.g.:

```python
def linear_motion(t):
    return 0.01 + 0.005*t  # example: arc grows linearly
```

3. **Run the simulation**  

```python
from simulator import Simulator
import movements  # contains your arc_length function

sim = Simulator(
    V_source=3000.0,
    Z_source=100.0,
    file_path="transport_params.csv",
    t_start=0.0,
    t_end=1.0,
    steps=10000,
    arc_length_fun=movements.linear_motion
)

sim.run()
sim.plot()
```

4. **Inspect results**  
The simulation stores results in `sim.results`, a dictionary containing:

```python
sim.results.keys()
# ['electron_densities', 'arc_conductivity', 'arc_current', 'arc_voltage', 'reduced_fields', 'arc_lengths']
```
Each array has length equal to the number of time steps.

---

## How It Works

1. **Transport interpolation**  
   Log-log cubic interpolation of key parameters (mobility, ionization, attachment) is performed from the CSV data.

2. **Time grid**  
   An exponential time grid is generated to capture fast initial changes in the discharge.

3. **Electric field solving**  
   At each time step, `brentq` finds the electric field such that the simulated current matches the source voltage minus source impedance.

4. **Electron density update**  
   Electron density is updated iteratively using the difference between ionization and attachment frequencies.

5. **Derived quantities**  
   Current, voltage, power, arc length, reduced field, electron density, energy per bin, and SEI are computed and stored for plotting.

---

## Notes

- The code currently assumes a cylindrical arc with fixed cross-section.
- SEI (Specific Energy Input) is calculated in J/m³ and displayed in kJ/L in the plots.
- The first time step is removed in plots for clarity.
- Designed for readability and ease of modification rather than maximum performance.

---

## License

MIT License

