# Documentation - Electron Transport and Arc Simulation

This project implements a simulation of arc dynamics based on interpolated electron transport parameters from external data.  

The simulation includes:
- Interpolation of transport parameters as a function of reduced electric field \(E/N\).
- Resolution of the arc balance equation using Brent’s method (`scipy.optimize.brentq`).
- Time evolution of electron density, current, voltage, and power.
- Several motion laws for arc length.
- Visualization of results with `matplotlib`.

---

## Dependencies

```bash
pip install numpy pandas matplotlib scipy
```

---

## Main Classes

### `TransportInterpolator`

Class for loading and interpolating transport parameters.

#### Attributes
- `transport_params` : DataFrame containing transport parameters.
- `interpolators` : dictionary of log-log interpolation functions for each parameter.

#### Methods
- `__init__(file_path)`  
  Loads a CSV file with transport parameters.
  
- `get_parameters(E_over_N)`  
  Returns a dictionary of interpolated parameters for a given reduced electric field \(E/N\).

---

### `Simulator`

Class for time-domain simulation of an electric arc.

#### Key Attributes
- `densities` : dictionary of densities (initialized with electrons `e`).
- `n_0` : reference density (function of temperature and pressure).
- `arc_section` : arc cross-sectional area in m².
- `time_grid` : non-linear (exponentially spaced) time grid.
- Stored results:
  - `electron_densities`
  - `arc_conductivity`
  - `arc_current`
  - `arc_voltage`
  - `reduced_fields`
  - `arc_lengths`

#### Methods
- `__init__(V_source, Z_source, file_path, t_start, t_end, steps, arc_length_fun, alpha=5)`  
  Initializes the simulation with source parameters, time span, and arc motion function.

- `__iter__()`  
  Allows the object to be used in a `for` loop.

- `__next__()`  
  Performs one simulation step, updates physical quantities, and stores results.

- `residual(E, n_e_prev, dt)`  
  Residual function for Brent’s solver.  

- `solve(E_min, E_max, n_e_prev, dt)`  
  Solves the electric field with Brent and updates the electron density.

---

## Arc Motion Functions

Several arc length laws are available (all return position in meters):

- `linear_motion(t, total_time=1, total_distance=0.1)`  
  Linear growth.
  
- `quadratic_motion(t, total_time=1, total_distance=0.1)`  
  Quadratic growth.
  
- `log_motion(t, total_time=1, total_distance=0.1)`  
  Logarithmic growth (fast at the beginning).
  
- `exp_motion(t, total_time=1, total_distance=0.1)`  
  Exponential growth (slow then fast).
  
- `sinusoidal_motion(t, total_time=1, min_pos=0.001, max_pos=0.01)`  
  Sinusoidal motion between two bounds.

---

## How to Use

1. Export transport data from BOLSIG+

 - Open BOLSIG+ and set up your gas mixture and conditions.

 - Export the tabulated electron transport parameters as a CSV file (transport_params.csv), ensuring it contains all necessary columns (e.g., mean energy, mobility, ionization frequencies, etc.).

2. Initialize the Simulator

 - Import your desired arc motion function (linear_motion, quadratic_motion, etc.).

 - Create a Simulator instance using the source voltage, source impedance, CSV file path, time span, number of steps, and arc motion function.

3. Run the Simulation

 - Use a for loop to iterate over the simulator object.

 - Extract time, reduced field, electron density, current, and voltage from the simulator.

4. Visualize Results

 - Use the built-in plotting routines or customize your own to analyze current, voltage, power, reduced field, electron density, and energy distribution.

---

## Example Usage

```python
file_path = "transport_params.csv"
V_source = 3000.0
Z_source = 1000.0
t_start, t_end, steps = 0.0, 2.0, 10000

# Simulation with linear arc motion
simulator = Simulator(V_source, Z_source, file_path, t_start, t_end, steps, linear_motion)

for _ in simulator:
    pass

# Extract results
time = simulator.time_grid[:len(simulator.arc_current)]
E_over_N = np.array(simulator.reduced_fields)
electron_density = np.array(simulator.electron_densities)
arc_current = np.array(simulator.arc_current)
arc_voltage = np.array(simulator.arc_voltage)
```

---

## Visualization

The script automatically generates a 2x3 grid of plots:

1. **Arc Current**
2. **Arc Voltage**
3. **Dissipated Power**
4. **Reduced Field (E/N)**
5. **Electron Density**
6. **Histogram of dissipated energy vs reduced field**

---

## Notes
- The file `transport_params.csv` must contain the expected columns (transport parameters).  
- The simulation automatically stops if the arc extinguishes (based on voltage and current criteria).  
- The parameter `alpha` controls the time grid spacing (more or less concentrated at the beginning).  

---

## Possible Improvements
- Implement additional arc motion models.  
- Add options to save results in CSV or HDF5 format.  
- Enable comparison across different motion laws.  

---
