import pandas as pd
import numpy as np
import scipy.constants as consts
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import movements
import time
import warnings
from typing import Callable, Dict, Tuple

class TransportInterpolator:
    def __init__(self, file_path: str):
        """Load transport parameters and create log-log interpolators for selected columns."""
        self.transport_params = pd.read_csv(file_path, sep=",", skiprows=1)
        self.transport_params.columns = [
            "R#", "E/N_Td", "A1_mean_energy_eV", "A2_mobility_N", "A6_diffusion_N",
            "A11_energy_mobility_N", "A12_energy_diffusion_N", "A13_total_collision_freq_N",
            "A14_momentum_freq_N", "A16_total_ionization_freq_N", "A17_total_attachment_freq_N",
            "A18_Townsend_alpha_N", "A19_Townsend_eta_N", "A20_power_N", "A21_elastic_power_loss_N",
            "A22_inelastic_power_loss_N", "A23_growth_power_N", "A27_max_energy",
            "A28_n_iterations", "A29_n_grid_trials"
        ]

        self.interpolators: Dict[str, interp1d] = {}
        cols_to_interpolate = ["A2_mobility_N", "A16_total_ionization_freq_N", "A17_total_attachment_freq_N"]

        for col in cols_to_interpolate:
            x = self.transport_params["E/N_Td"].values
            y = self.transport_params[col].values
            mask = (x > 0) & (y > 0)
            if np.sum(mask) < len(x)//2:
                warnings.warn(f"More than half of the points for {col} are non-positive and ignored.")
            x, y = x[mask], y[mask]
            self.interpolators[col] = interp1d(np.log(x), np.log(y), kind='cubic', fill_value="extrapolate")

    def get_parameters(self, E_over_N: float) -> Dict[str, float]:
        """Return interpolated transport parameters at given reduced field E/N."""
        return {col: np.exp(interp(np.log(E_over_N))) for col, interp in self.interpolators.items()}


class Simulator:
    TOWNSEND = 1e-21

    def __init__(
        self, V_source: float, Z_source: float, file_path: str,
        t_start: float, t_end: float, steps: int,
        arc_length_fun: Callable[[float], float], alpha: float = 5
    ):
        self.transport_interpolator = TransportInterpolator(file_path)
        self.densities = {"e": 1.66e21}  # initial electron density
        self.n_0 = 101325 / (consts.Boltzmann * 3000)
        self.arc_section = np.pi * (1e-3)**2 / 4
        self.movement_fcn = arc_length_fun
        self.source_voltage = V_source
        self.source_impedance = Z_source
        self.t_start, self.t_end, self.steps = t_start, t_end, steps
        self.alpha = alpha

        # E field boundaries
        self.E_N_min_Td = 1.0
        self.E_N_max_Td = 1500.0
        self.E_min = self.E_N_min_Td * self.n_0 * Simulator.TOWNSEND
        self.E_max = self.E_N_max_Td * self.n_0 * Simulator.TOWNSEND

        # Time grid
        self.time_grid = self.exponential_grid(t_start, t_end, steps, alpha)

        # Results dictionary
        self.results = {key: np.zeros(steps) for key in [
            "electron_densities", "arc_conductivity", "arc_current", "arc_voltage",
            "reduced_fields", "arc_lengths"
        ]}

    def exponential_grid(self, t_start: float, t_end: float, steps: int, alpha: float) -> np.ndarray:
        """Generate an exponential time grid."""
        i = np.linspace(0, 1, steps)
        scale = (np.exp(alpha * i) - 1) / (np.exp(alpha) - 1)
        return t_start + (t_end - t_start) * scale

    def interpolate_params(self, E: float) -> Tuple[float, float]:
        """Return interpolated electron mobility and effective collision frequency difference."""
        E_N = E / (self.n_0 * Simulator.TOWNSEND)
        params = self.transport_interpolator.get_parameters(E_N)
        mu_e = params["A2_mobility_N"] / self.n_0
        delta_nu = (params["A16_total_ionization_freq_N"] - params["A17_total_attachment_freq_N"]) * self.n_0
        return mu_e, delta_nu

    def residual(self, E: float, n_e_prev: float, dt: float) -> float:
        mu_e, delta_nu = self.interpolate_params(E)
        n_e = max(n_e_prev / (1 - delta_nu * dt), 1e6)
        I = n_e * mu_e * consts.elementary_charge * E * self.arc_section
        return self.source_voltage - self.source_impedance * I - E * self.arc_length

    def solve(self, E_min: float, E_max: float, n_e_prev: float, dt: float) -> Tuple[float, float, float]:
        """Solve for E and n_e using brentq."""
        try:
            E_converged = brentq(self.residual, E_min, E_max, args=(n_e_prev, dt), xtol=1e-9, rtol=1e-9, maxiter=1000)
        except ValueError:
            raise RuntimeError("Brentq failed to converge.")
        mu_e, delta_nu = self.interpolate_params(E_converged)
        n_e_converged = max(n_e_prev / (1 - delta_nu * dt), 1e6)
        return E_converged, n_e_converged, mu_e

    def run(self):
        """Run the simulation."""
        for i, t in enumerate(self.time_grid):
            self.arc_length = max(self.movement_fcn(t), 1e-6)
            n_e_prev = self.densities["e"]

            E, n_e, mu_e = self.solve(self.E_min, self.E_max, n_e_prev, 1e-11)
            E_N = E / (self.n_0 * Simulator.TOWNSEND)
            self.densities["e"] = n_e
            conductivity = max(n_e * mu_e * consts.elementary_charge, 1e-20)
            I = conductivity * E * self.arc_section

            # Store results
            self.results["arc_lengths"][i] = self.arc_length
            self.results["electron_densities"][i] = n_e
            self.results["reduced_fields"][i] = E_N
            self.results["arc_conductivity"][i] = conductivity
            self.results["arc_current"][i] = I
            self.results["arc_voltage"][i] = E * self.arc_length

            if i % 200 == 0:
                print(f"t={t:.3e}s; U={E*self.arc_length:.2f}V; I={I:.2f}A; n_e={n_e:.2e}m^-3")

    # --- Helper for plotting ---
    @staticmethod
    def plot_line(ax, x, y, color: str, ylabel: str, xlabel: str, title: str):
        ax.plot(x, y, color=color)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, linestyle='--', alpha=0.5)

    def plot(self):
        """Plot simulation results."""
        time = self.time_grid[1:]
        # Remove first point for all results
        for key in self.results:
            self.results[key] = self.results[key][1:]

        power = self.results["arc_voltage"] * self.results["arc_current"]
        dt = np.diff(time, append=time[-1])
        energy_per_bin = power * dt

        # Energy histogram
        E_min, E_max = np.min(self.results["reduced_fields"]), np.max(self.results["reduced_fields"])
        bin_width = max(1.0, (E_max - E_min) / 20)
        bins = np.arange(E_min, E_max + bin_width, bin_width)
        energy_hist, bin_edges = np.histogram(self.results["reduced_fields"], bins=bins, weights=energy_per_bin)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        width = bin_edges[1] - bin_edges[0]

        # SEI
        volume = self.arc_section * self.results["arc_lengths"]
        SEI = energy_per_bin / volume
        SEI_mean = np.mean(SEI)
        total_energy = np.sum(energy_per_bin)

        # Subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Current and voltage
        ax1 = axes[0,0]
        line1, = ax1.plot(time, self.results["arc_current"], color='tab:blue', label='Current [A]')
        ax1.set_ylabel('Current [A]', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax1b = ax1.twinx()
        line2, = ax1b.plot(time, self.results["arc_voltage"], color='tab:red', label='Voltage [V]')
        ax1b.set_ylabel('Voltage [V]', color='tab:red')
        ax1b.tick_params(axis='y', labelcolor='tab:red')

        ax1.set_xlabel('Time [s]')
        ax1.set_title("Arc Current and Voltage")
        ax1.grid(True, linestyle='--', alpha=0.5)
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')

        # Other plots
        self.plot_line(axes[0,1], time, power, 'tab:orange', 'Power [W]', 'Time [s]', 'Dissipated Power')
        self.plot_line(axes[0,2], time, self.results["arc_lengths"], 'tab:purple', 'Arc Length [m]', 'Time [s]', 'Arc Length')
        self.plot_line(axes[1,0], time, self.results["reduced_fields"], 'tab:purple', 'Reduced Field [Td]', 'Time [s]', 'Reduced Electric Field')
        axes[1,0].axhline(np.mean(self.results["reduced_fields"]), color='tab:green', linestyle='--', label=f"Mean E = {np.mean(self.results['reduced_fields']):.1f} Td")
        axes[1,0].legend()

        self.plot_line(axes[1,1], time, self.results["electron_densities"], 'tab:blue', 'Electron Density [m^-3]', 'Time [s]', 'Electron Density')
        ax6 = axes[1,2]
        ax6.bar(bin_centers, energy_hist, width=width, color='tab:green', alpha=0.7)
        ax6.set_xlabel('Reduced Field E/N [Td]')
        ax6.set_ylabel('Dissipated Energy [J]')
        ax6.set_title('Energy Dissipated per Reduced Field Bin')
        ax6.grid(True, linestyle='--', alpha=0.5)
        ax6.text(0.95, 0.95,
                 f"Total Energy = {total_energy/1e3:.2e} kJ\nMean SEI = {SEI_mean/1e6:.2e} kJ/L",
                 transform=ax6.transAxes, ha='right', va='top', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    file_path = "transport_params.csv"
    simulator = Simulator(3000.0, 100.0, file_path, 0.0, 1.0, 100000, movements.linear_motion)
    tic = time.perf_counter()
    simulator.run()
    toc = time.perf_counter()
    print(f"Simulation took {toc - tic:.2f} s to run.")
    simulator.plot()
