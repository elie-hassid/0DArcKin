import pandas as pd
import numpy as np
import scipy.constants as consts
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from movements import *
import time

class TransportInterpolator:
    def __init__(self, file_path):
        self.transport_params = pd.read_csv(file_path, sep=",", skiprows=1)
        self.transport_params.columns = [
            "R#", "E/N_Td", "A1_mean_energy_eV", "A2_mobility_N", "A6_diffusion_N",
            "A11_energy_mobility_N", "A12_energy_diffusion_N", "A13_total_collision_freq_N",
            "A14_momentum_freq_N", "A16_total_ionization_freq_N", "A17_total_attachment_freq_N",
            "A18_Townsend_alpha_N", "A19_Townsend_eta_N", "A20_power_N", "A21_elastic_power_loss_N",
            "A22_inelastic_power_loss_N", "A23_growth_power_N", "A27_max_energy",
            "A28_n_iterations", "A29_n_grid_trials"
        ]

        self.interpolators = {}

        # To speed up the simulation, only interpolate mobility, ionization and attachment.
        # Speeds up the simulation by a huge factor
        cols_to_interpolate = self.transport_params.columns[[3, 9, 10]]

        for col in cols_to_interpolate:
            x = self.transport_params["E/N_Td"].values
            y = self.transport_params[col].values
            mask = (x > 0) & (y > 0)
            x = x[mask]
            y = y[mask]
            self.interpolators[col] = interp1d(
                np.log(x), np.log(y), kind='cubic', fill_value="extrapolate"
            )

    def get_parameters(self, E_over_N):
        params = {}
        for col, interp_func in self.interpolators.items():
            params[col] = np.exp(interp_func(np.log(E_over_N)))
        return params

class Simulator:
    TOWNSEND = 1e-21

    def __init__(self, V_source, Z_source, file_path, t_start, t_end, steps, arc_length_fun, alpha=5):
        self.transport_interpolator = TransportInterpolator(file_path)
        self.densities = {"e": 1e20}  # Initial density
        self.n_0 = 101325 / (consts.Boltzmann * 3000)
        self.arc_section = np.pi*(1e-3)**2 / 4
        self.movement_fcn = arc_length_fun
        self.source_voltage = V_source
        self.source_impedance = Z_source
        self.t_start = t_start
        self.t_end = t_end
        self.steps = steps

        # Boundaries on E
        self.E_N_max_Td = 1500.0
        self.E_N_min_Td = 1.0
        self.E_min = self.E_N_min_Td * self.n_0 * Simulator.TOWNSEND
        self.E_max = self.E_N_max_Td * self.n_0 * Simulator.TOWNSEND

        self.time_grid = self.exponential_grid(t_start, t_end, steps, alpha)

        # Results
        self.electron_densities = np.zeros(steps)
        self.arc_conductivity = np.zeros(steps)
        self.arc_current = np.zeros(steps)
        self.arc_voltage = np.zeros(steps)
        self.reduced_fields = np.zeros(steps)
        self.arc_lengths = np.zeros(steps)

    def exponential_grid(self, t_start, t_end, steps, alpha):
        """Generates an exponential time grid."""
        # Initialize a linear grid
        i = np.linspace(0,1,steps)
        # Scales it to an exponential grid with density controlled by alpha
        scale = (np.exp(alpha * i) - 1) / (np.exp(alpha) - 1)
        # Stretches the grid to [t_start, t_end]
        return t_start + (t_end - t_start) * scale
    
    def interpolate_params(self, E):
        """Quicker way to returns interpolated params mu_e and delta_nu."""
        E_N = E / (self.n_0 * Simulator.TOWNSEND)
        transport_params = self.transport_interpolator.get_parameters(E_N)
        mu_e = transport_params["A2_mobility_N"] / self.n_0
        nu_ion = transport_params["A16_total_ionization_freq_N"] * self.n_0
        nu_att = transport_params["A17_total_attachment_freq_N"] * self.n_0
        delta_nu = nu_ion - nu_att
        return mu_e, delta_nu
    
    def residual(self, E, n_e_prev, dt):
        """Residue function for brentq."""
        mu_e, delta_nu = self.interpolate_params(E)
        n_e = max(n_e_prev / (1 - delta_nu * dt), 1e6)
        I = n_e * mu_e * consts.elementary_charge * E * self.arc_section
        return self.source_voltage - self.source_impedance * I - E * self.arc_length

    def solve(self, E_min, E_max, n_e_prev, dt):
        """Determine the electric field E with brentq, then compute the corresponding electronic density n_e."""
        try:
            E_converged = brentq(self.residual, E_min, E_max, args=(n_e_prev, dt), xtol=1e-9, rtol=1e-9, maxiter=1000)
        except ValueError:
            print("Brentq failed.")
            E_converged = E_min

        # Update n_e
        mu_e, delta_nu = self.interpolate_params(E_converged)
        n_e_converged = max(n_e_prev / (1 - delta_nu * dt), 1e6)
        return E_converged, n_e_converged
    
    def run(self):
        """Run the simulation."""
        for i in range(self.steps):
            t = self.time_grid[i]
            # dt = self.time_grid[self.current_index + 1] - t
            self.arc_length = max(self.movement_fcn(t), 1e-6)

            n_e_prev = self.densities["e"]

            E_converged, n_e_converged = self.solve(self.E_min, self.E_max, n_e_prev, 1e-11) # Reduce or increase this value depending on the convergence speed.
            E_N = E_converged / (self.n_0 * Simulator.TOWNSEND)

            self.densities["e"] = n_e_converged
            transport_params = self.transport_interpolator.get_parameters(E_N)
            mu_e = transport_params["A2_mobility_N"] / self.n_0
            conductivity = max(n_e_converged * mu_e * consts.elementary_charge, 1e-20)

            I = conductivity * E_converged * self.arc_section

            # Saving
            self.arc_lengths[i]        = self.arc_length
            self.electron_densities[i] = n_e_converged
            self.reduced_fields[i]     = E_N
            self.arc_conductivity[i]   = conductivity
            self.arc_current[i]        = I
            self.arc_voltage[i]        = E_converged * self.arc_length

    def axis_limits_stable(data, min_span=1e-2, margin=0.05):
        """Limit zooming in the case the variation is small."""
        data_min, data_max = np.min(data), np.max(data)
        span = data_max - data_min
        if span < min_span:
            mid = 0.5 * (data_max + data_min)
            data_min = mid - 0.5 * min_span
            data_max = mid + 0.5 * min_span
        span = data_max - data_min
        return data_min - margin*span, data_max + margin*span

    def plot(self):
        """Plots the results from the simulation."""
        time = simulator.time_grid[:len(simulator.arc_current)]

        # --- Power and time steps ---
        time = np.array(time)
        power = self.arc_voltage * self.arc_current
        dt = np.diff(time, append=time[-1])  # Last dt repeated to simplify

        # --- Energy dissipated per reduced field zone histogram. ---
        energy_per_bin = power * dt  # énergie dissipée à chaque pas
        E_min, E_max = np.min(self.reduced_fields), np.max(self.reduced_fields)
        bin_width = max(1.0, (E_max - E_min) / 20)  # largeur minimale 1 Td
        bins = np.arange(E_min, E_max + bin_width, bin_width)

        # Energy sum in each bin.
        energy_hist, bin_edges = np.histogram(self.reduced_fields, bins=bins, weights=energy_per_bin)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Current
        ax1 = axes[0,0]
        ax1.plot(time, self.arc_current, color='tab:blue')
        ax1.set_ylabel('Courant [A]')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.set_title("Courant dans l'arc")

        # Voltage
        ax2 = axes[0,1]
        ax2.plot(time, self.arc_voltage, color='tab:red')
        ax2.set_ylabel('Tension [V]')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_title("Tension dans l'arc")

        # Power
        ax3 = axes[0,2]
        ax3.plot(time, power, color='tab:orange')
        ax3.set_ylabel("Puissance [W]")
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.set_title("Puissance dissipée")

        # Reduced field
        ax4 = axes[1,0]
        ax4.plot(time, self.reduced_fields, color='tab:purple', label='E/N')
        E_mean = np.mean(self.reduced_fields)
        ax4.axhline(E_mean, color='tab:green', linestyle='--', label=f"E moyen = {E_mean:.1f} Td")
        ax4.set_xlabel('Temps [s]')
        ax4.set_ylabel("Champ réduit [Td]")
        ax4.grid(True, linestyle='--', alpha=0.5)
        ax4.set_title("Champ réduit")
        ax4.legend()

        # Electronic density
        electron_density = np.array(self.electron_densities)
        ax5 = axes[1,1]
        ax5.plot(time, electron_density, color='tab:blue')
        ax5.set_xlabel('Temps [s]')
        ax5.set_ylabel("Densité électronique [m⁻³]")
        ax5.grid(True, linestyle='--', alpha=0.5)
        ax5.set_title("Densité électronique")

        # Energy deposited in reduced field zones histogram
        ax6 = axes[1,2]
        ax6.bar(bin_centers, energy_hist, width=(bin_edges[1]-bin_edges[0]), color='tab:green', alpha=0.7)
        ax6.set_xlabel('Champ réduit E/N [Td]')
        ax6.set_ylabel('Énergie dissipée [J]')
        ax6.grid(True, linestyle='--', alpha=0.5)
        ax6.set_title('Énergie dissipée par zone de champ réduit')

        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    
    file_path = "transport_params.csv"

    V_source = 3000.0
    Z_source = 1000.0
    t_start = 0.0
    t_end = 1
    steps = 10000

    simulator = Simulator(V_source, Z_source, file_path, t_start, t_end, steps, linear_motion)
    tic = time.perf_counter()
    simulator.run()
    toc = time.perf_counter()
    print(f"Simulation took {toc - tic} s to run.")
    simulator.plot()


