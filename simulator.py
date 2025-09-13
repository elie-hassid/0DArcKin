import pandas as pd
import numpy as np
import scipy.constants as consts
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from movements import *

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
        cols_to_interpolate = self.transport_params.columns[2:]

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
        self.t_start = t_start
        self.t_end = t_end
        self.steps = steps
        self.alpha = alpha
        self.movement_fcn = arc_length_fun
        self.source_voltage = V_source
        self.source_impedance = Z_source

        i = np.arange(steps)
        self.time_grid = t_start + (t_end - t_start) * (np.exp(alpha * i / (steps-1)) - 1) / (np.exp(alpha) - 1)
        self.current_index = 0

        # Results
        self.electron_densities = []
        self.arc_conductivity = []
        self.arc_current = []
        self.arc_voltage = []
        self.reduced_fields = []
        self.arc_lengths = []

    def __iter__(self):
        return self

    def residual(self, E, n_e_prev, dt):
        """Residue function for brentq."""
        E_N = E / (self.n_0 * Simulator.TOWNSEND)
        transport_params = self.transport_interpolator.get_parameters(E_N)
        mu_e = transport_params["A2_mobility_N"] / self.n_0
        nu_ion = transport_params["A16_total_ionization_freq_N"] * self.n_0
        nu_att = transport_params["A17_total_attachment_freq_N"] * self.n_0
        delta_nu = nu_ion - nu_att

        n_e = max(n_e_prev / (1 - delta_nu * dt), 1e6)

        # courant
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
        E_N = E_converged / (self.n_0 * Simulator.TOWNSEND)
        transport_params = self.transport_interpolator.get_parameters(E_N)
        nu_ion = transport_params["A16_total_ionization_freq_N"] * self.n_0
        nu_att = transport_params["A17_total_attachment_freq_N"] * self.n_0
        delta_nu = nu_ion - nu_att
        n_e_converged = max(n_e_prev / (1 - delta_nu * dt), 1e6)

        return E_converged, n_e_converged

    def __next__(self):
        if self.current_index >= self.steps - 1:
            raise StopIteration

        t = self.time_grid[self.current_index]
        dt = self.time_grid[self.current_index + 1] - t
        self.arc_length = max(self.movement_fcn(t), 1e-6)
        self.arc_lengths.append(self.arc_length)

        n_e_prev = self.densities["e"]

        # Boundaries on E
        E_N_max_Td = 1500.0
        E_N_min_Td = 1.0
        E_min = E_N_min_Td * self.n_0 * Simulator.TOWNSEND
        E_max = E_N_max_Td * self.n_0 * Simulator.TOWNSEND

        E_converged, n_e_converged = self.solve(E_min, E_max, n_e_prev, 1e-11) # Reduce or increase this value depending on the convergence speed.
        E_N = E_converged / (self.n_0 * Simulator.TOWNSEND)

        # Saving
        self.densities["e"] = n_e_converged
        self.electron_densities.append(n_e_converged)
        self.reduced_fields.append(E_N)

        transport_params = self.transport_interpolator.get_parameters(E_N)
        mu_e = transport_params["A2_mobility_N"] / self.n_0
        conductivity = max(n_e_converged * mu_e * consts.elementary_charge, 1e-20)
        self.arc_conductivity.append(conductivity)

        I = conductivity * E_converged * self.arc_section
        self.arc_current.append(I)
        self.arc_voltage.append(E_converged * self.arc_length)

        print(f"Time {t:.6e}s: U = {E_converged * self.arc_length:.3e} V, "
              f"I = {I:.3e} A, n_e = {n_e_converged:.4e} m^-3")

        # Critère d'extinction
        if (self.source_voltage - E_converged * self.arc_length)/self.source_voltage < 1e-3 and I < 1:
            print("Arc extinguishing before the end of the simulation.")
            raise StopIteration

        self.current_index += 1
        return t


if __name__ == "__main__":
    
    file_path = "transport_params.csv"

    V_source = 3000.0
    Z_source = 1000.0
    t_start = 0.0
    t_end = 2
    steps = 10000

    simulator = Simulator(V_source, Z_source, file_path, t_start, t_end, steps, linear_motion)

    for _ in simulator:
        pass

    time = simulator.time_grid[:len(simulator.arc_current)]

    E_over_N = np.array(simulator.reduced_fields)
    electron_density = np.array(simulator.electron_densities)
    arc_current = np.array(simulator.arc_current)
    arc_voltage = np.array(simulator.arc_voltage)

    # --- Limit zooming in the case the variation is small. ---
    def axis_limits_stable(data, min_span=1e-2, margin=0.05):
        data_min, data_max = np.min(data), np.max(data)
        span = data_max - data_min
        if span < min_span:
            mid = 0.5 * (data_max + data_min)
            data_min = mid - 0.5 * min_span
            data_max = mid + 0.5 * min_span
        span = data_max - data_min
        return data_min - margin*span, data_max + margin*span

    # --- Power and time steps ---
    arc_voltage = np.array(arc_voltage)
    arc_current = np.array(arc_current)
    time = np.array(time)
    power = arc_voltage * arc_current
    dt = np.diff(time, append=time[-1])  # Last dt repeated to simplify

    # --- Energy dissipated per reduced field zone histogram. ---
    E_over_N = np.array(E_over_N)
    energy_per_bin = power * dt  # énergie dissipée à chaque pas
    E_min, E_max = np.min(E_over_N), np.max(E_over_N)
    bin_width = max(1.0, (E_max - E_min) / 20)  # largeur minimale 1 Td
    bins = np.arange(E_min, E_max + bin_width, bin_width)

    # Energy sum in each bin.
    energy_hist, bin_edges = np.histogram(E_over_N, bins=bins, weights=energy_per_bin)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Current
    ax1 = axes[0,0]
    ax1.plot(time, arc_current, color='tab:blue')
    ax1.set_ylabel('Courant [A]')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title("Courant dans l'arc")

    # Voltage
    ax2 = axes[0,1]
    ax2.plot(time, arc_voltage, color='tab:red')
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
    ax4.plot(time, E_over_N, color='tab:purple', label='E/N')
    E_mean = np.mean(E_over_N)
    ax4.axhline(E_mean, color='tab:green', linestyle='--', label=f"E moyen = {E_mean:.1f} Td")
    ax4.set_xlabel('Temps [s]')
    ax4.set_ylabel("Champ réduit [Td]")
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.set_title("Champ réduit")
    ax4.legend()

    # Electronic density
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


