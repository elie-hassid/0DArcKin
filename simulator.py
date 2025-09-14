import pandas as pd
import numpy as np
import scipy.constants as consts
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import movements
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
        self.densities = {"e": 1.5e21}  # initial density
        self.n_0 = 101325 / (consts.Boltzmann * 3000)
        self.arc_section = np.pi*(1e-3)**2 / 4
        self.movement_fcn = arc_length_fun
        self.source_voltage = V_source
        self.source_impedance = Z_source
        self.t_start = t_start
        self.t_end = t_end
        self.steps = steps

        # Thermionic emission parameters
        self.cathode_temp = 2500.0  # K
        self.cathode_phi  = 4.5     # eV
        self.Richardson_const = 1.2e6  # A/m^2/K^2

        # Boundaries on E
        self.E_N_max_Td = 1500.0
        self.E_N_min_Td = 1.0
        self.E_min = self.E_N_min_Td * self.n_0 * Simulator.TOWNSEND
        self.E_max = self.E_N_max_Td * self.n_0 * Simulator.TOWNSEND

        self.time_grid = self.exponential_grid(t_start, t_end, steps, alpha)

        # Store results
        self.electron_densities = np.zeros(steps)
        self.arc_conductivity   = np.zeros(steps)
        self.arc_current        = np.zeros(steps)
        self.arc_current_plasma = np.zeros(steps)
        self.arc_current_therm  = np.zeros(steps)
        self.arc_voltage        = np.zeros(steps)
        self.reduced_fields     = np.zeros(steps)
        self.arc_lengths        = np.zeros(steps)

    def exponential_grid(self, t_start, t_end, steps, alpha):
        i = np.linspace(0,1,steps)
        scale = (np.exp(alpha * i) - 1) / (np.exp(alpha) - 1)
        return t_start + (t_end - t_start) * scale

    def interpolate_params(self, E):
        E_N = E / (self.n_0 * Simulator.TOWNSEND)
        transport_params = self.transport_interpolator.get_parameters(E_N)
        mu_e = transport_params["A2_mobility_N"] / self.n_0
        nu_ion = transport_params["A16_total_ionization_freq_N"]
        nu_att = transport_params["A17_total_attachment_freq_N"]
        delta_nu = (nu_ion - nu_att) * self.n_0
        return mu_e, delta_nu

    def thermionic_current(self):
        J_th = self.Richardson_const * self.cathode_temp**2 \
               * np.exp(-consts.e * self.cathode_phi / (consts.Boltzmann * self.cathode_temp))
        return J_th * self.arc_section  # A

    def residual(self, E, n_e_prev, dt):
        mu_e, delta_nu = self.interpolate_params(E)
        n_th = self.thermionic_current() / (consts.elementary_charge * self.arc_section * self.arc_length)
        n_e = max((n_e_prev + n_th * dt) / (1 - delta_nu * dt), 1e6)
        I_plasma = n_e * mu_e * consts.elementary_charge * E * self.arc_section
        I_th = self.thermionic_current()
        return self.source_voltage - self.source_impedance * (I_plasma + I_th) - E * self.arc_length

    def solve(self, E_min, E_max, n_e_prev, dt):
        try:
            E_converged = brentq(self.residual, E_min, E_max, args=(n_e_prev, dt), xtol=1e-9, rtol=1e-9, maxiter=1000)
        except ValueError:
            exit("Brentq failed.")
        mu_e, delta_nu = self.interpolate_params(E_converged)
        n_th = self.thermionic_current() / (consts.elementary_charge * self.arc_section * self.arc_length)
        n_e_converged = max((n_e_prev + n_th * dt) / (1 - delta_nu * dt), 1e6)
        return E_converged, n_e_converged, mu_e

    def run(self):
        for i in range(self.steps):
            t = self.time_grid[i]
            self.arc_length = max(self.movement_fcn(t), 1e-6)
            n_e_prev = self.densities["e"]

            E_converged, n_e_converged, mu_e = self.solve(self.E_min, self.E_max, n_e_prev, 1e-11)
            E_N = E_converged / (self.n_0 * Simulator.TOWNSEND)
            self.densities["e"] = n_e_converged
            conductivity = max(n_e_converged * mu_e * consts.elementary_charge, 1e-20)
            I_plasma = conductivity * E_converged * self.arc_section
            I_th = self.thermionic_current()
            I_total = I_plasma + I_th

            self.arc_lengths[i]        = self.arc_length
            self.electron_densities[i] = n_e_converged
            self.reduced_fields[i]     = E_N
            self.arc_conductivity[i]   = conductivity
            self.arc_current[i]        = I_total
            self.arc_current_plasma[i] = I_plasma
            self.arc_current_therm[i]  = I_th
            self.arc_voltage[i]        = E_converged * self.arc_length

            if i % 200 == 0:
                print(f"t={t:.3e}s; U={E_converged*self.arc_length:.2f}V; I_tot={I_total:.2f}A; I_plasma={I_plasma:.2f}A; I_th={I_th:.2f}A; n_e={n_e_converged:.2e}m^-3")

    def plot(self):
        time = self.time_grid[1:]
        arc_current = self.arc_current[1:]          # Total current
        arc_voltage = self.arc_voltage[1:]
        reduced_fields = self.reduced_fields[1:]
        electron_densities = self.electron_densities[1:]
        arc_lengths = self.arc_lengths[1:]

        power = arc_voltage * arc_current
        dt = np.diff(time, append=time[-1])
        energy_per_bin = power * dt

        E_min, E_max = np.min(reduced_fields), np.max(reduced_fields)
        bin_width = max(1.0, (E_max - E_min) / 20)
        bins = np.arange(E_min, E_max + bin_width, bin_width)
        energy_hist, bin_edges = np.histogram(reduced_fields, bins=bins, weights=energy_per_bin)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        width = bin_edges[1] - bin_edges[0]

        volume = self.arc_section * arc_lengths
        SEI = energy_per_bin / volume
        SEI_mean = np.mean(SEI)
        total_energy = np.sum(energy_per_bin)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Total Current and Voltage
        ax1 = axes[0,0]
        line_current, = ax1.plot(time, arc_current, color='tab:blue', label='Total Current [A]')
        ax1.set_ylabel('Current [A]', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax1b = ax1.twinx()
        line_voltage, = ax1b.plot(time, arc_voltage, color='tab:red', label='Voltage [V]')
        ax1b.set_ylabel('Voltage [V]', color='tab:red')
        ax1b.tick_params(axis='y', labelcolor='tab:red')

        ax1.set_xlabel('Time [s]')
        ax1.set_title("Arc Total Current and Voltage")
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Combine legends
        lines = [line_current, line_voltage]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')

        # Power
        ax2 = axes[0,1]
        ax2.plot(time, power, color='tab:orange')
        ax2.set_ylabel("Power [W]")
        ax2.set_title("Dissipated Power")
        ax2.grid(True, linestyle='--', alpha=0.5)

        # Arc length
        ax3 = axes[0,2]
        ax3.plot(time, arc_lengths, color='tab:purple')
        ax3.set_ylabel("Arc Length [m]")
        ax3.set_title("Arc Length")
        ax3.grid(True, linestyle='--', alpha=0.5)

        # Reduced field
        ax4 = axes[1,0]
        ax4.plot(time, reduced_fields, color='tab:purple', label='E/N')
        E_mean = np.mean(reduced_fields)
        ax4.axhline(E_mean, color='tab:green', linestyle='--', label=f"Mean E = {E_mean:.1f} Td")
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel("Reduced Field [Td]")
        ax4.set_title("Reduced Electric Field")
        ax4.grid(True, linestyle='--', alpha=0.5)
        ax4.legend()

        # Electron density
        ax5 = axes[1,1]
        ax5.plot(time, electron_densities, color='tab:blue')
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel("Electron Density [m⁻³]")
        ax5.set_title("Electron Density")
        ax5.grid(True, linestyle='--', alpha=0.5)

        # Energy histogram
        ax6 = axes[1,2]
        ax6.bar(bin_centers, energy_hist, width=width, color='tab:green', alpha=0.7)
        ax6.set_xlabel('Reduced Field E/N [Td]')
        ax6.set_ylabel('Dissipated Energy [J]')
        ax6.set_title('Energy Dissipated per Reduced Field Bin')
        ax6.grid(True, linestyle='--', alpha=0.5)
        ax6.text(0.95, 0.95,
                f"Total Energy = {(total_energy/1e3):.2e} kJ\nMean SEI = {(SEI_mean / 1e6):.2e} kJ/L",
                transform=ax6.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

        fig.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    file_path = "transport_params.csv"
    V_source = 3000.0
    Z_source = 100.0
    t_start = 0.0
    t_end = 1
    steps = 100000

    simulator = Simulator(V_source, Z_source, file_path, t_start, t_end, steps, movements.sinusoidal_motion)
    tic = time.perf_counter()
    simulator.run()
    toc = time.perf_counter()
    print(f"Simulation took {toc - tic:.2f} s.")
    simulator.plot()
