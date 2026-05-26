import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
import io

class AsteroidDensityAnalyzer:
    """
    Generalized pipeline for spectral unmixing using external RELAB databases.
    """
    def __init__(self, bulk_density, sigma_bulk_density, 
                 incidence_deg=30, emission_deg=0, phase_deg=30,
                 B0=0.67, h=0.066, b=0.29, c=0.45):
        self.rho_bulk = bulk_density
        self.sigma_rho_bulk = sigma_bulk_density
        self.results = {}

        self.i, self.e, self.g = np.radians(incidence_deg), np.radians(emission_deg), np.radians(phase_deg)
        self.mu0, self.mu = np.cos(self.i), np.cos(self.e)
        self.B0, self.h, self.b, self.c = B0, h, b, c

    def _phase_function(self, g):
        P_f = (1 - self.b**2) / (1 + self.b**2 + 2 * self.b * np.cos(g))**(1.5)
        P_b = (1 - self.b**2) / (1 + self.b**2 - 2 * self.b * np.cos(g))**(1.5)
        return self.c * P_f + (1 - self.c) * P_b

    def _opposition_effect(self, g):
        if g == 0: return 1 + self.B0
        return 1 + (self.B0 / (1 + (1 / self.h) * np.tan(g / 2)))

    def _h_function(self, x, w):
        gamma = np.sqrt(1 - w)
        r0 = (1 - gamma) / (1 + gamma)
        if x == 0: x = 1e-10 
        term = r0 + (1 - 0.5 * r0 - r0 * x) * np.log((1 + x) / x)
        return 1.0 / (1.0 - (1.0 - gamma) * x * term)

    def hapke_forward_model(self, w):
        P_g = self._phase_function(self.g)
        B_g = self._opposition_effect(self.g)
        multiple_scattering = (self._h_function(self.mu0, w) * self._h_function(self.mu, w)) - 1.0
        return (w / 4.0) * (self.mu0 / (self.mu0 + self.mu)) * (P_g * B_g + multiple_scattering)

    def reflectance_to_albedo(self, R_array):
        w_array = np.zeros_like(R_array)
        for idx, R_obs in np.ndenumerate(R_array):
            res = minimize_scalar(lambda w: (self.hapke_forward_model(w) - R_obs)**2, 
                                  bounds=(1e-4, 0.9999), method='bounded')
            w_array[idx] = res.x
        return w_array

    def unmix_spectra(self, target_R, library_R):
        target_w = self.reflectance_to_albedo(target_R)
        library_w = np.array([self.reflectance_to_albedo(R) for R in library_R])
        
        num_endmembers = library_w.shape[0]
        
        constraints = {'type': 'eq', 'fun': lambda f: np.sum(f) - 1.0}
        bounds = [(0, 1) for _ in range(num_endmembers)]
        initial_guess = np.ones(num_endmembers) / num_endmembers
        
        # Optimize
        result = minimize(lambda f: np.sum((target_w - np.dot(f, library_w))**2), 
                          initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success: raise ValueError(f"Optimization failed: {result.message}")
        f_i = result.x
        
        # RMSE
        w_model = np.dot(f_i, library_w)
        R_model = np.array([self.hapke_forward_model(w) for w in w_model])
        rmse = np.sqrt(np.mean((target_R - R_model)**2))
        
        # ERROR FIX: Uncertainty proportional to abundance to prevent phantom error propagation
        # Adds a baseline 1% error floor, scaled by the fraction present + global RMSE
        sigma_f_i = (f_i * rmse) + 0.01 
        
        self.results.update({'mass_fractions': f_i, 'sigma_mass_fractions': sigma_f_i, 'rmse': rmse})
        return f_i, sigma_f_i

    def calculate_densities(self, f_i, sigma_f_i, rho_i, sigma_rho_i):
        # Filter out minerals with exactly 0% fraction to prevent divide-by-zero or math errors
        active_indices = f_i > 1e-4
        f_act, sig_f_act = f_i[active_indices], sigma_f_i[active_indices]
        rho_act, sig_rho_act = rho_i[active_indices], sigma_rho_i[active_indices]

        # Re-normalize active fractions just in case
        f_act = f_act / np.sum(f_act)

        Y = np.sum(f_act / rho_act)
        rho_grain = 1.0 / Y
        
        sigma_Y = np.sqrt(np.sum(((1.0 / rho_act) * sig_f_act)**2 + ((-f_act / (rho_act**2)) * sig_rho_act)**2))
        sigma_rho_grain = sigma_Y / (Y**2)
        
        delta_rho = self.rho_bulk - rho_grain
        self.results.update({
            'rho_grain': rho_grain,
            'sigma_rho_grain': sigma_rho_grain,
            'delta_rho': delta_rho,
            'sigma_delta_rho': np.sqrt(self.sigma_rho_bulk**2 + sigma_rho_grain**2),
            'porosity': 1 - (self.rho_bulk / rho_grain)
        })
        return self.results

    def load_and_run(self, target_R, relab_csv_path):
        """Loads RELAB data from a CSV, separates metadata from spectra, and runs the pipeline."""
        df = pd.read_csv(relab_csv_path)
        
        mineral_names = df['Mineral_Name'].values
        rho_i = df['Density_gcm3'].values
        sigma_rho_i = df['Density_Uncertainty'].values
        
        # All columns starting from the 4th column are assumed to be spectral wavelengths
        library_R = df.iloc[:, 3:].values
        
        print(f"Loaded {len(mineral_names)} endmembers from RELAB database.")
        
        # Execute pipeline
        f_i, sigma_f_i = self.unmix_spectra(target_R, library_R)
        self.calculate_densities(f_i, sigma_f_i, rho_i, sigma_rho_i)
        
        # Print top contributors
        print("\n--- TOP MINERAL CONTRIBUTORS ---")
        for idx in np.argsort(f_i)[::-1]:
            if f_i[idx] > 0.01: # Only print minerals > 1% abundance
                print(f"{mineral_names[idx]}: {f_i[idx]*100:.1f}%")

# ==========================================
# USAGE DEMONSTRATION WITH EXTERNAL FILE
# ==========================================
if __name__ == "__main__":
    # We simulate a CSV file here for the demonstration so the code is self-contained.
    # In reality, this would just be: analyzer.load_and_run(target_R, "relab_database.csv")
    csv_content = """Mineral_Name,Density_gcm3,Density_Uncertainty,Wv_0.7,Wv_1.0,Wv_1.2,Wv_1.5,Wv_2.0
Orthopyroxene_RELAB,3.40,0.02,0.28,0.12,0.22,0.26,0.18
Clinopyroxene_RELAB,3.30,0.02,0.25,0.15,0.20,0.24,0.14
Anorthite_RELAB,2.70,0.02,0.45,0.44,0.42,0.40,0.38
Forsterite_RELAB,3.30,0.02,0.30,0.18,0.25,0.28,0.29
Metallic_Iron_RELAB,7.87,0.01,0.15,0.16,0.17,0.18,0.19"""
    
    mock_csv = io.StringIO(csv_content)
    
    vesta_analyzer = AsteroidDensityAnalyzer(bulk_density=3.456, sigma_bulk_density=0.035)
    vesta_target_R = np.array([0.285, 0.165, 0.245, 0.270, 0.190])
    
    vesta_analyzer.load_and_run(vesta_target_R, mock_csv)
    
    # Print the physical results
    print(f"\nFinal Crustal Density: {vesta_analyzer.results['rho_grain']:.3f} g/cm^3")
    print(f"Final Porosity: {vesta_analyzer.results['porosity']:.4f}")