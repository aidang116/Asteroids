import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize, minimize_scalar

# Initialize the module-level logger used across the analysis pipeline.
logger = logging.getLogger("AsteroidDensityAnalyzer")

class AsteroidDensityAnalyzer:
    """Pipeline for asteroid spectral unmixing and grain-density estimation.

    This class converts reflectance spectra into Hapke single-scattering
    albedo, performs constrained spectral unmixing against a mineral library,
    computes a harmonic mass-balance grain density, and propagates uncertainty
    through the entire inversion chain using Tikhonov-stabilized covariance mapping.
    """

    def __init__(self, bulk_density, sigma_bulk_density,
                 incidence_deg=30, emission_deg=0, phase_deg=30,
                 B0=0.67, h=0.066, b=0.29, c=0.45, lambda_damping=1e-4):
        # Bulk density inputs are stored for completeness and future diagnostics.
        self.rho_bulk = bulk_density
        self.sigma_rho_bulk = sigma_bulk_density
        self.results = {}

        # Convert photometric geometry from degrees to radians.
        self.i = np.radians(incidence_deg)
        self.e = np.radians(emission_deg)
        self.g = np.radians(phase_deg)
        self.mu0 = np.cos(self.i)
        self.mu = np.cos(self.e)

        # Hapke model parameters for opposition and phase behavior.
        self.B0 = B0
        self.h = h
        self.b = b
        self.c = c
        
        # Regularization hyperparameter to penalize collinear covariance explosion.
        self.lambda_damping = lambda_damping

    def _phase_function(self, g):
        """Compute Hapke bidirectional phase function for a given phase angle."""
        P_f = (1 - self.b**2) / (1 + self.b**2 + 2 * self.b * np.cos(g))**1.5
        P_b = (1 - self.b**2) / (1 + self.b**2 - 2 * self.b * np.cos(g))**1.5
        return self.c * P_f + (1 - self.c) * P_b

    def _opposition_effect(self, g):
        """Compute the Hapke opposition surge factor."""
        if g == 0:
            return 1 + self.B0
        return 1 + (self.B0 / (1 + (1 / self.h) * np.tan(g / 2)))

    def _h_function(self, x, w):
        """Compute the Hapke multiple-scattering term H(x, w)."""
        gamma = np.sqrt(np.clip(1 - w, 0, 1))
        r0 = (1 - gamma) / (1 + gamma)
        if x <= 0:
            x = 1e-10
        term = r0 + (1 - 0.5 * r0 - r0 * x) * np.log((1 + x) / x)
        return 1.0 / (1.0 - (1.0 - gamma) * x * term)

    def hapke_forward_model(self, w):
        """Forward Hapke model: single-scattering albedo -> reflectance."""
        P_g = self._phase_function(self.g)
        B_g = self._opposition_effect(self.g)
        multiple_scattering = (self._h_function(self.mu0, w) * self._h_function(self.mu, w)) - 1.0
        return (w / 4.0) * (self.mu0 / (self.mu0 + self.mu)) * (P_g * B_g + multiple_scattering)

    def reflectance_to_albedo(self, R_array):
        """Convert a reflectance vector into Hapke single-scattering albedo values.

        Each observed reflectance point is inverted using bounded scalar
        minimization so that the forward model best matches the measurement.
        """
        w_array = np.zeros_like(R_array)
        for idx, R_obs in np.ndenumerate(R_array):
            res = minimize_scalar(
                lambda w: (self.hapke_forward_model(w) - R_obs) ** 2,
                bounds=(1e-4, 0.9999),
                method='bounded'
            )
            w_array[idx] = res.x
        return w_array

    def route_spectrum(self, target_R, name):
        """Choose the best endmember library based on target spectrum shape."""
        max_reflectance = np.max(target_R)

        # Estimate the local continuum at 1.0 µm from the 0.7 and 1.2 µm points.
        continuation_slope = (1.0 - 0.7) / (1.2 - 0.7)
        continuum_1_0 = target_R[0] + continuation_slope * (target_R[2] - target_R[0])
        band_depth_1_0 = 1.0 - (target_R[1] / continuum_1_0)

        logger.info(
            f"[{name}] Routing evaluation -> Max Reflectance: {max_reflectance:.4f}, "
            f"1.0um Band Depth: {band_depth_1_0:.4f}"
        )

        if max_reflectance < 0.12 or (max_reflectance < 0.15 and band_depth_1_0 < 0.05):
            selected = "primitive_aqueous_library.csv"
        else:
            selected = "stony_metallic_library.csv"

        logger.info(f"[{name}] Logic gate assigned library file: {selected}")
        return selected

    def unmix_and_propagate(self, target_R, library_R, mineral_names, rho_i, sigma_rho_i, name):
        """Perform spectral unmixing and propagate density uncertainty through the fit."""
        logger.info(f"[{name}] Starting Hapke inversion. Converting target reflectance vector to single-scattering albedo (w)...")
        target_w = self.reflectance_to_albedo(target_R)
        logger.info(f"[{name}] Target albedo (w) vector: {np.round(target_w, 4)}")

        logger.info(f"[{name}] Converting library database reflectance curves to single-scattering albedo (w)...")
        library_w = np.array([self.reflectance_to_albedo(R) for R in library_R])

        num_endmembers = library_w.shape[0]
        constraints = {'type': 'eq', 'fun': lambda f: np.sum(f) - 1.0}
        bounds = [(0, 1) for _ in range(num_endmembers)]
        initial_guess = np.ones(num_endmembers) / num_endmembers

        logger.info(f"[{name}] Initializing SLSQP optimization bounded unmixing (Sum-to-One constrained)...")
        result = minimize(
            lambda f: np.sum((target_w - np.dot(f, library_w)) ** 2),
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        f_i = result.x

        # Keep only the endmembers with meaningful fractional contributions.
        active = f_i > 1e-4
        M_act = library_w[active, :]
        f_act = f_i[active]
        rho_act = rho_i[active]
        sig_rho_act = sigma_rho_i[active]
        act_names = mineral_names[active]

        logger.info(f"[{name}] Inversion Complete. Active minerals detected: {len(act_names)} out of {num_endmembers}")
        for n, f in zip(act_names, f_act):
            logger.info(f"[{name}]    - Contributor: {n} -> Fraction: {f*100:.2f}%")

        # Compute the albedo-space model residuals and estimate fit variance.
        w_model = np.dot(f_act, M_act)
        residuals = target_w - w_model
        dof = max(1, M_act.shape[1] - M_act.shape[0])
        residual_variance = np.sum(residuals**2) / dof
        logger.info(
            f"[{name}] Spectral Fit Performance -> Albedo-space Residual Variance "
            f"(s^2): {residual_variance:.6e} (DOF={dof})"
        )

        # --- IMPLEMENTED TIKHONOV REGULARIZATION ---
        # Compute the covariance of the active fractions via a stabilized pseudo-inverse.
        logger.info(f"[{name}] Mapping fractional parameter covariance using Tikhonov-stabilized Moore-Penrose pseudo-inverse...")
        design_matrix = np.dot(M_act, M_act.T)
        damping_matrix = self.lambda_damping * np.eye(design_matrix.shape[0])
        
        C_f_act = residual_variance * np.linalg.pinv(design_matrix + damping_matrix)
        if C_f_act.ndim == 1:
            C_f_act = np.atleast_2d(C_f_act)

        # Use harmonic averaging to compute grain density from mass fractions.
        Y = np.sum(f_act / rho_act)
        rho_grain = 1.0 / Y
        logger.info(f"[{name}] Harmonic mass-balance computed raw Grain Density: {rho_grain:.4f} g/cm^3")

        # Propagate uncertainty from both spectral fractions and library densities.
        deriv_f = - (rho_grain ** 2) / rho_act
        deriv_rho = (rho_grain ** 2) * f_act / (rho_act ** 2)

        var_f_part = np.dot(deriv_f.T, np.dot(C_f_act, deriv_f))
        var_rho_part = np.sum((deriv_rho * sig_rho_act) ** 2)

        logger.info(f"[{name}] Uncertainty Breakdown -> Covariant Spectral Variance Fragment: {var_f_part:.6e}")
        logger.info(f"[{name}] Uncertainty Breakdown -> Independent Database Density Variance Fragment: {var_rho_part:.6e}")

        sigma_rho_grain = np.sqrt(max(1e-6, var_f_part + var_rho_part))
        logger.info(f"[{name}] Total Combined Analytical Uncertainty (Quadrature): +/- {sigma_rho_grain:.4f} g/cm^3")

        return rho_grain, sigma_rho_grain

    def analyze_asteroid(self, target_R, name="Unknown"):
        """Run the full asteroid analysis pipeline for a single spectrum."""
        logger.info(f"==================== STARTING ANALYSIS RUN FOR: {name} ====================")
        selected_csv = self.route_spectrum(target_R, name)

        try:
            df = pd.read_csv(selected_csv)
        except FileNotFoundError:
            logger.error(f"[{name}] CRITICAL FILE FAILURE: Missing required file '{selected_csv}'")
            raise FileNotFoundError(f"Missing file '{selected_csv}'")

        mineral_names = df['Mineral_Name'].values
        rho_i = df['Density_gcm3'].values
        sigma_rho_i = df['Density_Uncertainty'].values

        # Library reflectance columns begin after the density metadata.
        library_R = df.iloc[:, 3:].values

        rho_grain, sigma_rho_grain = self.unmix_and_propagate(
            target_R, library_R, mineral_names, rho_i, sigma_rho_i, name
        )

        self.results = {'rho_grain': rho_grain, 'sigma_rho_grain': sigma_rho_grain}
        logger.info(f"==================== COMPLETED ANALYSIS RUN FOR: {name} ====================\n")
        return f"{rho_grain:.3f} +/- {sigma_rho_grain:.3f} g/cm^3"