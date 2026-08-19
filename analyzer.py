"""
Spectroscopic Bulk Inversion Framework for Minor Planetary Bodies.

This module provides an engine for quantitative mineralogical unmixing and 
bulk physical parameter derivation (grain density, porosity, core mass fraction) 
using Hapke radiative transfer approximations and SLSQP constrained optimization.
"""

import os
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Suppress expected optimization warnings from boundary conditions in scipy SLSQP
warnings.filterwarnings("ignore", message=".*Values in x were outside bounds.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=".*slsqp.*")


def configure_logger(log_path: str = "log.txt", verbose: bool = True) -> logging.Logger:
    """
    Configures file and stream logging handlers for the inversion pipeline.

    Parameters
    ----------
    log_path : str
        Path to the output log file.
    verbose : bool
        If True, log outputs to stdout as well as to file.

    Returns
    -------
    logging.Logger
        Configured Logger object instance.
    """
    logger = logging.getLogger("BulkInversion")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Initialize file logging handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Initialize console stream handler if verbosity is requested
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class SpectroscopicBulkInversion:
    """
    Spectroscopic and geophysical unmixing inversion model for asteroid spectra.

    Derives mineral volume fractions, crustal grain densities, macroporosity,
    and potential core mass/volume fractions based on reflectance spectra.
    """

    # Minimum albedo threshold below which data quality is considered insufficient
    MIN_ALBEDO_THRESHOLD: float = 0.01
    
    # Standard iron-nickel endmember bulk density (g/cm^3)
    IRON_CORE_DENSITY: float = 7.87

    def __init__(
        self,
        stony_lib_path: str = "stony_metallic_library.csv",
        primitive_lib_path: str = "primitive_aqueous_library.csv",
        logger: Optional[logging.Logger] = None,
        verbose: bool = True,
        n_mc: int = 200,
    ) -> None:
        """
        Initializes the inversion model by loading spectrum libraries.

        Parameters
        ----------
        stony_lib_path : str
            Path to CSV file containing stony/metallic mineral endmembers.
        primitive_lib_path : str
            Path to CSV file containing primitive/hydrated mineral endmembers.
        logger : Optional[logging.Logger]
            Existing logging instance, or None to generate a new logger.
        verbose : bool
            Enable verbose logging output.
        n_mc : int
            Number of Monte Carlo iterations for parameter error propagation.
        """
        self.verbose = verbose
        self.logger = logger if logger is not None else configure_logger(verbose=verbose)
        self.stony_lib_path = stony_lib_path
        self.primitive_lib_path = primitive_lib_path
        self.n_mc = n_mc

        # Load endmember spectral libraries
        self.stony_lib, self.stony_ssa, self.stony_wvs = self._load_library(stony_lib_path)
        self.primitive_lib, self.primitive_ssa, self.primitive_wvs = self._load_library(primitive_lib_path)

        if self.verbose:
            self.logger.info("=" * 80)
            self.logger.info("INITIALIZING PHYSICALLY RIGOROUS BULK INVERSION ENGINE")
            self.logger.info("=" * 80)

    def _load_library(self, path: str) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Reads spectral endmember library CSV and converts reflectance to Single Scattering Albedo (SSA).

        Parameters
        ----------
        path : str
            Path to the endmember library CSV.

        Returns
        -------
        Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]
            Parsed DataFrame, calculated matrix of SSA values (N_minerals x N_wavelengths), 
            and sorted wavelength values in micrometers.
        """
        if not os.path.exists(path):
            if self.verbose:
                self.logger.error(f"Library file missing: {path}")
            return None, None, None

        df = pd.read_csv(path)
        
        # Standardize mineral name header
        name_columns = [col for col in df.columns if "mineral_name" in col.lower()]
        if name_columns:
            df = df.rename(columns={name_columns[0]: "Mineral_Name"})

        # Identify and sort wavelength columns (formatted as 'Wv_X.X' or 'wv_X.X')
        wv_cols = [col for col in df.columns if str(col).lower().startswith("wv_")]
        wavelengths = np.array([float(str(col).lower().replace("wv_", "")) for col in wv_cols])

        # Ensure strict ascending order for wavelength channels
        sorted_indices = np.argsort(wavelengths)
        wavelengths = wavelengths[sorted_indices]
        wv_cols = [wv_cols[i] for i in sorted_indices]

        # Extract reflectance array and calculate Hapke Single Scattering Albedo (SSA)
        reflectance_matrix = df[wv_cols].values.astype(float)
        ssa_matrix = self.reflectance_to_ssa(reflectance_matrix)

        return df, ssa_matrix, wavelengths

    @staticmethod
    def extract_field(series: pd.Series, keys: List[str], default: Any = None) -> Any:
        """
        Extracts a scalar value from a Pandas Series matching candidate key aliases.

        Parameters
        ----------
        series : pd.Series
            Target data row.
        keys : List[str]
            List of alias strings to search for in series index.
        default : Any
            Fallback value if no valid field matches.

        Returns
        -------
        Any
            Extracted field value cast to target default type.
        """
        for key in keys:
            # Case-exact match check
            if key in series.index and not pd.isna(series[key]):
                val = series[key]
                try:
                    if isinstance(default, float):
                        return float(val)
                    if isinstance(default, int):
                        return int(val)
                    return str(val)
                except (ValueError, TypeError):
                    return default
            
            # Case-insensitive match check
            for col in series.index:
                if str(col).lower() == key.lower() and not pd.isna(series[col]):
                    val = series[col]
                    try:
                        if isinstance(default, float):
                            return float(val)
                        if isinstance(default, int):
                            return int(val)
                        return str(val)
                    except (ValueError, TypeError):
                        return default

        return default

    def validate_target_constraints(self, target: pd.Series) -> bool:
        """
        Validates target parameter constraints and raises exceptions for invalid targets.

        Parameters
        ----------
        target : pd.Series
            Target profile containing observational physical properties.

        Returns
        -------
        bool
            Returns True if all boundary conditions are satisfied.
        """
        name = str(self.extract_field(target, ["Asteroid_Name", "name", "designation"], "Unknown"))
        taxonomy = str(self.extract_field(target, ["Taxonomic_Type", "taxonomy", "tax_type"], "")).upper().strip()

        # Step 1: Reject cometary targets due to active outgassing and non-static surface regolith
        is_comet = (
            taxonomy in {"COMET", "C/", "P/", "COM"}
            or name.startswith("1P/")
            or name.startswith("67P/")
            or "COMET " in name.upper()
        )
        if is_comet:
            raise ValueError(
                f"Target '{name}': Cometary bodies are excluded from spectroscopic bulk inversion "
                f"due to active volatile outgassing and non-static surface regolith processes."
            )

        # Step 2: Reject targets below minimum geometric albedo threshold
        albedo = float(self.extract_field(target, ["Albedo", "pV", "pv", "p_v"], 0.15))
        if albedo < self.MIN_ALBEDO_THRESHOLD:
            raise ValueError(
                f"Target '{name}': Geometric albedo pV ({albedo:.3f}) is below the minimum threshold "
                f"({self.MIN_ALBEDO_THRESHOLD:.2f}) required for effective spectroscopic inversion."
            )

        # Step 3: Validate physical positivity of bulk density
        bulk_density = float(self.extract_field(target, ["Bulk_Density_gcm3", "bulk_density", "density"], 2.5))
        if bulk_density <= 0.0:
            raise ValueError(f"Target '{name}': Bulk density ({bulk_density} g/cm³) must be strictly positive.")

        # Step 4: Ensure sufficient spectral resolution (minimum 5 channels)
        wavelengths, _ = self._extract_spectrum(target)
        if len(wavelengths) < 5:
            raise ValueError(f"Target '{name}': Insufficient valid spectral channels ({len(wavelengths)} < 5).")

        return True

    @staticmethod
    def reflectance_to_ssa(r: np.ndarray) -> np.ndarray:
        """
        Converts isotropic reflectance (R) to Single Scattering Albedo (w) using Hapke theory:
        w = 4 * R / (1 + R)^2
        """
        r_clipped = np.clip(r, 1e-4, 0.999)
        return 4.0 * r_clipped / ((1.0 + r_clipped) ** 2)

    @staticmethod
    def ssa_to_reflectance(w: np.ndarray) -> np.ndarray:
        """
        Converts Single Scattering Albedo (w) to isotropic reflectance (R) using Hapke theory:
        gamma = sqrt(1 - w)
        R = (1 - gamma) / (1 + gamma)
        """
        w_clipped = np.clip(w, 1e-6, 0.9999)
        gamma = np.sqrt(1.0 - w_clipped)
        return (1.0 - gamma) / (1.0 + gamma)

    def _select_library(self, target: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Selects stony vs primitive spectral endmember library based on taxonomy and albedo."""
        taxonomy = str(self.extract_field(target, ["Taxonomic_Type", "taxonomy", "tax_type"], "")).upper().strip()
        albedo = float(self.extract_field(target, ["Albedo", "pV", "pv", "p_v"], 0.15))

        primitive_types = ("C", "B", "G", "F", "D", "P", "T")
        is_primitive = any(taxonomy.startswith(p) for p in primitive_types)

        # Select primitive library for carbonaceous/primitive classes without high-albedo overrides
        if (is_primitive or albedo < 0.08) and not taxonomy.startswith("E") and not taxonomy.startswith("M"):
            if self.primitive_lib is not None:
                return self.primitive_lib, self.primitive_ssa, self.primitive_wvs
            return self.stony_lib, self.stony_ssa, self.stony_wvs

        if self.stony_lib is not None:
            return self.stony_lib, self.stony_ssa, self.stony_wvs
        return self.primitive_lib, self.primitive_ssa, self.primitive_wvs

    def _extract_spectrum(self, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts valid sorted wavelength and reflectance arrays from a target row."""
        wv_cols = [col for col in target.index if str(col).lower().startswith("wv_")]
        wavelengths, reflectances = [], []

        for col in wv_cols:
            try:
                wv = float(str(col).lower().replace("wv_", ""))
                refl = float(target[col])
                if not np.isnan(refl) and not np.isinf(refl) and refl > 0.001:
                    wavelengths.append(wv)
                    reflectances.append(refl)
            except (ValueError, TypeError):
                continue

        sort_order = np.argsort(wavelengths)
        return np.array(wavelengths)[sort_order], np.array(reflectances)[sort_order]

    def _optimize_unmixing(
        self,
        target_wv: np.ndarray,
        target_refl: np.ndarray,
        lib_ssa: np.ndarray,
        albedo: float,
        phase_angle: float,
        n_minerals: int,
        initial_guess: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Executes SLSQP constrained optimization to determine mineral volume fractions,
        spectral scale factor, and space weathering continuum slope offset.
        """
        # Find index corresponding to V-band wavelength (~0.55 um)
        v_band_idx = np.argmin(np.abs(target_wv - 0.55))

        # Phase function correction approximation
        phase_rad = np.radians(phase_angle)
        phase_factor = max(0.1, np.cos(phase_rad / 2.0) ** 2) if phase_angle > 0 else 1.0

        def loss_function(params: np.ndarray) -> float:
            fractions = params[:n_minerals]
            weathering = params[n_minerals]
            scale = params[n_minerals + 1]

            # Enforce unit sum normalization on mineral fractions
            sum_f = np.sum(fractions)
            norm_fractions = fractions / sum_f if sum_f > 0 else np.ones(n_minerals) / n_minerals

            # Compute linear mix in SSA space and convert back to reflectance
            mix_ssa = np.dot(norm_fractions, lib_ssa)
            mix_refl = self.ssa_to_reflectance(mix_ssa)

            # Apply exponential continuum reddening modifier
            reddening = np.exp(weathering * (target_wv - 0.55))
            model_refl = scale * mix_refl * reddening * phase_factor

            # Mean squared spectral error
            mse = np.mean((target_refl - model_refl) ** 2)

            # Penalty term for deviation from observed V-band albedo
            albedo_penalty = 0.0
            if albedo > 0:
                v_model_albedo = scale * mix_refl[v_band_idx]
                albedo_penalty = 10.0 * ((v_model_albedo - albedo) / albedo) ** 2

            # Penalty terms for extreme scaling behavior
            scale_penalty = 0.0
            if scale < 0.1:
                scale_penalty = 100.0 * (0.1 - scale) ** 2
            elif scale > 3.0:
                scale_penalty = 100.0 * (scale - 3.0) ** 2

            # L2 Ridge regularization on fractional concentration vector
            ridge_reg = 1e-4 * np.sum(norm_fractions ** 2)

            return mse + albedo_penalty + scale_penalty + ridge_reg

        # Constraint: sum of mineral volume fractions must equal 1.0
        constraints = ({"type": "eq", "fun": lambda p: np.sum(p[:n_minerals]) - 1.0},)
        bounds = [(0.0, 1.0)] * n_minerals + [(-0.5, 1.0), (0.1, 3.0)]

        if initial_guess is not None and len(initial_guess) == n_minerals + 2:
            params0 = initial_guess
        else:
            uniform_fractions = np.ones(n_minerals) / n_minerals
            params0 = np.append(uniform_fractions, [0.05, max(0.80, albedo * 3.0)])

        result = minimize(
            loss_function,
            params0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        # Non-negativity clipping and re-normalization
        best_fractions = np.maximum(0, result.x[:n_minerals])
        sum_f = np.sum(best_fractions)
        best_fractions = best_fractions / sum_f if sum_f > 0 else np.ones(n_minerals) / n_minerals

        s_opt = float(result.x[n_minerals])
        k_opt = float(result.x[n_minerals + 1])

        return best_fractions, s_opt, k_opt, float(result.fun)

    def invert_single_target(self, target: pd.Series) -> Dict[str, Any]:
        """
        Inverts spectral properties for a single body, deriving physical bulk density,
        macroporosity, core parameters, and performing Monte Carlo error analysis.
        """
        name = self.extract_field(target, ["Asteroid_Name", "name", "designation"], "Unknown")
        taxonomy = str(self.extract_field(target, ["Taxonomic_Type", "taxonomy", "tax_type"], "Unknown"))
        bulk_density = float(self.extract_field(target, ["Bulk_Density_gcm3", "bulk_density", "density"], 2.5))
        bulk_density_err = float(self.extract_field(target, ["Bulk_Density_Uncertainty", "bulk_density_uncertainty", "density_err"], 0.1))
        albedo = float(self.extract_field(target, ["Albedo", "pV", "pv", "p_v"], 0.15))
        phase_angle = float(self.extract_field(target, ["Phase_Angle_deg", "phase_angle", "phase"], 0.0))

        if self.verbose:
            self.logger.info("-" * 60)
            self.logger.info(f"STARTING INVERSION: Target={name} | Taxonomy={taxonomy} | pV={albedo:.3f}")

        lib_df, lib_ssa_raw, lib_wvs = self._select_library(target)
        if lib_df is None or lib_ssa_raw is None:
            raise ValueError(f"Target '{name}': Spectral endmember library unavailable.")

        target_wv, target_refl = self._extract_spectrum(target)
        n_minerals = len(lib_df)
        densities = lib_df["Density_gcm3"].values.astype(float)
        mineral_names = lib_df["Mineral_Name"].values

        # Interpolate endmember SSA spectra onto target wavelength channels
        lib_ssa = np.zeros((n_minerals, len(target_wv)))
        for i in range(n_minerals):
            lib_ssa[i, :] = np.interp(target_wv, lib_wvs, lib_ssa_raw[i, :], left=lib_ssa_raw[i, 0], right=lib_ssa_raw[i, -1])

        # Optimize spectral unmixing
        best_fractions, s_opt, k_opt, loss_val = self._optimize_unmixing(
            target_wv, target_refl, lib_ssa, albedo, phase_angle, n_minerals
        )

        # Derive grain density of crustal material
        crust_grain_density = float(np.sum(best_fractions * densities))

        # Reconstruct best-fit model spectrum
        mix_ssa = np.dot(best_fractions, lib_ssa)
        mix_refl = self.ssa_to_reflectance(mix_ssa)
        phase_rad = np.radians(phase_angle)
        phase_factor = max(0.1, np.cos(phase_rad / 2.0) ** 2) if phase_angle > 0 else 1.0
        model_refl = k_opt * mix_refl * np.exp(s_opt * (target_wv - 0.55)) * phase_factor

        # Goodness-of-fit metrics
        residuals = target_refl - model_refl
        k_active = int(np.sum(best_fractions > 0.001)) + 2
        dof = max(1, len(target_wv) - k_active)
        sse = float(np.sum(residuals ** 2))
        rmse = float(np.sqrt(sse / dof))

        # Monte Carlo error propagation loop
        rng = np.random.default_rng(seed=42)
        mc_grain_densities, mc_density_diffs = [], []
        mc_macroporosities, mc_core_mass_fracs = [], []

        is_primitive = any(taxonomy.upper().startswith(p) for p in ("C", "B", "G", "F", "D", "P", "T")) or albedo < 0.08
        eta_micro = 0.25 if is_primitive else 0.10
        core_density = self.IRON_CORE_DENSITY
        noise_std = max(0.002, min(0.010, rmse * 0.2))

        opt_params0 = np.append(best_fractions, [s_opt, k_opt])

        for _ in range(self.n_mc):
            perturbed_refl = np.maximum(1e-4, target_refl + rng.normal(0, noise_std, size=len(target_refl)))
            perturbed_bulk = max(0.1, rng.normal(bulk_density, bulk_density_err))
            v_idx = np.argmin(np.abs(target_wv - 0.55))
            perturbed_albedo = max(0.01, albedo * (perturbed_refl[v_idx] / max(1e-4, target_refl[v_idx])))

            p_fracs, _, _, _ = self._optimize_unmixing(
                target_wv, perturbed_refl, lib_ssa, perturbed_albedo, phase_angle, n_minerals, initial_guess=opt_params0
            )
            p_grain = float(np.sum(p_fracs * densities))
            mc_grain_densities.append(p_grain)
            mc_density_diffs.append(p_grain - perturbed_bulk)

            if perturbed_bulk <= p_grain:
                p_macro = max(0.0, 1.0 - (perturbed_bulk / (p_grain * (1.0 - eta_micro))))
                p_core = 0.0
            else:
                p_macro = 0.0
                if abs(core_density - p_grain) > 1e-4 and core_density > p_grain:
                    v_c = min(1.0, max(0.0, (perturbed_bulk - p_grain) / (core_density - p_grain)))
                    p_core = min(1.0, max(0.0, v_c * (core_density / perturbed_bulk)))
                else:
                    p_core = 0.0

            mc_macroporosities.append(p_macro)
            mc_core_mass_fracs.append(p_core)

        # Aggregate parameter statistics and uncertainties
        crust_grain_err = float(np.std(mc_grain_densities))
        density_diff = float(crust_grain_density - bulk_density)
        density_diff_err = float(np.std(mc_density_diffs))

        total_porosity = max(0.0, 1.0 - (bulk_density / crust_grain_density)) if bulk_density <= crust_grain_density else 0.0

        if bulk_density <= crust_grain_density:
            macroporosity = max(0.0, 1.0 - (bulk_density / (crust_grain_density * (1.0 - eta_micro))))
            core_vol_frac = 0.0
            core_mass_frac = 0.0
        else:
            macroporosity = 0.0
            if abs(core_density - crust_grain_density) > 1e-4 and core_density > crust_grain_density:
                core_vol_frac = min(1.0, max(0.0, (bulk_density - crust_grain_density) / (core_density - crust_grain_density)))
                core_mass_frac = min(1.0, max(0.0, core_vol_frac * (core_density / bulk_density)))
            else:
                core_vol_frac = 0.0
                core_mass_frac = 0.0

        porosity_err = float(np.std(mc_macroporosities))
        core_mass_err = float(np.std(mc_core_mass_fracs))

        # Information-theoretic criteria calculation
        n_obs = len(target_wv)
        chi2_r = float(sse / (dof * (0.01 ** 2)))
        aic = float(n_obs * np.log(max(1e-10, sse / n_obs)) + 2 * k_active)
        denom = n_obs - k_active - 1
        aicc = float(aic + (2 * k_active * (k_active + 1)) / denom) if denom > 0 else float(aic)
        bic = float(n_obs * np.log(max(1e-10, sse / n_obs)) + k_active * np.log(n_obs))

        if self.verbose:
            self.logger.info(
                f"OPTIMIZATION CONVERGED: Target={name} | Best Loss={loss_val:.6f} | "
                f"Scale={k_opt:.4f} | Space Weathering={s_opt:.4f}"
            )
            self.logger.info("INVERTED MINERAL COMPOSITION BREAKDOWN:")
            for m in range(n_minerals):
                if best_fractions[m] > 0.001:
                    self.logger.info(f"  - {mineral_names[m]:<35} : {best_fractions[m]*100.0:6.2f}%")
            self.logger.info("GEOPHYSICAL & STATISTICAL PARAMETERS:")
            self.logger.info(f"  - Crustal Grain Density : {crust_grain_density:.3f} +/- {crust_grain_err:.3f} g/cm3")
            self.logger.info(f"  - Density Differential  : {density_diff:.3f} +/- {density_diff_err:.3f} g/cm3")
            self.logger.info(f"  - Total Porosity        : {total_porosity*100.0:.2f}%")
            self.logger.info(f"  - Macroporosity         : {macroporosity*100.0:.2f}% +/- {porosity_err*100.0:.2f}%")
            self.logger.info(f"  - Core Volume Fraction  : {core_vol_frac*100.0:.2f}%")
            self.logger.info(f"  - Core Mass Fraction    : {core_mass_frac*100.0:.2f}% +/- {core_mass_err*100.0:.2f}%")
            self.logger.info(f"  - Goodness-of-Fit       : Chi2_r={chi2_r:.4f} | SSE={sse:.6f} | AIC={aic:.2f} | AICc={aicc:.2f} | BIC={bic:.2f}")

        return {
            "Status": "VALIDATION PASS",
            "Exclusion_Reason": "",
            "Crustal_Grain_Density": crust_grain_density,
            "Crustal_Grain_Error": crust_grain_err,
            "Density_Differential": density_diff,
            "Density_Differential_Error": density_diff_err,
            "Total_Porosity_Fraction": total_porosity,
            "Macroporosity_Fraction": macroporosity,
            "Macroporosity_Error": porosity_err,
            "Core_Volume_Fraction": core_vol_frac,
            "Core_Mass_Fraction": core_mass_frac,
            "Core_Mass_Error": core_mass_err,
            "Reduced_Chi_Squared": chi2_r,
            "Spectral_SSE": sse,
            "AIC": aic,
            "AICc": aicc,
            "BIC": bic,
        }

    def process_batch(self, csv_path: str) -> pd.DataFrame:
        """Processes a batch CSV dataset of target spectra and outputs inverted results."""
        if not os.path.exists(csv_path):
            if self.verbose:
                self.logger.error(f"Batch target file not found: {csv_path}")
            return pd.DataFrame()

        df_targets = pd.read_csv(csv_path)
        records = df_targets.to_dict("records")
        results = []

        if self.verbose:
            self.logger.info(f"PROCESSING BATCH FILE: {csv_path} ({len(records)} targets)")

        for record in records:
            target = pd.Series(record)
            name = self.extract_field(target, ["Asteroid_Name", "name", "designation"], "Unknown")
            taxonomy = self.extract_field(target, ["Taxonomic_Type", "taxonomy", "tax_type"], "Unknown")

            entry = {
                "Asteroid_Name": name,
                "Taxonomic_Type": taxonomy,
                "Status": "VALIDATION PASS",
                "Exclusion_Reason": "",
            }

            try:
                if self.validate_target_constraints(target):
                    entry.update(self.invert_single_target(target))
            except Exception as exc:
                entry.update(
                    {
                        "Status": "EXCLUDED" if isinstance(exc, ValueError) else "FAILED",
                        "Exclusion_Reason": str(exc),
                        "Crustal_Grain_Density": 0.0,
                        "Crustal_Grain_Error": 0.0,
                        "Density_Differential": 0.0,
                        "Density_Differential_Error": 0.0,
                        "Total_Porosity_Fraction": 0.0,
                        "Macroporosity_Fraction": 0.0,
                        "Macroporosity_Error": 0.0,
                        "Core_Volume_Fraction": 0.0,
                        "Core_Mass_Fraction": 0.0,
                        "Core_Mass_Error": 0.0,
                        "Reduced_Chi_Squared": 0.0,
                        "Spectral_SSE": 0.0,
                        "AIC": 0.0,
                        "AICc": 0.0,
                        "BIC": 0.0,
                    }
                )

            results.append(entry)

        if self.verbose:
            self.logger.info("BATCH INVERSION COMPLETED SUCCESSFULLY.")

        return pd.DataFrame(results)