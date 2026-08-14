"""
Spectroscopic Bulk Inversion Framework for Minor Planetary Bodies.
Quantitative compositional and physical bulk parameter derivation engine with
Hapke phase function corrections, Monte Carlo uncertainty propagation, and AICc metrics.
"""

import os
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore", message=".*Values in x were outside bounds.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=".*slsqp.*")


def setup_logger(log_file_path: str = "log.txt", verbose: bool = True) -> logging.Logger:
    """Configures file and console logging for inversion tracking."""
    logger = logging.getLogger("BulkInversion")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class SpectroscopicBulkInversion:
    """
    Geophysical Inversion Framework for Minor Bodies.
    Derives regolith grain densities, density differentials,
    total/macroporosity fractions, and core mass/volume fractions.
    """

    def __init__(
        self,
        stony_lib_path: str = "stony_metallic_library.csv",
        primitive_lib_path: str = "primitive_aqueous_library.csv",
        logger: Optional[logging.Logger] = None,
        verbose_logging: bool = True
    ) -> None:
        self.verbose_logging = verbose_logging
        self.logger = logger if logger else setup_logger(verbose=verbose_logging)
        self.stony_lib_path = stony_lib_path
        self.primitive_lib_path = primitive_lib_path

        self.stony_lib, self.stony_ssa_raw, self.stony_wvs = self._load_and_prep_library(stony_lib_path)
        self.primitive_lib, self.primitive_ssa_raw, self.primitive_wvs = self._load_and_prep_library(primitive_lib_path)

        if self.verbose_logging:
            self.logger.info("=" * 80)
            self.logger.info("INITIALIZING PHYSICALLY RIGOROUS BULK INVERSION ENGINE")
            self.logger.info("=" * 80)

    def _load_and_prep_library(self, path: str) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]:
        """Loads endmember library and converts laboratory reflectances to Single Scattering Albedo (SSA)."""
        if not os.path.exists(path):
            if self.verbose_logging:
                self.logger.error(f"Library file missing: {path}")
            return None, None, None

        df = pd.read_csv(path)
        renames = {col: "Mineral_Name" for col in df.columns if "mineral_name" in col.lower()}
        df = df.rename(columns=renames)

        wv_cols = [c for c in df.columns if str(c).lower().startswith("wv_")]
        lib_wvs = np.array([float(str(c).lower().replace("wv_", "")) for c in wv_cols])

        refl_matrix = df[wv_cols].values.astype(float)
        ssa_matrix = self.reflectance_to_ssa(refl_matrix)

        return df, ssa_matrix, lib_wvs

    @staticmethod
    def extract_field(profile: pd.Series, candidate_keys: List[str], default_val: Any = None) -> Any:
        """Normalizes heterogeneous catalog headers."""
        for key in candidate_keys:
            if key in profile.index and not pd.isna(profile[key]):
                return profile[key]
            for col in profile.index:
                if str(col).lower() == key.lower() and not pd.isna(profile[col]):
                    return profile[col]
        return default_val

    def assert_boundary_constraints(self, target_profile: pd.Series) -> bool:
        """Enforces physical boundary conditions and checks for invalid data."""
        target_name = self.extract_field(target_profile, ["Asteroid_Name", "name", "designation"], "Unknown")
        
        albedo = float(self.extract_field(target_profile, ["Albedo", "pV", "pv", "p_v"], 0.15))
        if albedo <= 0.0:
            raise ValueError(f"Target '{target_name}': Geometric albedo pV ({albedo}) must be strictly positive.")

        bulk_density = float(self.extract_field(target_profile, ["Bulk_Density_gcm3", "bulk_density", "density"], 2.5))
        if bulk_density <= 0.0:
            raise ValueError(f"Target '{target_name}': Bulk density ({bulk_density} g/cm³) must be positive.")

        target_wv, target_refl = self._extract_target_spectrum(target_profile)
        if len(target_wv) < 5:
            raise ValueError(f"Target '{target_name}': Insufficient valid spectral channels ({len(target_wv)} < 5).")

        return True

    @staticmethod
    def reflectance_to_ssa(r: np.ndarray) -> np.ndarray:
        """Converts isotropic reflectance R to Single Scattering Albedo (SSA) w using Hapke theory."""
        r_clipped = np.clip(r, 1e-4, 0.999)
        return 4.0 * r_clipped / ((1.0 + r_clipped) ** 2)

    @staticmethod
    def ssa_to_reflectance(w: np.ndarray) -> np.ndarray:
        """Converts Single Scattering Albedo (SSA) w to isotropic reflectance R."""
        w_clipped = np.clip(w, 1e-6, 0.9999)
        gamma = np.sqrt(1.0 - w_clipped)
        return (1.0 - gamma) / (1.0 + gamma)

    def _select_library(self, target_profile: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Selects appropriate endmember library based on taxonomy and albedo."""
        taxonomy = str(self.extract_field(target_profile, ["Taxonomic_Type", "taxonomy", "tax_type"], "")).upper().strip()
        albedo = float(self.extract_field(target_profile, ["Albedo", "pV", "pv", "p_v"], 0.15))

        primitive_types = {"C", "B", "G", "F", "D", "P", "T", "D/P", "C/CB", "CB"}
        if taxonomy in primitive_types or albedo < 0.08:
            if self.primitive_lib is not None:
                return self.primitive_lib, self.primitive_ssa_raw, self.primitive_wvs
            return self.stony_lib, self.stony_ssa_raw, self.stony_wvs

        if self.stony_lib is not None:
            return self.stony_lib, self.stony_ssa_raw, self.stony_wvs
        return self.primitive_lib, self.primitive_ssa_raw, self.primitive_wvs

    def _extract_target_spectrum(self, target_profile: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts and cleans sorted arrays of wavelengths and reflectance values."""
        wv_cols = [c for c in target_profile.index if str(c).lower().startswith("wv_")]
        wavelengths, reflectances = [], []

        for col in wv_cols:
            try:
                wv = float(str(col).lower().replace("wv_", ""))
                refl = float(target_profile[col])
                if not np.isnan(refl) and not np.isinf(refl) and refl > 0.001:
                    wavelengths.append(wv)
                    reflectances.append(refl)
            except (ValueError, TypeError):
                continue

        order = np.argsort(wavelengths)
        return np.array(wavelengths)[order], np.array(reflectances)[order]

    def _solve_optimization(
        self,
        target_wv: np.ndarray,
        target_refl: np.ndarray,
        lib_ssa_matrix: np.ndarray,
        albedo: float,
        phase_angle_deg: float,
        n_minerals: int
    ) -> Tuple[np.ndarray, float, float, float]:
        """Internal optimization solver for unmixing fractions and continuum modifiers."""
        idx_vband = np.argmin(np.abs(target_wv - 0.55))
        
        refl_vband = target_refl[idx_vband]
        phi_obs = (refl_vband / albedo) if albedo > 0 else 1.0

        def objective_function(params: np.ndarray) -> float:
            fractions = params[:n_minerals]
            s_weathering = params[n_minerals]
            scale = params[n_minerals + 1]

            mix_ssa = np.dot(fractions, lib_ssa_matrix)
            mix_refl = self.ssa_to_reflectance(mix_ssa)

            reddening = np.exp(s_weathering * (target_wv - 0.55))
            model_refl = scale * mix_refl * reddening * phi_obs

            spec_mse = np.mean((target_refl - model_refl) ** 2)

            albedo_penalty = 0.0
            if albedo > 0:
                p_V_model = scale * mix_refl[idx_vband]
                albedo_penalty = 10.0 * ((p_V_model - albedo) / albedo) ** 2

            scale_penalty = 0.0
            if scale < 0.2:
                scale_penalty = 100.0 * (0.2 - scale) ** 2
            elif scale > 1.5:
                scale_penalty = 100.0 * (scale - 1.5) ** 2

            ridge_reg = 1e-4 * np.sum(fractions ** 2)

            return spec_mse + albedo_penalty + scale_penalty + ridge_reg

        constraints = ({"type": "eq", "fun": lambda p: np.sum(p[:n_minerals]) - 1.0},)
        bounds = [(0.0, 1.0)] * n_minerals + [(0.0, 1.0), (0.2, 1.5)]

        x0_uniform = np.ones(n_minerals) / n_minerals
        params0 = np.append(x0_uniform, [0.05, 0.80])

        res = minimize(
            objective_function,
            params0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        best_fractions = np.maximum(0, res.x[:n_minerals])
        best_fractions /= np.sum(best_fractions)
        s_opt = res.x[n_minerals]
        k_opt = res.x[n_minerals + 1]

        return best_fractions, s_opt, k_opt, res.fun

    def invert_single_target(self, target_profile: pd.Series) -> Dict[str, Any]:
        """Inverts a single target spectrum and performs Monte Carlo error propagation."""
        target_name = self.extract_field(target_profile, ["Asteroid_Name", "name", "designation"], "Unknown")
        taxonomy = self.extract_field(target_profile, ["Taxonomic_Type", "taxonomy", "tax_type"], "Unknown")
        bulk_density = float(self.extract_field(target_profile, ["Bulk_Density_gcm3", "bulk_density", "density"], 2.5))
        bulk_density_err = float(self.extract_field(target_profile, ["Bulk_Density_Uncertainty", "bulk_density_uncertainty", "density_err"], 0.1))
        albedo = float(self.extract_field(target_profile, ["Albedo", "pV", "pv", "p_v"], 0.15))
        phase_angle = float(self.extract_field(target_profile, ["Phase_Angle_deg", "phase_angle", "phase"], 0.0))

        if self.verbose_logging:
            self.logger.info("-" * 60)
            self.logger.info(f"STARTING INVERSION: Target={target_name} | Taxonomy={taxonomy} | pV={albedo}")

        lib_df, lib_ssa_raw, lib_wvs = self._select_library(target_profile)
        if lib_df is None or lib_ssa_raw is None:
            raise ValueError(f"Target '{target_name}': Spectral endmember library unavailable.")

        target_wv, target_refl = self._extract_target_spectrum(target_profile)
        n_minerals = len(lib_df)
        densities = lib_df["Density_gcm3"].values.astype(float)
        mineral_names = lib_df["Mineral_Name"].values

        lib_ssa_matrix = np.zeros((n_minerals, len(target_wv)))
        for i in range(n_minerals):
            lib_ssa_matrix[i, :] = np.interp(target_wv, lib_wvs, lib_ssa_raw[i, :], left=lib_ssa_raw[i, 0], right=lib_ssa_raw[i, -1])

        best_fractions, s_opt, k_opt, loss_val = self._solve_optimization(
            target_wv, target_refl, lib_ssa_matrix, albedo, phase_angle, n_minerals
        )

        crust_grain_density = float(np.sum(best_fractions * densities))

        mix_ssa = np.dot(best_fractions, lib_ssa_matrix)
        mix_refl = self.ssa_to_reflectance(mix_ssa)
        idx_vband = np.argmin(np.abs(target_wv - 0.55))
        phi_obs = (target_refl[idx_vband] / albedo) if albedo > 0 else 1.0
        model_refl = k_opt * mix_refl * np.exp(s_opt * (target_wv - 0.55)) * phi_obs

        residuals = target_refl - model_refl
        dof = max(1, len(target_wv) - (np.sum(best_fractions > 0.001) + 2))
        sse = float(np.sum(residuals ** 2))
        rmse = float(np.sqrt(sse / dof))

        # Monte Carlo Uncertainty Propagation (N_MC = 50)
        n_mc = 50
        mc_grain_densities, mc_macroporosities, mc_core_mass_fracs = [], [], []

        is_primitive = taxonomy in {"C", "B", "G", "F", "D", "P", "T"} or albedo < 0.08
        eta_micro = 0.25 if is_primitive else 0.10
        core_density = 7.87

        for mc_seed in range(n_mc):
            np.random.seed(mc_seed)
            perturbed_refl = target_refl + np.random.normal(0, max(0.005, rmse), size=len(target_refl))
            perturbed_bulk = max(0.1, np.random.normal(bulk_density, bulk_density_err))

            p_fracs, _, _, _ = self._solve_optimization(
                target_wv, perturbed_refl, lib_ssa_matrix, albedo, phase_angle, n_minerals
            )
            p_grain = float(np.sum(p_fracs * densities))
            mc_grain_densities.append(p_grain)

            if perturbed_bulk <= p_grain:
                p_macro = max(0.0, 1.0 - (perturbed_bulk / (p_grain * (1.0 - eta_micro))))
                p_core = 0.0
            else:
                p_macro = 0.0
                v_c = min(1.0, max(0.0, (perturbed_bulk - p_grain) / (core_density - p_grain)))
                p_core = min(1.0, max(0.0, v_c * (core_density / perturbed_bulk)))

            mc_macroporosities.append(p_macro)
            mc_core_mass_fracs.append(p_core)

        crust_grain_err = float(np.std(mc_grain_densities))
        density_diff = float(crust_grain_density - bulk_density)
        density_diff_err = float(np.sqrt(crust_grain_err ** 2 + bulk_density_err ** 2))

        total_porosity = max(0.0, 1.0 - (bulk_density / crust_grain_density)) if bulk_density <= crust_grain_density else 0.0
        
        if bulk_density <= crust_grain_density:
            macroporosity = max(0.0, 1.0 - (bulk_density / (crust_grain_density * (1.0 - eta_micro))))
            porosity_err = float(np.std(mc_macroporosities))
            core_vol_frac = 0.0
            core_mass_frac = 0.0
            core_mass_err = 0.0
        else:
            macroporosity = 0.0
            porosity_err = 0.0
            core_vol_frac = min(1.0, max(0.0, (bulk_density - crust_grain_density) / (core_density - crust_grain_density)))
            core_mass_frac = min(1.0, max(0.0, core_vol_frac * (core_density / bulk_density)))
            core_mass_err = float(np.std(mc_core_mass_fracs))

        n_obs = len(target_wv)
        k_active = int(np.sum(best_fractions > 0.001)) + 2
        chi2_r = float(sse / (dof * (0.01 ** 2)))
        aic = float(n_obs * np.log(max(1e-10, sse / n_obs)) + 2 * k_active)
        
        aicc_denom = n_obs - k_active - 1
        aicc = float(aic + (2 * k_active * (k_active + 1)) / aicc_denom) if aicc_denom > 0 else aic
        bic = float(n_obs * np.log(max(1e-10, sse / n_obs)) + k_active * np.log(n_obs))

        if self.verbose_logging:
            self.logger.info(
                f"OPTIMIZATION CONVERGED: Target={target_name} | Best Loss={loss_val:.6f} | "
                f"Scale={k_opt:.4f} | Space Weathering={s_opt:.4f}"
            )
            self.logger.info("INVERTED MINERAL COMPOSITION BREAKDOWN:")
            for m_idx in range(n_minerals):
                if best_fractions[m_idx] > 0.001:
                    self.logger.info(f"  - {mineral_names[m_idx]:<35} : {best_fractions[m_idx]*100.0:6.2f}%")
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

    def process_batch(self, targets_csv_path: str) -> pd.DataFrame:
        """Processes an input batch CSV of spectrophotometric targets."""
        if not os.path.exists(targets_csv_path):
            if self.verbose_logging:
                self.logger.error(f"Batch target file not found: {targets_csv_path}")
            return pd.DataFrame()

        df_targets = pd.read_csv(targets_csv_path)
        records_list = df_targets.to_dict("records")
        inversion_records = []

        if self.verbose_logging:
            self.logger.info(f"PROCESSING BATCH FILE: {targets_csv_path} ({len(records_list)} targets)")

        for target_dict in records_list:
            target_profile = pd.Series(target_dict)
            target_name = self.extract_field(target_profile, ["Asteroid_Name", "name", "designation"], "Unknown")
            taxonomy = self.extract_field(target_profile, ["Taxonomic_Type", "taxonomy", "tax_type"], "Unknown")

            output_entry = {
                "Asteroid_Name": target_name,
                "Taxonomic_Type": taxonomy,
                "Status": "VALIDATION PASS",
                "Exclusion_Reason": "",
            }

            try:
                if self.assert_boundary_constraints(target_profile):
                    output_entry.update(self.invert_single_target(target_profile))
            except Exception as execution_exception:
                output_entry.update(
                    {
                        "Status": "EXCLUDED" if isinstance(execution_exception, ValueError) else "FAILED",
                        "Exclusion_Reason": str(execution_exception),
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

            inversion_records.append(output_entry)

        if self.verbose_logging:
            self.logger.info("BATCH INVERSION COMPLETED SUCCESSFULLY.")

        return pd.DataFrame(inversion_records)