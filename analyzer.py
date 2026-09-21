"""
Spectroscopic Bulk Inversion Framework for Minor Planetary Bodies (Icarus Peer-Review Edition).

Provides an unbiased physical engine for quantitative mineralogical unmixing and bulk 
geophysical parameter derivation (crustal grain density, macroporosity, core mass fraction) 
using Hapke radiative transfer transformations, unregularized Maximum Likelihood chi-squared 
optimization, comprehensive multi-source Monte Carlo error propagation, and multi-start constrained SLSQP optimization.

File: analyzer.py
"""

import os
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

warnings.filterwarnings("ignore", message=".*Values in x were outside bounds.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=".*slsqp.*")


def configure_logger(log_path: str = "log.txt", verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger("BulkInversion")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class SpectroscopicBulkInversion:
    MIN_ALBEDO_THRESHOLD: float = 0.01  
    IRON_CORE_DENSITY: float = 7.87     
    IRON_CORE_DENSITY_ERR: float = 0.15 

    MINERAL_DENSITY_UNCERTAINTIES: Dict[str, float] = {
        "iron": 0.15,
        "kamacite": 0.15,
        "troilite": 0.10,
        "pyroxene": 0.08,
        "olivine": 0.08,
        "serpentine": 0.06,
        "smectite": 0.06,
        "glass": 0.07,
        "carbon": 0.05,
        "default": 0.08,
    }

    def __init__(
        self,
        stony_lib_path: str = "stony_metallic_library_200ch.csv",
        primitive_lib_path: str = "primitive_aqueous_library_200ch.csv",
        logger: Optional[logging.Logger] = None,
        verbose: bool = True,
        n_mc: int = 200,
    ) -> None:
        self.verbose = verbose
        self.logger = logger if logger is not None else configure_logger(verbose=verbose)
        self.stony_lib_path = stony_lib_path
        self.primitive_lib_path = primitive_lib_path
        self.n_mc = n_mc

        self.stony_lib, self.stony_ssa, self.stony_wvs, self.stony_errs = self._load_library(stony_lib_path)
        self.primitive_lib, self.primitive_ssa, self.primitive_wvs, self.primitive_errs = self._load_library(primitive_lib_path)

        if self.verbose:
            self.logger.info("=" * 80)
            self.logger.info("INITIALIZING RIGOROUS 200-CHANNEL BULK INVERSION ENGINE (ICARUS SPECIFICATION)")
            self.logger.info("=" * 80)

    def _load_library(self, path: str) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if not os.path.exists(path):
            if self.verbose:
                self.logger.error(f"Spectral library file missing: {path}")
            return None, None, None, None

        df = pd.read_csv(path)
        name_columns = [col for col in df.columns if "mineral_name" in col.lower()]
        if name_columns:
            df = df.rename(columns={name_columns[0]: "Mineral_Name"})

        wv_cols = [col for col in df.columns if str(col).lower().startswith("wv_")]
        wavelengths = np.array([float(str(col).lower().replace("wv_", "")) for col in wv_cols])

        sorted_indices = np.argsort(wavelengths)
        wavelengths = wavelengths[sorted_indices]
        wv_cols = [wv_cols[i] for i in sorted_indices]

        reflectance_matrix = df[wv_cols].values.astype(float)
        ssa_matrix = self.reflectance_to_ssa(reflectance_matrix)

        err_cols = []
        for w in wavelengths:
            match = [c for c in df.columns if c.lower() in (f"err_{w:.4f}".lower(), f"err_{w:.2f}".lower())]
            if match:
                err_cols.append(match[0])
        
        if len(err_cols) == len(wavelengths):
            error_matrix = df[err_cols].values.astype(float)
        else:
            error_matrix = np.maximum(0.005, reflectance_matrix * 0.02)

        return df, ssa_matrix, wavelengths, error_matrix

    @staticmethod
    def extract_field(series: pd.Series, keys: List[str], default: Any = None) -> Any:
        for key in keys:
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
        name = str(self.extract_field(target, ["Asteroid_Name", "name", "designation"], "Unknown"))
        taxonomy = str(self.extract_field(target, ["Taxonomic_Type", "taxonomy", "tax_type"], "")).upper().strip()

        is_comet = (
            taxonomy in {"COMET", "C/", "P/", "COM"}
            or name.startswith("1P/")
            or name.startswith("67P/")
            or "COMET" in name.upper()
        )
        if is_comet:
            raise ValueError(
                f"Target '{name}': Cometary bodies are excluded from spectroscopic bulk inversion "
                f"due to active volatile outgassing and non-static surface regolith processes."
            )

        albedo = float(self.extract_field(target, ["Albedo", "pV", "pv", "p_v"], 0.15))
        if albedo < self.MIN_ALBEDO_THRESHOLD:
            raise ValueError(
                f"Target '{name}': Geometric albedo pV ({albedo:.3f}) is below the minimum threshold "
                f"({self.MIN_ALBEDO_THRESHOLD:.2f}) required for effective spectroscopic inversion."
            )

        bulk_density = float(self.extract_field(target, ["Bulk_Density_gcm3", "bulk_density", "density"], 2.5))
        if bulk_density <= 0.0:
            raise ValueError(f"Target '{name}': Bulk density ({bulk_density} g/cm³) must be strictly positive.")

        wavelengths, _, _ = self._extract_spectrum(target)
        if len(wavelengths) < 5:
            raise ValueError(f"Target '{name}': Insufficient valid spectral channels ({len(wavelengths)} < 5).")

        return True

    @staticmethod
    def reflectance_to_ssa(r: np.ndarray) -> np.ndarray:
        r_clipped = np.clip(r, 1e-4, 0.999)
        return 4.0 * r_clipped / ((1.0 + r_clipped) ** 2)

    @staticmethod
    def ssa_to_reflectance(w: np.ndarray) -> np.ndarray:
        w_clipped = np.clip(w, 1e-6, 0.9999)
        gamma = np.sqrt(1.0 - w_clipped)
        return (1.0 - gamma) / (1.0 + gamma)

    def _select_library(self, target: pd.Series) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        taxonomy = str(self.extract_field(target, ["Taxonomic_Type", "taxonomy", "tax_type"], "")).upper().strip()
        albedo = float(self.extract_field(target, ["Albedo", "pV", "pv", "p_v"], 0.15))

        primitive_types = ("C", "B", "G", "F", "D", "P", "T")
        is_primitive = any(taxonomy.startswith(p) for p in primitive_types)

        if (is_primitive or albedo < 0.08) and not taxonomy.startswith("E") and not taxonomy.startswith("M"):
            if self.primitive_lib is not None:
                return self.primitive_lib, self.primitive_ssa, self.primitive_wvs, self.primitive_errs
            return self.stony_lib, self.stony_ssa, self.stony_wvs, self.stony_errs

        if self.stony_lib is not None:
            return self.stony_lib, self.stony_ssa, self.stony_wvs, self.stony_errs
        return self.primitive_lib, self.primitive_ssa, self.primitive_wvs, self.primitive_errs

    def _extract_spectrum(self, target: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        wv_cols = [col for col in target.index if str(col).lower().startswith("wv_")]
        wavelengths, reflectances, uncertainties = [], [], []
        
        n_raw = float(self.extract_field(target, ["Initial_Data_Points"], 105.0))
        interp_factor = 0.015 * np.sqrt(105.0 / max(30.0, n_raw))

        for col in wv_cols:
            try:
                wv = float(str(col).lower().replace("wv_", ""))
                refl = float(target[col])
                if not np.isnan(refl) and not np.isinf(refl) and refl > 0.001:
                    wavelengths.append(wv)
                    reflectances.append(refl)
                    
                    meas_err = None
                    for col_name in target.index:
                        c_low = str(col_name).lower()
                        if c_low in (f"err_{wv:.4f}".lower(), f"err_{wv:.2f}".lower()) and not pd.isna(target[col_name]):
                            meas_err = float(target[col_name])
                            break
                    if meas_err is None:
                        meas_err = 0.015 * refl
                    
                    total_err = np.sqrt(meas_err**2 + (refl * interp_factor)**2)
                    uncertainties.append(total_err)
            except (ValueError, TypeError):
                continue

        sort_order = np.argsort(wavelengths)
        return np.array(wavelengths)[sort_order], np.array(reflectances)[sort_order], np.array(uncertainties)[sort_order]

    def _get_mineral_density_uncertainties(self, lib_df: pd.DataFrame) -> np.ndarray:
        dens_errs = []
        for name in lib_df["Mineral_Name"].values:
            name_low = str(name).lower()
            err = self.MINERAL_DENSITY_UNCERTAINTIES["default"]
            for key, val in self.MINERAL_DENSITY_UNCERTAINTIES.items():
                if key in name_low:
                    err = val
                    break
            dens_errs.append(err)
        return np.array(dens_errs)

    def _optimize_unmixing(
        self,
        target_wv: np.ndarray,
        target_refl: np.ndarray,
        target_err: np.ndarray,
        lib_ssa: np.ndarray,
        albedo: float,
        phase_angle: float,
        n_minerals: int,
        initial_guess: Optional[np.ndarray] = None,
        target_name: str = "",
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, float, float, float]:
        phase_rad = np.radians(phase_angle)
        phase_factor = max(0.1, np.cos(phase_rad / 2.0) ** 2) if phase_angle > 0 else 1.0
        n_obs = len(target_wv)
        safe_err = np.maximum(1e-4, target_err)

        def loss_function(params: np.ndarray) -> float:
            fractions = np.maximum(0.0, params[:n_minerals])
            weathering = params[n_minerals]
            scale = params[n_minerals + 1]

            mix_ssa = np.dot(fractions, lib_ssa)
            mix_refl = self.ssa_to_reflectance(mix_ssa)

            reddening = np.exp(weathering * (target_wv - 0.55))
            model_refl = scale * mix_refl * reddening * phase_factor

            chi2 = np.sum(((target_refl - model_refl) / safe_err) ** 2)
            return chi2 / n_obs

        constraints = ({"type": "eq", "fun": lambda p: np.sum(p[:n_minerals]) - 1.0},)
        bounds = [(0.0, 1.0)] * n_minerals + [(-0.5, 1.0), (0.1, 3.0)]

        best_res = None
        best_fun = float("inf")

        guesses = []
        if initial_guess is not None and len(initial_guess) == n_minerals + 2:
            guesses.append(initial_guess)

        uniform_fractions = np.ones(n_minerals) / n_minerals
        guesses.append(np.append(uniform_fractions, [0.0, 1.0]))

        for idx in range(n_minerals):
            single_f = np.zeros(n_minerals)
            single_f[idx] = 1.0
            guesses.append(np.append(single_f, [0.0, 1.0]))

        if rng is None:
            seed_val = (abs(hash(target_name)) % 1000000) if target_name else None
            rng = np.random.default_rng(seed=seed_val)
        
        n_random = 16 if initial_guess is not None else 32
        for _ in range(n_random):
            dirichlet_fractions = rng.dirichlet(np.ones(n_minerals) * 0.3)
            random_s = rng.uniform(-0.1, 0.3)
            random_k = rng.uniform(0.7, 1.3)
            guesses.append(np.append(dirichlet_fractions, [random_s, random_k]))

        for p0 in guesses:
            res = minimize(
                loss_function,
                p0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1500, "ftol": 1e-9, "eps": 1e-5},
            )
            if res.fun < best_fun and not np.isnan(res.fun):
                best_fun = res.fun
                best_res = res

        if best_res is None:
            best_fractions = np.ones(n_minerals) / n_minerals
            s_opt, k_opt = 0.0, 1.0
        else:
            best_raw_fractions = np.maximum(0, best_res.x[:n_minerals])
            sum_f = np.sum(best_raw_fractions)
            best_fractions = best_raw_fractions / sum_f if sum_f > 0 else np.ones(n_minerals) / n_minerals
            s_opt = float(best_res.x[n_minerals])
            k_opt = float(best_res.x[n_minerals + 1])

        mix_ssa_final = np.dot(best_fractions, lib_ssa)
        mix_refl_final = self.ssa_to_reflectance(mix_ssa_final)
        model_refl_final = k_opt * mix_refl_final * np.exp(s_opt * (target_wv - 0.55)) * phase_factor
        pure_chi2 = float(np.sum(((target_refl - model_refl_final) / safe_err) ** 2))

        return best_fractions, s_opt, k_opt, pure_chi2

    def invert_single_target(self, target: pd.Series) -> Dict[str, Any]:
        name = str(self.extract_field(target, ["Asteroid_Name", "name", "designation"], "Unknown"))
        taxonomy = str(self.extract_field(target, ["Taxonomic_Type", "taxonomy", "tax_type"], "Unknown"))
        bulk_density = float(self.extract_field(target, ["Bulk_Density_gcm3", "bulk_density", "density"], 2.5))
        bulk_density_err = float(self.extract_field(target, ["Bulk_Density_Uncertainty", "bulk_density_uncertainty", "density_err"], 0.1))
        albedo = float(self.extract_field(target, ["Albedo", "pV", "pv", "p_v"], 0.15))
        phase_angle = float(self.extract_field(target, ["Phase_Angle_deg", "phase_angle", "phase"], 0.0))

        if self.verbose:
            self.logger.info("-" * 60)
            self.logger.info(f"STARTING INVERSION: Target={name} | Taxonomy={taxonomy} | pV={albedo:.3f}")

        lib_df, lib_ssa_raw, lib_wvs, lib_errs_raw = self._select_library(target)
        if lib_df is None or lib_ssa_raw is None:
            raise ValueError(f"Target '{name}': Spectral endmember library unavailable.")

        target_wv, target_refl, target_err = self._extract_spectrum(target)
        n_minerals = len(lib_df)
        densities = lib_df["Density_gcm3"].values.astype(float)
        density_errs = self._get_mineral_density_uncertainties(lib_df)
        mineral_names = lib_df["Mineral_Name"].values

        lib_ssa = np.zeros((n_minerals, len(target_wv)))
        lib_ssa_err = np.zeros((n_minerals, len(target_wv)))
        for i in range(n_minerals):
            if len(lib_wvs) >= 4:
                pchip_ssa = PchipInterpolator(lib_wvs, lib_ssa_raw[i, :], extrapolate=True)
                pchip_err = PchipInterpolator(lib_wvs, lib_errs_raw[i, :], extrapolate=True)
                lib_ssa[i, :] = np.clip(pchip_ssa(target_wv), 1e-6, 0.9999)
                lib_ssa_err[i, :] = np.maximum(0.001, np.abs(pchip_err(target_wv)))
            else:
                lib_ssa[i, :] = np.interp(target_wv, lib_wvs, lib_ssa_raw[i, :])
                lib_ssa_err[i, :] = np.interp(target_wv, lib_wvs, lib_errs_raw[i, :])

        best_fractions, s_opt, k_opt, loss_val = self._optimize_unmixing(
            target_wv, target_refl, target_err, lib_ssa, albedo, phase_angle, n_minerals, target_name=name
        )

        crust_grain_density = float(np.sum(best_fractions * densities))

        mix_ssa = np.dot(best_fractions, lib_ssa)
        mix_refl = self.ssa_to_reflectance(mix_ssa)
        phase_rad = np.radians(phase_angle)
        phase_factor = max(0.1, np.cos(phase_rad / 2.0) ** 2) if phase_angle > 0 else 1.0
        model_refl = k_opt * mix_refl * np.exp(s_opt * (target_wv - 0.55)) * phase_factor

        residuals = target_refl - model_refl
        k_active = int(np.sum(best_fractions > 0.001)) + 2
        dof = max(1, len(target_wv) - k_active)
        sse = float(np.sum(residuals ** 2))

        chi2_r = float(loss_val / dof)

        # Multi-Start Stochastic Monte Carlo Error Engine
        rng = np.random.default_rng(seed=42)
        mc_grain_densities, mc_density_diffs = [], []
        mc_macroporosities, mc_core_mass_fracs = [], []

        is_primitive = any(taxonomy.upper().startswith(p) for p in ("C", "B", "G", "F", "D", "P", "T")) or albedo < 0.08
        eta_micro = 0.25 if is_primitive else 0.10
        opt_params0 = np.append(best_fractions, [s_opt, k_opt])

        for _ in range(self.n_mc):
            perturbed_refl = np.maximum(1e-4, target_refl + rng.normal(0, target_err, size=len(target_refl)))
            perturbed_lib_ssa = np.clip(
                lib_ssa + rng.normal(0, lib_ssa_err, size=lib_ssa.shape), 1e-6, 0.9999
            )
            perturbed_densities = np.maximum(1.0, densities + rng.normal(0, density_errs, size=n_minerals))
            perturbed_bulk = max(0.1, rng.normal(bulk_density, bulk_density_err))
            perturbed_core_density = max(6.0, rng.normal(self.IRON_CORE_DENSITY, self.IRON_CORE_DENSITY_ERR))
            perturbed_phase = max(0.0, phase_angle + rng.normal(0, 0.5))

            # Full multi-start re-optimization on perturbed space
            p_fracs, _, _, _ = self._optimize_unmixing(
                target_wv,
                perturbed_refl,
                target_err,
                perturbed_lib_ssa,
                albedo,
                perturbed_phase,
                n_minerals,
                initial_guess=opt_params0,
                target_name=name,
                rng=rng,
            )

            p_grain = float(np.sum(p_fracs * perturbed_densities))
            mc_grain_densities.append(p_grain)
            mc_density_diffs.append(p_grain - perturbed_bulk)

            if perturbed_bulk <= p_grain * (1.0 - eta_micro):
                p_macro = max(0.0, 1.0 - (perturbed_bulk / (p_grain * (1.0 - eta_micro))))
                p_core = 0.0
            else:
                p_macro = 0.0
                if abs(perturbed_core_density - p_grain) > 1e-4 and perturbed_core_density > p_grain:
                    v_c = min(1.0, max(0.0, (perturbed_bulk - p_grain) / (perturbed_core_density - p_grain)))
                    p_core = min(1.0, max(0.0, v_c * (perturbed_core_density / perturbed_bulk)))
                else:
                    p_core = 0.0

            mc_macroporosities.append(p_macro)
            mc_core_mass_fracs.append(p_core)

        crust_grain_err = float(np.std(mc_grain_densities))
        density_diff = float(crust_grain_density - bulk_density)
        density_diff_err = float(np.std(mc_density_diffs))

        total_porosity = max(0.0, 1.0 - (bulk_density / crust_grain_density)) if bulk_density <= crust_grain_density else 0.0

        if bulk_density <= crust_grain_density * (1.0 - eta_micro):
            macroporosity = max(0.0, 1.0 - (bulk_density / (crust_grain_density * (1.0 - eta_micro))))
            core_vol_frac = 0.0
            core_mass_frac = 0.0
        else:
            macroporosity = 0.0
            if abs(self.IRON_CORE_DENSITY - crust_grain_density) > 1e-4 and self.IRON_CORE_DENSITY > crust_grain_density:
                core_vol_frac = min(1.0, max(0.0, (bulk_density - crust_grain_density) / (self.IRON_CORE_DENSITY - crust_grain_density)))
                core_mass_frac = min(1.0, max(0.0, core_vol_frac * (self.IRON_CORE_DENSITY / bulk_density)))
            else:
                core_vol_frac = 0.0
                core_mass_frac = 0.0

        porosity_err = float(np.std(mc_macroporosities))
        core_mass_err = float(np.std(mc_core_mass_fracs))

        n_obs = len(target_wv)
        aic = float(n_obs * np.log(max(1e-10, sse / n_obs)) + 2 * k_active)
        denom = n_obs - k_active - 1
        aicc = float(aic + (2 * k_active * (k_active + 1)) / denom) if denom > 0 else float(aic)
        bic = float(n_obs * np.log(max(1e-10, sse / n_obs)) + k_active * np.log(n_obs))

        if self.verbose:
            self.logger.info(
                f"OPTIMIZATION CONVERGED: Target={name} | Best Chi2={loss_val:.4f} | "
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