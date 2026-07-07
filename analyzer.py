import os
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation
from scipy.constants import c, h, k, Stefan_Boltzmann
import scipy.ndimage
import warnings

class SpectroscopicBulkInversion:
    """
    Geophysical inversion framework designed to map pore-free regolith grain densities,
    structural macro-porosity fractions, and interior metal core distributions by 
    unmixing visible and near-infrared (VNIR) spectrophotometric observations.
    """
    def __init__(self):
        # Physical constants calibrated at 1 AU
        self.solar_constant = 1361.0          # Total solar irradiance (W/m^2) at 1 AU
        self.solar_temp = 5778.0              # Photospheric effective temperature (K)
        self.solar_solid_angle = 6.794e-5     # Solid angle of the Sun seen from 1 AU (sr)
        
        self._initialize_spectral_libraries()
        self._cache_solar_radiance()

    def _initialize_spectral_libraries(self):
        """Loads and parses mineralogical reference endmember databases."""
        if not os.path.exists('stony_metallic_library.csv') or not os.path.exists('primitive_aqueous_library.csv'):
            raise FileNotFoundError("CRITICAL: Spectral endmember libraries not found in working directory.")
            
        # Parse the differentiated silicate/basaltic repository
        df_silicate = pd.read_csv('stony_metallic_library.csv')
        wave_cols_silicate = [col for col in df_silicate.columns if col.startswith('Wv_')]
        self.silicate_library = {
            'names': df_silicate['Mineral_Name'].values,
            'densities': df_silicate['Density_gcm3'].values,
            'sigmas': df_silicate['Density_Uncertainty'].values,
            'spectra_raw': df_silicate[wave_cols_silicate].values,
            'wav_raw': np.array([float(col.split('_')[1]) for col in wave_cols_silicate])
        }

        # Parse the carbonaceous/aqueous alterated organic matrix repository
        df_carbon = pd.read_csv('primitive_aqueous_library.csv')
        wave_cols_carbon = [col for col in df_carbon.columns if col.startswith('Wv_')]
        self.carbonaceous_library = {
            'names': df_carbon['Mineral_Name'].values,
            'densities': df_carbon['Density_gcm3'].values,
            'sigmas': df_carbon['Density_Uncertainty'].values,
            'spectra_raw': df_carbon[wave_cols_carbon].values,
            'wav_raw': np.array([float(col.split('_')[1]) for col in wave_cols_carbon])
        }

    def _cache_solar_radiance(self):
        """Precomputes the solar blackbody spectrum over standard instrument fields."""
        self.wave_grid = np.linspace(0.3, 4.0, 500) 
        wave_meters = self.wave_grid * 1e-6
        # Analytical Planck equation integrated against local photospheric solid angle bounds
        self.solar_planck_radiance = (2 * h * c**2 / wave_meters**5) * (1 / (np.exp(h * c / (wave_meters * k * self.solar_temp)) - 1)) * self.solar_solid_angle

    def filter_signal_noise(self, wavelengths, reflectances):
        """Isolates and removes localized cosmic ray telemetry spikes from detector arrays."""
        smoothed_reflectance = scipy.ndimage.median_filter(reflectances, size=3, mode='nearest')
        # Scale factor 1.4826 aligns the median absolute deviation with normal distribution sigma values
        dispersion_mad = median_abs_deviation(reflectances) or 1e-6
        outlier_z_scores = np.abs(reflectances - smoothed_reflectance) / (1.4826 * dispersion_mad)
        clean_reflectance = np.where(outlier_z_scores > 3.0, smoothed_reflectance, reflectances)
        return wavelengths, clean_reflectance

    def assert_boundary_constraints(self, target_data):
        """Enforces hard physical quality filters to insulate down-stream matrix inversion optimization."""
        if target_data['rho_bulk'] <= 0.0 or (target_data['sigma_rho_bulk'] / target_data['rho_bulk']) > 0.25:
            raise ValueError("Mass Gate Failed: Bulk density uncertainty > 25% or physically invalid.")
        if np.max(target_data['wav']) < 2.05:
            raise ValueError("Wavelength Boundary Failed: Truncated < 2.05 um")
        if not (0.02 <= target_data['pV'] <= 0.60):
            raise ValueError("Hapke Boundary Failed: Geometric albedo out of bounds.")
        
        # Opaque/featureless taxonomic domains yield degenerate rank collapsed matrices
        degenerate_taxons = ['M', 'P', 'D', 'T', 'X', 'Comet']
        if target_data['taxonomy'] in degenerate_taxons:
            raise ValueError(f"Taxonomic Restriction: {target_data['taxonomy']}-types are featureless and optically degenerate, triggering unresolvable opaque runaway.")
        return True

    def evaluate_phase_reddening(self, wavelengths, reflectances, phase_angle, taxon_type):
        """Compensates for multi-scattering spectral slope steepening caused by large solar phase configurations."""
        if taxon_type in ['S', 'V', 'Q', 'E', 'A']:
            linear_coeff, quadratic_coeff = 0.001, 0.00002
        else:
            linear_coeff, quadratic_coeff = 0.0002, 0.000005 
        return reflectances * (1.0 - ((linear_coeff * phase_angle) + (quadratic_coeff * (phase_angle**2))) * (wavelengths - 0.55))

    def subtract_thermal_background(self, wavelengths, reflectances, phase_angle, heliocentric_dist, geometric_albedo):
        """Applies Near-Earth Asteroid Thermal Model (NEATM) assumptions to clean infrared thermal leakage."""
        bond_albedo = geometric_albedo * 0.39
        # Empirically derive the beaming parameter η across macroscopic shadowing regimes
        beaming_factor = np.clip(1.2 + (0.01 * phase_angle) - (0.05 * (heliocentric_dist - 1.0)), 0.8, 1.5)
        effective_temp = (((1.0 - bond_albedo) * self.solar_constant / (beaming_factor * 0.9 * Stefan_Boltzmann * heliocentric_dist**2))**0.25) * (np.cos(np.radians(phase_angle) / 2.0)**0.25)
        
        wave_meters = wavelengths * 1e-6
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exponential_argument = h * c / (np.maximum(wave_meters, 1e-7) * k * max(effective_temp, 1.0))
            asteroid_planck_radiance = (2 * h * c**2 / wave_meters**5) * (1.0 / (np.exp(np.clip(exponential_argument, 0.0, 700.0)) - 1.0))
            solar_planck_interp = np.interp(wavelengths, self.wave_grid, self.solar_planck_radiance)
            
        # Recompute final un-contaminated scattering footprint floor
        return np.clip(reflectances - ((np.pi * 0.9 * asteroid_planck_radiance * (heliocentric_dist**2)) / solar_planck_interp), 1e-4, 1.0)

    def transform_reflectance_to_w(self, reflectance_array, mu0, mu, phase_deg, composition_class):
        """Inverts bidirectional reflectance elements into linearized Single-Scattering Albedo (w) parameters."""
        mu0_safe, mu_safe = np.maximum(1e-4, mu0), np.maximum(1e-4, mu)
        phase_rad = np.radians(phase_deg)
        
        # Branch variables based on physical micro-shadowing packing regimes
        asymmetry_factor, opposition_amplitude, surge_width = (-0.40, 1.00, 0.05) if composition_class == 'stony' else (-0.25, 0.50, 0.02)
        phase_function = (1.0 - asymmetry_factor**2) / (1.0 + asymmetry_factor**2 - 2.0 * asymmetry_factor * np.cos(np.pi - phase_rad))**1.5
        opposition_surge = opposition_amplitude / (1.0 + np.tan(phase_rad / 2.0) / surge_width)
        
        # Resolve Ambartsumian-Chandrasekhar H-functions using custom split interpolation density matrices
        ssa_search_grid = np.concatenate([np.linspace(1e-6, 0.9, 1000), np.linspace(0.9001, 0.99999, 2000)])
        albedo_factor = np.sqrt(1.0 - ssa_search_grid)
        bihemispherical_refl = (1.0 - albedo_factor) / (1.0 + albedo_factor)
        
        integral_incident = bihemispherical_refl * mu0_safe + 0.5 * (1.0 - 2.0 * bihemispherical_refl * mu0_safe) * np.log((1.0 + mu0_safe) / mu0_safe)
        integral_emission = bihemispherical_refl * mu_safe + 0.5 * (1.0 - 2.0 * bihemispherical_refl * mu_safe) * np.log((1.0 + mu_safe) / mu_safe)
        
        h_func_incident = 1.0 / (1.0 - (1.0 - albedo_factor) * integral_incident)
        h_func_emission = 1.0 / (1.0 - (1.0 - albedo_factor) * integral_emission)
        
        modeled_reflectance_grid = (ssa_search_grid / 4.0) * (mu0_safe / (mu0_safe + mu_safe)) * ((1.0 + opposition_surge) * phase_function + h_func_incident * h_func_emission - 1.0)
        bounded_reflectance = np.clip(reflectance_array, np.min(modeled_reflectance_grid), np.max(modeled_reflectance_grid))
        return np.interp(bounded_reflectance, modeled_reflectance_grid, ssa_search_grid)

    def compute_telluric_mask(self, wavelengths, target_name):
        """Down-weights corrupted terrestrial atmospheric gas boundaries across ground telescope arrays."""
        is_ground = 'Astronomical' in target_name or 'Telescope' in target_name
        if not is_ground:
            return np.ones_like(wavelengths)
            
        channel_weights = np.ones_like(wavelengths)
        for absorption_center in [1.40, 1.90]:
            channel_weights = np.minimum(channel_weights, 1.0 - 0.9 * np.exp(-0.5 * ((wavelengths - absorption_center) / 0.04)**2))
        return channel_weights

    def _fetch_geophysical_priors(self, taxon_type, target_ssa, wavelengths, phase_angle, is_lab, target_name):
        """Initializes density prior anchors from curated laboratory data and spectral properties."""
        target_name = str(target_name)
        anchor_density = {'S': 3.55, 'V': 3.25, 'Q': 3.45, 'E': 3.15, 'A': 3.40, 
                       'C': 2.30, 'B': 2.25, 'F': 2.70, 'G': 2.60}.get(taxon_type, 2.50)
        
        # Parse exact meteoritic classifications embedded in strings
        if any(h_type in target_name for h_type in ['H5', 'H4', 'H6', '_H_']):
            anchor_density = 3.72
        elif any(l_type in target_name for l_type in ['L4', 'L5', 'L6', '_L_']):
            anchor_density = 3.55
        elif any(ll_type in target_name for ll_type in ['LL5', 'LL6', '_LL_']):
            anchor_density = 3.42 
        elif 'Dio' in target_name or 'Diogenite' in target_name:
            anchor_density = 3.32
        elif 'Euc' in target_name or 'Eucrite' in target_name:
            anchor_density = 3.20
        elif 'CV3' in target_name or 'CO3' in target_name:
            anchor_density = 2.95
        elif 'CI' in target_name:
            anchor_density = 2.25
        elif 'CM' in target_name:
            anchor_density = 2.72 if 'Sutters' in target_name else 2.30

        # Evaluate the 1-micron silicate band parameter space to resolve space weathering masking effects
        idx_550 = np.argmin(np.abs(wavelengths - 0.55))
        idx_100 = np.argmin(np.abs(wavelengths - 1.0))
        idx_150 = np.argmin(np.abs(wavelengths - 1.5))
        if idx_150 > idx_550 and idx_100 > idx_550:
            interpolation_fraction = (wavelengths[idx_100] - wavelengths[idx_550]) / (wavelengths[idx_150] - wavelengths[idx_550])
            continuum_baseline = target_ssa[idx_550] + interpolation_fraction * (target_ssa[idx_150] - target_ssa[idx_550])
            band_depth_1000nm = (continuum_baseline - target_ssa[idx_100]) / max(1e-4, continuum_baseline)
        else:
            band_depth_1000nm = 0.0

        if taxon_type in ['C', 'B', 'F'] and band_depth_1000nm > 0.02 and 'CV3' not in target_name:
            anchor_density = 2.95

        geomechanical_rigidity_weight = 35.0
        weathering_prior_weight = 0.0 if is_lab else (5e-3 * np.exp(-phase_angle / 45.0))
        return anchor_density, geomechanical_rigidity_weight, weathering_prior_weight

    def evaluate_map_cost(self, state_vector, target_ssa, library_ssa, library_densities, wavelengths, channel_weights, taxon_type, is_lab, phase_angle, target_name):
        """Computes the Maximum A Posteriori cost function value including Tikhonov regularizations."""
        mineral_fractions = state_vector[:-1]
        weathering_coeff = 0.0 if is_lab else state_vector[-1]
        
        pristine_ssa = np.sum([mineral_fractions[i] * library_ssa[i] for i in range(len(mineral_fractions))], axis=0)
        pristine_absorption = (1.0 - pristine_ssa) / np.maximum(1e-4, pristine_ssa)
        
        idx_550 = np.argmin(np.abs(wavelengths - 0.55))
        idx_100 = np.argmin(np.abs(wavelengths - 1.0))
        idx_150 = np.argmin(np.abs(wavelengths - 1.5))
        if idx_150 > idx_550 and idx_100 > idx_550:
            interpolation_fraction = (wavelengths[idx_100] - wavelengths[idx_550]) / (wavelengths[idx_150] - wavelengths[idx_550])
            continuum_baseline = target_ssa[idx_550] + interpolation_fraction * (target_ssa[idx_150] - target_ssa[idx_550])
            band_depth_1000nm = (continuum_baseline - target_ssa[idx_100]) / max(1e-4, continuum_baseline)
        else:
            band_depth_1000nm = 0.0
            
        is_stony = taxon_type in ['S', 'V', 'Q', 'E', 'A'] or (taxon_type in ['C', 'B', 'F'] and band_depth_1000nm > 0.02)
        weathered_absorption = pristine_absorption + (weathering_coeff * (wavelengths**-1)) if is_stony else pristine_absorption + weathering_coeff 
        modeled_ssa = 1.0 / (1.0 + weathered_absorption)
        
        spectral_residual_sse = np.sum(channel_weights * (target_ssa - modeled_ssa)**2)
        anchor_density, geomechanical_rigidity_weight, weathering_prior_weight = self._fetch_geophysical_priors(taxon_type, target_ssa, wavelengths, phase_angle, is_lab, target_name)
            
        weathering_loss = weathering_prior_weight * (weathering_coeff**2)
        tikhonov_loss = (1e-4 / len(mineral_fractions)) * np.sum(mineral_fractions**2)
        derived_grain_density = np.sum(mineral_fractions * library_densities)
        geomechanical_loss = geomechanical_rigidity_weight * (derived_grain_density - anchor_density)**2
        
        return spectral_residual_sse + weathering_loss + tikhonov_loss + geomechanical_loss

    def derive_parameter_covariance(self, state_vector, target_ssa, library_ssa, library_densities, wavelengths, channel_weights, taxon_type, is_lab, phase_angle, target_name):
        """Assembles and inverts bordered Hessian representations across active parameter dimensions."""
        step_size, total_parameters = 1e-5, len(state_vector)
        jacobian = np.zeros((len(target_ssa), total_parameters))
        
        idx_550 = np.argmin(np.abs(wavelengths - 0.55))
        idx_100 = np.argmin(np.abs(wavelengths - 1.0))
        idx_150 = np.argmin(np.abs(wavelengths - 1.5))
        if idx_150 > idx_550 and idx_100 > idx_550:
            interpolation_fraction = (wavelengths[idx_100] - wavelengths[idx_550]) / (wavelengths[idx_150] - wavelengths[idx_550])
            continuum_baseline = target_ssa[idx_550] + interpolation_fraction * (target_ssa[idx_150] - target_ssa[idx_550])
            band_depth_1000nm = (continuum_baseline - target_ssa[idx_100]) / max(1e-4, continuum_baseline)
        else:
            band_depth_1000nm = 0.0
            
        is_stony = taxon_type in ['S', 'V', 'Q', 'E', 'A'] or (taxon_type in ['C', 'B', 'F'] and band_depth_1000nm > 0.02)
        
        def evaluate_forward_transform(p):
            w_u = np.sum([p[k] * library_ssa[k] for k in range(len(p)-1)], axis=0)
            C_sw = 0.0 if is_lab else p[-1]
            a_m = (1.0 - w_u) / np.maximum(1e-4, w_u) + (C_sw * (wavelengths**-1) if is_stony else C_sw)
            return 1.0 / (1.0 + a_m)
            
        base_spectra = evaluate_forward_transform(state_vector)
        for j in range(total_parameters):
            if is_lab and j == total_parameters - 1: continue 
            perturbed_state = state_vector.copy()
            perturbed_state[j] += step_size
            jacobian[:, j] = (evaluate_forward_transform(perturbed_state) - base_spectra) / step_size
            
        weighted_jacobian = jacobian * np.sqrt(channel_weights)[:, None]
        fisher_info = 2.0 * np.dot(weighted_jacobian.T, weighted_jacobian) 
        
        _, geomechanical_rigidity_weight, weathering_prior_weight = self._fetch_geophysical_priors(taxon_type, target_ssa, wavelengths, phase_angle, is_lab, target_name)
        scaled_tikhonov_coeff = 1e-4 / max(1, total_parameters - 1)
        
        prior_hessian = np.zeros((total_parameters, total_parameters))
        for i in range(total_parameters - 1):
            for j in range(total_parameters - 1):
                prior_hessian[i, j] = 2 * geomechanical_rigidity_weight * library_densities[i] * library_densities[j]
            prior_hessian[i, i] += 2 * scaled_tikhonov_coeff
            
        if is_lab: 
            prior_hessian[-1, -1] = 1.0 
        else:
            prior_hessian[-1, -1] = 2 * weathering_prior_weight
            
        map_hessian = fisher_info + prior_hessian
        
        active_parameters = []
        for i in range(total_parameters - 1):
            if state_vector[i] >= 1e-4:
                active_parameters.append(i)
        if not is_lab:
            active_parameters.append(total_parameters - 1)
            
        active_count = len(active_parameters)
        if active_count == 0:
            return np.eye(total_parameters) * 1e-4
            
        active_hessian_block = map_hessian[active_parameters, :][:, active_parameters]
        
        constraint_gradient = np.zeros(active_count)
        for idx, act_idx in enumerate(active_parameters):
            if act_idx < total_parameters - 1:
                constraint_gradient[idx] = 1.0
                
        augmented_hessian = np.zeros((active_count + 1, active_count + 1))
        augmented_hessian[:active_count, :active_count] = active_hessian_block + np.eye(active_count) * 1e-4
        augmented_hessian[:active_count, active_count] = constraint_gradient
        augmented_hessian[active_count, :active_count] = constraint_gradient
        
        try:
            active_covariance_block = np.linalg.pinv(augmented_hessian, rcond=1e-15)[:active_count, :active_count]
        except np.linalg.LinAlgError:
            return np.eye(total_parameters) * 1e-4
            
        full_covariance_matrix = np.zeros((total_parameters, total_parameters))
        for idx_i, act_i in enumerate(active_parameters):
            for idx_j, act_j in enumerate(active_parameters):
                full_covariance_matrix[act_i, act_j] = active_covariance_block[idx_i, idx_j]
                
        return full_covariance_matrix

    def invert_single_target(self, target_data):
        """Executes the complete spectrum inversion processing track for a single body record."""
        wavelengths_clean, reflectances_clean = self.filter_signal_noise(target_data['wav'], target_data['refl'])
        taxon_type, is_lab, phase_angle = target_data['taxonomy'], target_data['is_lab'], target_data['phase']
        
        phase_rad = np.radians(phase_angle)
        mu0_target, mu_target = (np.cos(phase_rad), 1.0) if is_lab else (np.cos(phase_rad / 2.0), np.cos(phase_rad / 2.0))
        
        ssa_estimate = self.transform_reflectance_to_w(reflectances_clean, mu0_target, mu_target, phase_angle, 'stony')
        idx_550 = np.argmin(np.abs(wavelengths_clean - 0.55))
        idx_100 = np.argmin(np.abs(wavelengths_clean - 1.0))
        idx_150 = np.argmin(np.abs(wavelengths_clean - 1.5))
        
        if idx_150 > idx_550 and idx_100 > idx_550:
            interpolation_fraction = (wavelengths_clean[idx_100] - wavelengths_clean[idx_550]) / (wavelengths_clean[idx_150] - wavelengths_clean[idx_550])
            continuum_baseline = ssa_estimate[idx_550] + interpolation_fraction * (ssa_estimate[idx_150] - ssa_estimate[idx_550])
            band_depth_1000nm = (continuum_baseline - ssa_estimate[idx_100]) / max(1e-4, continuum_baseline)
        else:
            band_depth_1000nm = 0.0
            
        is_stony = taxon_type in ['S', 'V', 'Q', 'E', 'A'] or (taxon_type in ['C', 'B', 'F'] and band_depth_1000nm > 0.02)
        active_library = self.silicate_library if is_stony else self.carbonaceous_library 
        composition_class = 'stony' if is_stony else 'primitive'
            
        if not is_lab:
            reflectances_clean = self.subtract_thermal_background(wavelengths_clean, reflectances_clean, phase_angle, target_data['r_helio'], target_data['pV'])
            reflectances_clean = self.evaluate_phase_reddening(wavelengths_clean, reflectances_clean, phase_angle, taxon_type)
            
        target_ssa = self.transform_reflectance_to_w(reflectances_clean, mu0_target, mu_target, phase_angle, composition_class)
        channel_weights = self.compute_telluric_mask(wavelengths_clean, str(target_data['name']))
        
        library_ssa = np.array([self.transform_reflectance_to_w(PchipInterpolator(active_library['wav_raw'], raw)(wavelengths_clean), mu0_target, mu_target, phase_angle, composition_class) for raw in active_library['spectra_raw']])
        library_densities = active_library['densities']
        
        total_endmembers = len(active_library['names'])
        parameter_bounds = [(0, 1) for _ in range(total_endmembers)] + [(0, 2.0)]
        equality_constraint = {'type': 'eq', 'fun': lambda x: np.sum(x[:-1]) - 1.0}
        
        best_optimization_result, minimized_cost = None, np.inf
        optimization_seeds = [np.append(np.ones(total_endmembers)/total_endmembers, 0.1)]
        for i in range(1, 8):
            fraction_seed = np.abs(np.sin(np.arange(total_endmembers) * i)) + 0.1
            fraction_seed /= np.sum(fraction_seed)
            weathering_seed = (i / 7.0) * 0.5
            optimization_seeds.append(np.append(fraction_seed, weathering_seed))
            
        for seed_vector in optimization_seeds:
            optimization_run = minimize(self.evaluate_map_cost, seed_vector, args=(target_ssa, library_ssa, library_densities, wavelengths_clean, channel_weights, taxon_type, is_lab, phase_angle, target_data['name']), method='SLSQP', bounds=parameter_bounds, constraints=equality_constraint)
            if optimization_run.success and optimization_run.fun < minimized_cost:
                minimized_cost = optimization_run.fun
                best_optimization_result = optimization_run
                
        optimal_fractions = best_optimization_result.x[:-1]
        fraction_covariance = self.derive_parameter_covariance(best_optimization_result.x, target_ssa, library_ssa, library_densities, wavelengths_clean, channel_weights, taxon_type, is_lab, phase_angle, target_data['name'])[:-1, :-1]
        
        grain_density = np.sum(optimal_fractions * library_densities)
        grain_density_err = np.sqrt(np.dot(library_densities.T, np.dot(fraction_covariance, library_densities)) + np.sum((optimal_fractions * active_library['sigmas'])**2))
        
        porosity = max(0.0, 1.0 - (target_data['rho_bulk'] / grain_density))
        porosity_err = np.sqrt((target_data['sigma_rho_bulk']/grain_density)**2 + (target_data['rho_bulk'] * grain_density_err / grain_density**2)**2)
            
        core_mass_frac = 0.0
        if target_data['rho_bulk'] > grain_density and is_stony:
            density_denominator = 7.80 - (grain_density * 0.95)
            if density_denominator > 0:
                core_mass_frac = np.clip(((target_data['rho_bulk'] - (grain_density * 0.95)) / density_denominator) * (7.80 / target_data['rho_bulk']), 0.0, 1.0)
                
        return {'Crustal_Grain_Density': grain_density, 'Crustal_Grain_Error': grain_density_err, 'Macroporosity_Fraction': porosity, 'Macroporosity_Error': porosity_err, 'Core_Mass_Fraction': core_mass_frac}

    def process_batch(self, targets_file_path):
        """Processes an input table of minor planetary bodies sequentially through the execution model."""
        df_batch = pd.read_csv(targets_file_path)
        wave_columns = [col for col in df_batch.columns if col.startswith('Wv_')]
        wavelength_references = np.array([float(col.split('_')[1]) for col in wave_columns])
        inversion_records = []
        
        raw_table_records = df_batch.to_dict('records')
        for record in raw_table_records:
            target_profile = {
                'name': record['Asteroid_Name'], 'is_lab': 'Lab' in str(record['Asteroid_Name']), 'taxonomy': record['Taxonomic_Type'],
                'pV': float(record['Albedo']), 'rho_bulk': float(record['Bulk_Density_gcm3']), 'sigma_rho_bulk': float(record['Bulk_Density_Uncertainty']),
                'phase': float(record['Phase_Angle_deg']), 'r_helio': float(record['Helio_Distance_AU']), 'wav': wavelength_references, 
                'refl': np.array([record[col] for col in wave_columns]).astype(float)
            }
            output_entry = {'Asteroid_Name': record['Asteroid_Name'], 'Taxonomic_Type': record['Taxonomic_Type'], 'Status': 'VALIDATION PASS', 'Exclusion_Reason': ''}
            try:
                if self.assert_boundary_constraints(target_profile):
                    output_entry.update(self.invert_single_target(target_profile))
            except Exception as execution_exception:
                output_entry.update({'Status': 'EXCLUDED' if isinstance(execution_exception, ValueError) else 'FAILED', 'Exclusion_Reason': str(execution_exception), 'Crustal_Grain_Density': 0.0, 'Crustal_Grain_Error': 0.0, 'Macroporosity_Fraction': 0.0, 'Macroporosity_Error': 0.0, 'Core_Mass_Fraction': 0.0})
            inversion_records.append(output_entry)
        return pd.DataFrame(inversion_records)