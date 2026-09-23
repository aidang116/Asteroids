"""
analyzer.py
Spectral unmixing and physical property inversion routines for asteroid spectra.
"""
import numpy as np
import pandas as pd
import scipy.optimize as opt

def calculate_channel_noise(ssa_matrix, window_size=5):
    noise = np.zeros_like(ssa_matrix)
    pad = window_size // 2
    for i in range(ssa_matrix.shape[1]):
        spec = ssa_matrix[:, i]
        padded = np.pad(spec, (pad, pad), mode='edge')
        moving_std = np.array([np.std(padded[j:j + window_size]) for j in range(len(spec))])
        noise[:, i] = np.maximum(moving_std, 0.002)
    return noise

def reflectance_to_ssa(R):
    R_clipped = np.clip(R, 0.0, 0.9999)
    gamma = (1.0 - R_clipped) / (1.0 + 2.0 * R_clipped)
    return 1.0 - gamma**2

def ssa_to_reflectance(w):
    w_clipped = np.clip(w, 0.0, 0.9999)
    gamma = np.sqrt(np.maximum(0.0, 1.0 - w_clipped))
    return (1.0 - gamma) / (1.0 + 2.0 * gamma)

def calculate_phase_reddening_coefficient(albedo):
    a_clamped = np.clip(albedo, 0.02, 0.60)
    return 0.00012 + (0.00048 / (1.0 + np.exp(-18.0 * (a_clamped - 0.09))))

def calculate_thermal_emission_weights(helio_dist, albedo, wv_vals):
    if not np.isfinite(helio_dist) or helio_dist <= 0:
        helio_dist = 1.0
    if not np.isfinite(albedo) or albedo <= 0:
        albedo = 0.10
        
    S_0 = 1361.0
    sigma = 5.670374e-8
    eta = 1.20
    eps = 0.90
    
    T_ss = ((1.0 - albedo) * S_0 / (eta * eps * sigma * (helio_dist**2)))**0.25
    
    weights = np.ones_like(wv_vals)
    if T_ss > 280.0:
        thermal_factor = (T_ss / 300.0)**4 * np.maximum(0.0, wv_vals - 1.8)**2
        weights = 1.0 / (1.0 + 0.8 * thermal_factor)
        
    return weights

def calculate_l1_sparsity_penalty(albedo):
    a_clamped = np.clip(albedo, 0.02, 0.50)
    return 5e-5 + 1e-4 * np.exp(-12.0 * (a_clamped - 0.03))


class AsteroidAnalyzer:
    def __init__(self, df_stony, df_prim, wavelength_cols=None):
        self.df_stony = df_stony
        self.df_prim = df_prim
        
        if wavelength_cols is None:
            self.wv_cols = [c for c in df_stony.columns if c.startswith('Wv_')]
        else:
            self.wv_cols = wavelength_cols
            
        self.wv_vals = np.array([float(c.replace('Wv_', '')) for c in self.wv_cols])
        self.norm_idx = np.argmin(np.abs(self.wv_vals - 0.55))
        
        self.E_stony_raw = df_stony[self.wv_cols].values.T
        self.E_prim_raw = df_prim[self.wv_cols].values.T
        
        norm_stony = np.where(self.E_stony_raw[self.norm_idx, :] > 0, self.E_stony_raw[self.norm_idx, :], 1.0)
        norm_prim = np.where(self.E_prim_raw[self.norm_idx, :] > 0, self.E_prim_raw[self.norm_idx, :], 1.0)
        
        self.E_stony_norm = self.E_stony_raw / norm_stony
        self.E_prim_norm = self.E_prim_raw / norm_prim
        
        self.E_stony_ssa = reflectance_to_ssa(self.E_stony_norm)
        self.E_prim_ssa = reflectance_to_ssa(self.E_prim_norm)
        
        self.E_stony_err = calculate_channel_noise(self.E_stony_ssa)
        self.E_prim_err = calculate_channel_noise(self.E_prim_ssa)
        
        self.primitive_taxa = {'C', 'CH', 'CG', 'CGH', 'CB', 'B', 'F', 'CF', 'D', 'P', 'T'}
        self.stony_taxa = {'S', 'SQ', 'SL', 'SK', 'SA', 'SR', 'K', 'A', 'O', 'L', 'LD', 'V', 'Q', 'M', 'E', 'R'}

    def compute_band_parameters(self, wv_vals, y_norm):
        def get_smoothed_val(target_wv):
            idx = np.argmin(np.abs(wv_vals - target_wv))
            start_idx = max(0, idx - 1)
            end_idx = min(len(wv_vals), idx + 2)
            return float(np.median(y_norm[start_idx:end_idx]))

        val_075 = get_smoothed_val(0.75)
        val_150 = get_smoothed_val(1.50)
        val_240 = get_smoothed_val(2.40)
        
        idx_1um = np.argmin(np.abs(wv_vals - 1.05))
        idx_2um = np.argmin(np.abs(wv_vals - 2.00))
        val_105 = float(np.median(y_norm[max(0, idx_1um - 1):min(len(wv_vals), idx_1um + 2)]))
        val_200 = float(np.median(y_norm[max(0, idx_2um - 1):min(len(wv_vals), idx_2um + 2)]))

        cont_1 = np.interp(1.05, [0.75, 1.50], [val_075, val_150])
        cont_2 = np.interp(2.00, [1.50, 2.40], [val_150, val_240])
        
        depth_1 = max(0.0, 1.0 - (val_105 / cont_1)) if cont_1 > 0 else 0.0
        depth_2 = max(0.0, 1.0 - (val_200 / cont_2)) if cont_2 > 0 else 0.0
        
        bar_approx = (depth_2 / depth_1) if depth_1 > 0.02 else np.nan
        return depth_1, depth_2, bar_approx

    def fcls_unmix_weathered_ssa(self, E_ssa, y_ssa, wv_vals, weights=None, l1_penalty=1e-4, delta_huber=0.005, n_starts=5):
        n_endmembers = E_ssa.shape[1]
        w_shift = wv_vals - 0.55
        
        if weights is None:
            weights = np.ones_like(y_ssa)
            
        def obj(params):
            f = params[:n_endmembers]
            S = params[n_endmembers]
            weathering_continuum = np.exp(-S * w_shift)
            y_model = weathering_continuum * (E_ssa @ f)
            r = y_model - y_ssa
            
            huber_r = (delta_huber ** 2) * (np.sqrt(1.0 + (r / delta_huber)**2) - 1.0)
            soft_sum_penalty = 10.0 * (np.sum(f) - 1.0)**2
            return np.sum(weights * huber_r) + soft_sum_penalty
            
        bounds = [(0.0, 1.0) for _ in range(n_endmembers)] + [(-0.15, 1.50)]
        cons = ({'type': 'eq', 'fun': lambda p: np.sum(p[:n_endmembers]) - 1.0})
        
        best_res = None
        best_fun = np.inf
        
        s_inits = [-0.05, 0.0, 0.10, 0.30, 0.60]
        starts = [np.append(np.full(n_endmembers, 1.0 / n_endmembers), s_val) for s_val in s_inits[:n_starts]]
        
        for p0 in starts:
            res = opt.minimize(obj, p0, method='SLSQP', bounds=bounds, constraints=cons)
            if res.success and res.fun < best_fun:
                best_fun = res.fun
                best_res = res
                
        if best_res is not None:
            f = np.clip(best_res.x[:n_endmembers], 0, 1)
            f[f < 0.005] = 0.0
            f_sum = np.sum(f)
            if f_sum > 0:
                f /= f_sum
            S_opt = best_res.x[n_endmembers]
            return f, S_opt, best_res.fun
        else:
            f0 = np.full(n_endmembers, 1.0 / n_endmembers)
            return f0, 0.0, np.sum(weights * (E_ssa @ f0 - y_ssa)**2)

    def analyze_asteroid(self, ast_row, n_mc=150, random_seed=42, min_albedo=0.03, max_missing_ratio=0.20, max_rmse=0.15):
        asteroid_name = ast_row['Asteroid_Name'] if 'Asteroid_Name' in ast_row else 'Unknown_Asteroid'
        tax_type = str(ast_row['Taxonomic_Type']).strip() if 'Taxonomic_Type' in ast_row else 'Unknown'
        density_source = str(ast_row['Density_Source']).strip() if 'Density_Source' in ast_row else 'Unknown'
        phase_angle = float(ast_row['Phase_Angle_deg']) if 'Phase_Angle_deg' in ast_row else 0.0
        helio_dist = float(ast_row['Helio_Distance_AU']) if 'Helio_Distance_AU' in ast_row else 1.0
        albedo = float(ast_row['Albedo']) if 'Albedo' in ast_row else np.nan
        
        def make_exclusion_result(reason):
            return {
                'Asteroid_Name': asteroid_name,
                'Status': 'Excluded',
                'Exclusion_Reason': reason,
                'Taxonomic_Type': tax_type,
                'Density_Source': density_source,
                'Library_Used': 'N/A',
                'RMSE_Fit': np.nan,
                'Chi2_Reduced': np.nan,
                'Band_I_Depth': np.nan,
                'Band_II_Depth': np.nan,
                'BAR_Approx': np.nan,
                'Space_Weathering_Slope_S': np.nan,
                'Space_Weathering_Slope_Err': np.nan,
                'Grain_Density_gcm3': np.nan,
                'Grain_Density_Uncertainty': np.nan,
                'Bulk_Density_gcm3': float(ast_row['Bulk_Density_gcm3']) if 'Bulk_Density_gcm3' in ast_row else np.nan,
                'Bulk_Density_Uncertainty': float(ast_row['Bulk_Density_Uncertainty']) if 'Bulk_Density_Uncertainty' in ast_row else np.nan,
                'Macroporosity': np.nan,
                'Macroporosity_Uncertainty': np.nan,
                'Macroporosity_Status': 'Excluded',
                'Mineral_Abundances': {},
                'Mineral_Abundance_Errors': {},
                'Mineral_Abundances_P16': {},
                'Mineral_Abundances_P84': {}
            }

        if np.isnan(albedo):
            return make_exclusion_result('Missing or unmeasured albedo value')
        elif albedo < min_albedo:
            return make_exclusion_result(f'Albedo too low for feature discrimination ({albedo:.4f} < {min_albedo})')

        if isinstance(ast_row, dict):
            y_raw = np.array([float(ast_row[c]) for c in self.wv_cols])
        else:
            y_raw = ast_row[self.wv_cols].values.astype(float)

        n_channels = len(self.wv_cols)
        nan_count = np.sum(np.isnan(y_raw))
        missing_ratio = nan_count / float(n_channels)
        
        if missing_ratio > max_missing_ratio:
            return make_exclusion_result(f'Incomplete spectral data ({nan_count}/{n_channels} missing channels)')

        valid_raw = y_raw[np.isfinite(y_raw)]
        if len(valid_raw) == 0 or np.max(valid_raw) <= 0 or np.min(valid_raw) < 0:
            return make_exclusion_result('Invalid spectral reflectance values')

        norm_val = y_raw[self.norm_idx] if self.norm_idx < len(y_raw) else np.nan
        if np.isnan(norm_val) or norm_val <= 0:
            norm_val = np.nanmean(valid_raw) if len(valid_raw) > 0 else np.nan
            if np.isnan(norm_val) or norm_val <= 0:
                return make_exclusion_result('Normalization anchor wavelength (0.55 um) is invalid')

        rho_bulk = float(ast_row['Bulk_Density_gcm3']) if 'Bulk_Density_gcm3' in ast_row else np.nan
        rho_bulk_err = float(ast_row['Bulk_Density_Uncertainty']) if 'Bulk_Density_Uncertainty' in ast_row else 0.0
        if np.isnan(rho_bulk_err):
            rho_bulk_err = 0.0

        if np.isnan(rho_bulk) or rho_bulk <= 0:
            return make_exclusion_result('Invalid or missing bulk density')
        elif rho_bulk > 8.0:
            return make_exclusion_result(f'Unphysical bulk density ({rho_bulk:.2f} g/cm3)')
        elif rho_bulk_err > 0 and (rho_bulk_err / rho_bulk) > 0.50:
            return make_exclusion_result('Excessive bulk density uncertainty (> 50% relative error)')

        placeholder_keywords = ['model', 'neowise', 'placeholder', 'default', 'assumed', 'estimated']
        is_placeholder_density = any(kw in density_source.lower() for kw in placeholder_keywords)

        y_norm = y_raw / norm_val
        b1_depth, b2_depth, bar_approx = self.compute_band_parameters(self.wv_vals, y_norm)

        if b1_depth < 0.015 and b2_depth < 0.015:
            return make_exclusion_result(f'Featureless spectrum lacking bands (b1={b1_depth:.4f}, b2={b2_depth:.4f})')
        
        if np.isfinite(phase_angle) and phase_angle > 5.0:
            k_phase = calculate_phase_reddening_coefficient(albedo)
            phase_slope_corr = k_phase * (phase_angle - 5.0)
            phase_continuum = np.exp(-phase_slope_corr * (self.wv_vals - 0.55))
            y_norm = y_norm * phase_continuum

        thermal_weights = calculate_thermal_emission_weights(helio_dist, albedo, self.wv_vals)
        telluric_mask = ((self.wv_vals >= 1.35) & (self.wv_vals <= 1.42)) | ((self.wv_vals >= 1.80) & (self.wv_vals <= 1.92))
        weights = thermal_weights.copy()
        weights[telluric_mask] *= 0.50

        l1_penalty = calculate_l1_sparsity_penalty(albedo)
        y_ssa = reflectance_to_ssa(y_norm)
        
        valid_st = np.isfinite(y_ssa) & np.all(np.isfinite(self.E_stony_ssa), axis=1)
        valid_pr = np.isfinite(y_ssa) & np.all(np.isfinite(self.E_prim_ssa), axis=1)
        
        f_st, S_st, rss_st = self.fcls_unmix_weathered_ssa(self.E_stony_ssa[valid_st], y_ssa[valid_st], self.wv_vals[valid_st], weights=weights[valid_st], l1_penalty=l1_penalty)
        f_pr, S_pr, rss_pr = self.fcls_unmix_weathered_ssa(self.E_prim_ssa[valid_pr], y_ssa[valid_pr], self.wv_vals[valid_pr], weights=weights[valid_pr], l1_penalty=l1_penalty)
        
        n_st = np.sum(valid_st)
        n_pr = np.sum(valid_pr)
        k_st = self.E_stony_ssa.shape[1] + 1
        k_pr = self.E_prim_ssa.shape[1] + 1
        
        aic_st = n_st * np.log(max(rss_st / n_st, 1e-10)) + 2 * k_st
        aic_pr = n_pr * np.log(max(rss_pr / n_pr, 1e-10)) + 2 * k_pr
        
        rmse_st = np.sqrt(rss_st / n_st)
        rmse_pr = np.sqrt(rss_pr / n_pr)
        
        tax = tax_type.upper()
        if tax in self.primitive_taxa:
            lib_used = "Primitive/Aqueous"
            df_lib = self.df_prim
            E_ssa, E_err = self.E_prim_ssa, self.E_prim_err
            valid_m = valid_pr
            f_best, S_best, rmse_best = f_pr, S_pr, rmse_pr
            weights_m = weights[valid_pr]
            k_best = k_pr
        elif tax in self.stony_taxa:
            lib_used = "Stony/Metallic"
            df_lib = self.df_stony
            E_ssa, E_err = self.E_stony_ssa, self.E_stony_err
            valid_m = valid_st
            f_best, S_best, rmse_best = f_st, S_st, rmse_st
            weights_m = weights[valid_st]
            k_best = k_st
        else:
            if aic_st <= aic_pr:
                lib_used = "Stony/Metallic (Auto-AIC)"
                df_lib = self.df_stony
                E_ssa, E_err = self.E_stony_ssa, self.E_stony_err
                valid_m = valid_st
                f_best, S_best, rmse_best = f_st, S_st, rmse_st
                weights_m = weights[valid_st]
                k_best = k_st
            else:
                lib_used = "Primitive/Aqueous (Auto-AIC)"
                df_lib = self.df_prim
                E_ssa, E_err = self.E_prim_ssa, self.E_prim_err
                valid_m = valid_pr
                f_best, S_best, rmse_best = f_pr, S_pr, rmse_pr
                weights_m = weights[valid_pr]
                k_best = k_pr
                
        if rmse_best > max_rmse:
            return make_exclusion_result(f'Poor unmixing fit (RMSE = {rmse_best:.4f})')

        densities = df_lib['density_gcm3'].values.astype(float)
        density_errs_clean = np.nan_to_num(df_lib['density_err_gcm3'].values.astype(float), nan=0.0)
        
        mass_weights_unnorm = f_best * densities
        w_best = mass_weights_unnorm / np.sum(mass_weights_unnorm)
        rho_grain = 1.0 / np.sum(w_best / densities)
        
        if is_placeholder_density:
            macroporosity = np.nan
            macroporosity_status = 'Not Calculated (Placeholder bulk density)'
        elif np.isfinite(rho_bulk) and rho_grain > 0:
            if rho_bulk > rho_grain:
                macroporosity = np.nan
                macroporosity_status = 'Unconstrained (Bulk density exceeds grain density)'
            else:
                macroporosity = 1.0 - (rho_bulk / rho_grain)
                macroporosity_status = 'Valid'
        else:
            macroporosity = np.nan
            macroporosity_status = 'Invalid Input Data'
        
        weathering_cont = np.exp(-S_best * (self.wv_vals[valid_m] - 0.55))
        residuals = y_ssa[valid_m] - (weathering_cont * (E_ssa[valid_m] @ f_best))
        
        n_valid = np.sum(valid_m)
        dof = max(1, n_valid - k_best)
        sigma_res = np.std(residuals)
        if sigma_res <= 0:
            sigma_res = 0.005
        chi2_red = np.sum(weights_m * (residuals / sigma_res)**2) / dof

        if len(residuals) > 2:
            phi_ar1 = np.clip(np.corrcoef(residuals[:-1], residuals[1:])[0, 1], -0.8, 0.8)
            if np.isnan(phi_ar1):
                phi_ar1 = 0.65
        else:
            phi_ar1 = 0.65
            
        np.random.seed(random_seed)
        mc_f, mc_S, mc_rho_grain, mc_phi = [], [], [], []
        
        for _ in range(n_mc):
            white_noise = np.random.normal(0, sigma_res, size=n_valid)
            ar_noise = np.zeros_like(white_noise)
            ar_noise[0] = white_noise[0]
            for t in range(1, len(ar_noise)):
                ar_noise[t] = phi_ar1 * ar_noise[t-1] + np.sqrt(1.0 - phi_ar1**2) * white_noise[t]
                
            y_pert = np.clip(y_ssa[valid_m] + ar_noise, 0.0, 1.0)
            E_pert = np.maximum(E_ssa[valid_m] + np.random.normal(0, E_err[valid_m]), 0.0)
            rho_pert = np.random.normal(densities, density_errs_clean)
            rho_pert = np.maximum(rho_pert, 0.5)
            
            f_p, S_p, _ = self.fcls_unmix_weathered_ssa(E_pert, y_pert, self.wv_vals[valid_m], weights=weights_m, l1_penalty=l1_penalty)
            
            w_p = (f_p * rho_pert) / np.sum(f_p * rho_pert)
            rho_g_p = 1.0 / np.sum(w_p / rho_pert)
            
            if not is_placeholder_density and np.isfinite(rho_bulk):
                rho_b_p = np.random.normal(rho_bulk, rho_bulk_err) if rho_bulk_err > 0 else rho_bulk
                phi_p = (1.0 - (rho_b_p / rho_g_p)) if (rho_g_p > 0 and rho_b_p <= rho_g_p) else np.nan
            else:
                phi_p = np.nan
            
            mc_f.append(f_p)
            mc_S.append(S_p)
            mc_rho_grain.append(rho_g_p)
            mc_phi.append(phi_p)
            
        mc_f = np.array(mc_f)
        f_err = np.std(mc_f, axis=0)
        f_p16 = np.percentile(mc_f, 16, axis=0)
        f_p84 = np.percentile(mc_f, 84, axis=0)
        
        valid_phi = np.array(mc_phi)[np.isfinite(mc_phi)]
        macroporosity_err = np.std(valid_phi, ddof=0) if len(valid_phi) > 1 else np.nan
        
        return {
            'Asteroid_Name': asteroid_name,
            'Status': 'Analyzed',
            'Exclusion_Reason': 'None',
            'Taxonomic_Type': tax_type,
            'Density_Source': density_source,
            'Library_Used': lib_used,
            'RMSE_Fit': rmse_best,
            'Chi2_Reduced': chi2_red,
            'Band_I_Depth': b1_depth,
            'Band_II_Depth': b2_depth,
            'BAR_Approx': bar_approx,
            'Space_Weathering_Slope_S': S_best,
            'Space_Weathering_Slope_Err': np.std(mc_S),
            'Grain_Density_gcm3': rho_grain,
            'Grain_Density_Uncertainty': np.std(mc_rho_grain),
            'Bulk_Density_gcm3': rho_bulk,
            'Bulk_Density_Uncertainty': rho_bulk_err,
            'Macroporosity': macroporosity,
            'Macroporosity_Uncertainty': macroporosity_err,
            'Macroporosity_Status': macroporosity_status,
            'Mineral_Abundances': dict(zip(df_lib['mineral_name'], f_best)),
            'Mineral_Abundance_Errors': dict(zip(df_lib['mineral_name'], f_err)),
            'Mineral_Abundances_P16': dict(zip(df_lib['mineral_name'], f_p16)),
            'Mineral_Abundances_P84': dict(zip(df_lib['mineral_name'], f_p84))
        }