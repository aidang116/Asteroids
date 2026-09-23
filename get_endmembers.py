import os
import re
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# 1. Load density reference data robustly from densities.csv
def load_density_reference(csv_path="densities.csv"):
    density_table = {}
    if not os.path.exists(csv_path):
        print(f"[WARNING] '{csv_path}' not found. Densities will default to NaN.")
        return density_table
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        _ = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                name = parts[0]
                err_str = parts[-2]
                dens_str = parts[-3]
                
                try:
                    dens = float(dens_str)
                except ValueError:
                    dens = np.nan
                    
                try:
                    err = float(err_str)
                except ValueError:
                    err = np.nan
                    
                density_table[name] = {"density": dens, "err": err}
    return density_table

DENSITY_REF = load_density_reference("densities.csv")

# 2. RELAB mapping by code identifiers with regex boundaries
RELAB_PAIRINGS = {
    "Olivine_Fa50_Fe": r"\bc1ol02\b",
    "Orthopyroxene_En86": r"\bc1pp01\b",
    "Pigeonite_LowCa": r"\bc1px\b",
    "Plagioclase_Anorthite": r"\bc1pl01\b",
    "Mg_Al_Spinel": r"\bc1sp01\b",
    "FeNi_Metal_Kamacite": r"\bcasc06\b",
    "Troilite_FeS": r"\bc1su01\b",
    "Saponite_Smectite": r"\bc1cl02\b",
    "Cronstedtite_FePhyllosilicate": r"\bc1cr01\b",
    "Epsomite_Mg_Sulfate": r"\bc1sf01\b",
    "Carbonaceous_Matrix_1": r"\bc1cm01\b",
    "Magnesite_Carbonate": r"\bcabe256\b",
    "Carbonaceous_Matrix": r"\bcaca00\b"
}

# 3. Validated USGS full-path mapping
USGS_FULL_PATHS = {
    "Diopside_HighCa": "usgs_splib07/ASCIIdata/ASCIIdata_splib07a/ChapterM_Minerals/splib07a_Diopside_HS15.3B_Pyroxene_BECKc_AREF.txt",
    "Dolomite_Carbonate": "usgs_splib07/ASCIIdata/ASCIIdata_splib07a/ChapterM_Minerals/splib07a_Dolomite_HS102.3B_BECKb_AREF.txt",
    "Magnetite_Fe3O4": "usgs_splib07/ASCIIdata/ASCIIdata_splib07a/ChapterM_Minerals/splib07a_Magnetite_HS195.3B_BECKb_AREF.txt",
    "Water_Ice_H2O": "usgs_splib07/ASCIIdata/ASCIIdata_splib07a/ChapterL_Liquids/splib07a_H2O-Ice_GDS136_77K_BECKa_AREF.txt",
    "Phyllosilicate_Serpentine": "usgs_splib07/ASCIIdata/ASCIIdata_splib07a/ChapterM_Minerals/splib07a_Serpentine_HS318.3B_ASDFRC_AREF.txt",
    "Plagioclase_Anorthite":"usgs_splib07/ASCIIdata/ASCIIdata_splib07a/ChapterM_Minerals/splib07a_Anorthite_HS201.3B_Plagio_NIC4aa_RREF.txt",
    "Clinopyroxene_Augite": "usgs_splib07/ASCIIdata/ASCIIdata_splib07a/ChapterM_Minerals/splib07a_Augite_WS588_Pyroxene_NIC4cbb_RREF.txt",
    "Olivine_Fo90_Mg": "usgs_splib07/ASCIIdata/ASCIIdata_splib07a/ChapterM_Minerals/splib07a_Olivine_GDS71.b_Fo91_lt60um_NIC4cbu_RREF.txt",
}

USGS_WAVELENGTH_FILES = {
    "ASD": "usgs_splib07/ASCIIdata/ASCIIdata_splib07a/Chapter0_Wavelengths/splib07a_Wavelengths_ASD_0.2_to_3.0_microns.txt",
    "BECK": "usgs_splib07/ASCIIdata/ASCIIdata_splib07a/Chapter0_Wavelengths/splib07a_Wavelengths_BECK_0.2_to_3.0_microns.txt",
    "NIC": "usgs_splib07/ASCIIdata/ASCIIdata_splib07a/Chapter0_Wavelengths/splib07a_Wavelengths_NIC4_0.2_to_3.0_microns.txt"
}

WAVELENGTH_COLUMNS = [f"Wv_{w:.4f}" for w in np.linspace(0.4000, 2.5000, 161)]
TARGET_WAVELENGTH_VALS = np.array([float(col.replace("Wv_", "")) for col in WAVELENGTH_COLUMNS])

def load_usgs_wavelengths(root_dir="."):
    wavelength_grids = {}
    for key, rel_path in USGS_WAVELENGTH_FILES.items():
        file_path = os.path.join(root_dir, rel_path)
        if os.path.exists(file_path):
            waves = []
            with open(file_path, 'r', encoding='latin-1') as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str or line_str.startswith(('#', ';', 'Record=')):
                        continue
                    try:
                        val = float(line_str)
                        if val > 0:
                            waves.append(val)
                    except ValueError:
                        continue
            wavelength_grids[key] = np.array(waves)
    return wavelength_grids

def parse_relab_file(file_path):
    waves, refl = [], []
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith(('#', ';', '!', 'Name:', 'Dataset:', 'Description:')):
                    continue
                
                floats = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line_str)
                if len(floats) >= 2:
                    try:
                        w, r = float(floats[0]), float(floats[1])
                        if r > -1e30 and w > 0:
                            waves.append(w)
                            refl.append(r)
                    except ValueError:
                        continue
        return np.array(waves), np.array(refl)
    except Exception as e:
        print(f"  [ERROR RELAB] Could not parse {file_path}: {e}")
        return np.array([]), np.array([])

def parse_usgs_file(file_path, wavelength_grids):
    refls = []
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            for line_idx, line in enumerate(f):
                line_str = line.strip()
                if not line_str or line_idx == 0 or 'Record=' in line_str or 'Name:' in line_str:
                    continue
                try:
                    val = float(line_str)
                    if val > -1e30:
                        refls.append(val)
                except ValueError:
                    continue
                    
        refl_array = np.array(refls)
        if len(refl_array) == 0:
            return np.array([]), np.array([])
            
        filename_upper = os.path.basename(file_path).upper()
        w_grid = None
        for key in ["BECK", "ASD", "NIC"]:
            if key in filename_upper and key in wavelength_grids:
                if len(wavelength_grids[key]) == len(refl_array):
                    w_grid = wavelength_grids[key]
                    break

        if w_grid is None:
            w_grid = np.linspace(0.4000, 2.5000, len(refl_array))

        return w_grid, refl_array
    except Exception as e:
        print(f"  [ERROR USGS] Could not parse {file_path}: {e}")
        return np.array([]), np.array([])

def cubic_spline_interpolate_spectrum(raw_w, raw_r, target_vals):
    valid_mask = np.isfinite(raw_w) & np.isfinite(raw_r)
    w_clean, r_clean = raw_w[valid_mask], raw_r[valid_mask]
    if len(w_clean) < 4:
        return np.full(len(target_vals), np.nan)
        
    sort_idx = np.argsort(w_clean)
    w_sorted, r_sorted = w_clean[sort_idx], r_clean[sort_idx]
    
    try:
        cs = CubicSpline(w_sorted, r_sorted, extrapolate=False)
        interpolated = cs(target_vals)
        min_w, max_w = w_sorted[0], w_sorted[-1]
        interpolated[(target_vals < min_w) | (target_vals > max_w)] = np.nan
        return interpolated
    except Exception:
        return np.full(len(target_vals), np.nan)

def run_full_ingestion(root_dir="."):
    usgs_wv_grids = load_usgs_wavelengths(root_dir)
    stony_minerals = [
        "Olivine_Fo90_Mg", "Olivine_Fa50_Fe", "Orthopyroxene_En86", "Pigeonite_LowCa",
        "Clinopyroxene_Augite", "Diopside_HighCa", "Plagioclase_Anorthite", "Mg_Al_Spinel",
        "FeNi_Metal_Kamacite", "Troilite_FeS", "Dolomite_Carbonate", "Magnetite_Fe3O4"
    ]
    
    base_cols = ['mineral_name', 'density_gcm3', 'density_err_gcm3', 'database_source', 'sample_id'] + WAVELENGTH_COLUMNS
    collected_rows = {
        "endmembers_stony_metallic.csv": [],
        "endmembers_primitive_aqueous.csv": []
    }

    found_minerals = set()
    all_targets = list(RELAB_PAIRINGS.keys()) + list(USGS_FULL_PATHS.keys())

    print("\n--- PROCESSING USGS FILES ---")
    for mineral, rel_path in USGS_FULL_PATHS.items():
        file_path = os.path.join(root_dir, rel_path)
        if not os.path.exists(file_path):
            print(f"  [MISSING USGS] Path not found for '{mineral}': {file_path}")
            continue
            
        filename = os.path.basename(file_path)
        target_csv = "endmembers_stony_metallic.csv" if mineral in stony_minerals else "endmembers_primitive_aqueous.csv"
        
        raw_w, raw_r = parse_usgs_file(file_path, usgs_wv_grids)
        if len(raw_w) < 4:
            print(f"  [SKIPPED USGS] Insufficient points in '{filename}'")
            continue
            
        interpolated_values = cubic_spline_interpolate_spectrum(raw_w, raw_r, TARGET_WAVELENGTH_VALS)
        if np.isnan(interpolated_values).all():
            print(f"  [SKIPPED USGS] Interpolation NaN for '{filename}'")
            continue
            
        found_minerals.add(mineral)
        dens_info = DENSITY_REF.get(mineral, {"density": np.nan, "err": np.nan})
        
        row_data = {
            'mineral_name': mineral,
            'density_gcm3': dens_info["density"],
            'density_err_gcm3': dens_info["err"],
            'database_source': "USGS_Splib07",
            'sample_id': os.path.splitext(filename)[0]
        }
        for col, val in zip(WAVELENGTH_COLUMNS, interpolated_values):
            row_data[col] = float(val) if not np.isnan(val) else np.nan
        
        collected_rows[target_csv].append(row_data)
        print(f"  [SUCCESS USGS] Loaded '{filename}' -> '{mineral}'")

    print("\n--- PROCESSING RELAB FILES ---")
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(('.txt', '.asc')):
                all_files.append(os.path.join(dirpath, fn))

    for file_path in all_files:
        filename = os.path.basename(file_path)
        filename_lower = filename.lower()
        
        if "errorbar" in filename_lower or "readme" in filename_lower or "usgs" in file_path.lower():
            continue
            
        matched_mineral = None
        for mineral, pattern in RELAB_PAIRINGS.items():
            if mineral in found_minerals:
                continue
            if re.search(pattern, filename_lower):
                matched_mineral = mineral
                break
        
        if not matched_mineral:
            continue
            
        target_csv = "endmembers_stony_metallic.csv" if matched_mineral in stony_minerals else "endmembers_primitive_aqueous.csv"
        
        raw_w, raw_r = parse_relab_file(file_path)
        if len(raw_w) < 4:
            continue
            
        interpolated_values = cubic_spline_interpolate_spectrum(raw_w, raw_r, TARGET_WAVELENGTH_VALS)
        if np.isnan(interpolated_values).all():
            continue
            
        found_minerals.add(matched_mineral)
        dens_info = DENSITY_REF.get(matched_mineral, {"density": np.nan, "err": np.nan})
        
        row_data = {
            'mineral_name': matched_mineral,
            'density_gcm3': dens_info["density"],
            'density_err_gcm3': dens_info["err"],
            'database_source': "RELAB",
            'sample_id': os.path.splitext(filename)[0]
        }
        for col, val in zip(WAVELENGTH_COLUMNS, interpolated_values):
            row_data[col] = float(val) if not np.isnan(val) else np.nan
        
        collected_rows[target_csv].append(row_data)
        print(f"  [SUCCESS RELAB] Loaded '{filename}' -> '{matched_mineral}'")

    for csv_name, rows in collected_rows.items():
        df_out = pd.DataFrame(rows, columns=base_cols)
        df_out.to_csv(csv_name, index=False, na_rep='')
        print(f"\nSaved {len(rows)} records to {csv_name}")

    print(f"\n--- INGESTION REPORT ---")
    print(f"Successfully processed: {len(found_minerals)} / {len(all_targets)} minerals.")
    missing = set(all_targets) - found_minerals
    if missing:
        print(f"Still missing: {sorted(list(missing))}")

if __name__ == "__main__":
    run_full_ingestion()