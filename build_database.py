import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

"""Build a spectral library CSV by matching RELAB data to density metadata."""

def build_asteroid_database(relab_directory, density_csv_path, output_csv_path):
    """Build a combined database of interpolated RELAB spectra and reference densities.

    This function scans a RELAB data directory for text files, identifies each
    spectrum by matching its filename to a mineral density search term, and
    interpolates valid spectral measurements onto the standard pipeline
    wavelengths used by the asteroid analysis routines.
    """
    TARGET_WAVELENGTHS = [0.7, 1.0, 1.2, 1.5, 2.0]

    try:
        density_df = pd.read_csv(density_csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the density mapping file at {density_csv_path}")
        return

    results = []

    search_path = os.path.join(relab_directory, "**", "*.txt")
    filepaths = glob.glob(search_path, recursive=True)

    print(f"Found {len(filepaths)} potential RELAB files. Beginning processing...")

    for filepath in filepaths:
        filename = os.path.basename(filepath).lower()
        
        # --- Step A: Match the filename to a density database mineral entry ---
        matched_row = None
        for _, row in density_df.iterrows():
            if str(row['Search_Term']).lower() in filename:
                matched_row = row
                break

        if matched_row is None:
            # Skip files that cannot be linked to a known density record.
            continue

        # --- Step B: Parse the RELAB file while ignoring malformed lines and comments ---
        try:
            data = pd.read_csv(
                filepath,
                sep=r'\s+',
                comment='#',
                header=None,
                on_bad_lines='skip'
            )
            data = data.apply(pd.to_numeric, errors='coerce').dropna()

            if data.shape[1] < 2 or len(data) < 10:
                # Not enough reliable data points for a valid spectrum.
                continue

            wv = data.iloc[:, 0].values
            ref = data.iloc[:, 1].values

            # Convert spectra in nanometers to micrometers if necessary.
            if np.mean(wv) > 100:
                wv = wv / 1000.0

            # --- Step C: Interpolate the spectrum onto the pipeline's wavelength grid ---
            f_interp = interp1d(
                wv,
                ref,
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            ref_interpolated = f_interp(TARGET_WAVELENGTHS)

            if np.isnan(ref_interpolated).any():
                # The spectrum does not cover all required target wavelengths.
                continue

            # --- Step D: Build a clean row and store it for CSV export ---
            row_dict = {
                'Mineral_Name': f"{matched_row['Search_Term'].capitalize()}_{filename.split('.')[0]}",
                'Density_gcm3': matched_row['Density_gcm3'],
                'Density_Uncertainty': matched_row['Density_Uncertainty']
            }

            for i, w in enumerate(TARGET_WAVELENGTHS):
                row_dict[f'Wv_{w}'] = round(ref_interpolated[i], 4)

            results.append(row_dict)

        except Exception:
            # Drop any file that fails parsing rather than stopping the build.
            continue

    # 4. Save to final CSV
    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_csv_path, index=False)
        print(f"Success! Processed {len(results)} valid spectra into {output_csv_path}")
    else:
        print("No valid spectra were processed. Check your density CSV search terms.")

# ==========================================
# HOW TO RUN
# ==========================================
if __name__ == "__main__":
    # You must provide these paths on your local machine
    RELAB_FOLDER = "./relab_raw_data/"
    DENSITY_MAPPING = "./mineral_densities.csv" 
    OUTPUT_FILE = "./relab_expanded.csv"
    
    # build_asteroid_database(RELAB_FOLDER, DENSITY_MAPPING, OUTPUT_FILE)