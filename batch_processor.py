import os
import pandas as pd
import numpy as np
import logging
from analyzer import AsteroidDensityAnalyzer

"""High-level batch runner for asteroid density analysis."""

def setup_logging():
    """Configure root logging to write a single execution trace file named 'log'."""
    logging.basicConfig(
        filename='log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    print("Logging engine initialized. System tracking set to target output file: 'log'")


def run_batch_analysis():
    """Load asteroid targets and execute the density analysis pipeline for each."""
    targets_file = "asteroid_analysis_targets.csv"
    output_file = "results.csv"

    if not os.path.exists(targets_file):
        print(f"Error: Could not find '{targets_file}' in the current directory.")
        return

    # Initialize log file configuration
    setup_logging()

    print(f"Loading targets from {targets_file}...")
    targets_df = pd.read_csv(targets_file)
    results_list = []
    
    print("Beginning automated batch spectral unmixing and density analysis...")
    print("-" * 95)
    
    for idx, row in targets_df.iterrows():
        name = row['Asteroid_Name']
        ast_type = row['Taxonomic_Type']
        bulk_density = float(row['Bulk_Density_gcm3'])
        bulk_uncertainty = float(row['Bulk_Density_Uncertainty'])
        
        target_spectrum = np.array([
            row['Wv_0.7'], 
            row['Wv_1.0'], 
            row['Wv_1.2'], 
            row['Wv_1.5'], 
            row['Wv_2.0']
        ])
        
        analyzer = AsteroidDensityAnalyzer(
            bulk_density=bulk_density,
            sigma_bulk_density=bulk_uncertainty
        )

        try:
            # Run the analyzer and preserve the asteroid name for detailed logging.
            analyzer.analyze_asteroid(target_spectrum, name=name)

            calculated_grain_density = analyzer.results['rho_grain']
            calculated_grain_uncertainty = analyzer.results['sigma_rho_grain']

            density_deficit = calculated_grain_density - bulk_density
            density_deficit_uncertainty = np.sqrt(
                calculated_grain_uncertainty**2 + bulk_uncertainty**2
            )

            results_list.append({
                "Asteroid_Name": name,
                "Taxonomic_Type": ast_type,
                "Calculated_Grain_Density_gcm3": round(calculated_grain_density, 3),
                "Calculated_Grain_Density_Uncertainty_gcm3": round(calculated_grain_uncertainty, 3),
                "Bulk_Density_gcm3": round(bulk_density, 3),
                "Bulk_Density_Uncertainty_gcm3": round(bulk_uncertainty, 3),
                "Grain_Minus_Bulk_gcm3": round(density_deficit, 3),
                "Grain_Minus_Bulk_Uncertainty_gcm3": round(density_deficit_uncertainty, 3)
            })
            print(
                f"[{idx+1:02d}/30] Processed: {name:<18} ({ast_type:<2}) | "
                f"Grain-Bulk: {density_deficit:.3f} +/- {density_deficit_uncertainty:.3f} g/cm^3"
            )

        except Exception as e:
            print(f"[{idx+1:02d}/30] Failed to process {name}. Check log file for traceback.")
            logging.getLogger("AsteroidDensityAnalyzer").error(
                f"Execution crash on {name}: {str(e)}",
                exc_info=True
            )
            continue

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_file, index=False)
    print("-" * 95)
    print(f"Batch execution finished.\n -> Quantitative Spreadsheet: '{output_file}'\n -> Step-by-Step Readout:     'log'")

if __name__ == "__main__":
    run_batch_analysis()