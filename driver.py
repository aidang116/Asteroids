"""
driver.py
---------
Main Multi-Core Execution Pipeline for Asteroid Spectral Analysis & Physical Density Inversion.
==========================================================================================
Uses Python's concurrent.futures.ProcessPoolExecutor to parallelize Monte Carlo optimizations 
across available CPU cores.
"""

import os
import warnings
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from analyzer import AsteroidAnalyzer

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.optimize._slsqp_py")

def process_single_asteroid(args):
    idx, row, df_stony, df_prim = args
    analyzer = AsteroidAnalyzer(df_stony, df_prim)
    res = analyzer.analyze_asteroid(row, n_mc=150, random_seed=42 + idx)
    
    all_minerals = df_stony['mineral_name'].tolist() + df_prim['mineral_name'].tolist()
    
    flat_row = {
        'Index': idx,
        'Asteroid_Name': res['Asteroid_Name'],
        'Status': res.get('Status', 'Analyzed'),
        'Exclusion_Reason': res.get('Exclusion_Reason', 'None'),
        'Taxonomic_Type': res.get('Taxonomic_Type', 'Unknown'),
        'Density_Source': res.get('Density_Source', 'Unknown'),
        'Library_Used': res.get('Library_Used', 'N/A'),
        'RMSE_Fit': res.get('RMSE_Fit', np.nan),
        'Chi2_Reduced': res.get('Chi2_Reduced', np.nan),
        'Band_I_Depth': res.get('Band_I_Depth', np.nan),
        'Band_II_Depth': res.get('Band_II_Depth', np.nan),
        'BAR_Approx': res.get('BAR_Approx', np.nan),
        'Space_Weathering_Slope_S': res.get('Space_Weathering_Slope_S', np.nan),
        'Space_Weathering_Slope_Err': res.get('Space_Weathering_Slope_Err', np.nan),
        'Grain_Density_gcm3': res.get('Grain_Density_gcm3', np.nan),
        'Grain_Density_Uncertainty': res.get('Grain_Density_Uncertainty', np.nan),
        'Bulk_Density_gcm3': res.get('Bulk_Density_gcm3', np.nan),
        'Bulk_Density_Uncertainty': res.get('Bulk_Density_Uncertainty', np.nan),
        'Macroporosity': res.get('Macroporosity', np.nan),
        'Macroporosity_Uncertainty': res.get('Macroporosity_Uncertainty', np.nan),
        'Macroporosity_Status': res.get('Macroporosity_Status', 'N/A')
    }
    
    min_abundances = res.get('Mineral_Abundances', {})
    min_errors = res.get('Mineral_Abundance_Errors', {})
    min_p16 = res.get('Mineral_Abundances_P16', {})
    min_p84 = res.get('Mineral_Abundances_P84', {})
    
    for min_name in all_minerals:
        flat_row[f"Abund_{min_name}"] = min_abundances.get(min_name, np.nan)
        flat_row[f"Err_{min_name}"] = min_errors.get(min_name, np.nan)
        flat_row[f"P16_{min_name}"] = min_p16.get(min_name, np.nan)
        flat_row[f"P84_{min_name}"] = min_p84.get(min_name, np.nan)
        
    return flat_row

def run_pipeline():
    print("===============================================================")
    print("      ASTEROID SPECTRAL UNMIXING & DENSITY INVERSION          ")
    print("===============================================================")
    
    ast_file = 'asteroid_data_complete.csv'
    stony_file = 'endmembers_stony_metallic.csv'
    prim_file = 'endmembers_primitive_aqueous.csv'
    
    for fn in [ast_file, stony_file, prim_file]:
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Required input dataset missing: {fn}")
            
    df_ast = pd.read_csv(ast_file)
    df_stony = pd.read_csv(stony_file)
    df_prim = pd.read_csv(prim_file)
    
    print(f"Loaded {len(df_ast)} asteroid spectra.")
    print(f"Loaded Stony sub-library ({len(df_stony)} endmembers).")
    print(f"Loaded Primitive sub-library ({len(df_prim)} endmembers).")
    
    print("\nProcessing spectra using Parallel ProcessPoolExecutor...")
    tasks = [(idx, row.to_dict(), df_stony, df_prim) for idx, row in df_ast.iterrows()]
    
    results_list = []
    analyzed_count = 0
    excluded_count = 0
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_asteroid, task) for task in tasks]
        for future in as_completed(futures):
            res_row = future.result()
            results_list.append(res_row)
            if res_row.get('Status') == 'Excluded':
                excluded_count += 1
            else:
                analyzed_count += 1
                
            if len(results_list) % 50 == 0 or len(results_list) == len(df_ast):
                print(f"  Completed {len(results_list)}/{len(df_ast)} asteroids...")
                
    df_results = pd.DataFrame(results_list).sort_values('Index').drop(columns=['Index'])
    output_csv = 'asteroid_analysis_results.csv'
    df_results.to_csv(output_csv, index=False)
    
    print(f"\nSaved analysis results to '{output_csv}'.")
    print(f"Summary: {analyzed_count} asteroids successfully analyzed, {excluded_count} asteroids excluded by QC rules.")

if __name__ == '__main__':
    run_pipeline()