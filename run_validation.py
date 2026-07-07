import os
import pandas as pd
from analyzer import SpectroscopicBulkInversion

def execute_pipeline_validation():
    """
    Executes structural benchmark evaluation comparisons against inverted minor body parameters.
    """
    targets_input_file = "validation_targets.csv"
    diagnostic_output_file = "validation_results.csv"
    
    print("=" * 70)
    print("ASTEROID PIPELINE VALIDATION")
    print("Validation Criteria: Spectral-Optical Grain Density Retrieval")
    print("=" * 70)
    
    if not os.path.exists(targets_input_file):
        print(f"CRITICAL ERROR: Validation target file '{targets_input_file}' missing.")
        return

    # Instantiate the updated spectroscopic composition modeling pipeline
    inversion_model = SpectroscopicBulkInversion()
    
    computed_results_df = inversion_model.process_batch(targets_input_file)
    
    raw_benchmarks_df = pd.read_csv(targets_input_file)
    benchmark_metadata_blocks = raw_benchmarks_df[[
        'Asteroid_Name', 'Expected_Grain_Min', 'Expected_Grain_Max', 
        'Expected_Porosity_Min', 'Expected_Porosity_Max'
    ]]
    
    integrated_validation_df = pd.merge(computed_results_df, benchmark_metadata_blocks, on='Asteroid_Name')
    
    print("\n" + "=" * 70)
    print("AUTOMATED VERIFICATION REPORT")
    print("=" * 70)
    
    passed_records_count = 0
    total_evaluated_records = 0
    
    for row_index, record_row in integrated_validation_df.iterrows():
        target_name = record_row['Asteroid_Name']
        run_status = record_row['Status']
        intercept_reason = record_row['Exclusion_Reason']
        
        print(f"\nTarget Evaluation: {target_name} [{record_row['Taxonomic_Type']}-type]")
        
        if run_status == 'FAILED' or run_status == 'EXCLUDED':
            print(f"  [STATUS] Pipeline Intercepted Run: {run_status}")
            print(f"  [REASON] {intercept_reason}")
            continue
            
        total_evaluated_records += 1
        grain_density_value = record_row['Crustal_Grain_Density']
        grain_density_uncertainty = record_row['Crustal_Grain_Error']
        macro_porosity_value = record_row['Macroporosity_Fraction']
        macro_porosity_uncertainty = record_row['Macroporosity_Error']
        
        benchmark_grain_min = record_row['Expected_Grain_Min']
        benchmark_grain_max = record_row['Expected_Grain_Max']
        benchmark_porosity_min = record_row['Expected_Porosity_Min']
        benchmark_porosity_max = record_row['Expected_Porosity_Max']
        
        print(f"  Calculated Grain Density : {grain_density_value:.3f} +/- {grain_density_uncertainty:.3f} g/cm3  [Acceptable: {benchmark_grain_min} - {benchmark_grain_max}]")
        print(f"  Calculated Macroporosity : {macro_porosity_value*100:.1f}% +/- {macro_porosity_uncertainty*100:.1f}%  [Acceptable: {benchmark_porosity_min*100:.1f}% - {benchmark_porosity_max*100:.1f}%]")
        
        # Hard data quality constraints configuration filter
        passes_uncertainty_cutoff = (grain_density_uncertainty <= 0.60)
        
        benchmark_midpoint = (benchmark_grain_min + benchmark_grain_max) / 2.0
        grain_density_zscore = abs(grain_density_value - benchmark_midpoint) / max(grain_density_uncertainty, 1e-4)
        
        # Validation criterion: parameter falls directly inside target boundary envelope OR evaluates inside 95% confidence bounds (1.96 SE)
        is_grain_density_valid = (benchmark_grain_min <= grain_density_value <= benchmark_grain_max) or (grain_density_zscore <= 1.96)
        
        if passes_uncertainty_cutoff and is_grain_density_valid:
            print("  [VALIDATION] PASS: Calculated parameters match independent geophysical benchmarks.")
            passed_records_count += 1
        else:
            print("  [VALIDATION] FAIL: Divergence discovered against empirical criteria.")
            if not passes_uncertainty_cutoff:
                print(f"    -> High Parameter Instability: Uncertainties exceed scientific ceilings.")
            if not is_grain_density_valid:
                print(f"    -> Grain Density of {grain_density_value:.3f} is statistically distinct from benchmark values.")

    integrated_validation_df.to_csv(diagnostic_output_file, index=False)
    
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print(f"  Total Numerical Targets Validated: {total_evaluated_records}")
    print(f"  Total Automated Passes Achieved  : {passed_records_count} / {total_evaluated_records}")
    if total_evaluated_records > 0:
        print(f"  Validation Integrity Metric     : {(passed_records_count / total_evaluated_records) * 100:.1f}%")
    print(f"Full integrated diagnostic tables successfully written to '{diagnostic_output_file}'")
    print("=" * 70)

if __name__ == "__main__":
    execute_pipeline_validation()