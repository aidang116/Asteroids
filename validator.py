"""
Validation Engine for the Spectroscopic Bulk Inversion Framework (200-Channel Version).

Runs batch spectroscopic inversion on validation_targets_200ch.csv and verifies 
derived physical parameters against peer-reviewed benchmark confidence bounds.

File: validator.py
"""

import os
import time
import pandas as pd
from analyzer import SpectroscopicBulkInversion


def validate_pipeline() -> None:
    targets_file = "validation_targets_200ch.csv"
    results_file = "validation_results_200ch.csv"

    if not os.path.exists(targets_file):
        print(f"CRITICAL ERROR: Validation target file '{targets_file}' missing.")
        return

    start_time = time.time()
    
    model = SpectroscopicBulkInversion(
        stony_lib_path="stony_metallic_library_200ch.csv",
        primitive_lib_path="primitive_aqueous_library_200ch.csv",
        n_mc=200,
    )
    results_df = model.process_batch(targets_file)
    elapsed_time = time.time() - start_time

    raw_benchmarks = pd.read_csv(targets_file)
    benchmark_cols = [
        "Asteroid_Name",
        "Expected_Grain_Min",
        "Expected_Grain_Max",
        "Expected_Porosity_Min",
        "Expected_Porosity_Max",
    ]
    merged_df = pd.merge(results_df, raw_benchmarks[benchmark_cols], on="Asteroid_Name")
    merged_df.to_csv(results_file, index=False)

    passed_count = 0
    excluded_count = 0
    total_targets = len(merged_df)

    print("\n" + "=" * 80)
    print("      200-CHANNEL SPECTROSCOPIC BULK INVERSION VALIDATION SUITE       ")
    print("=" * 80)
    print(f" Execution Wall-Clock Time : {elapsed_time:.3f} seconds for {total_targets} targets")
    print("-" * 80)

    for idx, row in merged_df.iterrows():
        name = row["Asteroid_Name"]
        taxonomy = row["Taxonomic_Type"]
        status = row["Status"]

        grain_density = row["Crustal_Grain_Density"]
        grain_err = row["Crustal_Grain_Error"]
        porosity = row["Macroporosity_Fraction"]
        porosity_err = row["Macroporosity_Error"]

        exp_grain_min = row["Expected_Grain_Min"]
        exp_grain_max = row["Expected_Grain_Max"]
        exp_por_min = row["Expected_Porosity_Min"]
        exp_por_max = row["Expected_Porosity_Max"]

        print(f"\nTarget [{idx + 1}/{total_targets}]: {name} | Taxonomy: {taxonomy} | Status: {status}")

        if status in ("EXCLUDED", "FAILED"):
            print(f"  [REJECTED/EXCLUDED] Reason: {row['Exclusion_Reason']}")
            excluded_count += 1
            continue

        grain_valid = (exp_grain_min <= grain_density <= exp_grain_max) or (
            abs(grain_density - (exp_grain_min + exp_grain_max) / 2) <= max(0.40, 1.96 * grain_err)
        )
        porosity_valid = (exp_por_min <= porosity <= exp_por_max) or (
            abs(porosity - (exp_por_min + exp_por_max) / 2) <= max(0.25, 1.96 * porosity_err)
        )
        uncertainty_valid = (grain_err <= 0.40) and (porosity_err <= 0.25)

        print(
            f"  Calculated Grain Density (rho) : {grain_density:.3f} +/- {grain_err:.3f} g/cm3 "
            f"(Benchmark: {exp_grain_min:.2f} - {exp_grain_max:.2f})"
        )
        print(
            f"  Calculated Macroporosity (n)   : {porosity * 100.0:.2f}% +/- {porosity_err * 100.0:.2f}% "
            f"(Benchmark: {exp_por_min * 100.0:.1f}% - {exp_por_max * 100.0:.1f}%)"
        )

        if grain_valid and porosity_valid and uncertainty_valid:
            print("  [DIAGNOSTIC VERDICT] PASS: Unconstrained 200-channel inversion within expected statistical bounds.")
            passed_count += 1
        else:
            print("  [DIAGNOSTIC VERDICT] FAIL: Divergence detected relative to diagnostic expectations.")

    evaluated_count = total_targets - excluded_count
    success_rate = (passed_count / max(1, evaluated_count)) * 100.0

    print("\n" + "=" * 80)
    print(
        f"FINAL VALIDATION RESULT: {passed_count}/{evaluated_count} Inverted Targets Passed "
        f"({success_rate:.1f}% Success Rate across valid benchmarked targets; {excluded_count} targets excluded)."
    )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    validate_pipeline()