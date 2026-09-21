"""
Unbiased Execution Driver for Spectroscopic Bulk Inversion Framework.

Processes expanded target datasets (expanded_targets_200ch.csv), executes
200-channel unmixing and Monte Carlo error propagation, logs performance,
and outputs batch inversion results to a structured CSV without requiring
accepted benchmark values.

File: driver.py
"""

import os
import sys
import time
import logging
import pandas as pd
from analyzer import SpectroscopicBulkInversion, configure_logger


def run_bulk_inversion_driver(
    input_csv: str = "expanded_targets_200ch.csv",
    output_csv: str = "batch_inversion_results.csv",
    stony_lib: str = "stony_metallic_library_200ch.csv",
    primitive_lib: str = "primitive_aqueous_library_200ch.csv",
    log_file: str = "driver_execution.log",
    n_mc: int = 200,
) -> None:
    """Executes batch spectroscopic bulk inversion and records output telemetry."""
    logger = configure_logger(log_path=log_file, verbose=True)

    if not os.path.exists(input_csv):
        logger.error(f"CRITICAL ERROR: Input dataset file '{input_csv}' not found.")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("STARTING BATCH SPECTROSCOPIC BULK INVERSION ENGINE")
    logger.info(f"  Input Dataset : {input_csv}")
    logger.info(f"  Output File   : {output_csv}")
    logger.info(f"  Stony Library : {stony_lib}")
    logger.info(f"  Prim Library  : {primitive_lib}")
    logger.info(f"  Monte Carlo N : {n_mc}")
    logger.info("=" * 80)

    start_time = time.time()

    # Initialize unmixing inversion engine
    engine = SpectroscopicBulkInversion(
        stony_lib_path=stony_lib,
        primitive_lib_path=primitive_lib,
        logger=logger,
        verbose=True,
        n_mc=n_mc,
    )

    # Process batch targets
    results_df = engine.process_batch(input_csv)
    elapsed_time = time.time() - start_time

    if results_df.empty:
        logger.error("Inversion batch returned empty DataFrame. Halting process.")
        return

    # Write output dataset
    results_df.to_csv(output_csv, index=False)

    total_targets = len(results_df)
    passed_targets = len(results_df[results_df["Status"] == "VALIDATION PASS"])
    excluded_targets = len(results_df[results_df["Status"] == "EXCLUDED"])
    failed_targets = len(results_df[results_df["Status"] == "FAILED"])

    logger.info("\n" + "=" * 80)
    logger.info("          BATCH BULK INVERSION EXECUTION SUMMARY           ")
    logger.info("=" * 80)
    logger.info(f" Execution Wall-Clock Time : {elapsed_time:.3f} seconds")
    logger.info(f" Total Targets Processed   : {total_targets}")
    logger.info(f" Successfully Inverted    : {passed_targets}")
    logger.info(f" Physically Excluded       : {excluded_targets}")
    logger.info(f" Numerical Failures        : {failed_targets}")
    logger.info(f" Results Saved To          : {output_csv}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    run_bulk_inversion_driver()