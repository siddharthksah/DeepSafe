#!/usr/bin/env python3
"""
DeepSafe Meta-Feature Generator
===============================

Orchestrates the generation of meta-feature datasets for training stacking ensembles.
This component acts as a data ingestion pipeline that:
1.  Scans a target directory for labeled media (Real/Fake).
2.  Queries the distributed model microservices to obtain base probability scores.
3.  Aggregates these scores into a structured feature matrix (CSV) for the meta-learner.

Architectural Note:
This script is designed to be fault-tolerant. If a specific model microservice is unreachable
or fails for a subset of files, the pipeline continues, recording NaNs for those features.
This ensures that a single model failure does not halt the entire training data generation process,
though downstream imputers must handle these missing values.
"""

import os
import sys
import argparse
import json
import time
import gc
from typing import List, Dict, Optional, Any

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

# Ensure deepsafe_utils is importable regardless of execution context.
# This fallback is necessary when running the script directly from the project root
# without an installed package structure.
try:
    from deepsafe_utils.config_manager import ConfigManager
    from deepsafe_utils.api_client import APIClient
    from deepsafe_utils.media_handler import MediaHandler
except ImportError:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from deepsafe_utils.config_manager import ConfigManager
        from deepsafe_utils.api_client import APIClient
        from deepsafe_utils.media_handler import MediaHandler
    except ImportError as e:
        print(f"Critical Error: Failed to resolve deepsafe_utils dependency. {e}")
        sys.exit(1)


console = Console(width=120)

class MetaFeatureGenerator:
    """
    Manages the ETL process for meta-learning datasets.
    
    Attributes:
        media_type (str): The domain of operation (image, video, audio).
        config_manager (ConfigManager): Centralized configuration handler.
        api_client (APIClient): Interface for communicating with model microservices.
    """
    
    def __init__(self, media_type: str, config_manager: ConfigManager):
        self.media_type = media_type
        self.config_manager = config_manager
        # run_from_host=True implies we are running outside the docker network (e.g., local dev),
        # so we use localhost ports mapped in docker-compose.
        self.api_client = APIClient(config_manager, media_type, run_from_host=True)
        self.media_handler = MediaHandler(config_manager)
        self.base_model_names = list(config_manager.get_model_endpoints(media_type).keys())

        if not self.base_model_names:
            console.print(f"[bold red]Configuration Error: No base models defined for '{media_type}'.[/bold red]")
            sys.exit(1)

    def generate(self, input_dir: str, output_csv_path: str,
                 default_threshold: float, specific_models: Optional[List[str]] = None):
        """
        Executes the generation pipeline.

        Args:
            input_dir: Root directory containing 'Real' and 'Fake' subdirectories.
            output_csv_path: Destination for the resulting feature matrix.
            default_threshold: Decision threshold passed to models (mostly for logging/reference).
            specific_models: Optional filter to run only a subset of available models.
        """
        
        console.print(Panel(f"[bold cyan]Meta-Feature Generation Protocol ({self.media_type.capitalize()})[/bold cyan]\n"
                            f"Source: {input_dir}\n"
                            f"Target: {output_csv_path}\n"
                            f"Active Models: {specific_models or 'All configured'}",
                            title="Pipeline Configuration", border_style="blue", expand=False))

        # Discovery phase: Scan filesystem for valid media files and infer ground truth from directory structure.
        media_files_with_gt = self.media_handler.find_media_files(input_dir, self.media_type)
        if not media_files_with_gt:
            console.print(f"[bold red]Abort: No valid {self.media_type} files found in '{input_dir}'.[/bold red]")
            return

        # Determine the execution scope (subset of models vs all).
        models_to_query = self.base_model_names
        if specific_models:
            models_to_query = [m for m in specific_models if m in self.base_model_names]
            if not models_to_query:
                console.print(f"[bold red]Configuration Mismatch: Requested models {specific_models} are not configured for '{self.media_type}'.[/bold red]")
                return
            console.print(f"Scope restricted to: {models_to_query}")
        
        all_feature_data = [] 

        # Execution phase: Iterate through files and query models.
        # We use a rich progress bar for observability during long-running batch processes.
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as progress:
            total_files = len(media_files_with_gt)
            outer_task = progress.add_task(f"Processing {self.media_type} corpus...", total=total_files)

            for file_idx, (media_path, ground_truth_label) in enumerate(media_files_with_gt):
                media_file_name = os.path.basename(media_path)
                progress.update(outer_task, description=f"Processing: [cyan]{media_file_name}[/cyan]")

                # Pre-encode media to base64 once to avoid redundant I/O operations per model.
                encoded_media = self.media_handler.encode_media_to_base64(media_path)
                if not encoded_media:
                    console.print(f"[yellow]Skip: Encoding failed for {media_file_name}.[/yellow]")
                    progress.advance(outer_task)
                    continue

                # Feature vector initialization
                current_media_features: Dict[str, Any] = {
                    "media_path": media_path,
                    "media_name": media_file_name,
                    # Map string labels to numeric binary targets: Fake=1, Real=0.
                    "ground_truth": 1 if ground_truth_label == "Fake" else (0 if ground_truth_label == "Real" else -1)
                }
                
                # Initialize feature columns with NaN. This ensures structural consistency in the DataFrame
                # even if specific model queries fail.
                for model_name_cfg in self.base_model_names:
                    current_media_features[f"{model_name_cfg}_prob"] = np.nan

                # Query loop
                for model_name_query in models_to_query:
                    model_result = self.api_client.test_with_individual_model(
                        model_name_query, media_path, encoded_media, default_threshold
                    )
                    
                    if "error" not in model_result and model_result.get("probability") is not None:
                        current_media_features[f"{model_name_query}_prob"] = model_result["probability"]
                    else:
                        # Log failure but do not interrupt the pipeline. Robustness is key here.
                        error_msg = model_result.get('error', 'Invalid response payload')
                        console.print(f"[yellow]Model Failure: {model_name_query} on {media_file_name}. Reason: {error_msg}.[/yellow]", highlight=False)
                
                all_feature_data.append(current_media_features)
                progress.advance(outer_task)
                
                # Explicit garbage collection to prevent memory bloat during large dataset processing.
                gc.collect()

        if not all_feature_data:
            console.print("[bold red]Pipeline Failure: No features generated.[/bold red]")
            return

        # Data serialization and validation
        meta_features_df = pd.DataFrame(all_feature_data)
        
        # Filter invalid ground truth (should be handled by discovery, but defensive programming is good).
        meta_features_df = meta_features_df[meta_features_df['ground_truth'] != -1]

        if meta_features_df.empty:
            console.print("[bold red]Data Error: No valid labeled data remaining after processing.[/bold red]")
            return

        # Schema enforcement: Ensure all expected columns exist.
        expected_prob_cols = [f"{mn}_prob" for mn in self.base_model_names]
        for col in expected_prob_cols:
            if col not in meta_features_df.columns:
                meta_features_df[col] = np.nan 

        # Column ordering for readability and consistency.
        ordered_prob_cols = sorted([col for col in meta_features_df.columns if col.endswith('_prob')])
        final_cols_order = ["media_path", "media_name"] + ordered_prob_cols + ["ground_truth"]
        meta_features_df = meta_features_df[final_cols_order]

        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
            meta_features_df.to_csv(output_csv_path, index=False, float_format='%.6f')
            
            console.print(f"\n[bold green]Success: Dataset persisted to {os.path.abspath(output_csv_path)}[/bold green]")
            console.print(f"Dimensions: {meta_features_df.shape}")
            
            # Quality Assurance: Report missing values to inform downstream handling strategies.
            nan_summary_table = Table(title="Data Quality Report (Missing Values)", show_lines=True)
            nan_summary_table.add_column("Feature", style="cyan")
            nan_summary_table.add_column("Missing Count", style="magenta", justify="right")
            nan_summary_table.add_column("Missing %", style="yellow", justify="right")
            
            for col in ordered_prob_cols:
                nan_count = meta_features_df[col].isnull().sum()
                nan_percent = (nan_count / len(meta_features_df)) * 100 if len(meta_features_df) > 0 else 0
                nan_summary_table.add_row(col, str(nan_count), f"{nan_percent:.2f}%")
            console.print(nan_summary_table)

        except Exception as e:
            console.print(f"[bold red]I/O Error: Failed to write output CSV. {e}[/bold red]")

def main():
    parser = argparse.ArgumentParser(
        description="DeepSafe Meta-Feature Generator: ETL for Stacking Ensemble Training Data.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--media-type", type=str, choices=["image", "video", "audio"], required=True,
        help="Target domain. Defines the model registry subset."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Source directory. Must contain 'Real' and 'Fake' subdirectories for label inference."
    )
    parser.add_argument(
        "--output-csv", type=str, required=True,
        help="Destination path for the generated feature matrix."
    )
    parser.add_argument(
        "--threshold", type=float,
        help="Decision threshold override (0.0-1.0). Defaults to system config."
    )
    parser.add_argument(
        "--specific-models", type=str,
        help="Optional filter: Comma-separated list of model identifiers to query."
    )
    parser.add_argument(
        "--config-path", type=str, default=None,
        help=f"Configuration override path."
    )

    args = parser.parse_args()

    # Initialize configuration subsystem
    cfg_manager = ConfigManager(config_path=args.config_path)
    if not cfg_manager.is_config_loaded_successfully():
        sys.exit(1)

    default_thresh_from_config = cfg_manager.get_default("default_threshold", 0.5)
    query_threshold = args.threshold if args.threshold is not None else default_thresh_from_config
    
    specific_models_list = [m.strip() for m in args.specific_models.split(',')] if args.specific_models else None

    generator = MetaFeatureGenerator(args.media_type, cfg_manager)
    generator.generate(args.input_dir, args.output_csv, query_threshold, specific_models_list)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Process Interrupted by User.[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Fatal Error: {e}[/bold red]")
        import traceback
        console.print(Panel(traceback.format_exc(), title="Stack Trace", border_style="red"))
        sys.exit(1)