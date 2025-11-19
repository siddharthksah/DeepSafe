#!/usr/bin/env python3
"""
DeepSafe Unified Testing Tool & Experimentation Suite (deepsafe_test.py)
========================================================================

This script serves as a comprehensive command-line interface for testing,
evaluating, and managing the DeepSafe deepfake detection system. It interacts
with the main DeepSafe API and its individual model microservices for various
media types (image, video, audio).

Key Features:
-------------
1.  Health Checking: Verifies the status of the API gateway and all configured model services.
2.  Single Media Testing:
    - Test a media file against the main API's ensemble (voting, average, stacking).
    - Test a media file against a specific individual detection model.
3.  Batch Processing & Evaluation:
    - Process a directory of media files (with Real/Fake ground truth).
    - Run all or selected individual models for the specified media type.
    - Locally compute simple ensemble results (voting/average).
    - Calculate detailed performance metrics (Accuracy, Precision, Recall, F1, AUC, Specificity).
    - Generate visualizations (Confusion Matrices, ROC Curves, Probability Distributions).
4.  Model Comparison: Compares performance across multiple models and ensemble methods for a given media type.
5.  Performance Benchmarking: Measures client-side request latency for API and models for a given media type.
6.  API Stacking Ensemble Testing: Specifically evaluates the performance of the main
    API's deployed stacking ensemble on a batch of media files.
7.  Visualization Generation: Creates plots from saved result files or by processing a directory.

Core Usage Pattern:
-------------------
./deepsafe_test.py [COMMAND] --media-type [image|video|audio] [OPTIONS]

MEDIA TYPE ARGUMENT (--media-type):
  This argument is REQUIRED for most commands. It specifies the type of media
  being processed (image, video, or audio). This determines:
  - Which set of model endpoints from the configuration file will be used.
  - The types of files searched for in directory inputs (e.g., *.jpg for images, *.mp4 for videos).
  - How the main API is instructed to process the data (via a 'media_type' field in the payload).

Examples:
---------

1. Check System Health (for image models):
   ./deepsafe_test.py health --media-type image

2. Test a Single Image with the Main API (using default "stacking" ensemble):
   ./deepsafe_test.py test --media-type image --input /path/to/image.jpg

3. Test a Single Image with a Specific Individual Model:
   ./deepsafe_test.py test --media-type image --input /path/to/image.jpg --model npr_deepfakedetection

4. Test a Single Video with the Main API (using 'average' ensemble):
   ./deepsafe_test.py test --media-type video --input /path/to/video.mp4 --method average

5. Process a Batch of Images (evaluating all configured image models and local voting ensemble):
   # Assumes ./dataset/images_real_fake/Real/ and ./dataset/images_real_fake/Fake/ exist
   ./deepsafe_test.py batch --media-type image --input-dir ./dataset/images_real_fake/

6. Process a Batch of Audio files with a Specific Audio Model:
   ./deepsafe_test.py batch --media-type audio --input-dir ./dataset/audio_clips/ --model example_audio_model_1

7. Compare Model Performance for Videos:
   ./deepsafe_test.py compare --media-type video --input-dir ./dataset/eval_videos/ --output-dir ./reports/video_comparison/

8. Benchmark API and Image Model Latency:
   ./deepsafe_test.py benchmark --media-type image --input /path/to/sample_image.jpg --count 15

9. Test the Deployed API's Stacking Ensemble on a Batch of Images:
   ./deepsafe_test.py apistacktest --media-type image --input-dir ./dataset/eval_images/

10. Generate Visualizations from a Previous Batch Run's Results JSON:
    ./deepsafe_test.py visualize --input-path ./deepsafe_test_results/YYYYMMDD_HHMMSS/batch_results_image_my_dataset.json --output-dir ./viz_outputs/
    (Note: --media-type is not needed if --input-path is a file for visualize)

11. Process a Directory of Videos and then Generate All Visualizations:
    ./deepsafe_test.py visualize --media-type video --input-path ./dataset/test_videos/ --plot-type all

Configuration:
--------------
- API URLs, model endpoints (categorized by media type: image, video, audio),
  supported file extensions, and other default settings are defined in:
    `config/deepsafe_config.json`
- This configuration file is essential for the tool to operate correctly.
- Output is saved to a timestamped subdirectory. The base for this directory can be
  set via the global `--output-dir` argument or defaults to the value of
  `default_output_dir_base` in the configuration file.

Notes:
------
- Ensure all DeepSafe services (API gateway, model microservices for the
  relevant media type) are running (e.g., via `docker-compose up`) before
  using commands that interact with them.
- For commands processing directories (`batch`, `compare`, `apistacktest`,
  `visualize --input-path DIR`), the input directory should ideally contain
  'Real' and 'Fake' subdirectories for ground truth label inference.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Any, Tuple, Optional
import concurrent.futures # For future parallel processing enhancements
from datetime import datetime
from pathlib import Path
import gc
import uuid # For request IDs if needed locally, API usually assigns

import numpy as np
import pandas as pd # For data handling, esp. metrics and potential CSV outputs
from PIL import ImageFile # For image-specific settings

# Rich library for console output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.text import Text 
from rich.text import Text as RichText

# Custom utilities from deepsafe_utils package
from deepsafe_utils.config_manager import ConfigManager
from deepsafe_utils.api_client import APIClient
from deepsafe_utils.media_handler import MediaHandler
from deepsafe_utils.results_processor import ResultsProcessor
from deepsafe_utils.visualizer import Visualizer

# Allow loading truncated images (primarily for image media type)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Initialize console and config (ConfigManager is a singleton)
console = Console(width=120)
config_manager = ConfigManager()

# --- NpEncoder for JSON serialization ---
class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super(NpEncoder, self).default(o)

class DeepSafeTestOrchestrator:
    def __init__(self, media_type: Optional[str], cli_output_dir_base: Optional[str] = None):
        self.media_type = media_type # Can be None for commands like 'visualize' from file
        self.config_manager = ConfigManager() # Singleton

        if not self.config_manager.is_config_loaded_successfully():
            console.print("[bold red]FATAL: Configuration could not be loaded. Exiting.[/bold red]")
            sys.exit(1)

        if self.media_type and not self.config_manager.get_media_config(self.media_type):
            console.print(f"[bold red]Error: Media type '{self.media_type}' not defined in configuration {self.config_manager.config_path}.[/bold red]")
            available_types = list(self.config_manager.get('media_types', {}).keys())
            console.print(f"Available media types in config: {available_types if available_types else 'None found'}")
            sys.exit(1)

        self.api_client = APIClient(self.config_manager, self.media_type, run_from_host=True)
        self.media_handler = MediaHandler(self.config_manager)
        self.results_processor = ResultsProcessor(self.config_manager)

        # Determine base output directory
        _output_dir_base = cli_output_dir_base or self.config_manager.get_default("default_output_dir_base", "./deepsafe_test_results")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a unique directory for this specific run's outputs
        self.current_run_output_dir = os.path.join(_output_dir_base, timestamp)
        try:
            os.makedirs(self.current_run_output_dir, exist_ok=True)
            console.print(f"[cyan]Results for this run will be saved in: {os.path.abspath(self.current_run_output_dir)}[/cyan]")
        except OSError as e:
            console.print(f"[bold red]Error creating output directory {self.current_run_output_dir}: {e}[/bold red]")
            # Fallback to a local temp dir if creation fails, or exit
            self.current_run_output_dir = f"./deepsafe_temp_results_{timestamp}"
            os.makedirs(self.current_run_output_dir, exist_ok=True)
            console.print(f"[yellow]Fell back to temporary output directory: {self.current_run_output_dir}[/yellow]")

        self.visualizer = Visualizer(self.config_manager, self.current_run_output_dir)


    def _get_api_client_for_media_type(self, media_type_override: Optional[str] = None) -> APIClient:
        mt = media_type_override or self.media_type
        if not mt:
            raise ValueError("Media type must be specified for API client.")
        # Potentially cache APIClient instances per media_type if needed, but for CLI, new instance is fine.
        return APIClient(self.config_manager, mt)

    def execute_health_check(self):
        if not self.media_type: 
            console.print("[bold red]Health check requires --media-type to specify which models to check.[/bold red]")
            return

        console.print(Panel(f"[bold cyan]DeepSafe System Health Check (Media Type: {self.media_type.upper()})[/bold cyan]", border_style="blue", expand=False))
        client = self._get_api_client_for_media_type() 

        main_api_health = client.check_main_api_health(force_refresh=True) # force_refresh not used by client, but ok
        
        api_table = Table(title="DeepSafe System Status")
        api_table.add_column("Component", style="cyan", overflow="fold", min_width=30) # Increased min_width
        api_table.add_column("Status", style="magenta", min_width=25) # Increased min_width
        api_table.add_column("Details", style="green", overflow="fold", max_width=60) # Adjusted max_width

        # Get overall API status
        api_overall_status = main_api_health.get("overall_api_status", "UNKNOWN (Key 'overall_api_status' missing)")
        api_style = "green" if api_overall_status.lower() == "healthy" else "yellow" if api_overall_status.lower() == "degraded" else "red"
        api_table.add_row("API Gateway Overall", f"[bold {api_style}]{api_overall_status.upper()}[/bold {api_style}]", main_api_health.get("request_id", ""))
        
        # Get details for the specific media type
        media_type_details = main_api_health.get("media_type_details", {}).get(self.media_type, {})
        
        media_type_general_status = media_type_details.get("status", "UNKNOWN (Media type details missing or status key missing)")
        media_type_status_style = "green" if media_type_general_status.lower() == "healthy" else "yellow" if "degraded" in media_type_general_status.lower() else "red"
        api_table.add_row(f"'{self.media_type.upper()}' Processing Status", f"[bold {media_type_status_style}]{media_type_general_status.upper()}[/bold {media_type_status_style}]", "")


        # Get model statuses for the specific media type
        model_statuses_for_this_type = media_type_details.get("models", {}) # CORRECTED ACCESS

        api_table.add_section()
        api_table.add_row(RichText(f"Individual Models for '{self.media_type.upper()}' (Status via API Gateway)", style="bold blue"), "", "")

        configured_models_for_type = list(self.config_manager.get_model_endpoints(self.media_type).keys())
        
        if configured_models_for_type:
            if model_statuses_for_this_type: # Check if we got any model statuses for this type
                for model_name_cfg in configured_models_for_type:
                    model_health_info = model_statuses_for_this_type.get(model_name_cfg, {"status": "NOT_REPORTED_FOR_TYPE", "message": "Not in API health response for this media type."})
                    status = model_health_info.get("status", "unknown")
                    status_style = "green" if status.lower() == "healthy" else "yellow" if status.lower() in ["loading", "degraded", "degraded_not_loaded", "degraded_components_not_loaded"] else "red"
                    details = model_health_info.get("device", "") 
                    if "message" in model_health_info and model_health_info["message"]: # Check if message is not empty
                        details += f" | {model_health_info['message']}"
                    elif not details: # If no device and no message, provide a placeholder
                        details = "No further details from model health."
                    api_table.add_row(model_name_cfg, f"[bold {status_style}]{status.upper()}[/bold {status_style}]", details.strip(" | "))
            else:
                 api_table.add_row(f"No model statuses reported by API for '{self.media_type}'.", "-", "Check API health endpoint response structure.")
        else:
            api_table.add_row(f"No models configured locally for '{self.media_type}'.", "-", "-")

        api_table.add_section()
        stacking_loaded_for_type = media_type_details.get("stacking_ensemble_loaded", False)
        stacking_status_msg = "Available" if stacking_loaded_for_type else "Unavailable"
        stacking_style = "green" if stacking_loaded_for_type else "yellow"
        api_table.add_row(f"Stacking Ensemble for '{self.media_type.upper()}'", f"[bold {stacking_style}]{stacking_status_msg}[/bold {stacking_style}]", "")

        console.print(api_table)

    def execute_single_test(self, media_path: str, threshold: float,
                            ensemble_method: str, specific_model_name: Optional[str] = None,
                            output_file_path: Optional[str] = None):
        if not self.media_type: # Should be caught by argparse
            console.print("[bold red]Single test requires --media-type.[/bold red]"); return
        if not self.media_handler.validate_media_file(media_path, self.media_type): return

        console.print(Panel(f"[bold cyan]DeepSafe Single {self.media_type.capitalize()} Test[/bold cyan]\n"
                            f"Media File: {media_path}\nThreshold: {threshold:.2f}\n"
                            f"Output Base: {self.current_run_output_dir}",
                            title="Test Configuration", border_style="blue", expand=False))

        client = self._get_api_client_for_media_type()
        encoded_media = self.media_handler.encode_media_to_base64(media_path)
        if not encoded_media:
            console.print(f"[bold red]Failed to encode {media_path}. Aborting test.[/bold red]"); return

        test_results_payload = {
            "test_config": {"media_path": media_path, "media_type": self.media_type, "threshold": threshold, "timestamp": datetime.now().isoformat()}
        }
        final_output_path = output_file_path

        if specific_model_name:
            console.print(f"Testing with specific model: [magenta]{specific_model_name}[/magenta]")
            model_result = client.test_with_individual_model(specific_model_name, media_path, encoded_media, threshold)
            test_results_payload["specified_model_result"] = model_result
            self._display_single_model_result(model_result, specific_model_name)
            if not final_output_path:
                final_output_path = os.path.join(self.current_run_output_dir, f"test_{Path(media_path).stem}_{specific_model_name}.json")
        else:
            console.print(f"Testing with Main API (Ensemble: [magenta]{ensemble_method}[/magenta]) and all individual {self.media_type} models.")
            test_results_payload["test_config"]["ensemble_method_main_api"] = ensemble_method
            
            api_result = client.test_with_main_api(media_path, self.media_type, encoded_media, threshold, ensemble_method)
            test_results_payload["api_ensemble_result"] = api_result
            self._display_api_ensemble_result(api_result)

            individual_model_results = {}
            model_names_for_type = list(self.config_manager.get_model_endpoints(self.media_type).keys())
            if not model_names_for_type:
                console.print(f"[yellow]No individual models configured for '{self.media_type}'. Skipping.[/yellow]")
            else:
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), transient=True) as progress:
                    task = progress.add_task(f"[cyan]Testing individual models...", total=len(model_names_for_type))
                    for model_name in model_names_for_type:
                        progress.update(task, description=f"[cyan]Testing {model_name}...")
                        res = client.test_with_individual_model(model_name, media_path, encoded_media, threshold)
                        individual_model_results[model_name] = res
                        progress.advance(task)
                test_results_payload["individual_model_results"] = individual_model_results
                self._display_individual_model_results_table(individual_model_results)
            
            if not final_output_path:
                final_output_path = os.path.join(self.current_run_output_dir, f"test_{Path(media_path).stem}_all.json")

        if final_output_path:
            os.makedirs(os.path.dirname(os.path.abspath(final_output_path)), exist_ok=True)
            with open(final_output_path, 'w') as f: json.dump(test_results_payload, f, indent=2, cls=NpEncoder)
            console.print(f"[green]Test results saved to: {os.path.abspath(final_output_path)}[/green]")

    def _display_single_model_result(self, result_dict: Dict[str, Any], model_name: str):
        table = Table(title=f"Result for Model: {model_name}", show_header=False, box=None, padding=(0,1))
        table.add_column("Metric", style="cyan", justify="right")
        table.add_column("Value", style="magenta")
        if "error" in result_dict:
            table.add_row("Status", "[bold red]ERROR[/bold red]")
            table.add_row("Detail", Text(result_dict['error'], overflow="fold"))
        else:
            cl = result_dict.get("class", "N/A").upper(); pr = result_dict.get("probability", 0.0)
            it = result_dict.get("inference_time", result_dict.get("total_request_time")); ts = f"{it:.3f}s" if it else "N/A"
            col = "red" if cl == "FAKE" else "green" if cl == "REAL" else "yellow"
            table.add_row("Verdict:", f"[bold {col}]{cl}[/bold {col}]")
            table.add_row("P(Fake):", f"{pr:.3%}" if isinstance(pr, float) else "N/A")
            table.add_row("Time:", ts)
        console.print(Panel(table, title=f"[white]Model: {model_name}[/white]", border_style="dim blue", expand=False))


    def _display_api_ensemble_result(self, api_result: Dict[str, Any]):
        if "error" in api_result:
            console.print(Panel(f"[bold red]Error with Main API (Ensemble): {api_result['error']}[/bold red]", border_style="red", expand=False)); return
        
        v = api_result.get("verdict", "N/A").upper(); c = api_result.get("confidence_in_verdict", 0.0)
        pf = api_result.get("ensemble_score_is_fake", 0.0); em = api_result.get("ensemble_method_used", "N/A")
        at = api_result.get("total_inference_time_seconds"); ct = api_result.get('client_request_time', 0.0)
        col = "red" if v == "FAKE" else "green" if v == "REAL" else "yellow"
        
        table = Table(title=f"Main API Ensemble ({em.capitalize()}) Verdict", show_header=False, box=None, padding=(0,1))
        table.add_column("Metric", style="cyan", justify="right"); table.add_column("Value", style="magenta")
        table.add_row("Verdict:", f"[bold {col}]{v}[/bold {col}] (Confidence: {c:.3%})")
        table.add_row("P(Fake) Score:", f"{pf:.3%}")
        table.add_row("Base Fake Votes:", str(api_result.get('base_model_fake_votes', 'N/A')))
        table.add_row("Base Real Votes:", str(api_result.get('base_model_real_votes', 'N/A')))
        table.add_row("API Inf. Time:", f'{at:.3f}s' if at else 'N/A')
        table.add_row("Client Req. Time:", f'{ct:.3f}s')
        table.add_row("Request ID:", api_result.get('request_id', 'N/A'))
        console.print(Panel(table, title=f"[white]API Ensemble ({em.capitalize()})[/white]", border_style=col, expand=False))

    def _display_individual_model_results_table(self, model_results: Dict[str, Dict[str, Any]]):
        table = Table(title=f"Individual Model Results ({self.media_type.capitalize()})", show_lines=True)
        table.add_column("Model", style="cyan", overflow="fold", max_width=30)
        table.add_column("Verdict", style="magenta", justify="center")
        table.add_column("P(Fake)", style="yellow", justify="right")
        table.add_column("Time (s)", style="green", justify="right")

        for name, res in sorted(model_results.items()):
            if "error" in res:
                table.add_row(name, "[bold red]ERROR[/bold red]", "-", Text(res.get("error","?"),overflow="ellipsis",max_length=20))
            else:
                cl = res.get("class","N/A").upper(); pr = res.get("probability",0.0)
                it = res.get("inference_time",res.get("total_request_time")); ts = f"{it:.3f}" if it else "N/A"
                col = "red" if cl == "FAKE" else "green" if cl == "REAL" else "yellow"
                table.add_row(name, f"[bold {col}]{cl}[/bold {col}]", f"{pr:.3%}" if isinstance(pr,float) else "N/A", ts)
        console.print(table)

    def execute_batch_processing(self, input_dir: str, threshold: float, local_ensemble_method: str,
                                 specific_model_name: Optional[str] = None,
                                 output_file_path: Optional[str] = None,
                                 display_metrics: bool = True, generate_plots: bool = True):
        if not self.media_type:
            console.print("[bold red]Batch processing requires --media-type.[/bold red]"); return None

        console.print(Panel(f"[bold cyan]DeepSafe Batch Processing ({self.media_type.capitalize()})[/bold cyan]\n"
                            f"Input Dir: {input_dir}\nModel(s): {specific_model_name or 'All for ' + self.media_type}\n"
                            f"Threshold: {threshold:.2f}\nLocal Ensemble: {local_ensemble_method.capitalize()}\n"
                            f"Output Base: {self.current_run_output_dir}",
                            title="Batch Configuration", border_style="blue", expand=False))

        media_files = self.media_handler.find_media_files(input_dir, self.media_type)
        if not media_files: return None

        client = self._get_api_client_for_media_type()
        models_to_run = [specific_model_name] if specific_model_name else \
                        list(self.config_manager.get_model_endpoints(self.media_type).keys())
        if not models_to_run:
            console.print(f"[red]No models configured for {self.media_type}. Aborting batch.[/red]"); return None

        all_individual_results: List[Dict] = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as progress:
            for model_name in models_to_run:
                task_id = progress.add_task(f"Model [m]{model_name}[/m]", total=len(media_files))
                success_count = 0
                for media_path, gt_label in media_files:
                    fname = os.path.basename(media_path)
                    progress.update(task_id, description=f"Model [m]{model_name}[/m]: Processing {fname}")
                    encoded = self.media_handler.encode_media_to_base64(media_path)
                    res = {"error": "Encoding failed"} if not encoded else \
                          client.test_with_individual_model(model_name, media_path, encoded, threshold)
                    res.update({"ground_truth": gt_label, "media_path": media_path, "media_name": fname, "model_name": model_name})
                    if "error" not in res: success_count += 1
                    all_individual_results.append(res)
                    progress.advance(task_id)
                    gc.collect()
                console.print(f"Model [m]{model_name}[/m]: {success_count}/{len(media_files)} successful.")
                if self.config_manager.get_default("memory_optimization", True):
                    client.request_model_unload(model_name); time.sleep(0.5)

        results_by_media: Dict[str, List[Dict]] = {}
        for r in all_individual_results:
            p = r.get("media_path")
            if p: results_by_media.setdefault(p, []).append(r)
        
        final_results_list = list(all_individual_results)
        if local_ensemble_method and len(models_to_run) > 1:
            console.print(f"\nCalculating local '{local_ensemble_method}' ensemble results...")
            for single_media_outputs in results_by_media.values():
                if single_media_outputs: # Ensure there are results for this path
                    final_results_list.append(self.results_processor.compute_local_ensemble_results(
                        single_media_outputs, threshold, local_ensemble_method))
        
        final_out_fpath = output_file_path or os.path.join(self.current_run_output_dir, 
            f"batch_results_{self.media_type}_{Path(input_dir).name}{'_'+specific_model_name if specific_model_name else '_all'}.json")
        os.makedirs(os.path.dirname(os.path.abspath(final_out_fpath)), exist_ok=True)
        with open(final_out_fpath, 'w') as f: json.dump(final_results_list, f, indent=2, cls=NpEncoder)
        console.print(f"\n[green]Batch raw results saved to: {os.path.abspath(final_out_fpath)}[/green]")

        if display_metrics: self._calculate_and_display_metrics(final_results_list, "Batch Processing")
        if generate_plots:
            self.visualizer.plot_confusion_matrices(final_results_list)
            self.visualizer.plot_roc_curves(final_results_list)
            self.visualizer.plot_probability_distributions(final_results_list)
        
        console.print(f"[green]Batch for {self.media_type} completed. Results & plots in {os.path.abspath(self.current_run_output_dir)}[/green]")
        return {"results": final_results_list, "output_file": final_out_fpath}


    def _calculate_and_display_metrics(self, results_list: List[Dict[str, Any]], title_prefix: str):
        # (Implementation is the same as your previous version, just ensure it's part of the class)
        metrics_data = self.results_processor.calculate_batch_metrics(results_list)
        if not metrics_data:
            console.print("[yellow]No metrics could be calculated from the results.[/yellow]")
            return

        console.print(f"\n[bold cyan]{title_prefix} (Media: {self.media_type.upper()}):[/bold cyan]")
        metrics_table = Table(title=f"Performance Metrics (Sorted by AUC)", show_lines=True)
        metrics_table.add_column("Model/Method", style="cyan", overflow="fold", max_width=35)
        metrics_table.add_column("Count", style="blue", justify="right")
        metrics_table.add_column("Acc.", style="green", justify="right")
        metrics_table.add_column("Prec.", style="blue", justify="right")
        metrics_table.add_column("Rec.", style="magenta", justify="right")
        metrics_table.add_column("F1", style="yellow", justify="right")
        metrics_table.add_column("AUC", style="red", justify="right")
        metrics_table.add_column("Spec.", style="cyan", justify="right")

        sorted_metric_items = sorted(
            metrics_data.items(),
            key=lambda item: item[1].get('auc', -1) if isinstance(item[1], dict) and pd.notna(item[1].get('auc')) else -1,
            reverse=True
        )

        for model_method_name, metrics_dict in sorted_metric_items:
            if isinstance(metrics_dict, dict) and not metrics_dict.get("error"):
                metrics_table.add_row(
                    model_method_name,
                    str(metrics_dict.get('count', 'N/A')),
                    f"{metrics_dict.get('accuracy', np.nan):.4f}",
                    f"{metrics_dict.get('precision', np.nan):.4f}",
                    f"{metrics_dict.get('recall', np.nan):.4f}",
                    f"{metrics_dict.get('f1_score', np.nan):.4f}",
                    f"{metrics_dict.get('auc', np.nan):.4f}" if pd.notna(metrics_dict.get('auc')) else "N/A",
                    f"{metrics_dict.get('specificity', np.nan):.4f}"
                )
            else:
                error_msg = metrics_dict.get('error', 'Calc failed') if isinstance(metrics_dict, dict) else "Invalid data"
                count_val = metrics_dict.get('count', '-') if isinstance(metrics_dict, dict) else '-'
                metrics_table.add_row(model_method_name, str(count_val), Text(error_msg,overflow="ellipsis",max_length=10), "-", "-", "-", "-", "-")
        console.print(metrics_table)

        metrics_summary_filename = f"{title_prefix.lower().replace(' ', '_')}_{self.media_type}_metrics.json"
        metrics_summary_filepath = os.path.join(self.current_run_output_dir, metrics_summary_filename)
        with open(metrics_summary_filepath, 'w') as f_metrics:
            json.dump(metrics_data, f_metrics, indent=2, cls=NpEncoder)
        console.print(f"Metrics summary saved to [green]{os.path.abspath(metrics_summary_filepath)}[/green]")

    def execute_comparison(self, input_dir: str, threshold: float, local_ensemble_method: str,
                           specific_model_name: Optional[str] = None):
        if not self.media_type:
            console.print("[bold red]Model comparison requires --media-type.[/bold red]"); return

        console.print(Panel(f"[bold cyan]DeepSafe Model Comparison ({self.media_type.capitalize()})[/bold cyan]\n"
                            f"Input Dir: {input_dir}\nModel(s): {specific_model_name or 'All for ' + self.media_type}\n"
                            f"Threshold: {threshold:.2f}\nLocal Ensemble: {local_ensemble_method.capitalize()}\n"
                            f"Output Dir: {os.path.abspath(self.current_run_output_dir)}",
                            title="Comparison Configuration", border_style="blue", expand=False))

        raw_results_filename = f"compare_raw_outputs_{self.media_type}_{Path(input_dir).name}.json"
        raw_results_filepath = os.path.join(self.current_run_output_dir, raw_results_filename)

        batch_data = self.execute_batch_processing(
            input_dir, threshold, local_ensemble_method,
            specific_model_name, raw_results_filepath,
            display_metrics=True, generate_plots=True # Ensure metrics and plots are generated by batch
        )
        if batch_data and "results" in batch_data:
            console.print(f"\n[green]Model comparison for {self.media_type} complete. Details in {os.path.abspath(self.current_run_output_dir)}[/green]")
        else:
            console.print(f"[bold red]Model comparison failed as batch processing encountered an error.[/bold red]")

    def execute_api_stacking_test(self, input_dir: str, threshold: float):
        if not self.media_type:
            console.print("[bold red]API Stacking Test requires --media-type.[/bold red]"); return

        console.print(Panel(f"[bold cyan]DeepSafe API Stacking Ensemble Test ({self.media_type.capitalize()})[/bold cyan]\n"
                            f"Input Dir: {input_dir}\nAPI Threshold: {threshold:.2f}\n"
                            f"Output Dir: {os.path.abspath(self.current_run_output_dir)}\n"
                            f"Calls main API with ensemble_method='stacking' for each {self.media_type}.",
                            title="API Stacking Test Config", border_style="blue", expand=False))

        media_files = self.media_handler.find_media_files(input_dir, self.media_type)
        if not media_files: return

        client = self._get_api_client_for_media_type()
        all_api_results: List[Dict] = []

        with Progress(SpinnerColumn(),TextColumn("[progress.description]{task.description}"),BarColumn(),MofNCompleteColumn(),TimeElapsedColumn()) as progress:
            task_id = progress.add_task(f"[cyan]Querying API (stacking, {self.media_type})...", total=len(media_files))
            for media_path, gt_label in media_files:
                fname = os.path.basename(media_path)
                progress.update(task_id, description=f"Querying API for {fname}...")
                encoded = self.media_handler.encode_media_to_base64(media_path)
                if not encoded:
                    all_api_results.append({"error": "Encoding failed", "media_path": media_path, "media_name": fname, "ground_truth": gt_label, "model_name": "API_Stacking_Encoding_Error"})
                    progress.advance(task_id); continue

                api_res = client.test_with_main_api(media_path, self.media_type, encoded, threshold, "stacking")
                
                if "error" in api_res:
                    api_res.update({"media_path": media_path, "media_name": fname, "ground_truth": gt_label, "model_name": f"API_Stacking_Error_{self.media_type}"})
                    all_api_results.append(api_res)
                else:
                    prob_fake = api_res.get("ensemble_score_is_fake", 0.5) # Default if missing
                    verdict = api_res.get("verdict", "undetermined")
                    pred = 1 if verdict == "fake" else 0
                    formatted_res = {
                        "model_name": f"API_Stacking_Ensemble_{self.media_type}", "media_path": media_path, "media_name": fname,
                        "ground_truth": gt_label, "probability": float(prob_fake), "prediction": pred, "class": verdict,
                        "inference_time": api_res.get("total_inference_time_seconds")
                    }
                    all_api_results.append(formatted_res)
                    # Add individual model results if returned by API
                    for model_key, model_val in api_res.get("model_results", {}).items():
                        if isinstance(model_val, dict): # Ensure it's a dict
                             model_val.update({"media_path": media_path, "media_name": fname, "ground_truth": gt_label, "model_name": model_key})
                             all_api_results.append(model_val)
                progress.advance(task_id)

        if not all_api_results: console.print("[red]No results from API Stacking test.[/red]"); return

        raw_out_fpath = os.path.join(self.current_run_output_dir, f"apistacktest_raw_{self.media_type}_{Path(input_dir).name}.json")
        with open(raw_out_fpath, 'w') as f: json.dump(all_api_results, f, indent=2, cls=NpEncoder)
        console.print(f"\nAPI Stacking Test raw outputs: [green]{os.path.abspath(raw_out_fpath)}[/green]")

        self._calculate_and_display_metrics(all_api_results, f"API Stacking Test ({self.media_type.capitalize()})")
        self.visualizer.plot_confusion_matrices(all_api_results)
        self.visualizer.plot_roc_curves(all_api_results)
        self.visualizer.plot_probability_distributions(all_api_results)
        console.print(f"\n[green]API Stacking test for {self.media_type} complete. Results in {os.path.abspath(self.current_run_output_dir)}[/green]")

    def execute_benchmark(self, media_path: str, count: int):
        if not self.media_type: console.print("[red]Benchmark requires --media-type.[/red]"); return
        if not self.media_handler.validate_media_file(media_path, self.media_type): return

        console.print(Panel(f"[bold cyan]DeepSafe Performance Benchmark ({self.media_type.capitalize()})[/bold cyan]\n"
                            f"Media File: {media_path}\nIterations: {count}\n"
                            f"Output Dir: {os.path.abspath(self.current_run_output_dir)}",
                            title="Benchmark Configuration", border_style="blue", expand=False))

        client = self._get_api_client_for_media_type()
        encoded_media = self.media_handler.encode_media_to_base64(media_path)
        if not encoded_media: console.print(f"[red]Failed to encode {media_path}. Aborting.[/red]"); return
        
        results_data = {"api": {"raw_total_request_times": []}, "models": {}}
        def_thresh = self.config_manager.get_default("default_threshold", 0.5)
        def_ensemble = self.config_manager.get_default("default_ensemble_method", "stacking")

        console.print(f"\nBenchmarking Main API ({self.media_type}, ensemble: {def_ensemble}) with {count} iterations...")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TimeElapsedColumn(), transient=True) as p:
            task = p.add_task("API Benchmark", total=count)
            for _ in range(count):
                try:
                    res = client.test_with_main_api(media_path, self.media_type, encoded_media, def_thresh, def_ensemble)
                    if "client_request_time" in res: results_data["api"]["raw_total_request_times"].append(res["client_request_time"])
                    time.sleep(0.1) 
                except Exception as e: console.print(f"[yellow]API benchmark iter failed: {e}[/yellow]")
                p.advance(task)
        
        api_t = results_data["api"]["raw_total_request_times"]
        results_data["api"].update({
            "count": len(api_t), "mean_total_request_time": np.mean(api_t) if api_t else None,
            "median_total_request_time": np.median(api_t) if api_t else None,
            "min_total_request_time": np.min(api_t) if api_t else None,
            "max_total_request_time": np.max(api_t) if api_t else None,
            "std_total_request_time": np.std(api_t) if api_t else None,
        })

        models_for_type = list(self.config_manager.get_model_endpoints(self.media_type).keys())
        if models_for_type:
            console.print(f"\nBenchmarking {len(models_for_type)} individual {self.media_type} models ({count} iterations each)...")
            for model_name in models_for_type:
                model_t = []
                with Progress(SpinnerColumn(), TextColumn("{task.description}"), MofNCompleteColumn(), TimeElapsedColumn(), transient=True) as p:
                    task_m = p.add_task(f"Benchmarking [m]{model_name}[/m]", total=count)
                    for _ in range(count):
                        try:
                            res = client.test_with_individual_model(model_name, media_path, encoded_media, def_thresh)
                            if "total_request_time" in res: model_t.append(res["total_request_time"])
                            time.sleep(0.1)
                        except Exception as e: console.print(f"[yellow]Benchmark for {model_name} iter failed: {e}[/yellow]")
                        p.advance(task_m)
                results_data["models"][model_name] = {
                    "count": len(model_t), "mean_total_request_time": np.mean(model_t) if model_t else None,
                    "median_total_request_time": np.median(model_t) if model_t else None,
                    "min_total_request_time": np.min(model_t) if model_t else None,
                    "max_total_request_time": np.max(model_t) if model_t else None,
                    "std_total_request_time": np.std(model_t) if model_t else None,
                    "raw_total_request_times": model_t
                }
        
        self._display_benchmark_table(results_data)
        bench_json_path = os.path.join(self.current_run_output_dir, f"benchmark_results_{self.media_type}.json")
        with open(bench_json_path, 'w') as f: json.dump(results_data, f, indent=2, cls=NpEncoder)
        console.print(f"\nBenchmark results JSON: [green]{os.path.abspath(bench_json_path)}[/green]")
        self.visualizer.plot_benchmark_summary(results_data) # Visualizer handles subdirs
        console.print(f"\n[green]Benchmark for {self.media_type} complete. Plots in '{os.path.join(self.current_run_output_dir, 'benchmark_plots')}'[/green]")

    def _display_benchmark_table(self, benchmark_results: Dict[str, Any]):
        # (Implementation is the same as your previous version)
        table = Table(title=f"Benchmark Results (Total Request Time, {self.media_type.capitalize()})", show_lines=True)
        table.add_column("Component", style="cyan", overflow="fold", max_width=30)
        table.add_column("Count", style="blue", justify="right")
        table.add_column("Mean (s)", style="yellow", justify="right")
        table.add_column("Median (s)", style="green", justify="right")
        table.add_column("Min (s)", style="green", justify="right")
        table.add_column("Max (s)", style="red", justify="right")
        table.add_column("Std Dev (s)", style="magenta", justify="right")

        api_data = benchmark_results.get("api", {})
        if api_data.get("count", 0) > 0:
            table.add_row(
                "API (Ensemble)", str(api_data['count']),
                f"{api_data.get('mean_total_request_time', np.nan):.4f}", f"{api_data.get('median_total_request_time', np.nan):.4f}",
                f"{api_data.get('min_total_request_time', np.nan):.4f}", f"{api_data.get('max_total_request_time', np.nan):.4f}",
                f"{api_data.get('std_total_request_time', np.nan):.4f}"
            )
        
        models_data = benchmark_results.get("models", {})
        if models_data: table.add_section()
        for model_name, data in sorted(models_data.items()):
            if data.get("count", 0) > 0:
                table.add_row(
                    model_name, str(data['count']),
                    f"{data.get('mean_total_request_time', np.nan):.4f}", f"{data.get('median_total_request_time', np.nan):.4f}",
                    f"{data.get('min_total_request_time', np.nan):.4f}", f"{data.get('max_total_request_time', np.nan):.4f}",
                    f"{data.get('std_total_request_time', np.nan):.4f}"
                )
        console.print(table)

    def execute_visualization(self, input_path: str, plot_type: str):
        # media_type for orchestrator should be set if input_path is a directory
        console.print(Panel(f"[bold cyan]DeepSafe Visualization Generation[/bold cyan]\n"
                            f"Input Path: {input_path}\nMedia Type (if dir): {self.media_type or 'N/A (file input)'}\n"
                            f"Plot Type(s): {plot_type.capitalize()}\nOutput Dir: {os.path.abspath(self.current_run_output_dir)}",
                            title="Visualization Configuration", border_style="blue", expand=False))
        
        results_list: List[Dict[str, Any]] = []
        if os.path.isfile(input_path):
            try:
                with open(input_path, 'r') as f: data = json.load(f)
                if isinstance(data, list): results_list = data
                elif isinstance(data, dict): # Handle various dict structures
                    if "results" in data and isinstance(data["results"], list): results_list = data["results"]
                    elif "api_ensemble_result" in data: 
                        if data["api_ensemble_result"]: results_list.append(data["api_ensemble_result"])
                        results_list.extend(list(data.get("individual_model_results",{}).values()))
                    elif "specified_model_result" in data and data["specified_model_result"]: results_list.append(data["specified_model_result"])
                    else: console.print(f"[red]Unrecognized JSON structure in {input_path}.[/red]"); return
                else: console.print(f"[red]Input file {input_path} not a list or recognized dict.[/red]"); return
                console.print(f"Loaded {len(results_list)} results from [cyan]{input_path}[/cyan].")
            except Exception as e: console.print(f"[red]Error loading from {input_path}: {e}[/red]"); return
        elif os.path.isdir(input_path):
            if not self.media_type: console.print("[red]'visualize' from dir needs --media-type.[/red]"); return
            console.print(f"[cyan]Input is dir. Processing {self.media_type} files in '{input_path}'...[/cyan]")
            temp_fpath = os.path.join(self.current_run_output_dir, f"temp_viz_results_{self.media_type}_{Path(input_path).name}.json")
            batch_data = self.execute_batch_processing(input_path, 
                self.config_manager.get_default("default_threshold", 0.5), 
                self.config_manager.get_default("default_ensemble_method", "voting"), 
                output_file_path=temp_fpath, display_metrics=False, generate_plots=False # No metrics/plots from this sub-run
            )
            if batch_data and "results" in batch_data: results_list = batch_data["results"]
            else: console.print(f"[red]Failed to process dir '{input_path}' for visualization.[/red]"); return
        else: console.print(f"[red]Input path '{input_path}' not a valid file or directory.[/red]"); return

        if not results_list: console.print("[red]No results to visualize.[/red]"); return
        
        valid_results = [r for r in results_list if isinstance(r, dict) and "error" not in r]
        if not valid_results: console.print("[yellow]No valid (non-error) results for visualizations.[/yellow]"); return

        if plot_type in ["all", "confusion"]: self.visualizer.plot_confusion_matrices(valid_results)
        if plot_type in ["all", "roc"]: self.visualizer.plot_roc_curves(valid_results)
        if plot_type in ["all", "probability"]: self.visualizer.plot_probability_distributions(valid_results)
        
        console.print(f"\n[green]Visualizations generated in {os.path.abspath(self.current_run_output_dir)}[/green]")


# --- Main CLI Parsing and Execution ---
def main_cli():
    parser = argparse.ArgumentParser(
        description="DeepSafe Unified Testing Tool & Experimentation Suite.",
        formatter_class=argparse.RawTextHelpFormatter, # Allows newlines in help
        epilog=__doc__ # Use the module docstring as epilog
    )
    parser.add_argument( "--output-dir", type=str, default=None,
        help="Base directory for saving all results. A timestamped subfolder is created here. Overrides config.")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)

    # --- Parent parser for --media-type ---
    media_type_parser_parent = argparse.ArgumentParser(add_help=False)
    media_type_parser_parent.add_argument( "--media-type", type=str, choices=["image", "video", "audio"], required=True,
        help="Type of media to process (image, video, audio). REQUIRED for this command.")
    
    # --- Parent parser for optional --media-type ---
    media_type_parser_optional_parent = argparse.ArgumentParser(add_help=False)
    media_type_parser_optional_parent.add_argument( "--media-type", type=str, choices=["image", "video", "audio"],
        help="Type of media (image, video, audio). Required if --input-path is a directory for 'visualize'.")

    # --- Commands ---
    health_p = subparsers.add_parser("health", help="Check system health.", parents=[media_type_parser_parent])
    
    test_p = subparsers.add_parser("test", help="Test a single media file.", parents=[media_type_parser_parent])
    test_p.add_argument("--input", type=str, required=True, dest="media_path", help="Path to media file.")
    test_p.add_argument("--threshold", type=float, help="Classification threshold (0.0-1.0).")
    test_p.add_argument("--model", type=str, dest="specific_model_name", help="Specific model name to test.")
    test_p.add_argument("--method", type=str, dest="ensemble_method", choices=["voting", "average", "stacking"], help="Ensemble method for Main API call.")
    test_p.add_argument("--output-file", type=str, help="Specific output JSON file for single test results.")

    batch_p = subparsers.add_parser("batch", help="Process a batch of media files.", parents=[media_type_parser_parent])
    batch_p.add_argument("--input-dir", type=str, required=True, help="Input directory (Real/Fake subdirs ideal).")
    batch_p.add_argument("--model", type=str, dest="specific_model_name", help="Specific model to run, or all if omitted.")
    batch_p.add_argument("--threshold", type=float, help="Classification threshold.")
    batch_p.add_argument("--method", dest="local_ensemble_method", type=str, choices=["voting", "average"], help="Local ensemble method.")
    batch_p.add_argument("--output-file", type=str, help="Specific output JSON file for batch results.")

    compare_p = subparsers.add_parser("compare", help="Compare model performance on a batch.", parents=[media_type_parser_parent])
    compare_p.add_argument("--input-dir", type=str, required=True, help="Input directory (Real/Fake subdirs).")
    compare_p.add_argument("--model", type=str, dest="specific_model_name", help="Specific model, or all if omitted.")
    compare_p.add_argument("--threshold", type=float, help="Classification threshold.")
    compare_p.add_argument("--method", dest="local_ensemble_method", type=str, choices=["voting", "average"], help="Local ensemble method.")

    apistack_p = subparsers.add_parser("apistacktest", help="Test API's stacking ensemble on a batch.", parents=[media_type_parser_parent])
    apistack_p.add_argument("--input-dir", type=str, required=True, help="Input directory (Real/Fake subdirs).")
    apistack_p.add_argument("--threshold", type=float, help="Threshold for API's stacking decision.")

    benchmark_p = subparsers.add_parser("benchmark", help="Benchmark API and model latency.", parents=[media_type_parser_parent])
    benchmark_p.add_argument("--input", type=str, required=True, dest="media_path", help="Path to a media file for benchmark.")
    benchmark_p.add_argument("--count", type=int, default=10, help="Number of iterations.")

    visualize_p = subparsers.add_parser("visualize", help="Generate visualizations.", parents=[media_type_parser_optional_parent])
    visualize_p.add_argument("--input-path", type=str, required=True, help="Path to results JSON file OR a directory of media.")
    visualize_p.add_argument("--plot-type", type=str, default="all", choices=["all", "confusion", "roc", "probability"], help="Type of visualization.")

    args = parser.parse_args()

    # Set defaults from config if not provided in args
    if hasattr(args, "threshold") and args.threshold is None:
        args.threshold = config_manager.get_default("default_threshold", 0.5)
    if hasattr(args, "ensemble_method") and args.ensemble_method is None:
        args.ensemble_method = config_manager.get_default("default_ensemble_method", "stacking")
    if hasattr(args, "local_ensemble_method") and args.local_ensemble_method is None:
        args.local_ensemble_method = config_manager.get_default("default_local_ensemble_method", "voting") # Add to config if needed

    # Media type logic for orchestrator initialization
    orchestrator_media_type = args.media_type if hasattr(args, "media_type") and args.media_type else None
    if args.command == "visualize" and os.path.isdir(args.input_path) and not orchestrator_media_type:
        parser.error("Command 'visualize' requires --media-type when --input-path is a directory.")
    
    orchestrator = DeepSafeTestOrchestrator(orchestrator_media_type, cli_output_dir_base=args.output_dir)

    # Execute command
    try:
        if args.command == "health":
            orchestrator.execute_health_check()
        elif args.command == "test":
            if args.specific_model_name and args.specific_model_name not in config_manager.get_all_model_names(args.media_type):
                console.print(f"[bold red]Error: Model '{args.specific_model_name}' not in config for '{args.media_type}'.[/bold red]\nAvailable: {config_manager.get_all_model_names(args.media_type)}")
                sys.exit(1)
            orchestrator.execute_single_test(args.media_path, args.threshold, args.ensemble_method, args.specific_model_name, args.output_file)
        elif args.command == "batch":
            if args.specific_model_name and args.specific_model_name not in config_manager.get_all_model_names(args.media_type):
                console.print(f"[bold red]Error: Model '{args.specific_model_name}' not in config for '{args.media_type}'.[/bold red]"); sys.exit(1)
            orchestrator.execute_batch_processing(args.input_dir, args.threshold, args.local_ensemble_method, args.specific_model_name, args.output_file)
        elif args.command == "compare":
            if args.specific_model_name and args.specific_model_name not in config_manager.get_all_model_names(args.media_type):
                console.print(f"[bold red]Error: Model '{args.specific_model_name}' not in config for '{args.media_type}'.[/bold red]"); sys.exit(1)
            orchestrator.execute_comparison(args.input_dir, args.threshold, args.local_ensemble_method, args.specific_model_name)
        elif args.command == "apistacktest":
            orchestrator.execute_api_stacking_test(args.input_dir, args.threshold)
        elif args.command == "benchmark":
            orchestrator.execute_benchmark(args.media_path, args.count)
        elif args.command == "visualize":
            orchestrator.execute_visualization(args.input_path, args.plot_type)
        else: # Should not be reached if subparsers are correct
            console.print(f"[bold red]Unknown command: {args.command}[/bold red]"); parser.print_help(); sys.exit(1)

    except Exception as e:
        console.print(f"\n[bold red]An error occurred executing command '{args.command}': {e}[/bold red]")
        # import traceback
        # console.print(Panel(traceback.format_exc(), title="[bold red]Traceback[/bold red]", border_style="red"))
        # Re-raise to be caught by the main try-except block for consistent exit
        raise


if __name__ == "__main__":
    if not config_manager.is_config_loaded_successfully():
        console.print("[bold red]Exiting due to configuration load failure. Please check `config/deepsafe_config.json`.[/bold red]")
        sys.exit(1)
    try:
        main_cli()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user.[/bold yellow]")
        sys.exit(0)
    except ValueError as ve: # Catch config/input errors from orchestrator or utils
        console.print(f"\n[bold red]Error: {ve}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]An unexpected Toplevel error occurred: {e}[/bold red]")
        import traceback
        console.print(Panel(traceback.format_exc(), title="[bold red]Traceback[/bold red]", border_style="red"))
        sys.exit(1)