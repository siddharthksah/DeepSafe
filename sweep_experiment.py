#!/usr/bin/env python3
"""
DeepSafe Experiment Runner - CPU Optimized - Threshold Optimization

Runs a comprehensive evaluation of individual models and ensemble methods
within the DeepSafe system on a provided dataset (Fake/Real folders).
Evaluates performance across a range of thresholds to find the optimal
value for each method, aiding in model/ensemble selection and deployment tuning.

Usage:
  ./experiment.py --input_dir path/to/experiment/folder \
                  --output_dir path/to/output/results \
                  [--threshold_step 0.05]
"""

import os
import glob
import base64
import requests
import time
import json
import gc
import argparse
import sys
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel

# --- Configuration ---
console = Console()

# Model endpoints configuration (Match your docker-compose setup)
MODEL_ENDPOINTS = {
    "cnndetection": "http://localhost:5008/predict",
    "ganimagedetection": "http://localhost:5001/predict",
    "universalfakedetect": "http://localhost:5002/predict",
    "hifi_ifdl": "http://localhost:5003/predict",
    "npr_deepfakedetection": "http://localhost:5004/predict",
    "dmimagedetection": "http://localhost:5005/predict",
    "caddm": "http://localhost:5006/predict",
    "faceforensics_plus_plus": "http://localhost:5007/predict",
}

# We will calculate ensembles manually based on individual model probabilities
ENSEMBLE_METHODS_TO_CALC = ["voting", "average"]

REQUEST_TIMEOUT = 1200  # Increased timeout for CPU processing (seconds)
RETRY_DELAY = 5  # Seconds between retries
MAX_RETRIES = 2

# Default threshold to pass to individual APIs (just for API call structure, prediction is ignored)
DEFAULT_API_THRESHOLD = 0.5

# --- Helper Functions ---

def encode_image(image_path: str) -> str:
    """Encode an image file to base64."""
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        console.print(f"[bold red]Error encoding image {image_path}: {e}[/bold red]")
        return ""

def clear_memory():
    """Explicitly run garbage collection."""
    gc.collect()

def find_image_files(input_dir: str) -> List[Tuple[str, str]]:
    """Find image files in Fake and Real subdirs."""
    images = []
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    fake_dir = os.path.join(input_dir, "Fake")
    real_dir = os.path.join(input_dir, "Real")
    if not os.path.isdir(fake_dir): console.print(f"[bold red]Error: 'Fake' subdirectory not found in {input_dir}[/bold red]")
    else:
        for ext in supported_extensions: images.extend([(p, "Fake") for p in glob.glob(os.path.join(fake_dir, ext))])
    if not os.path.isdir(real_dir): console.print(f"[bold red]Error: 'Real' subdirectory not found in {input_dir}[/bold red]")
    else:
        for ext in supported_extensions: images.extend([(p, "Real") for p in glob.glob(os.path.join(real_dir, ext))])
    if not images: console.print(f"[bold red]Error: No images found in {input_dir}/Fake or {input_dir}/Real[/bold red]"); sys.exit(1)
    return images

def query_individual_model(model_name: str, api_url: str, image_b64: str) -> Dict[str, Any]:
    """Query a single model endpoint, focusing on getting the probability."""
    # Pass a dummy threshold; we only care about the returned probability
    payload = {"image": image_b64, "threshold": DEFAULT_API_THRESHOLD}
    result = {"model_name": model_name, "error": None, "probability": None, "inference_time": None}
    start_time = time.time()

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(api_url, json=payload, timeout=REQUEST_TIMEOUT)
            result["inference_time"] = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                result["probability"] = data.get("probability") # Primary goal
                # Get inference time if available under common keys
                if "inference_time" in data: result["inference_time"] = data["inference_time"]
                elif "inference_time_seconds" in data: result["inference_time"] = data["inference_time_seconds"]
                if result["probability"] is None or np.isnan(result["probability"]):
                     result["error"] = "Model returned invalid probability" # Treat invalid prob as error
                     console.print(f"[yellow]Warning: {model_name} returned invalid probability for an image.[/yellow]")
                return result # Return success or error after processing
            else:
                result["error"] = f"HTTP Error {response.status_code}: {response.text}"
                if attempt < MAX_RETRIES:
                    # console.print(f"[yellow]Warning:[/yellow] {model_name} failed ({result['error']}). Retrying {attempt+1}/{MAX_RETRIES}...")
                    time.sleep(RETRY_DELAY); continue
                return result # Return error after max retries

        except requests.exceptions.Timeout:
            result["error"] = "Request Timeout"; result["inference_time"] = time.time() - start_time
            if attempt < MAX_RETRIES:
                # console.print(f"[yellow]Warning:[/yellow] {model_name} timed out. Retrying {attempt+1}/{MAX_RETRIES}...")
                time.sleep(RETRY_DELAY); continue
            return result # Return error after max retries

        except Exception as e:
            result["error"] = f"Request Exception: {str(e)}"; result["inference_time"] = time.time() - start_time
            if attempt < MAX_RETRIES:
                # console.print(f"[red]Error:[/red] querying {model_name}: {e}. Retrying {attempt+1}/{MAX_RETRIES}...")
                time.sleep(RETRY_DELAY); continue
            return result # Return error after max retries
    return result # Should not be reached, but ensures return

def calculate_manual_ensembles(model_results: Dict[str, float], vote_threshold: float = 0.5) -> Dict[str, Optional[float]]:
    """Calculate ensemble probabilities from individual model probabilities."""
    ensemble_probs = {f"ensemble_{m}": None for m in ENSEMBLE_METHODS_TO_CALC}
    valid_probs = [p for p in model_results.values() if p is not None and not np.isnan(p)]

    if not valid_probs:
        return ensemble_probs # Return None if no valid model probs

    # Average ensemble probability
    if "average" in ENSEMBLE_METHODS_TO_CALC:
        ensemble_probs["ensemble_average"] = np.mean(valid_probs)

    # Voting ensemble probability (proportion voting fake based on a fixed internal threshold)
    if "voting" in ENSEMBLE_METHODS_TO_CALC:
        votes_fake = sum(1 for p in valid_probs if p >= vote_threshold)
        ensemble_probs["ensemble_voting"] = votes_fake / len(valid_probs) if valid_probs else 0.0

    return ensemble_probs

def calculate_metrics_for_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Optional[float]]:
    """Calculate binary classification metrics for a specific threshold."""
    metrics = {"threshold": threshold, "accuracy": None, "precision": None, "recall": None, "f1_score": None, "specificity": None}

    # Filter out samples where probability is invalid (NaN)
    valid_indices = ~np.isnan(y_prob)
    if not np.any(valid_indices): return metrics # Cannot calculate if no valid probabilities

    y_true_valid = y_true[valid_indices]
    y_prob_valid = y_prob[valid_indices]

    # Ensure y_true_valid still contains data after filtering based on y_prob
    if len(y_true_valid) == 0: return metrics

    y_pred_valid = (y_prob_valid >= threshold).astype(int) # Generate predictions based on threshold

    if len(np.unique(y_true_valid)) < 2:
        # Handle single class case
        metrics["accuracy"] = accuracy_score(y_true_valid, y_pred_valid)
        metrics["precision"] = precision_score(y_true_valid, y_pred_valid, labels=[0,1], pos_label=1, zero_division=0)
        metrics["recall"] = recall_score(y_true_valid, y_pred_valid, labels=[0,1], pos_label=1, zero_division=0)
        metrics["f1_score"] = f1_score(y_true_valid, y_pred_valid, labels=[0,1], pos_label=1, zero_division=0)
        try:
            cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except ValueError: metrics["specificity"] = None
    else:
        # Standard metrics calculation
        metrics["accuracy"] = accuracy_score(y_true_valid, y_pred_valid)
        metrics["precision"] = precision_score(y_true_valid, y_pred_valid, labels=[0,1], pos_label=1, zero_division=0)
        metrics["recall"] = recall_score(y_true_valid, y_pred_valid, labels=[0,1], pos_label=1, zero_division=0)
        metrics["f1_score"] = f1_score(y_true_valid, y_pred_valid, labels=[0,1], pos_label=1, zero_division=0)
        try:
             tn, fp, fn, tp = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1]).ravel()
             metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except ValueError: metrics["specificity"] = None

    return metrics

def find_optimal_threshold(threshold_metrics: List[Dict[str, float]], metric_to_optimize: str = 'f1_score') -> Tuple[Optional[float], Optional[Dict[str, float]]]:
    """Find the threshold yielding the best score for the specified metric."""
    if not threshold_metrics: return None, None

    best_score = -1.0
    optimal_threshold = None
    best_metrics_at_optimal = None

    # Filter out entries where the optimization metric is None or NaN
    valid_metrics = [m for m in threshold_metrics if m.get(metric_to_optimize) is not None and not np.isnan(m.get(metric_to_optimize))]
    if not valid_metrics:
        # If no valid scores for the primary metric, maybe return based on threshold=0.5? Or just None.
        # Let's return None for now.
        return None, None

    for metrics in valid_metrics:
        current_score = metrics[metric_to_optimize]
        if current_score >= best_score: # Use >= to favor higher thresholds in ties (less sensitive)
            # Tie-breaking: if F1 is the same, maybe prefer higher precision or closer to 0.5?
            # For now, just take the last one with the highest score (which implicitly favors higher threshold due to >=)
            best_score = current_score
            optimal_threshold = metrics['threshold']
            best_metrics_at_optimal = metrics

    return optimal_threshold, best_metrics_at_optimal


def plot_roc_curve(method_probs: Dict[str, np.ndarray], y_true_all: np.ndarray, output_dir: str):
    plt.figure(figsize=(12, 10))
    auc_scores = {}
    for method, y_prob in method_probs.items():
        valid_indices = ~np.isnan(y_true_all) & ~np.isnan(y_prob)
        if not np.any(valid_indices): continue
        y_true_valid = y_true_all[valid_indices]
        y_prob_valid = y_prob[valid_indices]
        if len(np.unique(y_true_valid)) < 2: continue
        try:
            fpr, tpr, _ = roc_curve(y_true_valid, y_prob_valid, pos_label=1)
            roc_auc = auc(fpr, tpr)
            auc_scores[method] = roc_auc
            if roc_auc is not None and not np.isnan(roc_auc):
                plt.plot(fpr, tpr, lw=2, label=f'{method} (AUC = {roc_auc:.3f})')
            else: plt.plot(fpr, tpr, lw=2, label=f'{method} (AUC = N/A)')
        except Exception as e:
            console.print(f"[yellow]Warning: Could not calculate ROC/AUC for {method}: {e}[/yellow]")
            auc_scores[method] = None
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)'); plt.ylabel('True Positive Rate (Recall / Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curves'); plt.legend(loc="lower right"); plt.grid(True)
    plot_path = os.path.join(output_dir, "roc_curves.png"); plt.savefig(plot_path); plt.close()
    console.print(f"[green]ROC curve plot saved to {plot_path}[/green]")
    return auc_scores # Return calculated AUCs


def plot_precision_recall_curve(method_probs: Dict[str, np.ndarray], y_true_all: np.ndarray, output_dir: str):
    plt.figure(figsize=(12, 10))
    ap_scores = {}
    for method, y_prob in method_probs.items():
        valid_indices = ~np.isnan(y_true_all) & ~np.isnan(y_prob)
        if not np.any(valid_indices): continue
        y_true_valid = y_true_all[valid_indices]
        y_prob_valid = y_prob[valid_indices]
        if len(np.unique(y_true_valid)) < 2: continue
        try:
            precision, recall, _ = precision_recall_curve(y_true_valid, y_prob_valid, pos_label=1)
            avg_precision = average_precision_score(y_true_valid, y_prob_valid, pos_label=1)
            ap_scores[method] = avg_precision
            if avg_precision is not None and not np.isnan(avg_precision):
                plt.plot(recall, precision, lw=2, label=f'{method} (AP = {avg_precision:.3f})')
            else: plt.plot(recall, precision, lw=2, label=f'{method} (AP = N/A)')
        except Exception as e:
             console.print(f"[yellow]Warning: Could not calculate Precision-Recall curve for {method}: {e}[/yellow]")
             ap_scores[method] = None
    plt.xlabel('Recall (Sensitivity)'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curves'); plt.legend(loc="lower left"); plt.grid(True); plt.ylim([0.0, 1.05])
    plot_path = os.path.join(output_dir, "precision_recall_curves.png"); plt.savefig(plot_path); plt.close()
    console.print(f"[green]Precision-Recall curve plot saved to {plot_path}[/green]")
    return ap_scores # Return calculated APs


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, optimal_threshold: Optional[float], method_name: str, output_dir: str):
    """Plot confusion matrix at the optimal threshold."""
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_prob)
    if not np.any(valid_indices): console.print(f"[yellow]Skipping CM for {method_name}: No valid data.[/yellow]"); return

    y_true_valid = y_true[valid_indices]
    y_prob_valid = y_prob[valid_indices]

    if optimal_threshold is None:
        # If no optimal found (e.g., F1 was always 0), maybe default to 0.5 or skip? Let's skip.
        console.print(f"[yellow]Skipping CM for {method_name}: No optimal threshold found.[/yellow]")
        return

    y_pred_valid = (y_prob_valid >= optimal_threshold).astype(int)
    if len(y_true_valid) == 0: console.print(f"[yellow]Skipping CM for {method_name}: Zero valid predictions.[/yellow]"); return

    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Real', 'Predicted Fake'], yticklabels=['Actual Real', 'Actual Fake'])
    plt.ylabel('Actual Label'); plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {method_name}\n(Optimal Thr={optimal_threshold:.3f}) | TN={tn}, FP={fp}, FN={fn}, TP={tp}')
    plot_path = os.path.join(output_dir, f"confusion_matrix_{method_name}_optimal.png"); plt.tight_layout(); plt.savefig(plot_path); plt.close()

# --- Main Execution Logic ---

def main():
    parser = argparse.ArgumentParser(description="Run DeepSafe evaluation experiment with threshold optimization.")
    parser.add_argument("--input_dir", required=True, help="Directory containing 'Fake' and 'Real' subfolders.")
    parser.add_argument("--output_dir", required=True, help="Directory to save results and plots.")
    parser.add_argument("--threshold_step", type=float, default=0.05, help="Step size for threshold evaluation range (e.g., 0.05).")
    args = parser.parse_args()

    start_timestamp = datetime.now().isoformat()

    # --- Setup ---
    if not os.path.isdir(args.input_dir): console.print(f"[bold red]Error: Input directory not found: {args.input_dir}[/bold red]"); sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots"); os.makedirs(plots_dir, exist_ok=True)
    data_dir = os.path.join(args.output_dir, "data"); os.makedirs(data_dir, exist_ok=True)

    threshold_range = np.arange(args.threshold_step, 1.0, args.threshold_step) # Start near 0, go up to near 1
    console.print(Panel(f"""
[bold cyan]DeepSafe Experiment Run (Threshold Optimization)[/bold cyan]
Input Directory: {args.input_dir}
Output Directory: {args.output_dir}
Threshold Range: {threshold_range[0]:.2f} to {threshold_range[-1]:.2f} (Step: {args.threshold_step})
Models: {', '.join(MODEL_ENDPOINTS.keys())}
Ensemble Methods: {', '.join(ENSEMBLE_METHODS_TO_CALC)}
Started: {start_timestamp}
""", title="Experiment Setup", border_style="blue"))
    console.print("[yellow]Note: This process runs sequentially on CPU and may take a very long time.[/yellow]")

    # --- Data Loading ---
    image_files = find_image_files(args.input_dir)
    num_images = len(image_files)
    console.print(f"Found {num_images} images ({len([f for f,l in image_files if l=='Fake'])} Fake, {len([f for f,l in image_files if l=='Real'])} Real).")

    # --- Prediction Phase (Collect Probabilities) ---
    raw_probabilities = [] # Store results like: {image_path, ground_truth, model_probs={...}, ensemble_probs={...}, errors={...}}
    total_tasks = num_images
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Collecting Probabilities...", total=total_tasks)
        for i, (img_path, ground_truth) in enumerate(image_files):
            img_name = os.path.basename(img_path)
            if (i+1) % 5 == 0 or i == 0: progress.update(task, description=f"[cyan]Image {i+1}/{num_images} ({img_name})")

            image_b64 = encode_image(img_path)
            if not image_b64: progress.advance(task); continue

            img_results = {"image_path": img_path, "image_name": img_name, "ground_truth": ground_truth,
                           "model_probs": {}, "ensemble_probs": {}, "errors": {}, "inference_times": {}}

            # Query individual models
            current_model_probs = {}
            for model_name, api_url in MODEL_ENDPOINTS.items():
                result = query_individual_model(model_name, api_url, image_b64)
                img_results["model_probs"][model_name] = result["probability"]
                img_results["inference_times"][model_name] = result["inference_time"]
                if result["error"]: img_results["errors"][model_name] = result["error"]
                current_model_probs[model_name] = result["probability"] # Keep track for ensemble calc
                clear_memory()

            # Calculate ensemble probabilities manually
            img_results["ensemble_probs"] = calculate_manual_ensembles(current_model_probs, vote_threshold=DEFAULT_API_THRESHOLD)
            # Estimate ensemble inference time roughly (sum of individual times - crude but gives an idea)
            valid_times = [t for t in img_results["inference_times"].values() if t is not None]
            approx_ensemble_time = sum(valid_times) if valid_times else None
            for ens_method in ENSEMBLE_METHODS_TO_CALC:
                img_results["inference_times"][f"ensemble_{ens_method}"] = approx_ensemble_time

            raw_probabilities.append(img_results)
            progress.advance(task)
            clear_memory() # After each image

    # Save raw probabilities (useful for detailed analysis)
    raw_prob_path = os.path.join(data_dir, "raw_probabilities.json")
    with open(raw_prob_path, 'w') as f: json.dump(raw_probabilities, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x if x is not None else None)
    console.print(f"\n[green]Raw probability results saved to {raw_prob_path}[/green]")


    # --- Analysis Phase (Calculate Metrics Across Thresholds) ---
    console.print("\n[bold cyan]Analyzing Performance Across Thresholds...[/bold cyan]")
    all_methods_metrics = {} # Key: method_name, Value: List of metric dicts per threshold
    method_probabilities = {} # Key: method_name, Value: numpy array of probabilities for all images
    y_true_map = {'Real': 0, 'Fake': 1}
    y_true_list = [y_true_map[res['ground_truth']] for res in raw_probabilities]
    y_true_all = np.array(y_true_list)

    eval_methods = list(MODEL_ENDPOINTS.keys()) + [f"ensemble_{m}" for m in ENSEMBLE_METHODS_TO_CALC]

    # Prepare probability arrays for each method
    for method in eval_methods:
         probs = []
         for res in raw_probabilities:
              if method.startswith("ensemble_"):
                   probs.append(res["ensemble_probs"].get(method))
              else:
                   probs.append(res["model_probs"].get(method))
         method_probabilities[method] = np.array(probs, dtype=float) # Ensure float, handles None -> nan


    # Calculate metrics for each threshold
    with Progress(TextColumn("Calculating metrics for {task.description} across thresholds..."), BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), console=console) as progress:
        task = progress.add_task("methods", total=len(eval_methods))
        for method in eval_methods:
            progress.update(task, description=method)
            all_methods_metrics[method] = []
            y_prob = method_probabilities[method]
            for threshold in threshold_range:
                metrics = calculate_metrics_for_threshold(y_true_all, y_prob, threshold)
                all_methods_metrics[method].append(metrics)
            progress.advance(task)

    # --- Find Optimal Thresholds and Best Metrics ---
    optimal_thresholds = {}
    best_metrics = {} # Store metrics at the optimal threshold + AUC
    auc_scores = {} # Store AUC separately

    console.print("\n[bold cyan]Finding Optimal Thresholds (Maximizing F1-Score)...[/bold cyan]")
    for method in eval_methods:
        threshold_metrics = all_methods_metrics[method]
        opt_thresh, best_metric_dict = find_optimal_threshold(threshold_metrics, metric_to_optimize='f1_score')
        optimal_thresholds[method] = opt_thresh
        # Initialize best_metrics dict for the method
        best_metrics[method] = {"optimal_threshold_f1": opt_thresh, "auc": None}
        if best_metric_dict:
             # Copy metrics achieved at the optimal threshold
             best_metrics[method].update({k: v for k, v in best_metric_dict.items() if k != 'threshold'})
        else:
             # If no optimal found, fill with N/A? Or calculate at 0.5? Let's put N/A.
             console.print(f"[yellow]Warning: No optimal threshold found for {method} based on F1-Score.[/yellow]")
             metrics_at_default = calculate_metrics_for_threshold(y_true_all, method_probabilities[method], 0.5)
             best_metrics[method].update({k: v for k, v in metrics_at_default.items() if k != 'threshold'})


    # --- Generate Plots (Requires Probabilities) ---
    console.print("\n[bold cyan]Generating Plots...[/bold cyan]")
    # Calculate AUC once per method (it's threshold independent)
    try:
        auc_scores = plot_roc_curve(method_probabilities, y_true_all, plots_dir)
        for method, auc_val in auc_scores.items():
            if method in best_metrics:
                best_metrics[method]["auc"] = auc_val
    except Exception as e:
         console.print(f"[bold red]Error generating ROC plot: {e}[/bold red]")


    # Generate Precision-Recall Curves
    try:
        plot_precision_recall_curve(method_probabilities, y_true_all, plots_dir)
    except Exception as e:
         console.print(f"[bold red]Error generating P-R plot: {e}[/bold red]")

    # Plot Confusion Matrices at Optimal Threshold
    console.print(f"Generating confusion matrices (at optimal F1 threshold) in: {plots_dir}")
    for method in eval_methods:
         plot_confusion_matrix(y_true_all, method_probabilities[method], optimal_thresholds.get(method), method, plots_dir)


    # --- Reporting Phase ---
    console.print("\n[bold cyan]Performance Summary at Optimal Threshold (Max F1-Score):[/bold cyan]")
    summary_table = Table(title="Optimal Performance Summary")
    summary_table.add_column("Method", style="cyan", no_wrap=True)
    summary_table.add_column("Optimal Thr\n(for F1)", style="yellow") # Clarify basis
    summary_table.add_column("AUC", style="white")
    summary_table.add_column("Best F1", style="yellow")
    summary_table.add_column("Accuracy\n(at opt F1 thr)", style="green") # Clarify
    summary_table.add_column("Precision\n(at opt F1 thr)", style="blue")
    summary_table.add_column("Recall\n(at opt F1 thr)", style="magenta")
    summary_table.add_column("Specificity\n(at opt F1 thr)", style="red")

    # Sort methods by AUC for presentation
    sorted_methods = sorted(eval_methods, key=lambda m: best_metrics.get(m, {}).get("auc", -1) if best_metrics.get(m, {}).get("auc") is not None else -1, reverse=True)


    final_summary_list = [] # For creating DataFrame and JSON

    for method in sorted_methods:
        metrics = best_metrics.get(method, {})
        opt_thr = metrics.get('optimal_threshold_f1')
        final_summary_list.append({ # Store for JSON
             "method": method,
             **metrics # Add all metrics found at optimal F1 threshold + AUC
        })
        summary_table.add_row(
            method,
            f"{opt_thr:.3f}" if opt_thr is not None else "N/A",
            f"{metrics.get('auc'):.3f}" if metrics.get('auc') is not None else "N/A",
            f"{metrics.get('f1_score'):.3f}" if metrics.get('f1_score') is not None else "N/A",
            f"{metrics.get('accuracy'):.3f}" if metrics.get('accuracy') is not None else "N/A",
            f"{metrics.get('precision'):.3f}" if metrics.get('precision') is not None else "N/A",
            f"{metrics.get('recall'):.3f}" if metrics.get('recall') is not None else "N/A",
            f"{metrics.get('specificity'):.3f}" if metrics.get('specificity') is not None else "N/A",
        )
    console.print(summary_table)

    # --- Ranking and Highlights ---
    # Convert best_metrics dict to DataFrame for easier ranking
    summary_df = pd.DataFrame(final_summary_list).set_index('method')
    summary_df_rank = summary_df.replace([None, 'N/A'], np.nan).astype(float) # Convert N/A to NaN for sorting

    # Rank by AUC first, then by Best F1
    ranked_df = summary_df_rank.sort_values(by=['auc', 'f1_score'], ascending=[False, False])

    console.print("\n[bold cyan]Ranked Performance (by AUC, then Best F1-Score):[/bold cyan]")
    ranked_table = Table(title="Ranked Optimal Performance")
    ranked_table.add_column("Rank", style="bold white")
    ranked_table.add_column("Method", style="cyan")
    ranked_table.add_column("AUC", style="white")
    ranked_table.add_column("Best F1", style="yellow")
    ranked_table.add_column("Optimal Thr", style="yellow")
    ranked_table.add_column("Recall", style="magenta")
    ranked_table.add_column("Precision", style="blue")

    for i, (method, row) in enumerate(ranked_df.iterrows()):
        ranked_table.add_row(
            str(i+1), method,
            f"{row['auc']:.3f}" if pd.notna(row['auc']) else "N/A",
            f"{row['f1_score']:.3f}" if pd.notna(row['f1_score']) else "N/A",
            f"{row['optimal_threshold_f1']:.3f}" if pd.notna(row['optimal_threshold_f1']) else "N/A",
            f"{row['recall']:.3f}" if pd.notna(row['recall']) else "N/A",
            f"{row['precision']:.3f}" if pd.notna(row['precision']) else "N/A"
        )
    console.print(ranked_table)

    # Identify top performers based on AUC ranking
    top_overall = ranked_df.index[0] if not ranked_df.empty else "N/A"
    top_ensemble = next((m for m in ranked_df.index if m.startswith("ensemble_")), "N/A")
    top_individual = next((m for m in ranked_df.index if not m.startswith("ensemble_")), "N/A")

    console.print("\n[bold cyan]Experiment Highlights:[/bold cyan]")
    console.print(f"- Overall Best Performer (by AUC): [bold magenta]{top_overall}[/bold magenta] (Achieved Best F1={ranked_df.loc[top_overall,'f1_score']:.3f} at Thr ≈ {ranked_df.loc[top_overall,'optimal_threshold_f1']:.3f})")
    console.print(f"- Best Ensemble Method (by AUC): [bold magenta]{top_ensemble}[/bold magenta] (Achieved Best F1={ranked_df.loc[top_ensemble,'f1_score']:.3f} at Thr ≈ {ranked_df.loc[top_ensemble,'optimal_threshold_f1']:.3f})")
    console.print(f"- Best Individual Model (by AUC): [bold magenta]{top_individual}[/bold magenta] (Achieved Best F1={ranked_df.loc[top_individual,'f1_score']:.3f} at Thr ≈ {ranked_df.loc[top_individual,'optimal_threshold_f1']:.3f})")

    # Add comparison details based on AUC
    if top_ensemble != "N/A" and top_individual != "N/A":
        top_ensemble_auc = ranked_df.loc[top_ensemble, 'auc']
        top_individual_auc = ranked_df.loc[top_individual, 'auc']
        if pd.notna(top_ensemble_auc) and pd.notna(top_individual_auc):
             if top_ensemble_auc > top_individual_auc:
                  console.print(f"- The best ensemble ({top_ensemble}, AUC: {top_ensemble_auc:.3f}) outperformed the best individual model ({top_individual}, AUC: {top_individual_auc:.3f}) in overall discrimination ability.")
             elif top_individual_auc > top_ensemble_auc:
                  console.print(f"- The best individual model ({top_individual}, AUC: {top_individual_auc:.3f}) outperformed the best ensemble ({top_ensemble}, AUC: {top_ensemble_auc:.3f}) in overall discrimination ability.")
             else:
                  console.print(f"- The best ensemble and individual model have similar AUC performance (≈{top_ensemble_auc:.3f}).")

    # --- Save Detailed Threshold Metrics and Summary ---
    final_output = {
        "experiment_metadata": {
            "timestamp": start_timestamp,
            "input_directory": args.input_dir,
            "output_directory": args.output_dir,
            "threshold_step": args.threshold_step,
            "tested_thresholds": threshold_range.tolist(),
            "num_images": num_images,
            "fake_count": len([f for f,l in image_files if l=='Fake']),
            "real_count": len([f for f,l in image_files if l=='Real']),
        },
        "metrics_per_threshold": all_methods_metrics, # Metrics for every threshold tested
        "best_performance_summary": final_summary_list, # Summary at optimal F1 threshold + AUC
        "performance_ranking_auc_f1": ranked_df.where(pd.notna(ranked_df), None).reset_index().to_dict(orient='records'), # Save ranked list (convert NaN to None)
        "top_performers_auc": { # Based on AUC ranking
            "overall_best": top_overall,
            "best_ensemble": top_ensemble,
            "best_individual": top_individual
        }
    }

    summary_path = os.path.join(data_dir, "threshold_experiment_summary.json")
    # Helper for JSON serialization
    def convert_numpy_nan(obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        elif isinstance(obj, (np.floating, np.float64)): return float(obj) if pd.notna(obj) else None # Convert NaN to None
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
        elif isinstance(obj, np.bool_): return bool(obj)
        return obj # Default pass-through

    with open(summary_path, 'w') as f:
        json.dump(final_output, f, indent=4, default=convert_numpy_nan)
    console.print(f"\n[green]Detailed threshold metrics and summary saved to {summary_path}[/green]")


    console.print(Panel("[bold green]Experiment Finished Successfully![/bold green]", border_style="green"))
    console.print(f"Results, stats, and plots saved in: {args.output_dir}")


if __name__ == "__main__":
    main()