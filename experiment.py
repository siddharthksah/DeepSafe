#!/usr/bin/env python3
"""
DeepSafe Experiment Runner - CPU Optimized - Focused on Model/Ensemble Selection

Runs a comprehensive evaluation of individual models and ensemble methods
within the DeepSafe system on a provided dataset (Fake/Real folders).
Outputs detailed metrics, rankings, and plots to aid in selecting the
best performing models and ensemble strategy for deepfake detection.

Usage:
  ./experiment.py --input_dir path/to/experiment/folder \
                  --output_dir path/to/output/results \
                  [--threshold 0.5]
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
    confusion_matrix, roc_curve, auc, precision_recall_curve
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

# Main API endpoint
MAIN_API_URL = "http://localhost:8000/predict"

# Ensemble methods to test
ENSEMBLE_METHODS = ["voting", "average"]

REQUEST_TIMEOUT = 1200  # Increased timeout for CPU processing (seconds)
RETRY_DELAY = 5  # Seconds between retries
MAX_RETRIES = 2

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

    if not os.path.isdir(fake_dir):
        console.print(f"[bold red]Error: 'Fake' subdirectory not found in {input_dir}[/bold red]")
    else:
        for ext in supported_extensions:
            images.extend([(p, "Fake") for p in glob.glob(os.path.join(fake_dir, ext))])

    if not os.path.isdir(real_dir):
        console.print(f"[bold red]Error: 'Real' subdirectory not found in {input_dir}[/bold red]")
    else:
        for ext in supported_extensions:
            images.extend([(p, "Real") for p in glob.glob(os.path.join(real_dir, ext))])

    if not images:
        console.print(f"[bold red]Error: No images found in {input_dir}/Fake or {input_dir}/Real[/bold red]")
        sys.exit(1)

    return images

def query_individual_model(model_name: str, api_url: str, image_b64: str, threshold: float) -> Dict[str, Any]:
    """Query a single model endpoint with retries."""
    payload = {"image": image_b64, "threshold": threshold}
    result = {"model_name": model_name, "error": None, "probability": None, "prediction": None, "inference_time": None}
    start_time = time.time()

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(api_url, json=payload, timeout=REQUEST_TIMEOUT)
            result["inference_time"] = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                result["probability"] = data.get("probability")
                if "prediction" in data:
                     result["prediction"] = data.get("prediction")
                elif result["probability"] is not None:
                     result["prediction"] = 1 if result["probability"] >= threshold else 0
                if "inference_time" in data:
                     result["inference_time"] = data["inference_time"]
                elif "inference_time_seconds" in data:
                     result["inference_time"] = data["inference_time_seconds"]
                return result
            else:
                result["error"] = f"HTTP Error {response.status_code}: {response.text}"
                # console.print(f"[yellow]Warning:[/yellow] {model_name} failed ({result['error']}). Attempt {attempt+1}/{MAX_RETRIES+1}")
                if attempt < MAX_RETRIES: time.sleep(RETRY_DELAY); continue
                return result

        except requests.exceptions.Timeout:
            result["error"] = "Request Timeout"
            result["inference_time"] = time.time() - start_time
            # console.print(f"[yellow]Warning:[/yellow] {model_name} timed out. Attempt {attempt+1}/{MAX_RETRIES+1}")
            if attempt < MAX_RETRIES: time.sleep(RETRY_DELAY); continue
            return result

        except Exception as e:
            result["error"] = f"Request Exception: {str(e)}"
            result["inference_time"] = time.time() - start_time
            # console.print(f"[red]Error:[/red] querying {model_name}: {e}. Attempt {attempt+1}/{MAX_RETRIES+1}")
            if attempt < MAX_RETRIES: time.sleep(RETRY_DELAY); continue
            return result

    return result

def query_api_ensemble(ensemble_method: str, image_b64: str, threshold: float) -> Dict[str, Any]:
    """Query the main API endpoint for an ensemble result."""
    payload = {"image": image_b64, "threshold": threshold, "ensemble_method": ensemble_method}
    result = {"ensemble_method": ensemble_method, "error": None, "probability": None, "prediction": None, "inference_time": None}
    start_time = time.time()

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(MAIN_API_URL, json=payload, timeout=REQUEST_TIMEOUT * 2) # Allow longer timeout for API
            result["inference_time"] = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                verdict = data.get("verdict", "undetermined")
                confidence = data.get("confidence", 0.0)
                if verdict == "fake":
                    result["probability"] = confidence # Probability of being FAKE
                    result["prediction"] = 1
                elif verdict == "real":
                     result["probability"] = 1.0 - confidence # Probability of being FAKE
                     result["prediction"] = 0
                else:
                     result["probability"] = 0.5
                     result["prediction"] = None
                result["inference_time"] = data.get("inference_time")
                return result
            else:
                result["error"] = f"HTTP Error {response.status_code}: {response.text}"
                # console.print(f"[yellow]Warning:[/yellow] API ({ensemble_method}) failed ({result['error']}). Attempt {attempt+1}/{MAX_RETRIES+1}")
                if attempt < MAX_RETRIES: time.sleep(RETRY_DELAY); continue
                return result

        except requests.exceptions.Timeout:
            result["error"] = "Request Timeout"
            result["inference_time"] = time.time() - start_time
            # console.print(f"[yellow]Warning:[/yellow] API ({ensemble_method}) timed out. Attempt {attempt+1}/{MAX_RETRIES+1}")
            if attempt < MAX_RETRIES: time.sleep(RETRY_DELAY); continue
            return result

        except Exception as e:
            result["error"] = f"Request Exception: {str(e)}"
            result["inference_time"] = time.time() - start_time
            # console.print(f"[red]Error:[/red] querying API ({ensemble_method}): {e}. Attempt {attempt+1}/{MAX_RETRIES+1}")
            if attempt < MAX_RETRIES: time.sleep(RETRY_DELAY); continue
            return result

    return result

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Calculate classification metrics.
    Args:
        y_true: Ground truth labels (0 for Real, 1 for Fake)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probabilities for the 'Fake' class (class 1)
    Returns:
        Dictionary of metrics.
    """
    metrics = {
        "accuracy": None, # Overall correctness
        "precision": None, # Of predicted Fakes, how many are actual Fakes? (TP / (TP + FP)) - Minimizes false accusations
        "recall": None, # Of actual Fakes, how many were caught? (TP / (TP + FN)) - Minimizes missed fakes (Sensitivity)
        "f1_score": None, # Harmonic mean of Precision and Recall - Good balance metric
        "specificity": None, # Of actual Reals, how many were correctly identified? (TN / (TN + FP)) - Minimizes misclassifying real images
        "auc": None, # Area under ROC curve - Overall model discriminative ability, threshold-independent
        "num_samples": len(y_true),
        "num_errors": 0 # Number of samples where prediction failed
    }

    # Filter out samples where prediction failed (y_pred is None or NaN)
    valid_indices = [i for i, p in enumerate(y_pred) if p is not None and not np.isnan(p)]

    if not valid_indices:
        metrics["num_errors"] = len(y_true)
        console.print("[yellow]Warning: No valid predictions found for this method. All metrics are N/A.[/yellow]")
        return metrics

    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices].astype(int) # Ensure integer type for metrics
    y_prob_valid = y_prob[valid_indices]

    metrics["num_errors"] = len(y_true) - len(valid_indices)

    if len(np.unique(y_true_valid)) < 2:
        console.print("[yellow]Warning: Only one class present in valid predictions. AUC and some other metrics may be undefined or misleading.[/yellow]")
        metrics["accuracy"] = accuracy_score(y_true_valid, y_pred_valid)
        metrics["precision"] = precision_score(y_true_valid, y_pred_valid, labels=[0,1], pos_label=1, zero_division=0)
        metrics["recall"] = recall_score(y_true_valid, y_pred_valid, labels=[0,1], pos_label=1, zero_division=0)
        metrics["f1_score"] = f1_score(y_true_valid, y_pred_valid, labels=[0,1], pos_label=1, zero_division=0)
        try:
            # Ensure labels include both 0 and 1 even if only one is present in y_true_valid/y_pred_valid
            cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except ValueError:
             metrics["specificity"] = None # Should not happen with labels=[0,1]
        metrics["auc"] = None # AUC not defined for single class
    else:
        # Standard metrics calculation
        metrics["accuracy"] = accuracy_score(y_true_valid, y_pred_valid)
        metrics["precision"] = precision_score(y_true_valid, y_pred_valid, labels=[0,1], pos_label=1, zero_division=0)
        metrics["recall"] = recall_score(y_true_valid, y_pred_valid, labels=[0,1], pos_label=1, zero_division=0)
        metrics["f1_score"] = f1_score(y_true_valid, y_pred_valid, labels=[0,1], pos_label=1, zero_division=0)
        try:
             tn, fp, fn, tp = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1]).ravel()
             metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        except ValueError:
             metrics["specificity"] = None
        # Calculate AUC
        valid_prob_indices = [i for i, p in enumerate(y_prob_valid) if p is not None and not np.isnan(p)]
        if len(valid_prob_indices) == len(y_true_valid):
             fpr, tpr, _ = roc_curve(y_true_valid, y_prob_valid, pos_label=1)
             metrics["auc"] = auc(fpr, tpr)
        else:
             console.print("[yellow]Warning: Some probabilities are invalid. Cannot calculate AUC accurately.[/yellow]")
             # Attempt AUC with valid probs only - might be less representative
             y_true_auc = y_true_valid[valid_prob_indices]
             y_prob_auc = y_prob_valid[valid_prob_indices]
             if len(np.unique(y_true_auc)) > 1:
                 fpr, tpr, _ = roc_curve(y_true_auc, y_prob_auc, pos_label=1)
                 metrics["auc"] = auc(fpr, tpr)
             else:
                 metrics["auc"] = None

    return metrics

def plot_roc_curve(results_df: pd.DataFrame, output_dir: str):
    """Plot ROC curves for all models and methods."""
    plt.figure(figsize=(12, 10))

    model_names = list(MODEL_ENDPOINTS.keys())
    eval_methods = model_names + [f"ensemble_{m}" for m in ENSEMBLE_METHODS]

    y_true_map = {'Real': 0, 'Fake': 1}
    # Ensure alignment and handle potential missing values robustly
    base_df = pd.DataFrame({
        'ground_truth': results_df['ground_truth'].map(y_true_map),
        'image_path': results_df['image_path']
    }).drop_duplicates(subset=['image_path']).set_index('image_path')


    for method in eval_methods:
        method_df = results_df[results_df['method'] == method][['image_path', 'probability']].copy()
        method_df = method_df.drop_duplicates(subset=['image_path']).set_index('image_path')

        # Merge probabilities with the base ground truth, ensuring alignment
        merged_df = base_df.join(method_df, how='left')
        valid_df = merged_df.dropna(subset=['probability', 'ground_truth']) # Drop if prob or truth is NaN

        if valid_df.empty:
            console.print(f"[yellow]Skipping ROC for {method}: No valid probabilities found.[/yellow]")
            continue

        y_true = valid_df['ground_truth'].values
        y_prob = valid_df['probability'].values

        if len(np.unique(y_true)) < 2:
             console.print(f"[yellow]Skipping ROC for {method}: Only one class present in valid data.[/yellow]")
             continue

        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)
        if roc_auc is not None and not np.isnan(roc_auc):
            plt.plot(fpr, tpr, lw=2, label=f'{method} (AUC = {roc_auc:.3f})')
        else:
            plt.plot(fpr, tpr, lw=2, label=f'{method} (AUC = N/A)')


    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plot_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(plot_path)
    plt.close()
    console.print(f"[green]ROC curve plot saved to {plot_path}[/green]")

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, method_name: str, output_dir: str):
    """Plot and save a confusion matrix."""
    valid_indices = [i for i, p in enumerate(y_pred) if p is not None and not np.isnan(p)]

    if not valid_indices:
        console.print(f"[yellow]Skipping Confusion Matrix for {method_name}: No valid predictions.[/yellow]")
        return

    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices].astype(int)

    # Ensure we have predictions to plot
    if len(y_true_valid) == 0:
         console.print(f"[yellow]Skipping Confusion Matrix for {method_name}: Zero valid predictions after filtering.[/yellow]")
         return

    # Create confusion matrix with explicit labels [0, 1]
    # This ensures the matrix has the correct 2x2 shape even if one class is missing in predictions/truth
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Real', 'Predicted Fake'],
                yticklabels=['Actual Real', 'Actual Fake'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {method_name}\nTN={tn}, FP={fp}, FN={fn}, TP={tp}')
    plot_path = os.path.join(output_dir, f"confusion_matrix_{method_name}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    # console.print(f"Confusion matrix for {method_name} saved to {plot_path}") # Reduce console noise

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run DeepSafe evaluation experiment.")
    parser.add_argument("--input_dir", required=True, help="Directory containing 'Fake' and 'Real' subfolders.")
    parser.add_argument("--output_dir", required=True, help="Directory to save results and plots.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (0.0 to 1.0).")
    args = parser.parse_args()

    start_timestamp = datetime.now().isoformat()

    if not os.path.isdir(args.input_dir):
        console.print(f"[bold red]Error: Input directory not found: {args.input_dir}[/bold red]")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    console.print(Panel(f"""
[bold cyan]DeepSafe Experiment Run[/bold cyan]
Input Directory: {args.input_dir}
Output Directory: {args.output_dir}
Classification Threshold: {args.threshold}
Models: {', '.join(MODEL_ENDPOINTS.keys())}
Ensemble Methods: {', '.join(ENSEMBLE_METHODS)}
Started: {start_timestamp}
""", title="Experiment Setup", border_style="blue"))
    console.print("[yellow]Note: This process runs sequentially on CPU and may take a very long time.[/yellow]")

    image_files = find_image_files(args.input_dir)
    num_images = len(image_files)
    console.print(f"Found {num_images} images ({len([f for f,l in image_files if l=='Fake'])} Fake, {len([f for f,l in image_files if l=='Real'])} Real).")

    all_results = []
    total_predictions = num_images * (len(MODEL_ENDPOINTS) + len(ENSEMBLE_METHODS))

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running Predictions...", total=total_predictions)
        for i, (img_path, ground_truth) in enumerate(image_files):
            img_name = os.path.basename(img_path)
            if (i+1) % 10 == 0 or i == 0: # Update description less frequently
                 progress.update(task, description=f"[cyan]Image {i+1}/{num_images} ({img_name})")

            image_b64 = encode_image(img_path)
            if not image_b64:
                 progress.advance(task, (len(MODEL_ENDPOINTS) + len(ENSEMBLE_METHODS))); continue

            for model_name, api_url in MODEL_ENDPOINTS.items():
                result = query_individual_model(model_name, api_url, image_b64, args.threshold)
                all_results.append({"image_path": img_path, "image_name": img_name, "ground_truth": ground_truth,
                                    "method": model_name, "probability": result["probability"], "prediction": result["prediction"],
                                    "inference_time": result["inference_time"], "error": result["error"]})
                progress.advance(task); clear_memory()

            for method in ENSEMBLE_METHODS:
                result = query_api_ensemble(method, image_b64, args.threshold)
                all_results.append({"image_path": img_path, "image_name": img_name, "ground_truth": ground_truth,
                                    "method": f"ensemble_{method}", "probability": result["probability"], "prediction": result["prediction"],
                                    "inference_time": result["inference_time"], "error": result["error"]})
                progress.advance(task); clear_memory()
            clear_memory()

    results_df = pd.DataFrame(all_results)
    raw_results_path = os.path.join(data_dir, "raw_results.csv")
    results_df.to_csv(raw_results_path, index=False)
    console.print(f"\n[green]Raw prediction results saved to {raw_results_path}[/green]")

    summary_stats = {}
    model_names = list(MODEL_ENDPOINTS.keys())
    eval_methods = model_names + [f"ensemble_{m}" for m in ENSEMBLE_METHODS]
    y_true_map = {'Real': 0, 'Fake': 1}

    console.print("\n[bold cyan]Calculating Performance Metrics...[/bold cyan]")
    metrics_list = [] # For creating DataFrame later

    # Base truth aligned by image_path index
    if 'ground_truth' not in results_df.columns:
         console.print("[bold red]Error: 'ground_truth' column not found in results. Cannot calculate metrics.[/bold red]")
         sys.exit(1)

    # Create a unique index if image paths might not be unique (unlikely but safe)
    results_df['unique_id'] = results_df['image_path'] + '_' + results_df['method']
    results_df = results_df.set_index('unique_id')

    # Create base truth series indexed like results_df
    base_truth = results_df[['image_path', 'ground_truth']].drop_duplicates(subset=['image_path'])
    base_truth['y_true'] = base_truth['ground_truth'].map(y_true_map)
    y_true_all_aligned = base_truth.set_index('image_path')['y_true']


    for method in eval_methods:
        method_df = results_df[results_df['method'] == method].copy()
        # Align predictions/probabilities using image_path
        method_df = method_df.join(y_true_all_aligned, on='image_path')

        y_true = method_df['y_true'].values
        y_prob = method_df['probability'].values
        y_pred = method_df['prediction'].values

        # Filter out NaNs which can occur if join failed or source data was NaN
        valid_indices_metrics = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true_filt = y_true[valid_indices_metrics]
        y_pred_filt = y_pred[valid_indices_metrics]

        valid_indices_auc = ~np.isnan(y_true) & ~np.isnan(y_prob)
        y_true_auc_filt = y_true[valid_indices_auc]
        y_prob_auc_filt = y_prob[valid_indices_auc]

        metrics = calculate_metrics(y_true_filt, y_pred_filt, y_prob_auc_filt)
        metrics["method"] = method # Add method name for DataFrame creation
        summary_stats[method] = metrics
        metrics_list.append(metrics)

         # Generate confusion matrix plot
        plot_confusion_matrix(y_true_filt, y_pred_filt, method, plots_dir)

    # --- Results Presentation & Ranking ---
    metrics_df = pd.DataFrame(metrics_list).set_index("method")
    metrics_df = metrics_df.fillna('N/A') # Replace NaN with N/A for display

    # Print overall metrics table
    display_table = Table(title=f"Model Performance Summary (Threshold: {args.threshold})")
    display_table.add_column("Method", style="cyan", no_wrap=True)
    display_table.add_column("AUC", style="white") # Area Under Curve (Overall discrimination)
    display_table.add_column("F1", style="yellow") # Balance Prec/Recall
    display_table.add_column("Acc", style="green") # Accuracy
    display_table.add_column("Prec", style="blue") # Precision (Min False Pos)
    display_table.add_column("Recall", style="magenta") # Recall (Min False Neg)
    display_table.add_column("Spec", style="red") # Specificity (True Neg Rate)
    display_table.add_column("Errors", style="grey50")

    for method, row in metrics_df.iterrows():
         display_table.add_row(
              method,
              f"{row['auc']:.3f}" if isinstance(row['auc'], (float, np.floating)) else str(row['auc']),
              f"{row['f1_score']:.3f}" if isinstance(row['f1_score'], (float, np.floating)) else str(row['f1_score']),
              f"{row['accuracy']:.3f}" if isinstance(row['accuracy'], (float, np.floating)) else str(row['accuracy']),
              f"{row['precision']:.3f}" if isinstance(row['precision'], (float, np.floating)) else str(row['precision']),
              f"{row['recall']:.3f}" if isinstance(row['recall'], (float, np.floating)) else str(row['recall']),
              f"{row['specificity']:.3f}" if isinstance(row['specificity'], (float, np.floating)) else str(row['specificity']),
              f"{int(row['num_errors'])}/{int(row['num_samples'])}" if isinstance(row['num_errors'], (int, np.integer)) else str(row['num_errors'])
         )
    console.print(display_table)


    # --- Ranking and Best Model/Ensemble Selection ---
    console.print("\n[bold cyan]Ranking Models and Ensemble Methods...[/bold cyan]")

    # Primary ranking metric: AUC (threshold-independent), Secondary: F1-Score
    # Convert 'N/A' back to NaN for sorting, handle potential errors
    metrics_df_rank = metrics_df.replace('N/A', np.nan).astype(float)
    ranked_df = metrics_df_rank.sort_values(by=['auc', 'f1_score'], ascending=[False, False])

    ranked_table = Table(title="Ranked Performance (by AUC, then F1)")
    ranked_table.add_column("Rank", style="bold white")
    ranked_table.add_column("Method", style="cyan")
    ranked_table.add_column("AUC", style="white")
    ranked_table.add_column("F1-Score", style="yellow")
    ranked_table.add_column("Recall", style="magenta")
    ranked_table.add_column("Precision", style="blue")

    for i, (method, row) in enumerate(ranked_df.iterrows()):
        ranked_table.add_row(
            str(i+1), method,
            f"{row['auc']:.3f}" if pd.notna(row['auc']) else "N/A",
            f"{row['f1_score']:.3f}" if pd.notna(row['f1_score']) else "N/A",
            f"{row['recall']:.3f}" if pd.notna(row['recall']) else "N/A",
            f"{row['precision']:.3f}" if pd.notna(row['precision']) else "N/A"
        )
    console.print(ranked_table)

    # Identify top performers
    top_method = ranked_df.index[0] if not ranked_df.empty else "N/A"
    top_ensemble = None
    top_individual = None

    # Find best ensemble
    ensemble_ranked = ranked_df[ranked_df.index.str.startswith('ensemble_')]
    top_ensemble = ensemble_ranked.index[0] if not ensemble_ranked.empty else "N/A"

    # Find best individual model
    individual_ranked = ranked_df[~ranked_df.index.str.startswith('ensemble_')]
    top_individual = individual_ranked.index[0] if not individual_ranked.empty else "N/A"

    console.print("\n[bold cyan]Experiment Highlights:[/bold cyan]")
    console.print(f"- Overall Best Performer (by AUC/F1): [bold magenta]{top_method}[/bold magenta]")
    console.print(f"- Best Ensemble Method: [bold magenta]{top_ensemble}[/bold magenta]")
    console.print(f"- Best Individual Model: [bold magenta]{top_individual}[/bold magenta]")

    # Add comparison details
    if top_ensemble != "N/A" and top_individual != "N/A":
        top_ensemble_auc = ranked_df.loc[top_ensemble, 'auc']
        top_individual_auc = ranked_df.loc[top_individual, 'auc']
        if pd.notna(top_ensemble_auc) and pd.notna(top_individual_auc):
             if top_ensemble_auc > top_individual_auc:
                  console.print(f"- The best ensemble ({top_ensemble}, AUC: {top_ensemble_auc:.3f}) outperformed the best individual model ({top_individual}, AUC: {top_individual_auc:.3f}).")
             elif top_individual_auc > top_ensemble_auc:
                  console.print(f"- The best individual model ({top_individual}, AUC: {top_individual_auc:.3f}) outperformed the best ensemble ({top_ensemble}, AUC: {top_ensemble_auc:.3f}).")
             else:
                  console.print(f"- The best ensemble and individual model have similar AUC performance (≈{top_ensemble_auc:.3f}).")

    # --- Save Summary Stats including Ranking ---
    summary_output = {
        "experiment_metadata": {
            "timestamp": start_timestamp,
            "input_directory": args.input_dir,
            "output_directory": args.output_dir,
            "classification_threshold": args.threshold,
            "num_images": num_images,
            "fake_count": len([f for f,l in image_files if l=='Fake']),
            "real_count": len([f for f,l in image_files if l=='Real']),
        },
        "performance_metrics": summary_stats, # Original metrics dict
        "performance_ranking_auc_f1": ranked_df.reset_index().to_dict(orient='records'), # Save ranked list
        "top_performers": {
            "overall_best": top_method,
            "best_ensemble": top_ensemble,
            "best_individual": top_individual
        }
    }

    summary_stats_path = os.path.join(data_dir, "summary_statistics.json")
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        elif isinstance(obj, (np.floating, np.float64)): return float(obj) if pd.notna(obj) else None # Handle NaN
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
        return obj

    with open(summary_stats_path, 'w') as f:
        json.dump(summary_output, f, indent=4, default=convert_numpy)
    console.print(f"\n[green]Summary statistics and rankings saved to {summary_stats_path}[/green]")

    # --- Generate Plots ---
    console.print("\n[bold cyan]Generating Plots...[/bold cyan]")
    try:
         plot_roc_curve(results_df, plots_dir)
         console.print(f"Confusion matrix plots saved in: {plots_dir}") # Printed earlier
    except Exception as e:
         console.print(f"[bold red]Error generating plots: {e}[/bold red]")


    console.print(Panel("[bold green]Experiment Finished Successfully![/bold green]", border_style="green"))
    console.print(f"Results, stats, and plots saved in: {args.output_dir}")


if __name__ == "__main__":
    main()