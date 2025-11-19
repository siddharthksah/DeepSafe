import numpy as np
import pandas as pd # Keep pandas for potential future use, though not strictly needed now
import os # For os.path.basename
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from typing import List, Dict, Any, Tuple, Optional
from rich.console import Console

console = Console()

class ResultsProcessor:
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def compute_local_ensemble_results(self,
                                 model_results_for_media_item: List[Dict[str, Any]],
                                 threshold: float,
                                 method: str = "voting") -> Dict[str, Any]:
        if not model_results_for_media_item:
            return {
                "error": "No model results provided for local ensemble.",
                "model_name": f"local_ensemble_{method}", # Include method in name
                "verdict": "undetermined", "probability": 0.5, "prediction": 0,
                "media_path": "unknown_path", "media_name": "unknown_name", "ground_truth": "Unknown"
            }

        first_result = model_results_for_media_item[0]
        media_path = first_result.get("media_path", "unknown_path")
        media_name = first_result.get("media_name", os.path.basename(media_path) if media_path != "unknown_path" else "unknown_name")
        ground_truth = first_result.get("ground_truth", "Unknown")

        valid_outputs = [
            r for r in model_results_for_media_item
            if isinstance(r, dict) and "error" not in r and
               r.get("probability") is not None and r.get("prediction") is not None
        ]

        if not valid_outputs:
            return {
                "error": "No valid model outputs for local ensemble.",
                "model_name": f"local_ensemble_{method}",
                "media_path": media_path, "media_name": media_name, "ground_truth": ground_truth,
                "verdict": "undetermined", "probability": 0.5, "prediction": 0
            }

        fake_votes = sum(1 for r in valid_outputs if r["prediction"] == 1)
        real_votes = sum(1 for r in valid_outputs if r["prediction"] == 0)
        total_valid_votes = len(valid_outputs)

        ensemble_prob_fake = 0.5
        verdict = "undetermined"

        if method == "voting":
            if total_valid_votes > 0:
                verdict = "fake" if fake_votes > real_votes else "real"
                ensemble_prob_fake = fake_votes / total_valid_votes
            else:
                ensemble_prob_fake = 0.5 # Keep as 0.5 if no valid votes
        elif method == "average":
            probabilities = [r["probability"] for r in valid_outputs] # Already checked for existence
            if probabilities:
                ensemble_prob_fake = sum(probabilities) / len(probabilities)
                verdict = "fake" if ensemble_prob_fake >= threshold else "real"
            else:
                 ensemble_prob_fake = 0.5
        else:
            console.print(f"[yellow]Unsupported local ensemble method '{method}'. Defaulting to undetermined for this item.[/yellow]")
            # verdict remains 'undetermined', ensemble_prob_fake remains 0.5

        prediction = 1 if verdict == "fake" else 0

        return {
            "model_name": f"local_ensemble_{method}",
            "media_path": media_path, "media_name": media_name, "ground_truth": ground_truth,
            "probability": float(ensemble_prob_fake),
            "prediction": int(prediction),
            "class": verdict,
            "fake_votes_local": fake_votes,
            "real_votes_local": real_votes,
            "total_valid_models_local": total_valid_votes,
            "inference_time": sum(r.get("inference_time", r.get("total_request_time", 0)) or 0 for r in valid_outputs)
        }

    def calculate_batch_metrics(self, all_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        metrics_by_model: Dict[str, Dict[str, Any]] = {}
        results_grouped: Dict[str, List[Dict[str, Any]]] = {}

        for res_item in all_results:
            model_name = res_item.get("model_name")
            if not model_name or "error" in res_item: continue
            if model_name not in results_grouped: results_grouped[model_name] = []
            results_grouped[model_name].append(res_item)

        for model_name, model_specific_results in results_grouped.items():
            y_true, y_pred_class, y_pred_proba = [], [], []
            for result in model_specific_results:
                if result.get("ground_truth") == "Unknown" or \
                   result.get("prediction") is None or \
                   result.get("probability") is None: continue
                y_true.append(1 if result["ground_truth"] == "Fake" else 0)
                y_pred_class.append(int(result["prediction"]))
                y_pred_proba.append(float(result["probability"]))

            current_metrics: Dict[str, Any] = { # Initialize with NaN/defaults
                "accuracy": np.nan, "precision": np.nan, "recall": np.nan,
                "f1_score": np.nan, "auc": np.nan, "specificity": np.nan,
                "confusion_matrix": [[0,0],[0,0]], "count": len(y_true), "error": None
            }

            if not y_true: current_metrics["error"] = "No valid ground truth labels found."
            elif len(set(y_true)) < 2 and len(y_true) > 0 : current_metrics["error"] = "Only one class present in ground truth."
            
            if current_metrics["error"]: # If error already set, skip calculations
                metrics_by_model[model_name] = current_metrics
                continue

            try:
                current_metrics["accuracy"] = accuracy_score(y_true, y_pred_class)
                current_metrics["precision"] = precision_score(y_true, y_pred_class, zero_division=0)
                current_metrics["recall"] = recall_score(y_true, y_pred_class, zero_division=0)
                current_metrics["f1_score"] = f1_score(y_true, y_pred_class, zero_division=0)

                if not (len(np.unique(y_pred_proba)) < 2 and len(y_pred_proba) == len(y_true)):
                    try: current_metrics["auc"] = auc(*roc_curve(y_true, y_pred_proba)[:2])
                    except ValueError: current_metrics["auc"] = np.nan # Keep NaN if ROC fails
                
                cm = confusion_matrix(y_true, y_pred_class, labels=[0,1]) # Ensure consistent label order
                current_metrics["confusion_matrix"] = cm.tolist()
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                    current_metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else: # Handle cases where cm might not be 2x2 (e.g. all one class prediction)
                    current_metrics["specificity"] = np.nan
                    # You might want to log a warning or adjust CM display if not 2x2
            except Exception as e:
                console.print(f"[bold red]Error calculating metrics for {model_name}: {e}[/bold red]")
                current_metrics["error"] = str(e)
            
            metrics_by_model[model_name] = current_metrics
        return metrics_by_model