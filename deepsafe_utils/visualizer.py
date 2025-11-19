import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import List, Dict, Any, Optional
from rich.console import Console

console = Console()

class Visualizer:
    def __init__(self, config_manager, output_dir_for_run: str):
        self.config_manager = config_manager
        self.output_dir_for_run = output_dir_for_run # Base for plots for this specific run
        os.makedirs(self.output_dir_for_run, exist_ok=True)

    def _group_results(self, results_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}
        for res_item in results_list:
            model_name = res_item.get("model_name")
            if not model_name or "error" in res_item or \
               res_item.get("ground_truth") == "Unknown" or \
               res_item.get("prediction") is None or \
               res_item.get("probability") is None:
                continue
            if model_name not in grouped: grouped[model_name] = []
            grouped[model_name].append(res_item)
        return grouped

    def plot_confusion_matrices(self, all_results: List[Dict[str, Any]]):
        grouped_results = self._group_results(all_results)
        if not grouped_results:
            console.print("[yellow]No valid results to plot confusion matrices.[/yellow]")
            return

        cm_dir = os.path.join(self.output_dir_for_run, "confusion_matrices")
        os.makedirs(cm_dir, exist_ok=True)

        for model_name, model_specific_results in grouped_results.items():
            y_true = [1 if r["ground_truth"] == "Fake" else 0 for r in model_specific_results]
            y_pred_class = [int(r["prediction"]) for r in model_specific_results]

            if not y_true or (len(set(y_true)) < 1 and len(y_true) > 0) : # Needs at least one class for CM, if y_true is empty, skip
                console.print(f"[yellow]Skipping CM for {model_name}: Insufficient or single-class data.[/yellow]")
                continue
            
            try:
                plt.figure(figsize=(7, 5.5)) # Slightly smaller default
                cm = confusion_matrix(y_true, y_pred_class, labels=[0,1]) # Ensure Real=0, Fake=1
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Predicted Real', 'Predicted Fake'],
                            yticklabels=['Actual Real', 'Actual Fake'],
                            annot_kws={"size": 12})
                plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
                plt.xlabel('Predicted Label', fontsize=12)
                plt.ylabel('True Label', fontsize=12)
                plt.tight_layout()
                plot_path = os.path.join(cm_dir, f"cm_{model_name.replace('/', '_').replace(' ', '_')}.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()
                # console.print(f"Confusion matrix for {model_name} saved to [green]{plot_path}[/green]")
            except Exception as e:
                console.print(f"[red]Error plotting CM for {model_name}: {e}[/red]")
        if os.listdir(cm_dir): console.print(f"Confusion matrices saved to [green]{cm_dir}[/green]")


    def plot_roc_curves(self, all_results: List[Dict[str, Any]]):
        grouped_results = self._group_results(all_results)
        if not grouped_results:
            console.print("[yellow]No valid results to plot ROC curves.[/yellow]")
            return

        plt.figure(figsize=(10, 8)) # Adjusted for better legend
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.500)')
        
        plot_count = 0
        for model_name, model_specific_results in grouped_results.items():
            y_true = [1 if r["ground_truth"] == "Fake" else 0 for r in model_specific_results]
            y_pred_proba = [float(r["probability"]) for r in model_specific_results]

            if not y_true or len(set(y_true)) < 2:
                # console.print(f"[yellow]Skipping ROC for {model_name}: Requires at least two classes in ground truth.[/yellow]")
                continue

            # Check for variance in predicted probabilities for this model
            if len(np.unique(y_pred_proba)) < 2 and len(y_pred_proba) == len(y_true):
                # console.print(f"[yellow]Skipping ROC for {model_name}: All predicted probabilities are the same.[/yellow]")
                continue
            
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc_val = auc(fpr, tpr)
                if pd.notna(roc_auc_val): # Only plot if AUC is valid
                    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc_val:.3f})')
                    plot_count +=1
            except Exception as e:
                console.print(f"[red]Error plotting ROC curve for {model_name}: {e}[/red]")
        
        if plot_count > 0:
            plt.xlim([-0.01, 1.0])
            plt.ylim([0.0, 1.01])
            plt.xlabel('False Positive Rate', fontsize=13)
            plt.ylabel('True Positive Rate', fontsize=13)
            plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=15)
            plt.legend(loc="lower right", fontsize='small', frameon=True, shadow=False) # Adjusted legend
            plt.grid(alpha=0.35)
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir_for_run, "combined_roc_curves.png")
            plt.savefig(plot_path, dpi=150)
            console.print(f"Combined ROC curves plot saved to [green]{plot_path}[/green]")
        else:
            console.print("[yellow]No valid ROC curves were plotted (e.g., due to single class in y_true or constant y_pred_proba).[/yellow]")
        plt.close()


    def plot_probability_distributions(self, all_results: List[Dict[str, Any]]):
        grouped_results = self._group_results(all_results)
        if not grouped_results:
            console.print("[yellow]No valid results to plot probability distributions.[/yellow]")
            return

        dist_dir = os.path.join(self.output_dir_for_run, "probability_distributions")
        os.makedirs(dist_dir, exist_ok=True)

        for model_name, model_specific_results in grouped_results.items():
            df_model = pd.DataFrame(model_specific_results)
            if 'probability' not in df_model.columns or 'ground_truth' not in df_model.columns: continue

            df_model['probability'] = pd.to_numeric(df_model['probability'], errors='coerce').dropna()
            if df_model.empty: continue
            
            real_probs = df_model[df_model['ground_truth'] == 'Real']['probability']
            fake_probs = df_model[df_model['ground_truth'] == 'Fake']['probability']

            if real_probs.empty and fake_probs.empty: continue

            plt.figure(figsize=(9, 5.5))
            sns.kdeplot(real_probs, fill=True, label="Actual Real (Prob Fake)", color="green", alpha=0.6, warn_singular=False)
            sns.kdeplot(fake_probs, fill=True, label="Actual Fake (Prob Fake)", color="red", alpha=0.6, warn_singular=False)
            
            plt.xlabel('Predicted Probability of Being Fake', fontsize=11)
            plt.ylabel('Density', fontsize=11)
            plt.title(f'Probability Distribution - {model_name}', fontsize=13)
            plt.legend(frameon=True)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.xlim(-0.05, 1.05)
            plot_path = os.path.join(dist_dir, f"dist_{model_name.replace('/', '_').replace(' ', '_')}.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
        if os.listdir(dist_dir): console.print(f"Probability distributions saved to [green]{dist_dir}[/green]")

    def plot_benchmark_summary(self, benchmark_data: Dict[str, Any]):
        output_plot_dir = os.path.join(self.output_dir_for_run, "benchmark_plots")
        os.makedirs(output_plot_dir, exist_ok=True)

        api_bench_data = benchmark_data.get("api", {})
        models_bench_data = benchmark_data.get("models", {})
        labels, mean_times, std_devs, all_times_raw = [], [], [], []

        if api_bench_data.get("mean_total_request_time") is not None and api_bench_data.get("count", 0) > 0:
            labels.append("API (Ensemble)")
            mean_times.append(api_bench_data["mean_total_request_time"])
            std_devs.append(api_bench_data.get("std_total_request_time", 0))
            all_times_raw.append(api_bench_data.get("raw_total_request_times", []))

        for model_name, data in sorted(models_bench_data.items()):
            if data.get("mean_total_request_time") is not None and data.get("count", 0) > 0:
                labels.append(model_name)
                mean_times.append(data["mean_total_request_time"])
                std_devs.append(data.get("std_total_request_time", 0))
                all_times_raw.append(data.get("raw_total_request_times", []))

        if not labels:
            console.print("[yellow]No valid benchmark data to plot.[/yellow]")
            return

        plt.figure(figsize=(max(8, len(labels) * 0.7), 6))
        bars = plt.bar(labels, mean_times, color='skyblue', yerr=std_devs, capsize=4, ecolor='grey', width=0.6)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(mean_times, default=1), f'{yval:.3f}s', ha='center', va='bottom', fontsize=8)
        
        plt.ylabel('Mean Total Request Time (s)', fontsize=11)
        plt.title('Client-Side Request Time Benchmark', fontsize=13)
        plt.xticks(rotation=40, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_dir, "benchmark_mean_times.png"), dpi=150)
        plt.close()
        console.print(f"Benchmark mean times plot saved to [green]{os.path.join(output_plot_dir, 'benchmark_mean_times.png')}[/green]")

        if any(all_times_raw):
            plt.figure(figsize=(max(8, len(labels) * 0.7), 6))
            bp = plt.boxplot(all_times_raw, labels=labels, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
            for patch in bp['boxes']: patch.set_facecolor('lightblue')
            plt.ylabel('Total Request Time (s)', fontsize=11)
            plt.title('Client-Side Request Time Distributions', fontsize=13)
            plt.xticks(rotation=40, ha='right', fontsize=9)
            plt.yticks(fontsize=9)
            plt.grid(True, axis='y', linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(output_plot_dir, "benchmark_distributions.png"), dpi=150)
            plt.close()
            console.print(f"Benchmark distributions plot saved to [green]{os.path.join(output_plot_dir, 'benchmark_distributions.png')}[/green]")
        else:
            console.print("[yellow]No raw time data available for distribution plot.[/yellow]")