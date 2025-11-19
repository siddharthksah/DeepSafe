#!/usr/bin/env python3
"""
DeepSafe Advanced Meta-Learner Training Suite (train_meta_learner_advanced.py)
==============================================================================

This script trains and evaluates various meta-learners (stacking ensembles)
for deepfake detection. It takes a CSV file of meta-features (outputs from
base deepfake detection models) and ground truth labels as input.

Key Features:
-------------
1.  Modality-Specific Training: Supports training separate meta-learners for
    different media types (image, video, audio) using the `--media-type` argument.
    This ensures that the meta-learner is optimized for the characteristics of
    the base models relevant to that modality.
2.  Data Preprocessing: Includes imputation for missing values (e.g., if a base
    model failed) and feature scaling.
3.  Multiple Meta-Learner Models: Trains and evaluates several standard classifiers
    (Logistic Regression, Random Forest, Gradient Boosting, SVC, KNN, Naive Bayes)
    and, if available, advanced models like XGBoost and LightGBM.
4.  Hyperparameter Optimization:
    - Supports Optuna for efficient hyperparameter search.
    - Falls back to GridSearchCV if Optuna is not installed or if specified.
5.  Comprehensive Evaluation:
    - Calculates Accuracy, F1-Score, Precision, Recall, and ROC AUC for each model.
    - Generates classification reports and confusion matrices.
    - Plots ROC curves for visual comparison of all trained meta-learners and
      simple ensemble baselines.
6.  Simple Ensemble Baselines: Also evaluates simple averaging and majority vote
    ensembles for comparison against more complex stacking models. Includes an
    option for optimized weighted averaging.
7.  Artifact Generation:
    - Saves all trained meta-learner models (e.g., .joblib files).
    - Saves the data preprocessor (imputer + scaler).
    - Saves the list of feature columns used during training.
    - Saves a summary of all experiment metrics in JSON format.
    - The final, best-performing trainable meta-learner and its associated
      preprocessors are saved with generic names inside media-type specific
      subfolders (e.g., api_artifacts_dir/image/deepsafe_meta_learner.joblib).
8.  Configurable Output: Allows specifying separate directories for general
    experiment outputs and for API-ready deployment artifacts.

CLI Usage:
----------
python train_meta_learner_advanced.py \\
    --media-type [image|video|audio] \\
    --meta-file /path/to/meta_features_[media_type].csv \\
    --output-dir ./meta_learning_experiment_runs/ \\
    --api-artifacts-dir ./api/meta_model_artifacts/ \\
    [--optimizer optuna|gridsearch] \\
    [--optuna-trials 50] \\
    [--weights /path/to/custom_weights.json]

Arguments:
----------
  --media-type {image,video,audio}
                        (Required) The type of media for which the meta-learner
                        is being trained. This affects output artifact naming.
  --meta-file META_FILE
                        (Required) Path to the CSV file containing meta-features
                        (base model outputs) and a 'ground_truth' column.
  --output-dir OUTPUT_DIR
                        Base directory for saving all experiment-related outputs
                        (logs, plots, individual model files from this run).
                        A timestamped, media-type-specific subdirectory will be created.
                        (Default: ./meta_learning_experiment_runs/)
  --api-artifacts-dir API_ARTIFACTS_DIR
                        Directory to save the final, API-ready deployment artifacts
                        (e.g., ./api/meta_model_artifacts/image/deepsafe_meta_learner.joblib).
                        (Default: ./api/meta_model_artifacts/)
  --optimizer {optuna,gridsearch}
                        Hyperparameter optimization strategy (Default: optuna).
  --optuna-trials N
                        Number of trials for Optuna optimization (Default: 50).
  --weights WEIGHTS_PATH_OR_JSON
                        Optional. Path to a JSON file or a JSON string defining
                        custom weights for the 'Provided_Weighted_Average' ensemble.
                        Keys should be base model names (without '_prob' suffix).

Example (Image Meta-Learner):
-----------------------------
python train_meta_learner_advanced.py \\
    --media-type image \\
    --meta-file ./meta_learning_data/meta_features_image.csv \\
    --output-dir ./ml_experiments_images \\
    --api-artifacts-dir ./deepsafe_private/api/meta_model_artifacts \\
    --optimizer optuna \\
    --optuna-trials 100

This will train image-specific meta-learners, save experiment details in
`./ml_experiments_images/experiments_image_YYYYMMDD_HHMMSS/`, and place
API-ready artifacts like `deepsafe_meta_learner.joblib` into
`./deepsafe_private/api/meta_model_artifacts/image/`.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os
import json
import argparse
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, SpinnerColumn, MofNCompleteColumn
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from typing import Optional, Dict, List, Any

# --- Optional Advanced Hyperparameter Optimization & Models ---
OPTIMIZER_CHOICE_DEFAULT = "optuna"

try:
    import optuna
    OPTIMIZER_AVAILABLE_OPTUNA = True
except ImportError:
    optuna = None
    OPTIMIZER_AVAILABLE_OPTUNA = False

from sklearn.model_selection import GridSearchCV 

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False

console = Console(width=120)

# --- Configuration ---
DEFAULT_EXPERIMENT_OUTPUT_DIR_BASE = "./meta_learning_experiment_runs"
DEFAULT_API_ARTIFACTS_DIR = "./api/meta_model_artifacts" 
DEFAULT_THRESHOLD_FOR_SIMPLE_ENSEMBLES = 0.5
N_OPTUNA_TRIALS_DEFAULT = 50
CV_FOLDS_DEFAULT = 5

# --- Helper Functions ---
class NpEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super(NpEncoder, self).default(o)

def evaluate_model_predictions(y_true: np.ndarray, y_pred_class: np.ndarray, y_pred_proba: Optional[np.ndarray], model_name: str ="Model") -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"name": model_name}
    try:
        metrics["accuracy"] = accuracy_score(y_true, y_pred_class)
        metrics["f1_score"] = f1_score(y_true, y_pred_class, zero_division=0)
        metrics["precision"] = precision_score(y_true, y_pred_class, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred_class, zero_division=0)
        
        roc_auc_val = np.nan
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            if not (len(np.unique(y_pred_proba)) < 2 and len(y_pred_proba) == len(y_true)):
                try: roc_auc_val = roc_auc_score(y_true, y_pred_proba)
                except ValueError: pass 
        metrics["roc_auc"] = roc_auc_val
            
        metrics["classification_report_dict"] = classification_report(y_true, y_pred_class, digits=4, zero_division=0, output_dict=True)
        metrics["confusion_matrix_list"] = confusion_matrix(y_true, y_pred_class).tolist()
        metrics["y_pred_test_classes_list"] = y_pred_class.tolist() if isinstance(y_pred_class, np.ndarray) else y_pred_class
        metrics["y_prob_test_scores_list"] = y_pred_proba.tolist() if y_pred_proba is not None and isinstance(y_pred_proba, np.ndarray) else y_pred_proba
    except Exception as e:
        console.print(f"[bold red]Error during evaluation for {model_name}: {e}[/bold red]")
        for m_key in ["accuracy", "f1_score", "precision", "recall", "roc_auc"]: metrics[m_key] = np.nan
        metrics["classification_report_dict"] = {}; metrics["confusion_matrix_list"] = []
    return metrics

def plot_roc_curves_all(experiment_results_dict: Dict[str, Dict[str, Any]], y_true_labels: np.ndarray, output_dir_path: str, media_type: str):
    plt.figure(figsize=(12, 10)) 
    plot_count = 0
    for model_key, result_data in experiment_results_dict.items():
        if 'y_prob_test_scores_list' in result_data and result_data['y_prob_test_scores_list'] is not None:
            proba_scores = np.array(result_data['y_prob_test_scores_list'])
            if len(np.unique(y_true_labels)) < 2 or (proba_scores.ndim > 0 and len(np.unique(proba_scores)) < 2 and len(proba_scores) == len(y_true_labels)):
                continue
            try:
                fpr, tpr, _ = roc_curve(y_true_labels, proba_scores)
                roc_auc_value = result_data.get('roc_auc', auc(fpr, tpr))
                if pd.notna(roc_auc_value):
                    plt.plot(fpr, tpr, lw=1.8, label=f'{model_key} (AUC = {roc_auc_value:.4f})')
                    plot_count +=1
            except ValueError as e:
                console.print(f"[yellow]Could not plot ROC for {model_key} ({media_type}): {e}[/yellow]")
    
    if plot_count > 0:
        plt.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--') 
        plt.xlim([-0.01, 1.0]); plt.ylim([0.0, 1.01]) 
        plt.xlabel('False Positive Rate', fontsize=13); plt.ylabel('True Positive Rate', fontsize=13)
        plt.title(f'Meta-Learner & Ensemble ROC Curves ({media_type.capitalize()})', fontsize=15)
        plt.legend(loc="lower right", fontsize='small', frameon=True) 
        plt.grid(alpha=0.35, linestyle=':') 
        plt.tight_layout()
        plot_path = os.path.join(output_dir_path, f"all_meta_learners_roc_curves_{media_type}.png")
        plt.savefig(plot_path, dpi=150) 
        console.print(f"Combined ROC curves plot for {media_type} saved to [green]{plot_path}[/green]")
    else:
        console.print(f"[yellow]No valid ROC curves to plot for {media_type}.[/yellow]")
    plt.close()

def optimize_average_weights_simple_grid(X_val_probs: np.ndarray, y_val_true: np.ndarray, num_base_models: int, weight_options: Optional[List[float]] = None) -> np.ndarray:
    if weight_options is None: weight_options = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    best_auc_val = -1.0; best_weights_val = np.ones(num_base_models)
    
    max_combinations_exhaustive = 5**4; num_random_samples_if_large = 2000 
    
    if num_base_models <= 0: 
        console.print("[yellow]No base models to optimize weights for. Returning default weights.[/yellow]")
        return best_weights_val

    if num_base_models <= 4 and len(weight_options)**num_base_models <= max_combinations_exhaustive :
        weight_candidates = list(itertools.product(weight_options, repeat=num_base_models))
        console.print(f"Optimizing average weights with exhaustive grid search ({len(weight_candidates)} trials).")
    else:
        console.print(f"[yellow]Optimizing average weights with random sampling ({num_random_samples_if_large} trials due to {num_base_models} models).[/yellow]")
        weight_candidates = [np.array(np.random.choice(weight_options, num_base_models)) for _ in range(num_random_samples_if_large)]

    with Progress(SpinnerColumn(),TextColumn("[progress.description]{task.description}"),BarColumn(),TextColumn("{task.percentage:>3.1f}%"),TimeElapsedColumn(), MofNCompleteColumn()) as progress:
        task = progress.add_task("Weight Grid Search", total=len(weight_candidates))
        for current_weights_tuple in weight_candidates:
            current_weights = np.array(current_weights_tuple)
            if np.sum(current_weights) == 0: progress.update(task, advance=1); continue
            
            if X_val_probs.shape[0] == 0: progress.update(task, advance=1); continue
            weighted_avg_probs_val_set = np.average(X_val_probs, axis=1, weights=current_weights)
            
            current_auc_val = 0.0
            if len(np.unique(y_val_true)) > 1 and not (len(np.unique(weighted_avg_probs_val_set)) < 2 and len(weighted_avg_probs_val_set) == len(y_val_true)):
                try: current_auc_val = roc_auc_score(y_val_true, weighted_avg_probs_val_set)
                except ValueError: pass
            if current_auc_val > best_auc_val: best_auc_val, best_weights_val = current_auc_val, current_weights
            progress.update(task, advance=1)
            
    console.print(f"Best weights from validation grid search: {best_weights_val.tolist()} with Val AUC: {best_auc_val:.4f}")
    return best_weights_val

# --- Main Experimentation Function ---
def run_meta_learning_experiments(
    meta_features_file: str,
    output_dir_base: str,
    api_artifacts_dir: str,
    media_type: str,
    optimizer_type: str,
    n_optuna_trials_config: int,
    provided_custom_weights: Optional[Dict[str, float]] = None
):
    global OPTIMIZER_CHOICE, N_OPTUNA_TRIALS 
    OPTIMIZER_CHOICE = optimizer_type
    N_OPTUNA_TRIALS = n_optuna_trials_config
    
    if OPTIMIZER_CHOICE == "optuna" and not OPTIMIZER_AVAILABLE_OPTUNA:
        console.print("[yellow]Optuna chosen but not installed. Falling back to GridSearchCV.[/yellow]")
        OPTIMIZER_CHOICE = "gridsearch"

    experiment_run_output_dir = os.path.join(output_dir_base, f"experiments_{media_type}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(experiment_run_output_dir, exist_ok=True)
    
    # Main API artifacts directory (parent for media-specific subfolders)
    os.makedirs(api_artifacts_dir, exist_ok=True)
    # Media-type specific subdirectory within the main api_artifacts_dir
    media_type_api_artifacts_subdir = os.path.join(api_artifacts_dir, media_type)
    os.makedirs(media_type_api_artifacts_subdir, exist_ok=True)


    console.rule(f"[bold cyan]DeepSafe Meta-Learning: {media_type.upper()} (Optimizer: {OPTIMIZER_CHOICE})[/bold cyan]")
    console.print(Panel(f"Meta-features: {meta_features_file}\n"
                        f"Experiment outputs: {os.path.abspath(experiment_run_output_dir)}\n"
                        f"API artifacts subfolder: {os.path.abspath(media_type_api_artifacts_subdir)}", 
                        title="Paths", border_style="dim blue", expand=False))
    all_experiment_results: Dict[str, Dict[str, Any]] = {}

    console.rule("[bold]1. Data Loading and Preprocessing[/bold]")
    try:
        df_meta = pd.read_csv(meta_features_file)
        console.print(f"Loaded {media_type} meta-features from: [cyan]{meta_features_file}[/cyan], shape: {df_meta.shape}")
    except Exception as e:
        console.print(f"[bold red]Fatal Error: Could not load meta-features file: {e}[/bold red]"); return

    base_model_prob_features = sorted([col for col in df_meta.columns if col.endswith('_prob')])
    if not base_model_prob_features:
        console.print("[bold red]Fatal Error: No base model probability columns (ending with '_prob') found in CSV.[/bold red]"); return
    
    console.print(f"Identified [magenta]{len(base_model_prob_features)}[/magenta] base model probability features: {base_model_prob_features}")
    
    temp_exp_feature_cols_path = os.path.join(experiment_run_output_dir, f"experiment_feature_columns_{media_type}.json")
    with open(temp_exp_feature_cols_path, 'w') as f: json.dump(base_model_prob_features, f, indent=2)

    X_meta_all = df_meta[base_model_prob_features].copy()
    y_meta_all = df_meta['ground_truth']

    cols_to_drop_all_nan = X_meta_all.columns[X_meta_all.isnull().all()].tolist()
    if cols_to_drop_all_nan:
        console.print(f"[yellow]Warning: Dropping fully NaN columns: {cols_to_drop_all_nan}[/yellow]")
        X_meta_all = X_meta_all.drop(columns=cols_to_drop_all_nan)
        base_model_prob_features = [col for col in base_model_prob_features if col not in cols_to_drop_all_nan]
        if not base_model_prob_features: console.print("[bold red]Fatal Error: All features became NaN after dropping some columns.[/bold red]"); return
        with open(temp_exp_feature_cols_path, 'w') as f: json.dump(base_model_prob_features, f, indent=2)

    X_meta_train_val, X_meta_test, y_meta_train_val, y_meta_test = train_test_split(
        X_meta_all, y_meta_all, test_size=0.25, random_state=42, stratify=y_meta_all if len(np.unique(y_meta_all)) > 1 else None
    )
    console.print(f"Data split: Meta-Train/Val shape {X_meta_train_val.shape}, Meta-Test shape {X_meta_test.shape}")

    ml_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    X_meta_train_val_processed = ml_preprocessor.fit_transform(X_meta_train_val)
    X_meta_test_processed = ml_preprocessor.transform(X_meta_test)
    
    joblib.dump(ml_preprocessor, os.path.join(experiment_run_output_dir, f"experiment_ml_preprocessor_{media_type}.joblib"))
    console.print(f"ML preprocessor for {media_type} (imputer + scaler) fitted and saved for this run.")

    imputer_for_simple_ensembles = ml_preprocessor.named_steps['imputer']
    X_meta_test_imputed_only_df = pd.DataFrame(imputer_for_simple_ensembles.transform(X_meta_test), columns=X_meta_test.columns)

    console.rule("[bold]2. Defining ML Meta-Learners and Hyperparameter Spaces[/bold]")
    models_and_param_spaces: Dict[str, Tuple[Any, Dict[str, Any]]] = {
        'LogisticRegression': (LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=3000), 
                               { 'C': (0.01, 1000.0, 'loguniform') if OPTIMIZER_CHOICE == "optuna" else [0.01, 0.1, 1, 10, 100, 500] }),
        'RandomForest': (RandomForestClassifier(random_state=42, class_weight='balanced'), 
                         { 'n_estimators': (100, 500, 'int') if OPTIMIZER_CHOICE == "optuna" else [100, 200, 300, 400],
                           'max_depth': (5, 25, 'int', True) if OPTIMIZER_CHOICE == "optuna" else [5, 10, 15, 20, None],
                           'min_samples_split': (2, 20, 'int') if OPTIMIZER_CHOICE == "optuna" else [2, 5, 10, 15],
                           'min_samples_leaf': (1, 15, 'int') if OPTIMIZER_CHOICE == "optuna" else [1, 5, 10, 15] }),
        'GradientBoosting': (GradientBoostingClassifier(random_state=42),
                             { 'n_estimators': (100, 500, 'int') if OPTIMIZER_CHOICE == "optuna" else [100, 200, 300, 400],
                               'learning_rate': (0.005, 0.2, 'loguniform') if OPTIMIZER_CHOICE == "optuna" else [0.01, 0.05, 0.1, 0.15],
                               'max_depth': (3, 10, 'int') if OPTIMIZER_CHOICE == "optuna" else [3, 5, 7, 9] }),
        'SVC_Linear': (SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced', max_iter=10000),
                       { 'C': (0.01, 100.0, 'loguniform') if OPTIMIZER_CHOICE == "optuna" else [0.1, 1, 10, 100] }),
        'KNeighbors': (KNeighborsClassifier(), {
            'n_neighbors': (3, 25, 'int', False, 2) if OPTIMIZER_CHOICE == "optuna" else [3, 5, 7, 11, 15, 19, 23],
            'weights': (['uniform', 'distance'], 'categorical') if OPTIMIZER_CHOICE == "optuna" else ['uniform', 'distance']
        }),
        'GaussianNB': (GaussianNB(), {})
    }
    if XGBOOST_AVAILABLE and XGBClassifier:
        models_and_param_spaces['XGBoost'] = (XGBClassifier(random_state=42, eval_metric='auc'), {
            'n_estimators': (100, 600, 'int') if OPTIMIZER_CHOICE == "optuna" else [100, 200, 300, 400, 500],
            'learning_rate': (0.005, 0.2, 'loguniform') if OPTIMIZER_CHOICE == "optuna" else [0.01, 0.05, 0.1],
            'max_depth': (3, 12, 'int') if OPTIMIZER_CHOICE == "optuna" else [3, 5, 7, 9, 11],
            'scale_pos_weight': ( (np.sum(y_meta_train_val==0)/np.sum(y_meta_train_val==1)) if np.sum(y_meta_train_val==1)>0 else 1.0 ,)
        })
    if LIGHTGBM_AVAILABLE and LGBMClassifier:
        models_and_param_spaces['LightGBM'] = (LGBMClassifier(random_state=42, class_weight='balanced', metric='auc', verbosity=-1), {
            'n_estimators': (100, 600, 'int') if OPTIMIZER_CHOICE == "optuna" else [100, 200, 300, 400, 500],
            'learning_rate': (0.005, 0.2, 'loguniform') if OPTIMIZER_CHOICE == "optuna" else [0.01, 0.05, 0.1],
            'num_leaves': (20, 150, 'int') if OPTIMIZER_CHOICE == "optuna" else [31, 50, 70, 100, 130]
        })

    console.rule(f"[bold]3. Training and Evaluating ML-based Meta-Learners ({media_type.capitalize()} Stacking)[/bold]")
    cv_strategy = StratifiedKFold(n_splits=CV_FOLDS_DEFAULT, shuffle=True, random_state=42)
    trained_ml_model_objects: Dict[str, Any] = {}

    for model_name_key, (model_instance_template, param_def) in models_and_param_spaces.items():
        console.rule(f"[bold blue]Optimizing & Training {media_type.capitalize()} Meta-Learner: {model_name_key}[/bold blue]", style="blue")
        start_train_time = time.time()
        best_estimator_for_model = None

        if not param_def: 
            model_instance_template.fit(X_meta_train_val_processed, y_meta_train_val)
            best_estimator_for_model = model_instance_template
            console.print(f"{model_name_key} fitted directly (no hyperparameters tuned).")
        elif OPTIMIZER_CHOICE == "optuna" and optuna:
            def optuna_objective(trial: optuna.Trial):
                current_params = {}
                for p_name, p_opts in param_def.items():
                    if isinstance(p_opts, tuple) and len(p_opts) >= 2: 
                        suggestion_type_or_values = p_opts[1] if p_name == 'weights' and p_opts[1] == 'categorical' else p_opts[2]
                        if suggestion_type_or_values == 'loguniform': current_params[p_name] = trial.suggest_float(p_name, p_opts[0], p_opts[1], log=True)
                        elif suggestion_type_or_values == 'uniform': current_params[p_name] = trial.suggest_float(p_name, p_opts[0], p_opts[1])
                        elif suggestion_type_or_values == 'int':
                            low, high = p_opts[0], p_opts[1]; can_be_none = p_opts[3] if len(p_opts) > 3 else False; step = p_opts[4] if len(p_opts) > 4 else 1
                            val = trial.suggest_int(p_name, low, high, step=step)
                            if can_be_none and trial.suggest_categorical(f"{p_name}_use_none", [True, False]): val = None
                            current_params[p_name] = val
                        elif suggestion_type_or_values == 'categorical': current_params[p_name] = trial.suggest_categorical(p_name, p_opts[0])
                        elif len(p_opts) == 1 and not isinstance(p_opts[0], list): current_params[p_name] = p_opts[0]
                        else: console.print(f"[red]Warning: Unknown Optuna parameter definition for {p_name}: {p_opts}[/red]")
                    else: 
                        if p_name in model_instance_template.get_params() and not isinstance(p_opts, tuple): current_params[p_name] = p_opts
                
                model_trial = model_instance_template.__class__(**model_instance_template.get_params())
                valid_model_params = model_trial.get_params().keys()
                filtered_current_params = {k: v for k, v in current_params.items() if k in valid_model_params}
                model_trial.set_params(**filtered_current_params)
                
                scores = []
                for train_idx, val_idx in cv_strategy.split(X_meta_train_val_processed, y_meta_train_val):
                    X_fold_train, X_fold_val = X_meta_train_val_processed[train_idx], X_meta_train_val_processed[val_idx]
                    y_fold_train, y_fold_val = y_meta_train_val.iloc[train_idx], y_meta_train_val.iloc[val_idx]
                    model_trial.fit(X_fold_train, y_fold_train)
                    if hasattr(model_trial, "predict_proba"):
                        try:
                            y_val_pred_proba = model_trial.predict_proba(X_fold_val)[:,1]
                            if len(np.unique(y_fold_val)) < 2 or (len(np.unique(y_val_pred_proba)) < 2 and len(y_val_pred_proba) == len(y_fold_val)):
                                scores.append(0.5) 
                            else: scores.append(roc_auc_score(y_fold_val, y_val_pred_proba))
                        except Exception: scores.append(0.0) 
                    else: scores.append(f1_score(y_fold_val, model_trial.predict(X_fold_val), zero_division=0)) 
                return np.mean(scores)

            study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
            study.optimize(optuna_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True, gc_after_trial=True)
            
            sklearn_best_params = {}
            for p_name_orig_def, p_opts_def in param_def.items():
                if p_name_orig_def in study.best_params: sklearn_best_params[p_name_orig_def] = study.best_params[p_name_orig_def]
                if len(p_opts_def) > 3 and p_opts_def[3] is True: 
                    if study.best_params.get(f"{p_name_orig_def}_use_none", False) is True:
                        sklearn_best_params[p_name_orig_def] = None
            console.print(f"Best Optuna params for {model_name_key} ({media_type}): {sklearn_best_params}")
            best_estimator_for_model = model_instance_template.__class__(**model_instance_template.get_params())
            best_estimator_for_model.set_params(**sklearn_best_params)
            best_estimator_for_model.fit(X_meta_train_val_processed, y_meta_train_val)
        else: 
            grid_search = GridSearchCV(model_instance_template, param_def, cv=cv_strategy, scoring='roc_auc', n_jobs=-1, verbose=0)
            grid_search.fit(X_meta_train_val_processed, y_meta_train_val)
            best_estimator_for_model = grid_search.best_estimator_
            console.print(f"Best GridSearchCV params for {model_name_key} ({media_type}): {grid_search.best_params_}")

        joblib.dump(best_estimator_for_model, os.path.join(experiment_run_output_dir, f"{model_name_key}_meta_learner_{media_type}.joblib"))
        trained_ml_model_objects[model_name_key] = best_estimator_for_model
        
        y_test_pred_classes = best_estimator_for_model.predict(X_meta_test_processed)
        y_test_pred_probas = best_estimator_for_model.predict_proba(X_meta_test_processed)[:,1] if hasattr(best_estimator_for_model, "predict_proba") else None
        metrics_results = evaluate_model_predictions(y_meta_test.values, y_test_pred_classes, y_test_pred_probas, model_name_key)
        all_experiment_results[model_name_key] = metrics_results
        train_time = time.time() - start_train_time
        console.print(f"[bold]{model_name_key} Test Set Perf. ({media_type}):[/bold] AUC: {metrics_results.get('roc_auc', np.nan):.4f}, F1: {metrics_results.get('f1_score', np.nan):.4f}, Acc: {metrics_results.get('accuracy', np.nan):.4f} (Train time: {train_time:.2f}s)")

    console.rule(f"[bold]4. Evaluating Simple Ensemble Baselines ({media_type.capitalize()} Meta-Test Set)[/bold]")
    avg_probs_meta_test = X_meta_test_imputed_only_df.mean(axis=1).values
    avg_preds_meta_test_classes = (avg_probs_meta_test >= DEFAULT_THRESHOLD_FOR_SIMPLE_ENSEMBLES).astype(int)
    all_experiment_results['Simple_Average_Prob'] = evaluate_model_predictions(y_meta_test.values, avg_preds_meta_test_classes, avg_probs_meta_test, "Simple_Average_Prob")
    console.print(f"[bold]Simple Average Prob Test ({media_type}):[/bold] AUC: {all_experiment_results['Simple_Average_Prob'].get('roc_auc', np.nan):.4f}, F1: {all_experiment_results['Simple_Average_Prob'].get('f1_score', np.nan):.4f}")

    binarized_X_meta_test = (X_meta_test_imputed_only_df.values >= DEFAULT_THRESHOLD_FOR_SIMPLE_ENSEMBLES).astype(int)
    num_models_for_vote = X_meta_test_imputed_only_df.shape[1]
    fake_votes_per_item_meta_test = binarized_X_meta_test.sum(axis=1)
    maj_vote_preds_meta_test_classes = (fake_votes_per_item_meta_test >= (num_models_for_vote / 2.0)).astype(int)
    maj_vote_prob_scores_meta_test = fake_votes_per_item_meta_test / num_models_for_vote if num_models_for_vote > 0 else np.full_like(fake_votes_per_item_meta_test, 0.5, dtype=float)
    all_experiment_results['Simple_Majority_Vote'] = evaluate_model_predictions(y_meta_test.values, maj_vote_preds_meta_test_classes, maj_vote_prob_scores_meta_test, "Simple_Majority_Vote")
    console.print(f"[bold]Simple Majority Vote Test ({media_type}):[/bold] AUC: {all_experiment_results['Simple_Majority_Vote'].get('roc_auc', np.nan):.4f}, F1: {all_experiment_results['Simple_Majority_Vote'].get('f1_score', np.nan):.4f}")

    if provided_custom_weights:
        current_weights_values = [provided_custom_weights.get(fc.replace("_prob", ""), 1.0) for fc in base_model_prob_features]
        current_weights_array = np.array(current_weights_values)
        
        if len(current_weights_array) == X_meta_test_imputed_only_df.shape[1] and np.sum(current_weights_array) > 0:
            prov_weighted_avg_probs_meta_test = np.average(X_meta_test_imputed_only_df.values, axis=1, weights=current_weights_array)
            prov_weighted_avg_preds_meta_test_classes = (prov_weighted_avg_probs_meta_test >= DEFAULT_THRESHOLD_FOR_SIMPLE_ENSEMBLES).astype(int)
            all_experiment_results['Provided_Weighted_Average'] = evaluate_model_predictions(y_meta_test.values, prov_weighted_avg_preds_meta_test_classes, prov_weighted_avg_probs_meta_test, "Provided_Weighted_Average")
            console.print(f"[bold]Provided Weighted Average Test ({media_type}):[/bold] AUC: {all_experiment_results['Provided_Weighted_Average'].get('roc_auc', np.nan):.4f}, F1: {all_experiment_results['Provided_Weighted_Average'].get('f1_score', np.nan):.4f}")
        else:
            console.print(f"[yellow]Warning: Mismatch in provided_custom_weights keys vs. features for {media_type}, or sum of weights is zero. Skipping.[/yellow]")
            
    X_train_val_imputed_for_opt_df = pd.DataFrame(ml_preprocessor.named_steps['imputer'].transform(X_meta_train_val), columns=base_model_prob_features)
    stratify_opt_split = y_meta_train_val if len(np.unique(y_meta_train_val)) > 1 else None
    X_opt_train_df, X_opt_val_df, y_opt_train_series, y_opt_val_series = train_test_split(
        X_train_val_imputed_for_opt_df, y_meta_train_val, test_size=0.33, random_state=123, stratify=stratify_opt_split
    )
    if X_opt_val_df.shape[0] > 10 and X_opt_val_df.shape[1] > 0: 
        console.print(f"Optimizing weights for averaging ({media_type}) using a validation split of meta-train data...")
        optimized_avg_weights = optimize_average_weights_simple_grid(
            X_opt_val_df.values, y_opt_val_series.values, X_opt_val_df.shape[1]
        )
        opt_w_avg_probs_meta_test = np.average(X_meta_test_imputed_only_df.values, axis=1, weights=optimized_avg_weights)
        opt_w_avg_preds_meta_test_classes = (opt_w_avg_probs_meta_test >= DEFAULT_THRESHOLD_FOR_SIMPLE_ENSEMBLES).astype(int)
        all_experiment_results['Optimized_Grid_Weighted_Average'] = evaluate_model_predictions(y_meta_test.values, opt_w_avg_preds_meta_test_classes, opt_w_avg_probs_meta_test, "Optimized_Grid_Weighted_Average")
        console.print(f"[bold]Optimized Grid Weighted Average Test ({media_type}):[/bold] AUC: {all_experiment_results['Optimized_Grid_Weighted_Average'].get('roc_auc', np.nan):.4f}, F1: {all_experiment_results['Optimized_Grid_Weighted_Average'].get('f1_score', np.nan):.4f}")
        
        # Save optimized weights to media-type specific subdirectory with generic name
        # (or keep media_type in name if preferred, but API loads generic name from subdir)
        # opt_weights_api_path_generic = os.path.join(media_type_api_artifacts_subdir, "optimized_grid_average_weights.json")
        # For now, keeping the original behavior of saving to main api_artifacts_dir with media_type in name
        opt_weights_api_path_typed = os.path.join(api_artifacts_dir, f"optimized_grid_average_weights_{media_type}.json")
        with open(opt_weights_api_path_typed, 'w') as f:
            json.dump({feat: w for feat, w in zip(base_model_prob_features, optimized_avg_weights)}, f, indent=2)
        console.print(f"Optimized weights for {media_type} saved to API artifacts: [green]{opt_weights_api_path_typed}[/green]")
    else:
        console.print(f"[yellow]Validation set for weight optimization ({media_type}) too small or no features. Skipping.[/yellow]")


    console.rule(f"[bold green]5. Overall Experiment Summary & Artifacts ({media_type.capitalize()})[/bold green]")
    summary_table = Table(title=f"Meta-Learner & Simple Ensemble Experiment Summary ({media_type.capitalize()} Meta-Test Set)")
    summary_table.add_column("Method/Model", style="cyan", overflow="fold", max_width=35)
    summary_table.add_column("Test AUC", style="magenta"); summary_table.add_column("Test F1", style="green")
    summary_table.add_column("Test Acc.", style="blue"); summary_table.add_column("Test Prec.", style="yellow")
    summary_table.add_column("Test Recall", style="red")

    sorted_results_list = sorted(
        all_experiment_results.items(),
        key=lambda item: item[1].get('roc_auc', -1) if pd.notna(item[1].get('roc_auc')) else -1,
        reverse=True
    )
    best_method_overall_name = "None"
    best_method_overall_auc = -1.0
    best_trainable_ml_model_for_api = None 

    for method_name_result, metrics_result in sorted_results_list:
        summary_table.add_row(
            method_name_result,
            f"{metrics_result.get('roc_auc', 'N/A'):.4f}" if pd.notna(metrics_result.get('roc_auc')) else "N/A",
            f"{metrics_result.get('f1_score', 'N/A'):.4f}", f"{metrics_result.get('accuracy', 'N/A'):.4f}",
            f"{metrics_result.get('precision', 'N/A'):.4f}", f"{metrics_result.get('recall', 'N/A'):.4f}"
        )
        current_auc_val_result = metrics_result.get('roc_auc', -1)
        if pd.notna(current_auc_val_result) and current_auc_val_result > best_method_overall_auc:
            best_method_overall_auc = current_auc_val_result
            best_method_overall_name = method_name_result
            if method_name_result in trained_ml_model_objects: 
                best_trainable_ml_model_for_api = trained_ml_model_objects[method_name_result]
    
    console.print(summary_table)
    console.print(f"\n[bold gold1]Best performing method overall for {media_type.upper()} (Test AUC): [white]{best_method_overall_name}[/white] (AUC: {best_method_overall_auc:.4f})[/bold gold1]")

    results_json_path = os.path.join(experiment_run_output_dir, f"all_experiments_metrics_summary_{media_type}.json")
    with open(results_json_path, 'w') as f: json.dump(all_experiment_results, f, indent=2, cls=NpEncoder)
    console.print(f"All experiment metrics summaries for {media_type} saved to [green]{results_json_path}[/green]")

    plot_roc_curves_all(all_experiment_results, y_meta_test.values, experiment_run_output_dir, media_type)

    console.print(f"\n[bold]Deployment Artifacts Preparation for {media_type.upper()} (in '{media_type_api_artifacts_subdir}'):[/bold]")

    joblib.dump(ml_preprocessor.named_steps['imputer'], os.path.join(media_type_api_artifacts_subdir, "deepsafe_meta_imputer.joblib"))
    joblib.dump(ml_preprocessor.named_steps['scaler'], os.path.join(media_type_api_artifacts_subdir, "deepsafe_meta_scaler.joblib"))
    
    api_feature_cols_path = os.path.join(media_type_api_artifacts_subdir, "deepsafe_meta_feature_columns.json")
    if os.path.exists(temp_exp_feature_cols_path):
        try:
            with open(temp_exp_feature_cols_path, 'r') as src_f, open(api_feature_cols_path, 'w') as dst_f:
                json.dump(json.load(src_f), dst_f, indent=2)
            console.print(f"Feature columns for {media_type} API saved to [green]{api_feature_cols_path}[/green]")
        except Exception as e: 
            console.print(f"[red]Error copying/saving feature columns file: {e}. Manual copy might be needed from {temp_exp_feature_cols_path} to {api_feature_cols_path}[/red]")
    else:
        console.print(f"[yellow]Temporary feature columns file {temp_exp_feature_cols_path} not found. API artifact for feature columns may be missing for {media_type}.[/yellow]")
    
    console.print(f"Common imputer, scaler, and feature columns for {media_type} saved for API in '{media_type_api_artifacts_subdir}'.")

    if best_trainable_ml_model_for_api:
        api_model_joblib_path = os.path.join(media_type_api_artifacts_subdir, "deepsafe_meta_learner.joblib")
        joblib.dump(best_trainable_ml_model_for_api, api_model_joblib_path)
        console.print(f"Best trainable ML meta-learner ([white]{best_method_overall_name}[/white]) for {media_type} saved as '{os.path.basename(api_model_joblib_path)}' in '{media_type_api_artifacts_subdir}'.")
        console.print(f"The 4 artifacts in '{media_type_api_artifacts_subdir}' are ready for the API.")
    elif best_method_overall_name.startswith(("Simple", "Provided", "Optimized")):
        console.print(f"[yellow]The overall best method for {media_type} ([white]{best_method_overall_name}[/white]) is rule-based.[/yellow]")
        console.print(f"[yellow]To deploy a trainable ML model, choose the best one from this run and ensure its '.joblib' is saved as 'deepsafe_meta_learner.joblib' inside '{media_type_api_artifacts_subdir}'.[/yellow]")
        
        opt_weights_main_dir_path = os.path.join(api_artifacts_dir, f"optimized_grid_average_weights_{media_type}.json")
        opt_weights_subdir_path_generic = os.path.join(media_type_api_artifacts_subdir, "optimized_grid_average_weights.json")
        
        if "Optimized_Grid_Weighted_Average" in best_method_overall_name:
            if os.path.exists(opt_weights_main_dir_path): 
                 console.print(f"  Optimized weights for this method are currently in '{opt_weights_main_dir_path}'. Consider standardizing its location if desired (e.g., to '{opt_weights_subdir_path_generic}').")
            elif os.path.exists(opt_weights_subdir_path_generic): # If you adjust saving logic for weights too
                 console.print(f"  Optimized weights for this method are in '{opt_weights_subdir_path_generic}'.")
    else:
        console.print(f"[bold red]Error: Could not determine a best trainable model to save for {media_type}. Please review results.[/bold red]")

    console.rule(f"[bold green]Experimentation Suite for {media_type.upper()} Completed[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Meta-Learning Experiments for DeepSafe Ensemble.")
    parser.add_argument(
        "--media-type", type=str, choices=["image", "video", "audio"], required=True,
        help="Type of media for which the meta-learner is being trained (image, video, or audio)."
    )
    parser.add_argument(
        "--meta-file", type=str, required=True,
        help="Path to the media-specific meta-features CSV (e.g., ./meta_data/meta_features_image.csv)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_EXPERIMENT_OUTPUT_DIR_BASE,
        help=f"Base directory for saving all experiment-related outputs (default: {DEFAULT_EXPERIMENT_OUTPUT_DIR_BASE})."
    )
    parser.add_argument(
        "--api-artifacts-dir", type=str, default=DEFAULT_API_ARTIFACTS_DIR,
        help=f"Directory to save final API-ready artifacts (default: {DEFAULT_API_ARTIFACTS_DIR})"
    )
    parser.add_argument(
        "--optimizer", type=str, choices=["optuna", "gridsearch"], default=OPTIMIZER_CHOICE_DEFAULT,
        help=f"Hyperparameter optimizer (default: {OPTIMIZER_CHOICE_DEFAULT})"
    )
    parser.add_argument(
        "--optuna-trials", type=int, default=N_OPTUNA_TRIALS_DEFAULT,
        help=f"Number of Optuna trials (default: {N_OPTUNA_TRIALS_DEFAULT})"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help='JSON string or path to JSON file for custom base model weights (for "Provided_Weighted_Average"). Keys should be base model names (e.g., "npr_deepfakedetection").'
    )
    
    args = parser.parse_args()
    
    if OPTIMIZER_CHOICE_DEFAULT == "optuna" and not OPTIMIZER_AVAILABLE_OPTUNA:
        console.print("[yellow]Default optimizer is Optuna, but it's not installed. GridSearchCV will be used if Optuna is chosen via CLI and not available.[/yellow]")
    if not XGBOOST_AVAILABLE: console.print("[yellow]XGBoost not installed. XGBoost experiments will be skipped if its block is reached.[/yellow]")
    if not LIGHTGBM_AVAILABLE: console.print("[yellow]LightGBM not installed. LightGBM experiments will be skipped if its block is reached.[/yellow]")

    custom_weights_dict_main = None
    if args.weights:
        try:
            if os.path.exists(args.weights):
                with open(args.weights, 'r') as f: custom_weights_dict_main = json.load(f)
            else:
                custom_weights_dict_main = json.loads(args.weights)
            console.print(f"Using provided custom base model weights: {custom_weights_dict_main}")
        except Exception as e_weights:
            console.print(f"[bold red]Error parsing --weights argument: {e_weights}. Proceeding without them.[/bold red]")

    run_meta_learning_experiments(
        meta_features_file=args.meta_file,
        output_dir_base=args.output_dir,
        api_artifacts_dir=args.api_artifacts_dir,
        media_type=args.media_type,
        optimizer_type=args.optimizer,
        n_optuna_trials_config=args.optuna_trials,
        provided_custom_weights=custom_weights_dict_main
    )