#!/usr/bin/env python3
"""
DeepSafe Batch Testing Tool

This script runs all deepfake detection models on all images in a specified directory
and generates a comprehensive report with detection results and performance metrics.

Usage:
  ./deepsafe_batch_test.py --assets FOLDER [--output FILE] [--threshold NUM] [--method METHOD] [--individual]
"""

import requests
import base64
import argparse
import time
import os
import json
import sys
from typing import Dict, Any, List
from datetime import datetime
import glob
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

# Model endpoints configuration
MODEL_ENDPOINTS = {
    "cnndetection": "http://localhost:5000/predict",
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

def encode_image(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')

def test_with_api(image_path: str, threshold: float = 0.5, ensemble_method: str = "voting") -> Dict[str, Any]:
    """Test image with the main DeepSafe API."""
    try:
        # Get image info
        from PIL import Image
        img = Image.open(image_path)
        img_width, img_height = img.size
        img_format = img.format
        img_size = os.path.getsize(image_path) / 1024  # KB
        
        # Encode image
        base64_image = encode_image(image_path)
        
        # Prepare request payload
        payload = {
            "image": base64_image,
            "threshold": threshold,
            "ensemble_method": ensemble_method
        }
        
        # Make request
        start_time = time.time()
        
        response = requests.post(
            MAIN_API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=240  # Increased timeout for initial model loading
        )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Add image info and timing
            result['image_path'] = image_path
            result['image_name'] = os.path.basename(image_path)
            result['image_size'] = img_size
            result['image_dimensions'] = f"{img_width}x{img_height}"
            result['image_format'] = img_format
            result['client_request_time'] = total_time
            
            return result
        else:
            return {
                "error": f"API request failed: {response.status_code}, {response.text}",
                "image_path": image_path,
                "image_name": os.path.basename(image_path)
            }
    except Exception as e:
        return {
            "error": str(e),
            "image_path": image_path,
            "image_name": os.path.basename(image_path)
        }

def test_with_individual_model(image_path: str, model_name: str, api_url: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Test image with an individual model endpoint."""
    try:
        # Encode image
        base64_image = encode_image(image_path)
        
        # Prepare request payload
        payload = {
            "image": base64_image,
            "threshold": threshold
        }
        
        # Make request
        start_time = time.time()
        
        # Use longer timeout for potentially slower models
        timeout = 120 if "universal" in model_name.lower() or "hifi" in model_name.lower() else 60
        
        response = requests.post(
            api_url, 
            json=payload, 
            timeout=timeout
        )
        
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Add model name and timing
            result["model_name"] = model_name
            result["total_request_time"] = total_time
            result["image_path"] = image_path
            result["image_name"] = os.path.basename(image_path)
            
            return result
        else:
            return {
                "error": f"API request failed: {response.status_code}, {response.text}",
                "model_name": model_name,
                "image_path": image_path,
                "image_name": os.path.basename(image_path)
            }
    except Exception as e:
        return {
            "error": str(e),
            "model_name": model_name,
            "image_path": image_path,
            "image_name": os.path.basename(image_path)
        }

def get_image_files(folder_path: str) -> List[str]:
    """Get all image files in the specified folder."""
    # Common image extensions
    extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp']
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f"*.{ext}")))
        image_files.extend(glob.glob(os.path.join(folder_path, f"*.{ext.upper()}")))
    
    return sorted(image_files)

def print_ensemble_results_table(results: List[Dict[str, Any]]) -> None:
    """Print a table of ensemble results for all images."""
    table = Table(title="DeepSafe Ensemble Results")
    
    # Add columns
    table.add_column("Image", style="cyan")
    table.add_column("Dimensions", style="blue")
    table.add_column("Size (KB)", style="blue")
    table.add_column("Verdict", style="magenta")
    table.add_column("Confidence", style="yellow")
    table.add_column("Fake Votes", style="red")
    table.add_column("Real Votes", style="green")
    table.add_column("Time (s)", style="cyan")
    
    # Add rows for each image
    for result in results:
        if "error" in result:
            table.add_row(
                result.get("image_name", "Unknown"),
                "Error", "Error",
                "[bold red]ERROR", "-", "-", "-", "-"
            )
            continue
            
        verdict = result.get('verdict', 'unknown').upper()
        confidence = result.get('confidence', 0.0)
        fake_votes = result.get('fake_votes', 0)
        real_votes = result.get('real_votes', 0)
        inference_time = result.get('inference_time', 0.0)
        dimensions = result.get('image_dimensions', 'Unknown')
        size = f"{result.get('image_size', 0):.1f}"
        
        verdict_color = "red" if verdict == "FAKE" else "green" if verdict == "REAL" else "yellow"
        
        table.add_row(
            result.get("image_name", "Unknown"),
            dimensions,
            size,
            f"[bold {verdict_color}]{verdict}",
            f"{confidence:.2f}",
            str(fake_votes),
            str(real_votes),
            f"{inference_time:.2f}"
        )
    
    console.print(table)

def print_individual_model_table(results: List[Dict[str, Dict[str, Any]]]) -> None:
    """Print a table of individual model results for all images."""
    # Get all model names
    model_names = sorted(MODEL_ENDPOINTS.keys())
    
    # Create a table per image
    for image_result in results:
        image_name = image_result.get("image_name", "Unknown")
        
        table = Table(title=f"Model Results for {image_name}")
        
        # Add columns
        table.add_column("Model", style="cyan")
        table.add_column("Result", style="magenta")
        table.add_column("Probability", style="yellow")
        table.add_column("Time (s)", style="green")
        
        # Add rows for each model
        for model_name in model_names:
            if model_name not in image_result or "error" in image_result[model_name]:
                table.add_row(model_name, "[bold red]ERROR", "-", "-")
                continue
                
            model_result = image_result[model_name]
            prediction_class = model_result.get("class", "unknown").upper()
            probability = model_result.get("probability", 0.0)
            inference_time = model_result.get("inference_time", model_result.get("inference_time_seconds", 0.0))
            
            class_color = "red" if prediction_class == "FAKE" else "green" if prediction_class == "REAL" else "yellow"
            
            table.add_row(
                model_name,
                f"[bold {class_color}]{prediction_class}",
                f"{probability:.4f}",
                f"{inference_time:.3f}"
            )
        
        console.print(table)
        console.print()  # Add spacing between tables

def print_model_summary_table(results: List[Dict[str, Dict[str, Any]]]) -> None:
    """Print a summary table of model accuracy across all images."""
    # Get all model names
    model_names = sorted(MODEL_ENDPOINTS.keys())
    
    # Calculate stats for each model
    model_stats = {}
    for model_name in model_names:
        model_stats[model_name] = {
            "total": 0,
            "errors": 0,
            "fake_predictions": 0,
            "real_predictions": 0,
            "avg_time": 0.0,
            "total_time": 0.0
        }
    
    # Gather stats from all image results
    for image_result in results:
        for model_name in model_names:
            if model_name not in image_result:
                continue
                
            model_result = image_result[model_name]
            model_stats[model_name]["total"] += 1
            
            if "error" in model_result:
                model_stats[model_name]["errors"] += 1
                continue
                
            prediction_class = model_result.get("class", "unknown").upper()
            
            if prediction_class == "FAKE":
                model_stats[model_name]["fake_predictions"] += 1
            elif prediction_class == "REAL":
                model_stats[model_name]["real_predictions"] += 1
                
            inference_time = model_result.get("inference_time", model_result.get("inference_time_seconds", 0.0))
            model_stats[model_name]["total_time"] += inference_time
    
    # Calculate averages
    for model_name in model_names:
        if model_stats[model_name]["total"] > 0:
            successful_runs = model_stats[model_name]["total"] - model_stats[model_name]["errors"]
            if successful_runs > 0:
                model_stats[model_name]["avg_time"] = model_stats[model_name]["total_time"] / successful_runs
    
    # Create summary table
    table = Table(title="Model Performance Summary")
    
    # Add columns
    table.add_column("Model", style="cyan")
    table.add_column("Success Rate", style="green")
    table.add_column("Fake %", style="red")
    table.add_column("Real %", style="blue")
    table.add_column("Avg. Time (s)", style="yellow")
    
    # Add rows for each model
    for model_name in model_names:
        stats = model_stats[model_name]
        
        if stats["total"] == 0:
            table.add_row(model_name, "N/A", "N/A", "N/A", "N/A")
            continue
            
        success_rate = (stats["total"] - stats["errors"]) / stats["total"] * 100
        
        if stats["total"] - stats["errors"] > 0:
            fake_percent = stats["fake_predictions"] / (stats["total"] - stats["errors"]) * 100
            real_percent = stats["real_predictions"] / (stats["total"] - stats["errors"]) * 100
        else:
            fake_percent = 0
            real_percent = 0
        
        table.add_row(
            model_name,
            f"{success_rate:.1f}%",
            f"{fake_percent:.1f}%",
            f"{real_percent:.1f}%",
            f"{stats['avg_time']:.3f}"
        )
    
    console.print(table)

def main():
    parser = argparse.ArgumentParser(
        description="DeepSafe Batch Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all images in the assets folder using the ensemble API
  ./deepsafe_batch_test.py --assets ./assets
  
  # Test with individual models and save results
  ./deepsafe_batch_test.py --assets ./assets --individual --output results.json
  
  # Change threshold and ensemble method
  ./deepsafe_batch_test.py --assets ./assets --threshold 0.7 --method average
  """
    )
    
    parser.add_argument('--assets', type=str, required=True, help='Path to folder containing images to test')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold (0.0 to 1.0)')
    parser.add_argument('--method', type=str, default="voting", choices=["voting", "average"], help='Ensemble method')
    parser.add_argument('--individual', action='store_true', help='Also test individual models (not just the ensemble API)')
    
    args = parser.parse_args()
    
    # Check if assets folder exists
    if not os.path.isdir(args.assets):
        console.print(f"[bold red]Error:[/bold red] Assets folder does not exist: {args.assets}")
        sys.exit(1)
    
    # Get all image files
    image_files = get_image_files(args.assets)
    
    if not image_files:
        console.print(f"[bold yellow]Warning:[/bold yellow] No image files found in {args.assets}")
        sys.exit(0)
    
    console.print(f"[bold]Found {len(image_files)} images in {args.assets}[/bold]")
    
    # Process all images
    ensemble_results = []
    individual_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn()
    ) as progress:
        # Test with DeepSafe API
        api_task = progress.add_task(f"[cyan]Testing images with DeepSafe API...", total=len(image_files))
        
        for i, image_path in enumerate(image_files):
            img_name = os.path.basename(image_path)
            progress.update(api_task, description=f"[cyan]Testing with API: {img_name} ({i+1}/{len(image_files)})")
            
            try:
                result = test_with_api(image_path, args.threshold, args.method)
                ensemble_results.append(result)
            except Exception as e:
                console.print(f"[bold red]Error processing {img_name}:[/bold red] {str(e)}")
                ensemble_results.append({
                    "error": str(e),
                    "image_path": image_path,
                    "image_name": img_name
                })
            
            progress.update(api_task, advance=1)
        
        # Test with individual models if requested
        if args.individual:
            individual_task = progress.add_task(f"[cyan]Testing with individual models...", total=len(image_files) * len(MODEL_ENDPOINTS))
            
            for i, image_path in enumerate(image_files):
                img_name = os.path.basename(image_path)
                img_result = {"image_path": image_path, "image_name": img_name}
                
                for model_name, api_url in MODEL_ENDPOINTS.items():
                    progress.update(individual_task, description=f"[cyan]Testing {img_name} with {model_name} ({i+1}/{len(image_files)})")
                    
                    try:
                        result = test_with_individual_model(image_path, model_name, api_url, args.threshold)
                        img_result[model_name] = result
                    except Exception as e:
                        console.print(f"[bold red]Error with {model_name} on {img_name}:[/bold red] {str(e)}")
                        img_result[model_name] = {
                            "error": str(e),
                            "model_name": model_name,
                            "image_path": image_path,
                            "image_name": img_name
                        }
                    
                    progress.update(individual_task, advance=1)
                
                individual_results.append(img_result)
    
    # Print results tables
    console.print("\n[bold]Ensemble Results:[/bold]")
    print_ensemble_results_table(ensemble_results)
    
    if args.individual:
        console.print("\n[bold]Individual Model Results:[/bold]")
        print_individual_model_table(individual_results)
        
        console.print("\n[bold]Model Performance Summary:[/bold]")
        print_model_summary_table(individual_results)
    
    # Save results to file if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "threshold": args.threshold,
                "ensemble_method": args.method,
                "image_count": len(image_files)
            },
            "ensemble_results": ensemble_results
        }
        
        if args.individual:
            output_data["individual_results"] = individual_results
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        console.print(f"\n[bold green]Results saved to {args.output}[/bold green]")
    
    console.print("\n[bold green]===== Batch Testing Complete =====[/bold green]")

if __name__ == "__main__":
    main()