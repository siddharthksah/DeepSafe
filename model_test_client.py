#!/usr/bin/env python3
"""
DeepSafe Model Test Client

A diagnostic tool for testing and debugging individual deepfake detection models
in the DeepSafe system. This client connects directly to individual model endpoints
rather than going through the main API.

Usage:
  ./model_test_client.py --image FILE [--url URL] [--all] [--health] [--threshold NUM]
"""

import requests
import base64
import time
import json
import argparse
import os
import sys
from typing import Optional, Dict, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

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

# Health endpoints
HEALTH_ENDPOINTS = {model: endpoint.replace("/predict", "/health") for model, endpoint in MODEL_ENDPOINTS.items()}

def encode_image(image_path: str) -> str:
    """Read an image file and encode it as base64."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')

def test_model(image_path: str, api_url: str, threshold: float = 0.5) -> Optional[dict]:
    """Test a deepfake detection model with an image."""
    # Find model name from URL
    model_name = "unknown"
    for name, url in MODEL_ENDPOINTS.items():
        if url == api_url:  # Check for exact match first
            model_name = name
            break
        elif name in api_url:  # Fallback check if name is in URL
            model_name = name
            break

    console.print(f"\n[bold cyan]Testing {model_name.upper()}[/bold cyan]")
    console.print(f"URL: {api_url}")
    console.print(f"Image: {image_path}")

    # Get image info
    try:
        from PIL import Image
        img = Image.open(image_path)
        img_width, img_height = img.size
        img_size = os.path.getsize(image_path) / 1024  # KB
        console.print(f"Image dimensions: {img_width}x{img_height}, Size: {img_size:.1f} KB")
    except:
        pass

    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            console.print(f"[bold red]Error:[/bold red] Image file {image_path} does not exist")
            return None

        # Encode the image
        with console.status(f"[bold green]Encoding image..."):
            encoded_image = encode_image(image_path)

        # Create request payload
        payload = {
            "image": encoded_image,
            "threshold": threshold
        }

        # Make request with timing
        console.print("Sending request...")
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=None)
            
            start_time = time.time()
            
            # Use longer timeout for potentially slower models
            timeout = 120 if "universal" in model_name.lower() or "hifi" in model_name.lower() else 60
            
            try:
                response = requests.post(api_url, json=payload, timeout=timeout)
                progress.update(task, completed=100)
                
                total_time = time.time() - start_time
                
                # Print results
                console.print(f"Status code: {response.status_code}")
                console.print(f"Total request time: {total_time:.4f} seconds")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Create result panel
                    prediction = result.get("prediction", -1)
                    class_label = result.get("class", "unknown").upper()
                    probability = result.get("probability", 0.0)
                    inference_time = result.get("inference_time", result.get("inference_time_seconds", 0.0))
                    
                    color = "red" if class_label == "FAKE" else "green" if class_label == "REAL" else "yellow"
                    
                    result_panel = Panel(
                        f"[bold]{class_label}[/bold]\n" +
                        f"Probability: {probability:.4f}\n" +
                        f"Inference time: {inference_time:.4f}s",
                        title=f"[bold]{model_name.upper()} Result",
                        border_style=color
                    )
                    
                    console.print(result_panel)
                    
                    # Add the model name and total request time to the result
                    result["model_name"] = model_name
                    result["total_request_time"] = total_time
                    
                    return result
                else:
                    console.print(f"[bold red]Error:[/bold red] {response.text}")
                    return None
                
            except requests.exceptions.Timeout:
                progress.update(task, completed=100)
                console.print(f"[bold red]Error:[/bold red] Request timed out after {timeout} seconds. The model may still be loading or processing took too long.")
                return None
            
    except requests.exceptions.ConnectionError:
        console.print(f"[bold red]Error:[/bold red] Could not connect to the API at {api_url}. Ensure the service is running.")
        return None
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {str(e)}")
        return None

def test_model_health(api_url: str, retry: bool = True) -> Optional[dict]:
    """Test the health endpoint of a model service with optional retry for loading state."""
    model_name = "unknown"
    base_url_predict = api_url  # Assume input is predict URL
    
    if "/predict" not in api_url:  # If base URL/health URL provided
        base_url_predict = api_url.replace("/health", "") + "/predict"

    for name, url in MODEL_ENDPOINTS.items():
        if url == base_url_predict:
            model_name = name
            break
        elif name in base_url_predict:
            model_name = name
            break

    console.print(f"\n[bold cyan]Health Check: {model_name.upper()}[/bold cyan]")

    # Construct health URL from predict URL or base URL
    base_url = base_url_predict.replace("/predict", "")
    health_url = f"{base_url}/health"
    console.print(f"Health URL: {health_url}")

    max_retries = 5 if retry else 1
    retry_delay = 10  # seconds

    for attempt in range(max_retries):
        try:
            with console.status(f"[bold green]Checking health (Attempt {attempt+1}/{max_retries})..."):
                response = requests.get(health_url, timeout=15)  # Increased timeout slightly
            
            console.print(f"Health check attempt {attempt+1}/{max_retries} -> Status code: {response.status_code}")

            if response.status_code == 200:
                health_data = response.json()
                status = health_data.get('status', 'unknown')
                
                # Format status with color
                if status == "healthy":
                    status_str = f"[bold green]{status}"
                elif status == "loading":
                    status_str = f"[bold yellow]{status}"
                else:
                    status_str = f"[bold red]{status}"
                    
                console.print(f"Health status: {status_str}")

                if 'device' in health_data:
                    console.print(f"Running on: {health_data['device']}")
                
                if 'model_loaded' in health_data:
                    console.print(f"Model loaded: {'[bold green]Yes' if health_data['model_loaded'] else '[bold red]No'}")

                if status == "loading" and retry and attempt < max_retries - 1:
                    console.print(f"[bold yellow]Model is still loading. Retrying in {retry_delay} seconds...[/bold yellow]")
                    time.sleep(retry_delay)
                    continue

                # If status is healthy or loading (and no more retries), return data
                return health_data
            else:
                console.print(f"[bold red]Health check failed:[/bold red] {response.text}")
                if retry and attempt < max_retries - 1:
                    console.print(f"[bold yellow]Retrying in {retry_delay} seconds...[/bold yellow]")
                    time.sleep(retry_delay)
                    continue
                return None  # Failed after retries or if no retry

        except requests.exceptions.Timeout:
            console.print(f"[bold red]Health check attempt {attempt+1}/{max_retries} -> Request timed out.[/bold red]")
            if retry and attempt < max_retries - 1:
                console.print(f"[bold yellow]Retrying in {retry_delay} seconds...[/bold yellow]")
                time.sleep(retry_delay)
                continue
            return None
        except requests.exceptions.ConnectionError:
            console.print(f"[bold red]Health check attempt {attempt+1}/{max_retries} -> Connection Error. Service might not be up.[/bold red]")
            if retry and attempt < max_retries - 1:
                console.print(f"[bold yellow]Retrying in {retry_delay} seconds...[/bold yellow]")
                time.sleep(retry_delay)
                continue
            return None  # Failed to connect after retries or if no retry
        except Exception as e:
            console.print(f"[bold red]Health check attempt {attempt+1}/{max_retries} -> Error: {str(e)}[/bold red]")
            if retry and attempt < max_retries - 1:
                console.print(f"[bold yellow]Retrying in {retry_delay} seconds...[/bold yellow]")
                time.sleep(retry_delay)
                continue
            return None  # Failed after retries or if no retry

    console.print(f"[bold red]Max retries ({max_retries}) reached for health check. Model '{model_name}' did not report healthy.[/bold red]")
    return None

def test_all_models(image_path: str, check_health: bool = True, threshold: float = 0.5) -> Dict[str, Optional[dict]]:
    """Test all individual models with the same image and return results."""
    results = {}

    if not os.path.exists(image_path):
        console.print(f"[bold red]Error:[/bold red] Image file {image_path} does not exist")
        return results

    # Sort models to test in a consistent order
    model_items = sorted(MODEL_ENDPOINTS.items())

    for model_name, api_url in model_items:
        is_healthy = True  # Default to true if health check is skipped
        
        if check_health:
            health_info = test_model_health(api_url, retry=True)  # Retry health check
            if health_info and health_info.get('status') == 'healthy':
                is_healthy = True
            else:
                is_healthy = False
                console.print(f"[bold yellow]Warning:[/bold yellow] {model_name} health check did not return 'healthy' status after retries. Skipping prediction.")

        if not check_health or is_healthy:  # Proceed if health check passed or was skipped
            result = test_model(image_path, api_url, threshold)
            results[model_name] = result
        else:
            results[model_name] = None  # Mark as failed due to health check failure

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DeepSafe Model Test Client - Diagnostic tool for testing individual deepfake detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a specific model
  ./model_test_client.py --image path/to/image.jpg --url http://localhost:5000/predict
  
  # Check health of a model
  ./model_test_client.py --url http://localhost:5000/health --health
  
  # Test all models with health checks
  ./model_test_client.py --image path/to/image.jpg --all --health
  
  # Test all models with custom threshold
  ./model_test_client.py --image path/to/image.jpg --all --threshold 0.7
  """
    )
    
    parser.add_argument('--image', type=str, help='Path to the image file')
    parser.add_argument('--url', type=str, default=None, help='URL of a specific model API predict endpoint to test (e.g., http://localhost:5003/predict)')
    parser.add_argument('--all', action='store_true', help='Test all configured models')
    parser.add_argument('--health', action='store_true', help='Check model health endpoint(s) before prediction (retries if loading)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold (0.0 to 1.0)')
    parser.add_argument('--output', type=str, help='Save results to JSON file')

    args = parser.parse_args()

    # Validate --image is provided when needed
    if not args.health and not args.image:
        console.print("[bold red]Error:[/bold red] --image argument is required for prediction tests")
        parser.print_help()
        sys.exit(1)

    if args.image and not os.path.exists(args.image):
        console.print(f"[bold red]Error:[/bold red] Image file specified does not exist: {args.image}")
        sys.exit(1)

    # If only --health is specified, we don't need an image
    if args.health and args.url and not args.image:
        health_info = test_model_health(args.url, retry=True)
        sys.exit(0 if health_info and health_info.get('status') == 'healthy' else 1)

    # Store all results for potential output file
    all_results = {}

    if args.all:
        # Test all individual models
        console.print("[bold]Running tests on all DeepSafe models[/bold]")
        results = test_all_models(args.image, args.health, args.threshold)
        all_results = results

        # Print summary
        console.print("\n[bold]Prediction Summary[/bold]")
        
        table = Table(title="Model Results Summary")
        table.add_column("Model", style="cyan")
        table.add_column("Result", style="magenta")
        table.add_column("Probability", style="yellow")
        table.add_column("Time (s)", style="green")
        
        for model, result in results.items():
            if result:
                prediction_class = result.get("class", "unknown").upper()
                probability = result.get("probability", -1)
                inf_time = result.get("inference_time", -1)
                if inf_time == -1:
                    inf_time = result.get("inference_time_seconds", -1)
                
                class_color = "red" if prediction_class == "FAKE" else "green" if prediction_class == "REAL" else "yellow"
                prob_str = f"{probability:.4f}" if probability != -1 else "N/A"
                time_str = f"{inf_time:.3f}" if inf_time != -1 else "N/A"
                
                table.add_row(
                    model.ljust(20), 
                    f"[bold {class_color}]{prediction_class}", 
                    prob_str, 
                    time_str
                )
            else:
                table.add_row(model.ljust(20), "[bold red]Failed or Skipped", "N/A", "N/A")
        
        console.print(table)

    elif args.url:
        # Test specific model URL provided by user
        # Validate if URL is one of the known predict endpoints
        if args.url not in MODEL_ENDPOINTS.values():
            console.print(f"[bold yellow]Warning:[/bold yellow] Provided URL '{args.url}' is not in the configured MODEL_ENDPOINTS.")

        if args.health:
            # Perform health check for the specific URL
            health_info = test_model_health(args.url, retry=True)
            if not health_info or health_info.get('status') != 'healthy':
                console.print(f"[bold red]Health check failed for {args.url}. Aborting prediction test.[/bold red]")
                sys.exit(1)

        # Proceed with prediction test
        result = test_model(args.image, args.url, args.threshold)
        if result:
            all_results[result.get("model_name", "custom")] = result

    else:
        # Default behavior: Test only the first model in the list if --all or --url not specified
        default_model_name = next(iter(MODEL_ENDPOINTS))
        default_model_url = MODEL_ENDPOINTS[default_model_name]
        console.print(f"[bold yellow]No specific model or --all flag provided. Testing default model: {default_model_name}[/bold yellow]")

        if args.health:
            # Health check for the default model
            health_info = test_model_health(default_model_url, retry=True)
            if not health_info or health_info.get('status') != 'healthy':
                console.print(f"[bold red]Health check failed for default model {default_model_name}. Aborting prediction test.[/bold red]")
                sys.exit(1)

        # Proceed with prediction test for the default model
        result = test_model(args.image, default_model_url, args.threshold)
        if result:
            all_results[default_model_name] = result

    # Save results to file if requested
    if args.output and all_results:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        console.print(f"\n[bold green]Results saved to {args.output}[/bold green]")

    console.print("\n[bold green]===== Test Client Finished =====[/bold green]")

    # Exit with success code if any model succeeded
    sys.exit(0 if any(result is not None for result in all_results.values()) else 1)