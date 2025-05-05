#!/usr/bin/env python3
"""
DeepSafe Client - Official client library for the DeepSafe Deepfake Detection API

This client provides a simple interface for detecting deepfakes using the 
DeepSafe ensemble of state-of-the-art detection models.

Usage:
  ./deepsafe_client.py health [--api URL]
  ./deepsafe_client.py detect --image FILE [--api URL] [--models LIST] [--threshold NUM] [--method METHOD] [--output FILE]
"""

import requests
import base64
import argparse
import time
import os
import json
import sys
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

console = Console()

class DeepSafeClient:
    """Client for the DeepSafe API for deepfake detection"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """Initialize the client with the API URL"""
        self.api_url = api_url
        self.base_headers = {
            "Content-Type": "application/json"
        }
    
    def check_health(self) -> Dict[str, Any]:
        """Check the health status of all models"""
        url = f"{self.api_url}/health"
        
        with console.status("[bold green]Checking DeepSafe system health..."):
            try:
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"Failed to check health: {response.status_code}, {response.text}")
            except requests.exceptions.ConnectionError:
                raise Exception(f"Connection error. Is the DeepSafe API running at {self.api_url}?")
            except Exception as e:
                raise Exception(f"Health check failed: {str(e)}")
    
    def encode_image(self, image_path: str) -> str:
        """Encode an image file to base64"""
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def detect_deepfake(self, 
                      image_path: str, 
                      models: Optional[List[str]] = None,
                      threshold: float = 0.5,
                      ensemble_method: str = "voting") -> Dict[str, Any]:
        """
        Detect if an image is a deepfake
        
        Args:
            image_path: Path to the image file
            models: List of specific models to use (all by default)
            threshold: Classification threshold (0.0 to 1.0)
            ensemble_method: Ensemble method ('voting' or 'average')
            
        Returns:
            Dictionary with detection results
        """
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Get image dimensions and size for reporting
        try:
            from PIL import Image
            img = Image.open(image_path)
            img_width, img_height = img.size
            img_size = os.path.getsize(image_path) / 1024  # KB
            img_format = img.format
        except:
            img_width, img_height, img_size, img_format = "Unknown", "Unknown", "Unknown", "Unknown"
        
        console.print(f"[bold]Image info:[/bold] {os.path.basename(image_path)} ({img_width}x{img_height}, {img_size:.1f}KB, {img_format})")
        
        # Encode image
        with console.status("[bold green]Encoding image..."):
            base64_image = self.encode_image(image_path)
        
        # Prepare request payload
        payload = {
            "image": base64_image,
            "threshold": threshold,
            "ensemble_method": ensemble_method
        }
        
        # Add models if specified
        if models:
            payload["models"] = models
        
        # Make request
        with console.status(f"[bold green]Sending request to DeepSafe API - {self.api_url}/predict..."):
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{self.api_url}/predict",
                    headers=self.base_headers,
                    json=payload,
                    timeout=120  # Increased timeout for initial model loading
                )
                
                total_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    result['client_request_time'] = total_time
                    return result
                else:
                    raise Exception(f"API request failed: {response.status_code}, {response.text}")
            except requests.exceptions.ConnectionError:
                raise Exception(f"Connection error. Is the DeepSafe API running at {self.api_url}?")
            except Exception as e:
                raise Exception(f"Detection failed: {str(e)}")
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print the detection results in a readable format"""
        
        # Create verdict panel
        verdict = results['verdict'].upper()
        confidence = results['confidence']
        color = "red" if verdict == "FAKE" else "green" if verdict == "REAL" else "yellow"
        
        verdict_panel = Panel(
            f"[bold]{verdict}[/bold] (Confidence: {confidence:.2f})\n" +
            f"Fake votes: {results['fake_votes']} | Real votes: {results['real_votes']}\n" +
            f"Ensemble method: {results['ensemble_method']}\n" +
            f"Total inference time: {results['inference_time']:.2f}s",
            title="[bold]VERDICT",
            border_style=color
        )
        
        console.print(verdict_panel)
        
        # Create table for individual model results
        table = Table(title="Individual Model Results")
        table.add_column("Model", style="cyan")
        table.add_column("Result", style="magenta")
        table.add_column("Probability", style="yellow")
        table.add_column("Time (s)", style="green")
        
        # Add rows for each model
        for model_name, result in results['model_results'].items():
            if "error" in result:
                table.add_row(model_name, "[bold red]ERROR", "-", "-")
            else:
                class_label = result.get("class", "unknown")
                probability = result.get("probability", 0.0)
                inference_time = result.get("inference_time", result.get("inference_time_seconds", 0.0))
                
                result_color = "red" if class_label.upper() == "FAKE" else "green"
                table.add_row(
                    model_name, 
                    f"[bold {result_color}]{class_label.upper()}", 
                    f"{probability:.2f}", 
                    f"{inference_time:.2f}"
                )
        
        console.print(table)

def main():
    parser = argparse.ArgumentParser(
        description="DeepSafe API Client for Deepfake Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check system health
  ./deepsafe_client.py health
  
  # Detect deepfakes in an image using all models
  ./deepsafe_client.py detect --image path/to/image.jpg
  
  # Use specific models with custom threshold and ensemble method
  ./deepsafe_client.py detect --image path/to/image.jpg --models cnndetection,universalfakedetect --threshold 0.6 --method average
  
  # Save detection results to a JSON file
  ./deepsafe_client.py detect --image path/to/image.jpg --output results.json
  """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Check health status of models")
    health_parser.add_argument("--api", type=str, default="http://localhost:8000", help="DeepSafe API URL")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect if an image is a deepfake")
    detect_parser.add_argument("--image", type=str, required=True, help="Path to the image file to analyze")
    detect_parser.add_argument("--api", type=str, default="http://localhost:8000", help="DeepSafe API URL")
    detect_parser.add_argument("--models", type=str, help="Comma-separated list of specific models to use")
    detect_parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold (0.0 to 1.0)")
    detect_parser.add_argument("--method", type=str, default="voting", choices=["voting", "average"], 
                              help="Ensemble method ('voting' or 'average')")
    detect_parser.add_argument("--output", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Initialize client
    api_url = args.api if hasattr(args, 'api') else "http://localhost:8000"
    client = DeepSafeClient(api_url=api_url)
    
    # Execute the appropriate command
    if args.command == "health":
        try:
            health_status = client.check_health()
            
            # Create table for model status
            table = Table(title="DeepSafe System Health Status")
            table.add_column("Model", style="cyan")
            table.add_column("Status", style="magenta")
            table.add_column("Device", style="yellow")
            table.add_column("Message", style="green")
            
            # Add rows for each model
            for model, status in health_status['models'].items():
                model_status = status.get('status', 'unknown')
                device = status.get('device', 'unknown')
                message = status.get('message', '')
                
                # Format with color
                if model_status == 'healthy':
                    status_str = f"[bold green]{model_status}"
                elif model_status == 'loading':
                    status_str = f"[bold yellow]{model_status}"
                else:
                    status_str = f"[bold red]{model_status}"
                
                table.add_row(model, status_str, device, message)
            
            # Add overall status row
            table.add_row(
                "[bold]OVERALL", 
                f"[bold {'green' if health_status['status'] == 'healthy' else 'red'}]{health_status['status'].upper()}", 
                "", 
                ""
            )
            
            console.print(table)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)
    
    elif args.command == "detect":
        try:
            # Parse models list if provided
            models = None
            if args.models:
                models = [m.strip() for m in args.models.split(",")]
            
            # Make detection request
            console.print(f"[bold]DeepSafe Analysis[/bold]: Running deepfake detection")
            
            results = client.detect_deepfake(
                image_path=args.image,
                models=models,
                threshold=args.threshold,
                ensemble_method=args.method
            )
            
            # Print results
            client.print_results(results)
            
            # Save results to file if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                console.print(f"\n[bold green]Results saved to {args.output}[/bold green]")
                
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)
    
    else:
        # If no command provided, show help
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()