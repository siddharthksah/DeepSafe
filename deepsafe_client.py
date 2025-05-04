import requests
import base64
import argparse
import time
import os
import json
from typing import Optional, List, Dict, Any

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
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to check health: {response.status_code}, {response.text}")
    
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
        
        # Encode image
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
        print(f"Sending request to {self.api_url}/predict...")
        start_time = time.time()
        
        response = requests.post(
            f"{self.api_url}/predict",
            headers=self.base_headers,
            json=payload
        )
        
        total_time = time.time() - start_time
        print(f"Request completed in {total_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            raise Exception(f"API request failed: {response.status_code}, {response.text}")
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print the detection results in a readable format"""
        print("\n===== DEEPFAKE DETECTION RESULTS =====")
        print(f"VERDICT: {results['verdict'].upper()}")
        print(f"Confidence: {results['confidence']:.2f}")
        print(f"Fake votes: {results['fake_votes']}")
        print(f"Real votes: {results['real_votes']}")
        print(f"Total inference time: {results['inference_time']:.2f} seconds")
        print(f"Ensemble method: {results['ensemble_method']}")
        
        # Process individual model results
        print("\nINDIVIDUAL MODEL RESULTS:")
        for model_name, result in results['model_results'].items():
            if "error" in result:
                print(f"- {model_name}: ERROR - {result['error']}")
            else:
                class_label = result.get("class", "unknown")
                probability = result.get("probability", 0.0)
                inference_time = result.get("inference_time", result.get("inference_time_seconds", 0.0))
                print(f"- {model_name}: {class_label.upper()} (prob: {probability:.2f}, time: {inference_time:.2f}s)")

def main():
    parser = argparse.ArgumentParser(description="DeepSafe API Client for Deepfake Detection")
    
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
            print("\n===== HEALTH STATUS =====")
            print(f"Overall status: {health_status['status']}")
            
            print("\nIndividual models:")
            for model, status in health_status['models'].items():
                model_status = status.get('status', 'unknown')
                device = status.get('device', 'unknown')
                
                # Format with color codes if in a terminal
                if model_status == 'healthy':
                    status_str = model_status  # Green in terminal
                elif model_status == 'loading':
                    status_str = model_status  # Yellow in terminal
                else:
                    status_str = model_status  # Red in terminal
                
                print(f"- {model}: {status_str} (device: {device})")
                
                # Show error message if available
                if 'message' in status:
                    print(f"  Message: {status['message']}")
        except Exception as e:
            print(f"Failed to check health: {str(e)}")
    
    elif args.command == "detect":
        try:
            # Parse models list if provided
            models = None
            if args.models:
                models = [m.strip() for m in args.models.split(",")]
            
            # Make detection request
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
                print(f"\nResults saved to {args.output}")
                
        except Exception as e:
            print(f"Error during detection: {str(e)}")
    
    else:
        # If no command provided, show help
        parser.print_help()

if __name__ == "__main__":
    main()