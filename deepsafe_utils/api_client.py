import requests
import time
import base64
import gc
import os 
from typing import Dict, Any, Optional, List
from rich.console import Console
import urllib.parse # For parsing URLs

console = Console()

MEDIA_TYPE_PAYLOAD_KEYS = {
    "image": "image_data",
    "video": "video_data",
    "audio": "audio_data"
}

class APIClient:
    def __init__(self, config_manager, media_type: Optional[str], run_from_host: bool = True):
        self.config_manager = config_manager
        self.media_type = media_type
        self.api_url_config = self.config_manager.get_api_url() # Store original config URL
        self.run_from_host = run_from_host

        # Determine the actual API URL to use
        if self.run_from_host:
            parsed_api_url = urllib.parse.urlparse(self.api_url_config)
            if parsed_api_url.hostname and parsed_api_url.hostname != "localhost" and parsed_api_url.hostname != "127.0.0.1":
                # If the hostname in config is a service name (e.g., "api"), replace it with localhost
                # This assumes the service name is just the hostname part without subdomains
                self.api_url = self.api_url_config.replace(f"http://{parsed_api_url.hostname}", "http://localhost")
            else:
                self.api_url = self.api_url_config # It's already localhost or an IP
        else:
            self.api_url = self.api_url_config


        if self.media_type:
            model_endpoints_config = self.config_manager.get_model_endpoints(media_type)
            health_endpoints_config = self.config_manager.get_health_endpoints(media_type)

            if self.run_from_host:
                self.model_endpoints = {}
                for name, url_str in model_endpoints_config.items():
                    try:
                        parsed_url = urllib.parse.urlparse(url_str)
                        # Only replace if hostname is not localhost or 127.0.0.1 (i.e., it's a service name)
                        if parsed_url.hostname and parsed_url.hostname != "localhost" and parsed_url.hostname != "127.0.0.1":
                            self.model_endpoints[name] = url_str.replace(f"http://{parsed_url.hostname}", "http://localhost")
                        else:
                            self.model_endpoints[name] = url_str
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not parse or transform URL '{url_str}' for model '{name}': {e}[/yellow]")
                        self.model_endpoints[name] = url_str # Keep original if parsing/replacement fails

                self.health_endpoints = {}
                for name, url_str in health_endpoints_config.items():
                    try:
                        parsed_url = urllib.parse.urlparse(url_str)
                        if parsed_url.hostname and parsed_url.hostname != "localhost" and parsed_url.hostname != "127.0.0.1":
                           self.health_endpoints[name] = url_str.replace(f"http://{parsed_url.hostname}", "http://localhost")
                        else:
                            self.health_endpoints[name] = url_str
                    except Exception as e:
                        console.print(f"[yellow]Warning: Could not parse or transform URL '{url_str}' for health '{name}': {e}[/yellow]")
                        self.health_endpoints[name] = url_str

            else: # Running inside Docker or when service names are desired
                self.model_endpoints = model_endpoints_config
                self.health_endpoints = health_endpoints_config
        else:
            self.model_endpoints = {}
            self.health_endpoints = {}

        self.timeout = self.config_manager.get_default("default_api_timeout_seconds", 1200)
        self.max_retries = self.config_manager.get_default("default_max_retries", 1)
        # console.print(f"[APIClient Debug] Initialized. Run_from_host: {self.run_from_host}, API URL: {self.api_url}, Model Endpoints: {self.model_endpoints}")


    def _make_request(self, url: str, method: str = "GET", json_payload: Optional[Dict] = None,
                      files: Optional[Dict] = None, data: Optional[Dict] = None,
                      params: Optional[Dict] = None,
                      attempt_num: int = 0) -> requests.Response:
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=self.timeout, params=params)
            elif method.upper() == "POST":
                response = requests.post(url, json=json_payload, files=files, data=data, timeout=self.timeout, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            console.print(f"[yellow]Request to {url} timed out (attempt {attempt_num+1}/{self.max_retries+1}).[/yellow]")
            if attempt_num < self.max_retries:
                time.sleep(1 * (attempt_num + 1)) # Simple linear backoff for test client
                return self._make_request(url, method, json_payload, files, data, params, attempt_num + 1)
            raise
        except requests.exceptions.HTTPError as e:
            # console.print(f"[red]HTTP error for {url}: {e.response.status_code} {e.response.text[:200]} (attempt {attempt_num+1}).[/red]")
            if attempt_num < self.max_retries and e.response and e.response.status_code in [429, 502, 503, 504]: # Ensure e.response exists
                 time.sleep(1 * (attempt_num + 1))
                 return self._make_request(url, method, json_payload, files, data, params, attempt_num + 1)
            raise
        except requests.exceptions.RequestException as e: # Catches ConnectionError, NameResolutionError etc.
            # console.print(f"[red]Request exception for {url}: {e} (attempt {attempt_num+1}).[/red]")
            if attempt_num < self.max_retries:
                time.sleep(1 * (attempt_num + 1))
                return self._make_request(url, method, json_payload, files, data, params, attempt_num + 1)
            raise

    def check_main_api_health(self, force_refresh=False) -> Dict[str, Any]: # force_refresh not used here, config reloaded on init
        health_url = f"{self.api_url}/health"
        # console.print(f"[APIClient Debug] Checking main API health at: {health_url}")
        try:
            response = self._make_request(health_url, "GET")
            return response.json()
        except Exception as e:
            console.print(f"[bold red]Error checking main API health at {health_url}: {e}[/bold red]")
            return {"status": "error", "detail": str(e), "url_attempted": health_url}

    def check_model_health(self, model_name: str) -> Dict[str, Any]:
        if not self.media_type:
             return {"status": "error", "detail": "Media type not set for APIClient to check model health."}
        if model_name not in self.health_endpoints:
            return {"status": "error", "detail": f"Unknown model or health endpoint not configured: {model_name} for media type {self.media_type}"}
        
        health_url = self.health_endpoints[model_name]
        # console.print(f"[APIClient Debug] Checking model health for {model_name} at: {health_url}")
        try:
            response = self._make_request(health_url, "GET")
            return response.json()
        except Exception as e:
            console.print(f"[bold red]Error checking model health for {model_name} at {health_url}: {e}[/bold red]")
            return {"status": "error", "detail": str(e), "url_attempted": health_url}

    def test_with_main_api(self, media_path: str, media_type_for_api: str, encoded_media: str,
                           threshold: float, ensemble_method: str,
                           selected_models_for_api: Optional[List[str]] = None) -> Dict[str, Any]:
        predict_url = f"{self.api_url}/predict"
        payload = {
            "media_type": media_type_for_api,
            "threshold": threshold,
            "ensemble_method": ensemble_method
        }
        
        payload_key = MEDIA_TYPE_PAYLOAD_KEYS.get(media_type_for_api)
        if not payload_key:
            return {"error": f"Unsupported media_type '{media_type_for_api}' for main API payload key mapping"}
        
        payload[payload_key] = encoded_media

        if selected_models_for_api:
            payload["models"] = selected_models_for_api

        # console.print(f"[cyan]Testing {media_type_for_api} with main API (ensemble: {ensemble_method}) at {predict_url}...[/cyan]")
        start_time = time.time()
        try:
            response = self._make_request(predict_url, "POST", json_payload=payload)
            result = response.json()
            result['client_request_time'] = time.time() - start_time
            result['media_path'] = media_path
            result['media_name'] = os.path.basename(media_path)
            return result
        except Exception as e:
            console.print(f"[bold red]Error testing with main API ({predict_url}): {e}[/bold red]")
            return {"error": str(e), "media_path": media_path, "media_name": os.path.basename(media_path), "url_attempted": predict_url}


    def test_with_individual_model(self, model_name: str, media_path: str, encoded_media: str, threshold: float) -> Dict[str, Any]:
        if not self.media_type:
             return {"error": "Media type not set for APIClient to test individual model."}
        
        if model_name not in self.model_endpoints:
            original_model_endpoints = self.config_manager.get_model_endpoints(self.media_type)
            if model_name not in original_model_endpoints:
                 return {"error": f"Unknown model: {model_name} for media type {self.media_type}. Not found in config.", "model_name": model_name, "media_path": media_path}
            else: 
                 return {"error": f"Could not determine host URL for model endpoint: {model_name}. Original URL: {original_model_endpoints[model_name]}", "model_name": model_name, "media_path": media_path}

        model_predict_url = self.model_endpoints[model_name]
        payload = {"threshold": threshold}
        
        payload_key = MEDIA_TYPE_PAYLOAD_KEYS.get(self.media_type)
        if not payload_key:
             return {"error": f"Unsupported media_type '{self.media_type}' for individual model {model_name} payload"}
        payload[payload_key] = encoded_media

        # console.print(f"[cyan]Testing {self.media_type} with {model_name} at {model_predict_url}...[/cyan]")
        start_time = time.time()
        try:
            response = self._make_request(model_predict_url, "POST", json_payload=payload)
            result = response.json()
            result["model_name"] = model_name
            result["total_request_time"] = time.time() - start_time
            result['media_path'] = media_path
            result['media_name'] = os.path.basename(media_path)
            return result
        except Exception as e:
            # console.print(f"[bold red]Error testing with model {model_name} at {model_predict_url}: {e}[/bold red]")
            return {"error": str(e), "model_name": model_name, "media_path": media_path, "media_name": os.path.basename(media_path), "url_attempted": model_predict_url}
        finally:
            gc.collect()

    def request_model_unload(self, model_name: str) -> bool:
        if not self.media_type:
             console.print(f"[yellow]Media type not set. Cannot determine health endpoint for {model_name}.[/yellow]")
             return False
        if model_name not in self.health_endpoints: 
            original_health_endpoints = self.config_manager.get_health_endpoints(self.media_type)
            if model_name not in original_health_endpoints:
                console.print(f"[yellow]No health endpoint defined for {model_name} (type: {self.media_type}), cannot send unload request.[/yellow]")
            else:
                console.print(f"[yellow]Could not determine host URL for {model_name} health endpoint. Original URL: {original_health_endpoints[model_name]}[/yellow]")
            return False

        unload_url = self.health_endpoints[model_name].replace("/health", "/unload")
        try:
            console.print(f"[cyan]Requesting unload for model {model_name} at {unload_url}...[/cyan]")
            response = requests.post(unload_url, timeout=20) 
            if response.status_code == 200:
                console.print(f"[green]Successfully requested unload of model {model_name}. Response: {response.text[:100]}[/green]")
                return True
            else:
                console.print(f"[yellow]Unload request for {model_name} failed or not supported. Status: {response.status_code}, Text: {response.text[:100]}[/yellow]")
                return False
        except requests.exceptions.ConnectionError:
            console.print(f"[yellow]Could not connect to {unload_url} for {model_name}. Model service might not have /unload or is down.[/yellow]")
            return False
        except Exception as e:
            console.print(f"[red]Error during unload request for {model_name}: {e}[/red]")
            return False