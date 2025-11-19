import json
import os
from typing import Dict, Any, Optional, List
from rich.console import Console

console = Console()

DEFAULT_CONFIG_PATH = "config/deepsafe_config.json" # Relative to project root

class ConfigManager:
    _instance = None

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            # Ensure the path is resolved correctly if called from different locations
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assumes utils is one level down
            cls._instance.config_path = config_path or os.path.join(project_root, DEFAULT_CONFIG_PATH)
            cls._instance.config = cls._instance.load_config()
        return cls._instance

    def load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            # console.print(f"[green]Configuration loaded successfully from: {self.config_path}[/green]")
            return config_data
        except FileNotFoundError:
            console.print(f"[bold red]ERROR: Configuration file not found at {self.config_path}. Please create it based on the template.[/bold red]")
            console.print(f"Attempted path: {os.path.abspath(self.config_path)}")
            # Fallback to a minimal default to prevent immediate crash, but issue a strong warning.
            return {
                "api_url": "http://localhost:8000",
                "media_types": {"image": {"model_endpoints": {}, "health_endpoints": {}, "supported_extensions": []}},
                "default_output_dir_base": "./deepsafe_test_results_default_fallback",
                "error": "Configuration file missing or invalid."
            }
        except json.JSONDecodeError:
            console.print(f"[bold red]ERROR: Invalid JSON in configuration file: {self.config_path}.[/bold red]")
            return {
                "api_url": "http://localhost:8000",
                "media_types": {"image": {"model_endpoints": {}, "health_endpoints": {}, "supported_extensions": []}},
                "default_output_dir_base": "./deepsafe_test_results_default_fallback",
                "error": "Invalid JSON in configuration."
            }

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.config.get(key, default)

    def get_api_url(self) -> str:
        return self.config.get("api_url", "http://localhost:8000")

    def get_media_config(self, media_type: str) -> Optional[Dict[str, Any]]:
        return self.config.get("media_types", {}).get(media_type)

    def get_model_endpoints(self, media_type: str) -> Dict[str, str]:
        media_cfg = self.get_media_config(media_type)
        return media_cfg.get("model_endpoints", {}) if media_cfg else {}

    def get_health_endpoints(self, media_type: str) -> Dict[str, str]:
        media_cfg = self.get_media_config(media_type)
        return media_cfg.get("health_endpoints", {}) if media_cfg else {}

    def get_supported_extensions(self, media_type: str) -> List[str]:
        media_cfg = self.get_media_config(media_type)
        return media_cfg.get("supported_extensions", []) if media_cfg else []

    def get_all_model_names(self, media_type: Optional[str] = None) -> List[str]:
        if media_type:
            return list(self.get_model_endpoints(media_type).keys())
        else: # Get all models across all types
            all_models = []
            for mt in self.config.get("media_types", {}).keys():
                all_models.extend(self.get_model_endpoints(mt).keys())
            return list(set(all_models)) # Unique model names

    def get_default(self, key: str, fallback_value: Any = None) -> Any:
        return self.config.get(key, fallback_value)

    def is_config_loaded_successfully(self) -> bool:
        return "error" not in self.config