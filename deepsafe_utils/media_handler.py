import os
import base64
import glob
from typing import List, Tuple, Optional
from rich.console import Console
from PIL import Image, ImageFile # Keep PIL for image-specific parts

console = Console()
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MediaHandler:
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def find_media_files(self, input_dir: str, media_type: str) -> List[Tuple[str, str]]:
        media_items = []
        supported_extensions = self.config_manager.get_supported_extensions(media_type)

        if not input_dir or not os.path.isdir(input_dir):
            console.print(f"[bold red]Input directory '{input_dir}' does not exist or is invalid.[/bold red]")
            return []
            
        if not supported_extensions:
            console.print(f"[bold red]No supported extensions defined for media type '{media_type}' in config.[/bold red]")
            return []

        fake_dir_options = [os.path.join(input_dir, "Fake"), os.path.join(input_dir, "fake")]
        real_dir_options = [os.path.join(input_dir, "Real"), os.path.join(input_dir, "real")]

        fake_dir = next((d for d in fake_dir_options if os.path.isdir(d)), None)
        real_dir = next((d for d in real_dir_options if os.path.isdir(d)), None)

        found_paths = set()
        if fake_dir and real_dir:
            console.print(f"Searching for {media_type} files in standard Real/Fake subdirectories...")
            for dir_path_actual, label in [(fake_dir, "Fake"), (real_dir, "Real")]:
                for ext_pattern in supported_extensions:
                    for media_file_path in glob.glob(os.path.join(dir_path_actual, "**", ext_pattern), recursive=True):
                        if media_file_path not in found_paths:
                            media_items.append((media_file_path, label))
                            found_paths.add(media_file_path)
        else:
            console.print(f"[yellow]Warning: Real/Fake subdirectories not fully found. Searching in '{input_dir}' directly and inferring labels for {media_type}s.[/yellow]")
            found_in_root = []
            for ext_pattern in supported_extensions:
                found_in_root.extend(glob.glob(os.path.join(input_dir, ext_pattern)))

            if found_in_root:
                for media_file_path in found_in_root:
                    if media_file_path in found_paths: continue # Already processed if, somehow, structure was partially there
                    file_name_lower = os.path.basename(media_file_path).lower()
                    if "fake" in file_name_lower:
                        media_items.append((media_file_path, "Fake"))
                    elif "real" in file_name_lower:
                        media_items.append((media_file_path, "Real"))
                    else:
                        media_items.append((media_file_path, "Real")) # Default label
                        # console.print(f"[yellow]Cannot infer label for {os.path.basename(media_file_path)}. Assuming Real.[/yellow]")
                    found_paths.add(media_file_path)
            else:
                console.print(f"[bold red]Error: No Real/Fake subdirectories found, and no {media_type} files found directly in {input_dir} matching extensions: {supported_extensions}[/bold red]")


        fake_count = sum(1 for _, label in media_items if label == "Fake")
        real_count = sum(1 for _, label in media_items if label == "Real")
        console.print(f"Found {len(media_items)} {media_type}(s): {fake_count} Fake, {real_count} Real.")

        if not media_items:
            console.print(f"[bold red]No {media_type} files found in '{input_dir}' based on configuration.[/bold red]")
        return media_items

    def encode_media_to_base64(self, media_path: str) -> Optional[str]:
        try:
            with open(media_path, "rb") as f:
                media_bytes = f.read()
            return base64.b64encode(media_bytes).decode('utf-8')
        except Exception as e:
            console.print(f"[bold red]Error encoding media '{media_path}': {e}[/bold red]")
            return None

    def validate_media_file(self, media_path: str, media_type: str) -> bool:
        """Basic validation for a media file (e.g., existence, basic format check for images)."""
        if not os.path.exists(media_path):
            console.print(f"[red]Media file not found: {media_path}[/red]")
            return False
        
        # Add more specific checks per media_type if needed
        if media_type == "image":
            try:
                img = Image.open(media_path)
                img.verify() # Verifies basic integrity for some formats
                return True
            except Exception as e:
                console.print(f"[red]Invalid or corrupt image file {media_path}: {e}[/red]")
                return False
        # For video/audio, basic existence check might be enough for now.
        # More complex validation (e.g., ffprobe) could be added later.
        return True