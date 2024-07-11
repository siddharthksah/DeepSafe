import unittest
import importlib
import os, sys
import time
import shutil
from unittest.mock import patch, MagicMock

# Import the functions you want to test
from main import clean, delete_module

class TestDeepFakeDetector(unittest.TestCase):

    def setUp(self):
        # Set up any necessary test data or configurations
        self.test_image_path = "./assets/Sanity_Check/delete.jpg"
        self.test_video_path = "./assets/Sanity_Check/delete.mp4"
        self.temp_dir = "./temp/"
        
        # Ensure the temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)

    def tearDown(self):
        # Clean up any test data or configurations
        clean()

    def test_model_lists(self):
        # Test that model lists are correctly populated
        models_list_image = []
        models_list_video = []
        for model in os.listdir("./models"):
            if model.endswith("_image"):
                models_list_image.append(model[:-6])
            elif model.endswith("_video"):
                models_list_video.append(model[:-6])
        
        self.assertTrue(len(models_list_image) > 0)
        self.assertTrue(len(models_list_video) > 0)

    @patch('importlib.import_module')
    def test_model_inference(self, mock_import):
        # Test model inference for both image and video
        mock_module = MagicMock()
        mock_module.demo.return_value = None
        mock_import.return_value = mock_module

        for model_type in ['image', 'video']:
            with self.subTest(model_type=model_type):
                model_name = f"test_{model_type}_model"
                module_name = f"models.{model_name}_{model_type}.demo"
                
                # Create a mock result file
                os.makedirs(f"models/{model_name}_{model_type}", exist_ok=True)
                with open(f"models/{model_name}_{model_type}/result.txt", 'w') as f:
                    f.write("0.75")

                start_time = time.time()
                importlib.import_module(module_name)
                inference_time = time.time() - start_time

                self.assertLess(inference_time, 1)  # Assert that inference takes less than 1 second

                # Check if result file was read correctly
                with open(f"models/{model_name}_{model_type}/result.txt", 'r') as f:
                    result = float(f.read())
                self.assertEqual(result, 0.75)

                # Clean up
                shutil.rmtree(f"models/{model_name}_{model_type}")

    def test_file_copy(self):
        # Test that files are correctly copied to temp directory
        for file_path in [self.test_image_path, self.test_video_path]:
            with self.subTest(file=file_path):
                filename = os.path.basename(file_path)
                shutil.copy2(file_path, self.temp_dir)
                self.assertTrue(os.path.exists(os.path.join(self.temp_dir, filename)))

    def test_clean_function(self):
        # Test the clean function
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        self.assertTrue(os.path.exists(test_file))
        clean()
        self.assertFalse(os.path.exists(test_file))

    @patch('sys.modules', new_callable=dict)
    def test_delete_module(self, mock_sys_modules):
        # Test the delete_module function
        module_name = "test_module"
        mock_module = MagicMock()
        mock_sys_modules[module_name] = mock_module

        self.assertIn(module_name, sys.modules)
        delete_module(module_name)
        self.assertNotIn(module_name, sys.modules)

        # Test deleting a non-existent module
        with self.assertRaises(ValueError):
            delete_module("non_existent_module")

if __name__ == '__main__':
    unittest.main()