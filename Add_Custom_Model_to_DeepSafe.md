
# Adding Your Custom Model to DeepSafe

To add your custom model to the DeepSafe platform for deepfake detection, follow these steps:

## Step 1: Clone the DeepSafe Repository

First, clone the DeepSafe repository from GitHub:

```bash
git clone https://github.com/siddharthksah/DeepSafe
```

## Step 2: Add Your Model Folder

Navigate to the `models` folder located in the root directory of the DeepSafe repository. Create a new folder for your model. The folder name should be in lowercase and end with `_image` for image models or `_video` for video models. For example, if your model detects deepfake images, you might name the folder `mycustommodel_image`.

```bash
cd DeepSafe/models
mkdir mycustommodel_image
```

## Step 3: Add Configuration File (Optional)

Inside your model folder, create a `deepsafe.config` file. This step is optional, but it allows you to specify whether your model supports GPU or should run on CPU. 

```bash
touch mycustommodel_image/deepsafe.config
```

Add the following content to `deepsafe.config` based on your model's requirements:

For CPU:
```text
device: cpu
```

For GPU:
```text
device: gpu
```

## Step 4: Create Python Files

### Create `predict.py`

Inside your model folder, create a file named `predict.py`. This script should take the `./temp` folder as input and save the output probability (1 for fake, 0 for real) in a file named `result.txt` within the model folder.

```bash
touch mycustommodel_image/predict.py
```

Here's a template for `predict.py`:

```python
import os
import sys

def predict():
    input_path = "./temp"
    output_path = "./mycustommodel_image/result.txt"
    
    # Add your model loading and prediction code here
    # For example:
    # model = load_model('path_to_your_model')
    # prediction = model.predict(input_path)
    
    # Assuming prediction is a float where 1 means fake and 0 means real
    prediction = 0.5  # Replace this with your actual prediction logic
    
    with open(output_path, 'w') as f:
        f.write(str(prediction))

if __name__ == "__main__":
    predict()
```

### Create `demo.py`

Next, create a file named `demo.py` inside your model folder. This script should use the `subprocess` module to run `predict.py`.

```bash
touch mycustommodel_image/demo.py
```

Here's a template for `demo.py`:

```python
import subprocess

def run_predict():
    subprocess.run(["python", "mycustommodel_image/predict.py"])

if __name__ == "__main__":
    run_predict()
```

## Step 5: Add Reference File

Create a `reference.txt` file inside your model folder. This file should contain information about your model, such as the name, GitHub URL, and license.

```bash
touch mycustommodel_image/reference.txt
```

Here's an example content for `reference.txt`:

```text
Name: My Custom Model
GitHub URL: https://github.com/yourusername/yourmodelrepo
License: MIT
```

## Step 6: Verify and Test

Ensure all files are correctly placed and configured. If done right, the DeepSafe web app will automatically load your model, and you should be able to use it in the app and run it.

## Folder Structure Example

Your model folder structure should look like this:

```
DeepSafe/
└── models/
    └── mycustommodel_image/
        ├── deepsafe.config (optional)
        ├── predict.py
        ├── demo.py
        └── reference.txt
```

## Final Step: Push to Repository

Once you've added your model, you can commit your changes and push them to your fork or the main repository if you have the necessary permissions.

```bash
git add .
git commit -m "Added my custom model"
git push origin main
```

You have now successfully added your custom model to the DeepSafe project.
