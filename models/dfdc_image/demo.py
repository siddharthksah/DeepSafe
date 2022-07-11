import warnings
warnings.filterwarnings("ignore")

import subprocess
def demo_dfdc():
    subprocess.run(["python3", "models/dfdc_image/dfdc_prediction.py"])
(demo_dfdc())