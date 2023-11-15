import os
import torch
from PIL import Image
from torchvision import transforms
from networks.resnet import resnet50

def predictor_CNN():
    model_path = "./weights/blur_jpg_prob0.5.pth"
    crop_size = None
    use_cpu = True

    # Load and prepare the model
    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu' if use_cpu else 'cuda')
    model.load_state_dict(state_dict['model'])
    model.eval()
    if not use_cpu:
        model.cuda()

    # Image transformation
    transformations = [transforms.CenterCrop(crop_size)] if crop_size else []
    transformations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_pipeline = transforms.Compose(transformations)

    # Process each image in the directory
    results = {}
    for file in os.listdir("./tempDir/"):
        image = Image.open(os.path.join("./tempDir", file)).convert('RGB')
        image_tensor = transform_pipeline(image).unsqueeze(0)
        if not use_cpu:
            image_tensor = image_tensor.cuda()

        with torch.no_grad():
            probability = model(image_tensor).sigmoid().item()

        # Store the result
        results[file] = format(1 - probability, ".2f")

    # Optionally: Delete the temporary directory
    # shutil.rmtree('tempDir', ignore_errors=True)

    return results

# Example usage
# results = predictor_CNN()
# print(results)
