from pathlib import Path
from PIL import Image
import torch
import torchvision
from torch import nn
from io import BytesIO
def create_effnetb0_model(num_classes: int = 2,
                          seed: int = 42):
    weights = torchvision.models.EfficientNet_B6_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b6(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(seed)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=2304, out_features=num_classes),
    )

    return model, transforms



def pred_and_store(image_bytes: bytes,
                   model: torch.nn.Module,
                   transform: torchvision.transforms,
                   ) -> str:
    device = "cpu"
    class_names = ['Brown_rust', 'Healthy']
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    transformed_image = transform(img).unsqueeze(0).to(device)
    model.to(device)
    model.eval()

    with torch.inference_mode():
        pred_logit = model(transformed_image)
        pred_prob = torch.softmax(pred_logit, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1)
        pred_class = class_names[pred_label.cpu().item()]

    return pred_class

def inference(image_bytes: bytes):
    model, model_transforms = create_effnetb0_model(num_classes=2)
    model.load_state_dict(torch.load("models/kt6.pth", weights_only=True, map_location=torch.device("cpu")))

    predicted_class = pred_and_store(image_bytes, model=model, transform=model_transforms)
    print(predicted_class)
    return predicted_class