import torch
from torchvision import transforms

from .model import vit_model


def image_inference(image):
    inference_model, classes, image_size = vit_model()
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    transformed_image = image_transform(image)
    inference_model.eval()
    with torch.no_grad():
        output = inference_model(transformed_image.unsqueeze(0))
        class_index = torch.argmax(output, dim=-1)
        predicted_class = classes[class_index]
        return predicted_class