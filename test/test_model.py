from src.model import vit_model
import torch
import toml

config = toml.load("configuration.toml")

def test_vit_model():
    inference_model, classes, image_size = vit_model()
    image = torch.randn(1, 3, image_size, image_size)
    inference_model.eval()
    with torch.no_grad():
        output = inference_model(image)
    assert output.shape == (1, 1000), f"Unexpected output shape: {output.shape}"