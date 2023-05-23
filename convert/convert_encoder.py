# import torch
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model_copy import WaveEncoder, WaveDecoder  # Assuming your WaveEncoder class is saved in a file called wave_encoder.py

import torch
import torch.onnx

# Load your pretrained model
wave_encoder = WaveEncoder()
wave_encoder.load_state_dict(torch.load("/home/jovyan/Color_Transfer/model_checkpoints/wave_encoder_cat5_l4.pth"))

wave_decoder = WaveDecoder()
wave_decoder.load_state_dict(torch.load("/home/jovyan/Color_Transfer/model_checkpoints/wave_decoder_cat5_l4.pth"))

# Create dummy variable to perform the forward pass
dummy_input_encoder = torch.randn(1, 3, 224, 224)  # Adjust the size as per your model's input size
dummy_input_decoder = torch.randn(1, 512, 14, 14)  # Adjust the size as per your model's input size
dummy_skips = {
    'pool1': [torch.randn(1, 64, 56, 56), torch.randn(1, 64, 56, 56), torch.randn(1, 64, 56, 56)],
    'pool2': [torch.randn(1, 128, 28, 28), torch.randn(1, 128, 28, 28), torch.randn(1, 128, 28, 28)],
    'pool3': [torch.randn(1, 256, 14, 14), torch.randn(1, 256, 14, 14), torch.randn(1, 256, 14, 14)],
    'conv1_2': torch.nn.functional.interpolate(torch.randn(1, 64, 56, 56), size=(112, 112)), # Resized from [56, 56] to [112, 112]
    'conv2_2': torch.nn.functional.interpolate(torch.randn(1, 128, 28, 28), size=(56, 56)),  # Resized from [28, 28] to [56, 56]
    'conv3_4': torch.nn.functional.interpolate(torch.randn(1, 256, 14, 14), size=(28, 28))  # Resized from [14, 14] to [28, 28]
}


dummy_input = {
    'x': dummy_input_decoder,
    'skips': dummy_skips
}

# Export the models to an ONNX file
torch.onnx.export(wave_encoder, dummy_input_encoder, "wave_encoder.onnx")
torch.onnx.export(wave_decoder, dummy_input, "wave_decoder.onnx")
