import torch
import torch.nn as nn
import torch.quantization

class SimpleQATModel(nn.Module):
    "\""
    A simple model demonstrating Quantization Aware Training (QAT) setup in PyTorch.
    "\""
    def __init__(self):
        super(SimpleQATModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = nn.Conv2d(1, 16, 3, 1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 26 * 26, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

def prepare_qat():
    model = SimpleQATModel()
    model.train()
    
    # Specify quantization configuration
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # Prepare model for QAT
    torch.quantization.prepare_qat(model, inplace=True)
    print("Model prepared for Quantization Aware Training.")
    
    # Convert to quantized model
    model.eval()
    quantized_model = torch.quantization.convert(model, inplace=False)
    print("Model successfully converted to INT8 quantized representation.")
    return quantized_model

if __name__ == "__main__":
    prepare_qat()