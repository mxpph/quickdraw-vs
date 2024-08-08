import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
print(f"Preds: {pred_probab}")
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")

# input_image = torch.rand(3,28,28)
# flat_image = nn.Flatten()(input_image)
# layer1 = nn.Linear(in_features=28*28, out_features=20)
# hidden1 = layer1(flat_image)
# hidden1 = nn.ReLU()(hidden1)

# layer1 = nn.Linear(in_features=28*28, out_features=20)

# seq_modules = nn.Sequential(
#     nn.Flatten(),
#     layer1,
#     nn.ReLU(),
#     nn.Linear(20, 10)
# )
# input_image = torch.rand(3,28,28)
# logits = seq_modules(input_image)
# softmax = nn.Softmax(dim=1)
# pred_probab = softmax(logits)
# print(pred_probab)