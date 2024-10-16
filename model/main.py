'''This file trains the model'''

import struct
from struct import unpack
import os
from os import listdir
from os.path import isfile, join
import time
import math
import random
import numpy as np
from io import StringIO
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from collections import OrderedDict

# from PIL import Image, ImageDraw, ImageOps

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.onnx
from torch.utils.data import DataLoader, TensorDataset
import onnx


import drawing
from drawing import *

# Reduce mem usage on remote training runs
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.cuda.empty_cache()

labels = []
values_dict = {}

datapath = os.path.join(os.getcwd(), "model", "trainingdata")
trianing_data_names = [f for f in listdir(datapath) if isfile(join(datapath, f))]
print(f"Item Order {trianing_data_names}")
items = len(trianing_data_names) # Number of items
print(f"Item Count: {items}")

labels = trianing_data_names

for i,v in enumerate(trianing_data_names):
    # shortened_name = v.replace("full_binary_","").replace(".bin","")
    labels[i] = v# shortened_name
    values_dict[v] = []

for item in values_dict.keys():
    i = 0
    for drawing in unpack_drawings(item):
        simplifiedVector = drawing["image"]
        raster = vector_to_raster([simplifiedVector])[0]
        values_dict[item].append(raster)
        i += 1
        if i > 1999:
            break

X = []
y = []

for key, value in enumerate(labels):
    print(value+"\n")
    data_i = values_dict[value]
    Xi = np.concatenate([data_i], axis = 0)
    yi = np.full((len(Xi), 1), key).ravel()

    X.append(Xi)
    y.append(yi)

# print(f"X: {(X)}\n")
# print(f"Y: {(y)}\n")

X = np.concatenate(X, axis = 0) # IMAGES
y = np.concatenate(y, axis = 0) # LABELS

print(f"X: {type(X[0])}, SHAPE:\n")
print(f"Y: {type(y[0])}\n")

def view_images_grid(X, y):
    _, axs = plt.subplots(5, 10, figsize=(20,10))

    for label_num in range(0,50):
        r_label = random.randint(0, len(X) - 1)
        image = X[r_label].reshape((28,28))  #reshape images
        i = label_num // 10
        j = label_num % 10
        axs[i,j].imshow(image) #plot the data
        axs[i,j].axis('off')
        axs[i,j].set_title(labels[y[r_label]])

    plt.show()

view_images_grid(X,y)

print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# class SimpleMLP(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size):
#         super(SimpleMLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_sizes[0])
#         self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#         self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
#         self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
#         self.fc5 = nn.Linear(hidden_sizes[3], output_size)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.relu(self.fc4(x))
#         x = self.fc5(x)
#         return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# reshape data for the cnn
def reshape_for_cnn(X, y):
    X = X.reshape(-1, 1, 28, 28)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    return X, y


# Define training and evaluation functions
def train_model(model, X_train, y_train, epochs=10, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # move data to same device as model
    device = next(model.parameters()).device
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

def train_cnn(model, X_train, y_train, X_val, y_val, epochs=10, learning_rate=0.001):
    model = model.to(device)
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_train, y_train = torch.from_numpy(X_train).float().to(device), torch.from_numpy(y_train).long().to(device)
    X_val = X_val.reshape(-1, 1, 28, 28)
    X_val, y_val = torch.from_numpy(X_val).float().to(device), torch.from_numpy(y_val).long().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            _, predicted = torch.max(val_outputs.data, 1)
            correct = (predicted == y_val).sum().item()
            total = y_val.size(0)

        print(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, '
              f'Val Loss: {val_loss.item():.4f}, '
              f'Val Accuracy: {100 * correct / total:.2f}%')

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Move data to the same device as the model
        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).long().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        y_test = torch.from_numpy(y_test).long().to(device)

        outputs_train = model(X_train)
        _, predicted_train = torch.max(outputs_train, 1)
        accuracy_train = (predicted_train == y_train).float().mean().item()
        print(f'Train Accuracy: {accuracy_train:.4f}')

        outputs_test = model(X_test)
        _, predicted_test = torch.max(outputs_test, 1)
        accuracy_test = (predicted_test == y_test).float().mean().item()
        print(f'Test Accuracy: {accuracy_test:.4f}')

    return accuracy_test

def evaluate_cnn(model, X_test, y_test):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # X_test, y_test = X_test.to(device), y_test.to(device)
    X_test, y_test = reshape_for_cnn(X_test,y_test)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_test).sum().item()
        total = y_test.size(0)

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def view_img(raster):
    grid = raster.reshape((28,28))
    plt.imshow(grid,cmap="gray")
    plt.show()

def get_pred(model, raster):
    device = next(model.parameters()).device # i dont know what this does but yeah
    raster_tensor = torch.tensor(raster, dtype=torch.float).unsqueeze(0).to(device) # use device for this
    model.eval()
    with torch.no_grad():
        outputs = model(raster_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_label = predicted.item()
    return labels[predicted_label]

def get_pred_cnn(model, raster):
    device = next(model.parameters()).device # i dont know what this does but yeah
    raster_tensor = torch.tensor(raster, dtype=torch.float).unsqueeze(0).to(device)
    raster_tensor = raster_tensor.reshape(1, 1, 28, 28)
    model.eval()
    with torch.no_grad():
        outputs = model(raster_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_label = predicted.item()
    return labels[predicted_label]


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

train_loader = reshape_for_cnn(X_train, y_train)
val_loader = reshape_for_cnn(X_val, y_val)
test_loader = reshape_for_cnn(X_test, y_test)

# Set up and train the model
input_size = 784
hidden_sizes = [600, 400, 160, 80]
output_size = items
# model = SimpleMLP(input_size, hidden_sizes, output_size).to(device) # use device
model = SimpleCNN(num_classes=len(labels)) # 16 cats

# train_model(model, X_train, y_train, epochs=200, learning_rate=0.025)
train_cnn(model, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.001)

# use sample input instead of 'torch.tensor(X_train[0], dtype=torch.float).unsqueeze(0)' to keep device consistent
# sampleinput = torch.randn(1, input_size, dtype=torch.float).to(device)
sampleinput = torch.randn(1, 1, 28, 28).to(device)

modelname = "CNN_cat16_v6-0_large_gputrain"
torch.save(model, modelname+".pt")
torch.onnx.export(model, sampleinput,
                  modelname+".onnx", export_params=True, do_constant_folding=True,
                  input_names = ['input'], output_names = ['output'])

# model = torch.load("model3_1_large.pt")


# Evaluate the model
# evaluate_model(model, X_train, y_train, X_test, y_test)
evaluate_cnn(model,X_test,y_test)

model.eval()

with torch.no_grad():
    # evaluate_model(model, X_train, y_train, X_test, y_test)
    evaluate_cnn(model,X_test,y_test)

    rawStrokes = svg_to_strokes(svg_string1)
    reformattedStrokes = svg_strokes_reformat(rawStrokes)
    simplifiedVector = simplify_strokes(reformattedStrokes)
    raster = vector_to_raster([simplifiedVector])[0]
    print(f"PREDICTED DRAWING: {get_pred_cnn(model,raster)}")
    view_img(raster)

    for i in range(61, 9771, 299):
        print(f"Actual: {labels[y_train[i]]}, Pred: {get_pred_cnn(model,X_train[i])}")
        view_img(X_train[i])
        # test_model(model,img, y_train[i])
