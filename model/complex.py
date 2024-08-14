import struct
from struct import unpack
import os
import time
import math
import random
import svgpathtools
from svgpathtools import svg2paths2
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
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.onnx

from ctypes.macholib import dyld # to fix cairo bug & not have to run: 
                                 # export DYLD_LIBRARY_PATH="/opt/homebrew/opt/cairo/lib:$DYLD_LIBRARY_PATH"
                                 # Should be macos only tho. thread: https://github.com/Kozea/CairoSVG/issues/354
dyld.DEFAULT_LIBRARY_FALLBACK.append("/opt/homebrew/lib")

import cairocffi as cairo

def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    country_code, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'country_code': country_code,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    path = os.getcwd()
    path = os.path.join(path,"model","trainingdata",filename)
    with open(path, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break


def vector_to_raster(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    padding and line_diameter are relative to the original 256x256 image.
    """
    
    original_side = 256.
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        
        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)        
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)
    
    return raster_images


def svg_to_strokes(svg_string):
    file_like = StringIO(svg_string)
    paths, attributes, svg_attributes = svg2paths2(file_like)
    strokes = []

    for path in paths:
        for segment in path:
            if isinstance(segment, svgpathtools.path.Line):
                strokes.append([(segment.start.real, segment.start.imag), (segment.end.real, segment.end.imag)])
            elif isinstance(segment, svgpathtools.path.CubicBezier):
                strokes.append([(segment.start.real, segment.start.imag),
                                (segment.control1.real, segment.control1.imag),
                                (segment.control2.real, segment.control2.imag),
                                (segment.end.real, segment.end.imag)])
            elif isinstance(segment, svgpathtools.path.QuadraticBezier):
                strokes.append([(segment.start.real, segment.start.imag),
                                (segment.control.real, segment.control.imag),
                                (segment.end.real, segment.end.imag)])
            elif isinstance(segment, svgpathtools.path.Arc):
                strokes.append([(segment.start.real, segment.start.imag),
                                (segment.radius.real, segment.radius.imag),
                                (segment.rotation),
                                (segment.arc),
                                (segment.sweep),
                                (segment.end.real, segment.end.imag)])
    return strokes

def svgStokesReformat(input_list):
    result = []

    for stroke in input_list:
        x_coords = [math.floor(coord[0]) for coord in stroke]
        y_coords = [math.floor(coord[1]) for coord in stroke]
        result.append((tuple(x_coords), tuple(y_coords)))
    
    return result

# Ramer-Douglas-Peucker algorithm
def rdp(points, epsilon):
    if len(points) < 3:
        return points

    start, end = points[0], points[-1]
    index = -1
    max_dist = 0

    for i in range(1, len(points) - 1):
        dist = np.abs(np.cross(end-start, points[i]-start) / np.linalg.norm(end-start))
        if dist > max_dist:
            index = i
            max_dist = dist

    if max_dist > epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
        return results
    else:
        return [start, end]

def resample_stroke(stroke, spacing=1.0):
    resampled = [stroke[0]]
    accumulated_distance = 0.0
    for i in range(1, len(stroke)):
        point_a = np.array(stroke[i - 1])
        point_b = np.array(stroke[i])
        segment_distance = euclidean(point_a, point_b)
        while accumulated_distance + segment_distance >= spacing:
            t = (spacing - accumulated_distance) / segment_distance
            new_point = point_a + t * (point_b - point_a)
            resampled.append(tuple(new_point))
            point_a = new_point
            segment_distance = euclidean(point_a, point_b)
            accumulated_distance = 0.0
        accumulated_distance += segment_distance
    resampled.append(stroke[-1])
    return resampled

def normalize_strokes(strokes):
    all_points = [point for stroke in strokes for point in zip(stroke[0], stroke[1])]
    min_x = min(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_x = max(point[0] for point in all_points)
    max_y = max(point[1] for point in all_points)

    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)
    scale = 255.0 / max_dim

    normalized_strokes = []
    for stroke in strokes:
        x_coords = [math.floor((x - min_x) * scale) for x in stroke[0]]
        y_coords = [math.floor((y - min_y) * scale) for y in stroke[1]]
        normalized_strokes.append((x_coords, y_coords))

    return normalized_strokes

def simplifyStrokes(input_strokes, epsilon=2.0):
    # Normalize strokes
    normalized_strokes = normalize_strokes(input_strokes)

    # Resample strokes
    resampled_strokes = [resample_stroke(list(zip(stroke[0], stroke[1]))) for stroke in normalized_strokes]

    # Simplify strokes
    simplified_strokes = []
    for stroke in resampled_strokes:
        stroke_array = np.array(stroke)
        simplified = rdp(stroke_array, epsilon)
        x_coords, y_coords = zip(*simplified)
        simplified_strokes.append((x_coords, y_coords))

    return simplified_strokes


svg_string = '''<svg width="813.6" height="731.7" xmlns="http://www.w3.org/2000/svg">
<path d="M 225.90962166314577 147.06121272302124 L 224.90912233531785 147.06121272302124 L 224.90912233531785 151.06287837534833 L 224.90912233531785 168.06995739773856 L 224.90912233531785 202.08411544251896 L 224.90912233531785 251.10451968352604 L 224.90912233531785 299.1245075114513" fill="none" stroke="black" stroke-width="2" />
<path d="M 473.03295563664295 161.0670425061661 L 473.03295563664295 163.06787533232966 L 473.03295563664295 182.07578718088342 L 473.03295563664295 231.0961914218905 L 475.0339542922988 307.12783881610557 L 478.0354522757826 382.1590697972388" fill="none" stroke="black" stroke-width="2" />
<path d="M 251.92260418667178 463.19279925686277 L 251.92260418667178 463.19279925686277 L 250.92210485884385 464.19321566994455 L 250.92210485884385 470.19571414843523 L 250.92210485884385 490.20404241007077 L 251.92260418667178 523.2177840417694 L 262.92809679277894 559.2327749127134 L 285.9395813328212 593.2469329574939 L 320.95705780679856 617.2569268714565 L 359.97653159208755 627.2610910022743 L 396.99500672172076 627.2610910022743 L 434.01348185135396 615.2560940452929 L 462.0274630305358 581.2419360005125 L 474.0334549644709 536.2231974118325 L 477.03495294795465 489.203625996989 L 470.0314576531592 437.1819725167366" fill="none" stroke="black" stroke-width="2" />
</svg>
'''

# strokes = svg_to_strokes(svg_string)

# print(simplifyStrokes(svgStokesReformat(strokes)))
# inp = simplifyStrokes(svgStokesReformat(strokes))
# raster = vector_to_raster([inp])[0]
# grid = raster.reshape((28,28))
# plt.imshow(grid,cmap="gray")
# plt.show()


################################
# HOW TO USE:

# FOR INPUT SVG:
# svgString = ```<svg ...>...</svg>```
# rawStrokes = svg_to_strokes(svg_string)
# reformattedStrokes = svgStokesReformat(rawStrokes)
# simplifiedVector = simplifyStrokes(reformattedStrokes)
# raster = vector_to_raster([simplifiedVector])[0]

# FOR DATASET:
# for drawing in unpack_drawings('full_binary_pencil.bin'):
# simplifiedVector = drawing["image"]
# raster = vector_to_raster([simplifiedVector])[0]

# VISUALIZATION:
# grid = raster.reshape((28,28))
# plt.imshow(grid,cmap="gray")
# plt.show()
################################



label_dict = {
    0: "basketball",
    1: "hammer",
    2: "paperclip",
    3: "pencil",
    4: "broom",
    5: "camera",
    6: "dog",
    7: "dresser",
    8: "hat",
    9: "hexagon",
}

values_dict = {
    "basketball": [],
    "hammer": [],
    "paperclip": [],
    "pencil": [],
    "broom": [],
    "camera": [],
    "dog": [],
    "dresser": [],
    "hat": [],
    "hexagon": [],
}

for item in values_dict.keys():
    i = 0
    for drawing in unpack_drawings('full_binary_'+str(item)+'.bin'):
        simplifiedVector = drawing["image"]
        raster = vector_to_raster([simplifiedVector])[0]
        values_dict[item].append(raster)
        i += 1
        if i > 1999:#9999:
            break


# randdrawing = values_dict["hammer"][2201].reshape((28,28))
# plt.imshow(randdrawing)
# plt.show()

X = []
y = []

for key, value in label_dict.items():
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
    fig, axs = plt.subplots(5, 10, figsize=(20,10))
    
    for label_num in range(0,50):
        r_label = random.randint(0, len(X) - 1)
        image = X[r_label].reshape((28,28))  #reshape images
        i = label_num // 10
        j = label_num % 10
        axs[i,j].imshow(image) #plot the data
        axs[i,j].axis('off')
        axs[i,j].set_title(label_dict[y[r_label]])

    plt.show()

view_images_grid(X,y)

def build_model(input_size, output_size, hidden_sizes, dropout = 0.0):
    '''
    Function creates deep learning model based on parameters passed.

    INPUT:
        input_size, output_size, hidden_sizes - layer sizes
        dropout - dropout (probability of keeping a node)

    OUTPUT:
        model - deep learning model
    '''

    # Build a feed-forward network
    model = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                        #   ('bn2', nn.BatchNorm1d(num_features=hidden_sizes[1])),
                          ('relu2', nn.ReLU()),
                        #   ('dropout', nn.Dropout(dropout)),
                          ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                        #   ('bn3', nn.BatchNorm1d(num_features=hidden_sizes[2])),
                          ('relu3', nn.ReLU()),
                          ('logits', nn.Linear(hidden_sizes[2], output_size))]))

    return model

# def shuffle(X_train, y_train):
#     """
#     Function which shuffles training dataset.
#     INPUT:
#         X_train - (tensor) training set
#         y_train - (tensor) labels for training set

#     OUTPUT:
#         X_train_shuffled - (tensor) shuffled training set
#         y_train_shuffled - (tensor) shuffled labels for training set
#     """
#     X_train_shuffled = X_train.numpy()
#     y_train_shuffled = y_train.numpy().reshape((X_train.shape[0], 1))

#     permutation = list(np.random.permutation(X_train.shape[0]))
#     X_train_shuffled = X_train_shuffled[permutation, :]
#     y_train_shuffled = y_train_shuffled[permutation, :].reshape((X_train.shape[0], 1))

#     X_train_shuffled = torch.from_numpy(X_train_shuffled).float()
#     y_train_shuffled = torch.from_numpy(y_train_shuffled).long()

#     return X_train_shuffled, y_train_shuffled

def fit_model(model, train_loader, epochs=100, learning_rate=0.003, weight_decay=0, optimizer='SGD'):
    print(f"Fitting model with epochs = {epochs}, learning rate = {learning_rate}")

    criterion = nn.CrossEntropyLoss()

    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        if (e+1) % 10 == 0:
            print(f"Epoch: {e+1}/{epochs}... Loss: {running_loss/len(train_loader):.4f}")
            running_loss = 0
                
def view_classify(img, ps):
    """
    Function for viewing an image and it's predicted classes
    with matplotlib.

    INPUT:
        img - (tensor) image file
        ps - (tensor) predicted probabilities for each class
    """
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(["basketball","hammer","paperclip","pencil","broom","camera","dog","dresser","hat","hexagon"], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()
    
def test_model(model, img, label):
    """
    Function creates test view of the model's prediction for image.

    INPUT:
        model - pytorch model
        img - (tensor) image from the dataset

    OUTPUT: None
    """

    # Convert 2D image to 1D vector
    # img = img.resize_(1, 784)

    # ps = get_preds(model, img)
    # view_classify(img.resize_(1, 28, 28), ps)

    img = img.view(1, 784)  # Ensure the image is a 1D vector of size 784
    
    # Forward pass through the model to get predictions
    ps = get_preds(model, img)
    
    # View the image and its classification
    view_classify(img.view(1, 28, 28), ps)
    
    # Print the predicted and actual labels
    _, predicted = torch.max(ps, 1)
    print(f'Predicted: {predicted.item()}, True: {label.item()}')


def get_preds(model, input):
    """
    Function to get predicted probabilities from the model for each class.

    INPUT:
        model - pytorch model
        input - (tensor) input vector

    OUTPUT:
        ps - (tensor) vector of predictions
    """

    # Turn off gradients to speed up this part
    with torch.no_grad():
        logits = model.forward(input)
    ps = F.softmax(logits, dim=1)
    return ps

def get_labels(pred):
    """
        Function to get the vector of predicted labels for the images in
        the dataset.

        INPUT:
            pred - (tensor) vector of predictions (probabilities for each class)
        OUTPUT:
            pred_labels - (numpy) array of predicted classes for each vector
    """

    pred_np = pred.numpy()
    pred_values = np.amax(pred_np, axis=1, keepdims=True)
    pred_labels = np.array([np.where(pred_np[i, :] == pred_values[i, :])[0] for i in range(pred_np.shape[0])])
    pred_labels = pred_labels.reshape(len(pred_np), 1)

    return pred_labels

def evaluate_model(model, train_loader, test_loader):
    """
    Function to print out train and test accuracy of the model.

    INPUT:
        model - pytorch model
        train - (tensor) train dataset
        y_train - (numpy) labels for train dataset
        test - (tensor) test dataset
        y_test - (numpy) labels for test dataset

    OUTPUT:
        accuracy_train - accuracy on train dataset
        accuracy_test - accuracy on test dataset
    """
    model.eval()
    
    def get_accuracy(loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return correct / total

    train_accuracy = get_accuracy(train_loader)
    test_accuracy = get_accuracy(test_loader)

    print(f"Accuracy score for train set is {train_accuracy:.4f}")
    print(f"Accuracy score for test set is {test_accuracy:.4f}")

    return train_accuracy, test_accuracy

# def evaluate_model(model, train, y_train, test, y_test):
#     model.eval()
#     train_pred = get_preds(model, train)
#     train_pred_labels = get_labels(train_pred)

#     for i in range(61, 9771, 299):
#         # _, p = torch.max(train_pred[i], 1)
#         print(f"Actual: {label_dict[y_train[i]]}, Pred: {label_dict[train_pred_labels[i].item()]}")
#         plt.imshow(train.numpy()[i].reshape((28,28)))
#         plt.show()

#     test_pred = get_preds(model, test)
#     test_pred_labels = get_labels(test_pred)

#     accuracy_train = accuracy_score(y_train, train_pred_labels)
#     accuracy_test = accuracy_score(y_test, test_pred_labels)

#     print("Y_TEST ",y_test)
#     print("PRED ",test_pred_labels)

#     print("Accuracy score for train set is {} \n".format(accuracy_train))
#     print("Accuracy score for test set is {} \n".format(accuracy_test))
#     model.train()

#     return accuracy_train, accuracy_test

def plot_learning_curve(input_size, output_size, hidden_sizes, train_loader, test_loader, learning_rate=0.003, weight_decay=0.0, dropout=0.0, n_chunks=1000, optimizer='SGD'):
    """
    Function to plot learning curve depending on the number of epochs.

    INPUT:
        input_size, output_size, hidden_sizes - model parameters
        train - (tensor) train dataset
        labels - (tensor) labels for train dataset
        y_train - (numpy) labels for train dataset
        test - (tensor) test dataset
        y_test - (numpy) labels for test dataset
        learning_rate - learning rate hyperparameter
        weight_decay - weight decay (regularization)
        dropout - dropout for hidden layer
        n_chunks - the number of minibatches to train the model
        optimizer - optimizer to be used for training (SGD or Adam)

    OUTPUT: None
    """
    train_acc = []
    test_acc = []

    for epochs in np.arange(100,101,1):
        model = build_model(input_size, output_size, hidden_sizes, dropout=dropout)

        fit_model(model, train_loader, epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, optimizer=optimizer)

        accuracy_train, accuracy_test = evaluate_model(model, train_loader, test_loader)
        model.train()

        train_acc.append(accuracy_train)
        test_acc.append(accuracy_test)
    
    return train_acc, test_acc

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1,shuffle=True)

train = torch.from_numpy(X_train).float()
labels = torch.from_numpy(y_train).long()
test = torch.from_numpy(X_test).float()
test_labels = torch.from_numpy(y_test).long()

# Data norming
# mean = train.mean()
# std = train.std()
# train = (train - mean) / std
# test = (test - mean) / std

# Set hyperparameters for our network
input_size = 784
hidden_sizes = [500, 250, 50]
output_size = 10

dropout = 0.2
weight_decay = 0.0
n_chunks = 100#700
learning_rate = 0.006
optimizer = 'SGD'

from torch.utils.data import TensorDataset, DataLoader

batch_size = len(train) // n_chunks  # Use the same batch size as in training

train_dataset = TensorDataset(train, labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(test, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = build_model(input_size, output_size, hidden_sizes, dropout = dropout)

train_acc, test_acc = plot_learning_curve(input_size, output_size, hidden_sizes, train_loader, test_loader, learning_rate=learning_rate, dropout=dropout, weight_decay=weight_decay, n_chunks=n_chunks, optimizer=optimizer)

x = np.arange(10, 10 * (len(train_acc) + 1), 10)
plt.plot(x, train_acc)
plt.plot(x, test_acc)
plt.legend(['train', 'test'], loc='upper left')
plt.title('Accuracy, learning_rate = ' + str(learning_rate), fontsize=14)
plt.xlabel('Number of epochs', fontsize=11)
plt.ylabel('Accuracy', fontsize=11)
plt.show()

accuracy_train, accuracy_test = evaluate_model(model, train_loader,test_loader)
print(f"trainacc: {accuracy_train}, testacc: {accuracy_test}")
torch.save(model, "model2_3_normie.pt")
# model = torch.load("model2_1_big.pt")

# for i in range(10):
#     print(f'Label {i}: {label_dict[y_train[i]]}')
model.eval()

with torch.no_grad():
    for i in range(61, 9771, 299):
        img = train[i].view(1, 784)
        print(f"L: {img.size()} Actual: {label_dict[y_train[i]]}, Pred: {label_dict[get_labels(get_preds(model,train))[i].item()]}")
        test_model(model,img, y_train[i])

# with torch.no_grad():
#     for i in range(61,9771,50):
#         test_model(model,test[i])

# torch.onnx.export(model,torch.from_numpy(X[0]).float().unsqueeze(0),"model.onnx",export_params=True,do_constant_folding=True,input_names = ['input'],output_names = ['output'])


# for drawing in unpack_drawings('full_binary_pencil.bin'):
#     raw = drawing["image"]
#     print(raw)
#     raster = vector_to_raster([raw])[0]
#     print(raster)
# #     grid = raster.reshape((28,28))
# #     plt.imshow(grid,cmap="gray")
# #     plt.show()
#     break
