import struct
from struct import unpack
import os
import time
import math
import random
import svgpathtools
from svgpathtools import svg2paths2
from os import listdir
from os.path import isfile, join
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
import onnx

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

svg_string1 = '''
<svg width="813.6" height="731.7" xmlns="http://www.w3.org/2000/svg"><path d="M 423.0079892452468,260.6084756078029 423.0079892452468,259.60805919472114 422.00748991741887,259.60805919472114 417.00499327827924,259.60805919472114 402.9980026886883,259.60805919472114 384.98901478778566,259.60805919472114 365.9795275590551,261.6088920208847 345.9695410024966,268.61180691245715 325.9595544459382,279.6163874563567 308.95106587286347,293.6222172395016 293.9435759554446,311.6297126749736 279.93658536585366,334.63929017585446 269.9315920875744,358.64928408981706 262.92809679277894,381.65886159069794 258.92609948146725,407.66968833082416 257.9256001536393,431.6796822447868 257.9256001536393,454.6892597456677 262.92809679277894,477.6988372465486 276.9350873823699,500.70841474742946 297.9455732667563,523.7179922483103 324.95905511811026,542.725904096864 356.9750336086038,557.7321502930907 392.99300941040906,567.7363144239085 431.0119838678702,572.7383964893174 469.03095832533126,572.7383964893174 507.0499327827924,564.7350651846632 542.0674092567697,547.7279861622729 574.0833877472633,526.7192414875557 601.0968695986172,500.70841474742946 622.1073554830036,469.69550594189434 638.1153447282504,436.6817643101957 646.1193393508738,405.6688555046606 650.1213366621855,377.6571959383709 651.1218359900134,350.6459527851629 646.1193393508738,327.636375284282 632.1123487612829,303.62638137031934 611.1018628768965,280.61680386943846 585.0888803533704,257.6072263685576 557.0748991741885,237.59889810692206 528.0604186671787,222.5926519106954 500.0464374879969,212.58848777987762 478.0354522757826,208.58682212755053 458.0254657192241,209.5872385406323" stroke="black" stroke-width="7" fill="none" /><path d="M 476.0344536201267,648.7700438835325 475.0339542922988,649.7704602966143 474.0334549644709,649.7704602966143 474.0334549644709,647.7696274704507 474.0334549644709,643.7679618181236 474.0334549644709,638.7658797527147 474.0334549644709,633.7637976873058 474.0334549644709,631.7629648611422 474.0334549644709,627.7612992088151 474.0334549644709,617.7571350779973 474.0334549644709,600.7500560556072 474.0334549644709,579.7413113808898 474.0334549644709,556.731733880009 475.0339542922988,530.7209071398828 478.0354522757826,504.7100803997565 479.0359516036105,480.7000864857939 480.0364509314384,460.6917582241583 480.0364509314384,443.6846792017681 478.0354522757826,427.6780165924597 473.03295563664295,414.6726032223966 467.0299596696754,404.6684390915788 460.02646437487994,396.66510778692464 453.0229690800845,388.6617764822704 445.0189744574611,382.6592780037797 434.01348185135396,376.6567795252891 420.006491261763,369.65386463371664 403.9985020165162,361.6505333290624 387.99051277126944,351.64636919824466 372.98302285385057,340.6417886543451 358.97603226425963,330.63762452352734 343.96854234684076,320.63346039270954 327.96055310159403,311.6297126749736 309.9515652006914,302.62596495723756 290.9420779719608,294.6226336525834 276.9350873823699,287.6197187610109 267.93059343191857,284.6184695217656" stroke="black" stroke-width="7" fill="none" /><path d="M 272.9330900710582,426.6776001793779 271.93259074323026,426.6776001793779 282.93808334933743,426.6776001793779 312.95306318417516,426.6776001793779 356.9750336086038,423.6763509401326 408.0004993278279,414.6726032223966 454.02346840791245,401.6671898523335 492.0424428653735,385.66052724302506 521.0569233723833,368.65344822063486 537.0649126176301,355.6480348505718 548.0704052237372,344.6434543066722 552.0724025350489,338.6409558281815" stroke="black" stroke-width="7" fill="none" /><path d="M 311.95256385634724,554.7309010538454 312.95306318417516,554.7309010538454 318.9560591511427,554.7309010538454 339.96654503552907,554.7309010538454 375.98452083733434,553.7304846407636 419.0059919339351,547.7279861622729 463.0279623583637,531.7213235529646 503.0479354714807,510.7125788782472 534.0634146341463,485.70216855120276 557.0748991741885,463.69300746340366 572.0823890916074,443.6846792017681 582.0873823698867,428.6784330055415 586.0893796811984,417.67385246164196 587.0898790090263,412.67177039623306" stroke="black" stroke-width="7" fill="none" /></svg>
'''

# strokes = svg_to_strokes(svg_string1)
# # print(simplifyStrokes(svgStokesReformat(strokes)))
# inp = simplifyStrokes(svgStokesReformat(strokes))
# raster = vector_to_raster([inp])[0]
# print(raster)
# grid = raster.reshape((28,28))
# plt.imshow(grid,cmap="gray")
# plt.show()

# test_canvas = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,76,255,255,255,255,143,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,128,255,255,128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,191,191,191,128,0,0,0,0,0,0,128,255,128,0,0,0,0,0,0,0,0,0,0,128,0,128,255,255,191,191,191,128,0,0,0,0,0,0,0,128,255,128,0,0,0,0,0,0,0,0,64,255,255,255,191,64,0,0,0,0,0,0,0,0,0,0,0,0,128,255,128,0,0,0,0,0,0,0,0,191,255,255,255,128,0,0,0,0,0,0,0,0,0,0,0,0,0,128,255,64,0,0,0,0,0,0,0,191,255,128,128,255,128,0,0,0,0,0,0,0,0,0,128,0,0,0,191,191,0,0,0,0,0,0,128,255,128,0,0,128,255,255,128,0,0,0,0,0,64,255,191,0,0,0,64,255,64,0,0,0,0,0,128,255,0,0,0,0,64,128,255,255,128,0,0,128,255,191,64,0,0,0,64,255,64,0,0,0,0,0,255,128,0,0,0,0,0,0,128,128,255,191,255,255,191,0,0,0,0,0,128,255,64,0,0,0,0,0,255,128,0,0,0,0,64,64,191,191,255,255,191,64,0,0,0,64,64,0,128,255,0,0,0,0,0,76,255,255,255,255,255,255,255,255,255,191,64,191,191,0,0,0,0,255,191,0,191,191,0,0,0,0,0,76,255,191,128,128,128,128,64,64,0,0,0,128,255,0,0,0,128,255,128,0,255,128,0,0,0,0,0,76,255,64,0,0,0,0,0,0,0,0,0,128,255,0,0,128,255,128,0,128,255,0,0,0,0,0,0,0,255,191,0,0,0,0,0,0,0,0,0,128,255,0,128,255,191,0,128,255,128,0,0,0,0,0,0,0,128,255,64,0,0,0,0,0,0,0,0,128,255,128,255,191,0,0,255,255,0,0,0,0,0,0,0,0,0,191,255,128,0,0,0,0,0,0,0,128,255,255,128,0,64,191,255,64,0,0,0,0,0,0,0,0,0,0,191,255,128,0,64,64,64,191,255,255,255,64,0,128,255,255,64,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,191,191,191,255,255,191,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,128,128,191,255,255,255,255,255,255,255,191,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,64,64,191,191,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,191,191,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,191,191,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,191,191,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# raster = np.array(test_canvas, dtype=np.float32).reshape(1, 784)
# grid = raster.reshape((28,28))
# plt.imshow(grid,cmap="gray")
# plt.show()


# import onnxruntime as ort

# session = ort.InferenceSession("model3_1_large.onnx")
# input_names = [inp.name for inp in session.get_inputs()]
# output_names = [out.name for out in session.get_outputs()]
# print("Input names:", input_names)
# print("Output names:", output_names)

# input_data = raster
# result = session.run(['output'], {'input': input_data})
# print(result)


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

label_dict = {}
values_dict = {}
items = 0

# modelCategories = ["airplane","angel","ant","anvil","apple","banana","basketball","broom","camera","dog", "dresser","hammer","hat","hexagon","paperclip","pencil"]

mypath = os.path.join(os.getcwd(), "model", "trainingdata")
arr = [f for f in listdir(mypath) if isfile(join(mypath, f))]
items = len(arr)
print(f"Items: {items}")

for i,v in enumerate(arr):
    label_dict[i] = v
    values_dict[v] = []

for item in values_dict.keys():
    i = 0
    for drawing in unpack_drawings(item):
        simplifiedVector = drawing["image"]
        raster = vector_to_raster([simplifiedVector])[0]
        values_dict[item].append(raster)
        i += 1
        if i > 34999:
            break

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

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Define training and evaluation functions
def train_model(model, X_train, y_train, epochs=10, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(torch.from_numpy(X_train).float())
        loss = criterion(outputs, torch.from_numpy(y_train).long())
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs1 = model(torch.from_numpy(X_train).float())
        _, predicted1 = torch.max(outputs1, 1)
        accuracy = (predicted1.numpy() == y_train).mean()
        print(f'Train Accuracy: {accuracy:.4f}')

        outputs = model(torch.from_numpy(X_test).float())
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted.numpy() == y_test).mean()
        print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

def view_img(raster):
    grid = raster.reshape((28,28))
    plt.imshow(grid,cmap="gray")
    plt.show()

def get_pred(model, raster):
    raster_tensor = torch.tensor(raster, dtype=torch.float).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(raster_tensor)
        _, predicted = torch.max(outputs, 1)
    
    predicted_label = predicted.item()
    return label_dict[predicted_label].replace("full_binary_","").replace(".bin","")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Set up and train the model
input_size = 784
hidden_sizes = [600, 400, 160, 80]
output_size = items
model = SimpleMLP(input_size, hidden_sizes, output_size)

train_model(model, X_train, y_train, epochs=200, learning_rate=0.03)

torch.save(model, "model3_4_large.pt")
torch.onnx.export(model,torch.tensor(X_train[0], dtype=torch.float).unsqueeze(0),"model4_3_large.onnx",export_params=True,do_constant_folding=True,input_names = ['input'],output_names = ['output'])

# model = torch.load("model3_1_large.pt")


# Evaluate the model
evaluate_model(model, X_train, y_train, X_test, y_test)

model.eval()

with torch.no_grad():
    evaluate_model(model, X_train, y_train, X_test, y_test)

    rawStrokes = svg_to_strokes(svg_string1)
    reformattedStrokes = svgStokesReformat(rawStrokes)
    simplifiedVector = simplifyStrokes(reformattedStrokes)
    raster = vector_to_raster([simplifiedVector])[0]
    print(f"PREDICTED DRAWING: {get_pred(model,raster)}")
    view_img(raster)

    for i in range(61, 9771, 299):
        print(f"Actual: {label_dict[y_train[i]]}, Pred: {get_pred(model,X_train[i])}")
        view_img(X_train[i])
        # test_model(model,img, y_train[i])