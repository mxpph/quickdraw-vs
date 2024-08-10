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


svg_string = '''
<svg width="813.6" height="731.7" xmlns="http://www.w3.org/2000/svg"><path d="M 314.954061839831 261.1086838143438 L 314.954061839831 261.1086838143438 L 313.9535625120031 260.108267401262 L 308.95106587286347 260.108267401262 L 299.94657192241215 260.108267401262 L 283.93858267716536 260.108267401262 L 261.927597464951 264.10993305358915 L 238.91611292490876 275.1145135974887 L 217.90562704052238 289.1203433806336 L 199.89663913961974 307.12783881610557 L 184.88914922220087 326.1357506646593 L 173.88365661609373 347.14449533937665 L 168.8811599769541 368.15324001409397 L 166.88016132129826 390.16240110189307 L 166.88016132129826 413.17197860277395 L 170.88215863260996 440.1832217559819 L 181.8876512387171 468.19488132227167 L 195.89464182830804 493.2052916493161 L 209.90163241789898 516.2148691501969 L 223.90862300748992 538.2240302379961 L 237.91561359708084 554.2306928473045 L 251.92260418667178 567.2361062173676 L 266.93009410409064 577.2402703481854 L 284.9390820049933 585.2436016528396 L 305.9495678893797 592.2465165444121 L 328.96105242942195 597.2485986098209 L 351.97253696946416 601.250264262148 L 375.98452083733434 603.2510970883116 L 398.9960053773766 603.2510970883116 L 422.00748991741887 602.2506806752298 L 444.0184751296332 597.2485986098209 L 470.0314576531592 589.2452673051667 L 495.0439408488573 578.2406867612672 L 517.0549260610717 566.2356898042858 L 539.0659112732859 553.2302764342227 L 558.0753985020165 540.2248630641596 L 572.0823890916074 528.2198661071783 L 582.0873823698867 517.2152855632787 L 591.091876320338 505.2102886062974 L 597.0948722873055 494.2057080623979 L 601.0968695986172 482.20071110541653 L 603.0978682542731 469.19529773535345 L 605.0988669099289 454.1890515391268 L 605.0988669099289 438.1823889298184 L 605.0988669099289 422.1757263205099 L 604.098367582101 407.16948012428327 L 600.0963702707893 393.1636503411384 L 594.0933743038217 380.15823697107527 L 588.0903783368542 369.15365642717575 L 579.0858843864029 358.1490758832762 L 570.0813904359516 348.1449117524584 L 561.0768964855002 341.14199686088597 L 552.0724025350489 333.1386655562318 L 541.0669099289418 325.13533425157755 L 529.0609179950067 317.13200294692336 L 517.0549260610717 310.1290880553509 L 506.04943345496446 304.1265895768602 L 494.04344152102937 299.1245075114513 L 482.0374495870943 293.1220090329607 L 471.0319569809871 288.1199269675518 L 460.02646437487994 284.1182613152247 L 448.0204724409449 280.11659566289757 L 437.01497983483773 276.1149300105705 L 426.00948722873056 273.11368077132516 L 416.0044939504513 270.1124315320798 L 407 268.11159870591626 L 398.9960053773766 266.1107658797527 L 391.99251008258113 265.1103494666709 L 384.98901478778566 264.10993305358915 L 376.98502016516227 263.10951664050737 L 366.980026886883 263.10951664050737 L 355.97453428077586 263.10951664050737 L 344.9690416746687 263.10951664050737 L 329.9615517572499 265.1103494666709" fill="none" stroke="black" stroke-width="2" /><path d="M 284.9390820049933 285.11867772830647 L 285.9395813328212 286.11909414138825 L 288.941079316305 293.1220090329607 L 293.9435759554446 307.12783881610557 L 299.94657192241215 324.13491783849577 L 303.94856923372384 344.1432461001313 L 304.94906856155177 365.1519907748486 L 304.94906856155177 385.16031903648417 L 304.94906856155177 402.16739805887437 L 304.94906856155177 415.1728114289375 L 303.94856923372384 429.17864121208237 L 299.94657192241215 443.18447099522723 L 292.9430766276167 453.188635126045 L 284.9390820049933 461.1919664306992 L 277.9355867101978 468.19488132227167 L 270.93209141540234 473.19696338768057 L 264.9290954484348 476.1982126269259 L 258.92609948146725 478.19904545308947 L 251.92260418667178 481.20029469233475" fill="none" stroke="black" stroke-width="2" /><path d="M 481.03695025926635 327.1361670777411 L 480.0364509314384 326.1357506646593 L 480.0364509314384 329.13699990390467 L 478.0354522757826 340.1415804478042 L 473.03295563664295 356.14824305711267 L 467.0299596696754 377.15698773183 L 463.0279623583637 399.16614881962903 L 461.02696370270786 418.17406066818285 L 461.02696370270786 437.1819725167366 L 461.02696370270786 455.1894679522086 L 461.02696370270786 472.1965469745988 L 461.02696370270786 486.20237675774365 L 464.02846168619163 501.20862295397035 L 469.03095832533126 515.2144527371152 L 476.0344536201267 529.2202825202601 L 485.03894757057805 543.226112303405 L 497.04493950451314 557.2319420865498 L 511.0519300941041 569.2369390435312 L 527.0599193393508 577.2402703481854 L 541.0669099289418 582.2423524135943" fill="none" stroke="black" stroke-width="2" /><path d="M 204.89913577875936 406.1690637112015 L 204.89913577875936 405.1686472981197 L 205.89963510658728 405.1686472981197 L 212.90313040138275 405.1686472981197 L 236.91511426925294 405.1686472981197 L 281.9375840215095 405.1686472981197 L 340.967044363357 405.1686472981197 L 402.9980026886883 405.1686472981197 L 461.02696370270786 405.1686472981197 L 510.05143076627616 405.1686472981197 L 552.0724025350489 405.1686472981197 L 583.0878816977146 405.1686472981197" fill="none" stroke="black" stroke-width="2" /></svg>'''

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
    # if str(item) != "hexagon":
        # break
    for drawing in unpack_drawings('full_binary_'+str(item)+'.bin'):
        simplifiedVector = drawing["image"]
        raster = vector_to_raster([simplifiedVector])[0]
        values_dict[item].append(raster)
        i += 1
        if i > 999:
            break

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

def get_preds(model, input):
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logits = model.forward(input)
    ps = F.softmax(logits, dim=1)
    return ps

X = []
y = []

for key, value in label_dict.items():
    data_i = values_dict[value]

    Xi = np.concatenate([data_i], axis = 0)

    yi = np.full((len(Xi), 1), key).ravel()
    
    X.append(Xi)
    y.append(yi)

X = np.concatenate(X, axis = 0) # IMAGES
y = np.concatenate(y, axis = 0) # LABELS

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

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
    
def test_model(model, img):
    """
    Function creates test view of the model's prediction for image.

    INPUT:
        model - pytorch model
        img - (tensor) image from the dataset

    OUTPUT: None
    """

    # Convert 2D image to 1D vector
    img = img.resize_(1, 784)

    ps = get_preds(model, img)
    view_classify(img.resize_(1, 28, 28), ps)

# print("RAHH",X[0])

model = torch.load("model7.pt")
with torch.no_grad():
    model.eval()
    for i in range(61,9771,50):
        test_model(model,X[i])
        # print(f"{np.concatenate([values_dict["hexagon"]],axis=0)}")
        # print(f"{type(np.concatenate(values_dict["hexagon"],axis=0))}")
        # print(f"{np.concatenate([values_dict["hexagon"]],axis=0)}")
        # print(f"{type(np.concatenate([values_dict["hexagon"]],axis=0))}")
        # print(f"{np.concatenate(values_dict["hexagon"][0],axis=0)}")
        # print(f"{type(np.concatenate(values_dict["hexagon"][0],axis=0))}")


        # output = model(X[i])
        # print(f"GOT: {label_dict[get_labels(get_preds(model,X[i:i+1]))[0][0]]} SHOULD: {label_dict[y[i]]}")
        # grid = X[i].reshape((28,28))
        # plt.imshow(grid,cmap="gray")
        # plt.show()
        # pred_probab = nn.Softmax(dim=1)(output)
        # print(f"Preds: {pred_probab}")
        # print(f"Get preds: {get_preds(model,torch.from_numpy(values_dict["hexagon"][i]).float().unsqueeze(0))}")
        # print("Max:",label_dict[torch.argmax(get_preds(model,torch.from_numpy(values_dict["hexagon"][i]).float().unsqueeze(0))).item()])

# print(type(np.random.randn(784)),np.random.randn(784).shape)

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")


# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 100),
#             nn.ReLU(),
#             nn.Linear(100, 10),
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
    

# model = NeuralNetwork().to(device)
# print(model)

# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# print(f"Preds: {pred_probab}")
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