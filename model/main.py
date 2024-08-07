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

strokes = svg_to_strokes(svg_string)

print(simplifyStrokes(svgStokesReformat(strokes)))
inp = simplifyStrokes(svgStokesReformat(strokes))
raster = vector_to_raster([inp])[0]
grid = raster.reshape((28,28))
plt.imshow(grid,cmap="gray")
plt.show()


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



# dict1 = {
#     0: "basketball",
#     1: "hammer",
#     2: "paperclip",
#     3: "pencil",
# }

# dict2 = {
#     "basketball": [],
#     "hammer": [],
#     "paperclip": [],
#     "pencil": [],
# }


# for item in dict2.keys():
#     i = 0
#     for drawing in unpack_drawings('full_binary_'+str(item)+'.bin'):
#         simplifiedVector = drawing["image"]
#         raster = vector_to_raster([simplifiedVector])[0]
#         dict2[item].append(raster)
#         i += 1
#         if i > 5000:
#             break

# for drawing in unpack_drawings('full_binary_pencil.bin'):
#     raw = drawing["image"]
#     print(raw)
#     raster = vector_to_raster([raw])[0]
#     print(raster)
# #     grid = raster.reshape((28,28))
# #     plt.imshow(grid,cmap="gray")
# #     plt.show()
#     break
