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

# Example usage
svg_string = '''<svg height="210" width="400">
  <path d="M150 0 L75 200 L225 200 Z" />
</svg>'''

strokes = svg_to_strokes(svg_string)

# print(strokes)
print(svgStokesReformat(strokes))
print(vector_to_raster([svgStokesReformat(strokes)])[0])
raster = vector_to_raster([svgStokesReformat(strokes)])[0]
grid = raster.reshape((28,28))
plt.imshow(grid,cmap="gray")
plt.show()



# print(strokes_array)

# for drawing in unpack_drawings('full_binary_pencil.bin'):
#     raw = drawing["image"]
#     print(raw)
#     raster = vector_to_raster([raw])[0]
#     print(raster)
# #     grid = raster.reshape((28,28))
# #     plt.imshow(grid,cmap="gray")
# #     plt.show()
#     break
