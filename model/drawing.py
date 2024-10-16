import os
import struct
from struct import unpack
import svgpathtools
from svgpathtools import svg2paths2
from ctypes.macholib import dyld
import numpy as np

# to fix cairo bug & not have to run:
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


# Array of drawing data objects from .bin file of training data
def unpack_drawings(filename):
    path = os.getcwd()
    path = os.path.join(path,"model","trainingdata",filename)
    with open(path, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break


# Taken from google quick, draw!'s dataset repository
def vector_to_raster(
        vector_images, side=28,
        line_diameter=16, # relative to the original 256x256 image
        padding=16, # relative to the original 256x256 image
        bg_color=(0,0,0),
        fg_color=(1,1,1)):

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


# Convert an SVG into an array of (raw/unformated) strokes
def svg_to_strokes(svg_string):
    file_like = StringIO(svg_string)
    paths, attributes, svg_attributes = svg2paths2(file_like)
    strokes = []

    for path in paths:
        for segment in path:
            if isinstance(segment, svgpathtools.path.Line):
                strokes.append([(segment.start.real, segment.start.imag),
                                (segment.end.real, segment.end.imag)])
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

# Format raw strokes (from svg_to_strokes) into what is used in the dataset
def svg_strokes_reformat(input_list):
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

def simplify_strokes(input_strokes, epsilon=2.0):
    # Normalize strokes
    normalized_strokes = normalize_strokes(input_strokes)

    # Resample strokes
    resampled_strokes = [resample_stroke(list(zip(stroke[0], stroke[1])))
                                        for stroke in normalized_strokes]

    # Simplify strokes
    simplified_strokes = []
    for stroke in resampled_strokes:
        stroke_array = np.array(stroke)
        simplified = rdp(stroke_array, epsilon)
        x_coords, y_coords = zip(*simplified)
        simplified_strokes.append((x_coords, y_coords))

    return simplified_strokes

# An example drawing SVG
svg_string1 = '''
<svg width="813.6" height="731.7" xmlns="http://www.w3.org/2000/svg"><path d="M 423.0079892452468,260.6084756078029 423.0079892452468,259.60805919472114 422.00748991741887,259.60805919472114 417.00499327827924,259.60805919472114 402.9980026886883,259.60805919472114 384.98901478778566,259.60805919472114 365.9795275590551,261.6088920208847 345.9695410024966,268.61180691245715 325.9595544459382,279.6163874563567 308.95106587286347,293.6222172395016 293.9435759554446,311.6297126749736 279.93658536585366,334.63929017585446 269.9315920875744,358.64928408981706 262.92809679277894,381.65886159069794 258.92609948146725,407.66968833082416 257.9256001536393,431.6796822447868 257.9256001536393,454.6892597456677 262.92809679277894,477.6988372465486 276.9350873823699,500.70841474742946 297.9455732667563,523.7179922483103 324.95905511811026,542.725904096864 356.9750336086038,557.7321502930907 392.99300941040906,567.7363144239085 431.0119838678702,572.7383964893174 469.03095832533126,572.7383964893174 507.0499327827924,564.7350651846632 542.0674092567697,547.7279861622729 574.0833877472633,526.7192414875557 601.0968695986172,500.70841474742946 622.1073554830036,469.69550594189434 638.1153447282504,436.6817643101957 646.1193393508738,405.6688555046606 650.1213366621855,377.6571959383709 651.1218359900134,350.6459527851629 646.1193393508738,327.636375284282 632.1123487612829,303.62638137031934 611.1018628768965,280.61680386943846 585.0888803533704,257.6072263685576 557.0748991741885,237.59889810692206 528.0604186671787,222.5926519106954 500.0464374879969,212.58848777987762 478.0354522757826,208.58682212755053 458.0254657192241,209.5872385406323" stroke="black" stroke-width="7" fill="none" /><path d="M 476.0344536201267,648.7700438835325 475.0339542922988,649.7704602966143 474.0334549644709,649.7704602966143 474.0334549644709,647.7696274704507 474.0334549644709,643.7679618181236 474.0334549644709,638.7658797527147 474.0334549644709,633.7637976873058 474.0334549644709,631.7629648611422 474.0334549644709,627.7612992088151 474.0334549644709,617.7571350779973 474.0334549644709,600.7500560556072 474.0334549644709,579.7413113808898 474.0334549644709,556.731733880009 475.0339542922988,530.7209071398828 478.0354522757826,504.7100803997565 479.0359516036105,480.7000864857939 480.0364509314384,460.6917582241583 480.0364509314384,443.6846792017681 478.0354522757826,427.6780165924597 473.03295563664295,414.6726032223966 467.0299596696754,404.6684390915788 460.02646437487994,396.66510778692464 453.0229690800845,388.6617764822704 445.0189744574611,382.6592780037797 434.01348185135396,376.6567795252891 420.006491261763,369.65386463371664 403.9985020165162,361.6505333290624 387.99051277126944,351.64636919824466 372.98302285385057,340.6417886543451 358.97603226425963,330.63762452352734 343.96854234684076,320.63346039270954 327.96055310159403,311.6297126749736 309.9515652006914,302.62596495723756 290.9420779719608,294.6226336525834 276.9350873823699,287.6197187610109 267.93059343191857,284.6184695217656" stroke="black" stroke-width="7" fill="none" /><path d="M 272.9330900710582,426.6776001793779 271.93259074323026,426.6776001793779 282.93808334933743,426.6776001793779 312.95306318417516,426.6776001793779 356.9750336086038,423.6763509401326 408.0004993278279,414.6726032223966 454.02346840791245,401.6671898523335 492.0424428653735,385.66052724302506 521.0569233723833,368.65344822063486 537.0649126176301,355.6480348505718 548.0704052237372,344.6434543066722 552.0724025350489,338.6409558281815" stroke="black" stroke-width="7" fill="none" /><path d="M 311.95256385634724,554.7309010538454 312.95306318417516,554.7309010538454 318.9560591511427,554.7309010538454 339.96654503552907,554.7309010538454 375.98452083733434,553.7304846407636 419.0059919339351,547.7279861622729 463.0279623583637,531.7213235529646 503.0479354714807,510.7125788782472 534.0634146341463,485.70216855120276 557.0748991741885,463.69300746340366 572.0823890916074,443.6846792017681 582.0873823698867,428.6784330055415 586.0893796811984,417.67385246164196 587.0898790090263,412.67177039623306" stroke="black" stroke-width="7" fill="none" /></svg>
'''


################################
# HOW TO USE:

# FOR INPUT SVG:
# svgString = ```<svg ...>...</svg>```
# rawStrokes = svg_to_strokes(svg_string)
# reformattedStrokes = svg_strokes_reformat(rawStrokes)
# simplifiedVector = simplify_strokes(reformattedStrokes)
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
