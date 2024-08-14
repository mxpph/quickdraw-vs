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


svg_string1 = '''
<svg width="813.6" height="731.7" xmlns="http://www.w3.org/2000/svg"><path d="M 278.93608603802573 170.0707902239021 L 277.9355867101978 172.07162305006565 L 272.9330900710582 178.07412152855633 L 267.93059343191857 185.07703642012876 L 263.92859612060687 192.0799513117012 L 259.92659880929517 198.08244979019187 L 258.92609948146725 202.08411544251896 L 258.92609948146725 204.08494826868252 L 259.92659880929517 204.08494826868252 L 261.927597464951 204.08494826868252 L 264.9290954484348 204.08494826868252 L 268.9310927597465 204.08494826868252 L 274.93408872671404 204.08494826868252 L 283.93858267716536 204.08494826868252 L 295.94457461110045 207.08619750792786 L 311.95256385634724 214.0891123995003 L 328.96105242942195 224.09327653031806 L 347.97053965815246 234.09744066113583 L 363.97852890339925 244.1016047919536 L 377.9855194929902 253.1053525096896 L 387.99051277126944 259.10785098818025 L 394.9940080660649 263.10951664050737 L 397.9955060495487 266.1107658797527 L 399.99650470520453 267.1111822928345 L 400.99700403303245 268.11159870591626 L 400.99700403303245 267.1111822928345 L 400.99700403303245 266.1107658797527 L 399.99650470520453 264.10993305358915 L 399.99650470520453 262.1091002274256 L 399.99650470520453 260.108267401262 L 399.99650470520453 258.10743457509847 L 399.99650470520453 256.1066017489349 L 399.99650470520453 253.1053525096896 L 399.99650470520453 250.10410327044428 L 399.99650470520453 245.10202120503538 L 399.99650470520453 239.09952272654473 L 400.99700403303245 234.09744066113583 L 401.9975033608604 230.09577500880874 L 403.9985020165162 227.0945257695634 L 403.9985020165162 226.09410935648162 L 403.9985020165162 225.09369294339984 L 403.9985020165162 224.09327653031806 L 403.9985020165162 222.0924437041545 L 403.9985020165162 221.09202729107272 L 403.9985020165162 220.09161087799095 L 402.9980026886883 220.09161087799095 L 400.99700403303245 219.0911944649092 L 397.9955060495487 218.0907780518274 L 394.9940080660649 217.09036163874563 L 390.9920107547532 215.08952881258207 L 385.9895141156136 212.08827957333673 L 378.9860188208181 209.08703033409142 L 369.9815248703668 204.08494826868252 L 359.97653159208755 199.08286620327365 L 348.9710389859804 193.08036772478297 L 339.96654503552907 188.0782856593741 L 331.96255041290567 184.07662000704698 L 324.95905511811026 180.07495435471986 L 318.9560591511427 176.07328870239277 L 314.954061839831 173.07203946314743 L 309.9515652006914 170.0707902239021 L 305.9495678893797 168.06995739773856 L 301.947570578068 166.069124571575 L 297.9455732667563 164.06829174541144 L 294.9440752832725 161.0670425061661 L 291.94257729978875 159.06620968000254 L 288.941079316305 158.0657932669208 L 285.9395813328212 158.0657932669208 L 282.93808334933743 158.0657932669208 L 279.93658536585366 158.0657932669208 L 277.9355867101978 158.0657932669208 L 274.93408872671404 158.0657932669208" fill="none" stroke="black" stroke-width="2" /><path d="M 400.99700403303245 235.09785707421761 L 401.9975033608604 235.09785707421761 L 405.9995006721721 235.09785707421761 L 414.00349529479547 235.09785707421761 L 425.00898790090264 235.09785707421761 L 438.01547916266566 234.09744066113583 L 454.02346840791245 235.09785707421761 L 470.0314576531592 239.09952272654473 L 485.03894757057805 244.1016047919536 L 500.0464374879969 249.1036868573625 L 514.0534280775879 255.10618533585316 L 529.0609179950067 262.1091002274256 L 545.0689072402535 269.11201511899804 L 563.0778951411561 277.1153464236522 L 579.0858843864029 285.11867772830647 L 593.0928749759938 292.1215926198789 L 606.0993662377568 299.1245075114513 L 617.104858843864 305.127005989942 L 626.1093527943153 309.1286716422691 L 633.1128480891108 312.12992088151447 L 638.1153447282504 316.1315865338416 L 643.11784136739 319.13283577308687 L 648.1203380065297 322.1340850123322 L 653.1228346456693 325.13533425157755 L 657.124831956981 328.1365834908229 L 661.1268292682927 331.1378327300682 L 666.1293259074323 334.13908196931357 L 671.1318225465719 337.1403312085589 L 676.1343191857115 340.1415804478042 L 680.1363164970232 344.1432461001313 L 683.137814480507 347.14449533937665 L 686.1393124639908 349.1453281655402 L 689.1408104474746 352.14657740478555 L 691.1418091031304 354.1474102309491 L 692.1423084309583 355.1478266440309 L 692.1423084309583 356.14824305711267 L 692.1423084309583 357.14865947019445 L 692.1423084309583 358.1490758832762 L 691.1418091031304 359.14949229635795 L 690.1413097753025 359.14949229635795 L 688.1403111196466 360.1499087094397 L 685.1388131361629 361.1503251225215 L 682.1373151526791 363.15115794868507 L 679.1358171691953 364.15157436176685 L 677.1348185135395 365.1519907748486 L 675.1338198578836 366.1524071879304 L 673.1328212022278 367.1528236010122 L 672.1323218743998 367.1528236010122 L 671.1318225465719 367.1528236010122 L 669.1308238909161 367.1528236010122 L 668.1303245630882 368.15324001409397 L 666.1293259074323 368.15324001409397 L 664.1283272517765 368.15324001409397 L 663.1278279239485 369.15365642717575 L 660.1263299404648 369.15365642717575 L 655.1238333013251 369.15365642717575 L 650.1213366621855 369.15365642717575 L 645.1188400230459 369.15365642717575 L 639.1158440560783 369.15365642717575 L 632.1123487612829 369.15365642717575 L 626.1093527943153 368.15324001409397 L 620.1063568273478 366.1524071879304 L 613.1028615325523 363.15115794868507 L 605.0988669099289 360.1499087094397 L 598.0953716151334 357.14865947019445 L 591.091876320338 353.1469938178673 L 583.0878816977146 349.1453281655402 L 576.0843864029191 345.1436625132131 L 568.0803917802957 340.1415804478042 L 560.0763971576723 335.13949838239535 L 552.0724025350489 331.1378327300682 L 544.0684079124255 328.1365834908229 L 537.0649126176301 324.13491783849577 L 530.0614173228346 321.1336685992504 L 521.0569233723833 316.1315865338416 L 511.0519300941041 312.12992088151447 L 503.0479354714807 308.12825522918735 L 496.0444401766852 305.127005989942 L 490.04144420971767 302.12575675069667 L 486.03944689840597 300.1249239245331 L 483.0379489149222 298.12409109836955 L 479.0359516036105 297.12367468528777 L 474.0334549644709 296.12325827220604 L 469.03095832533126 294.1224254460425 L 465.02896101401956 293.1220090329607 L 462.0274630305358 292.1215926198789 L 458.0254657192241 291.12117620679714 L 454.02346840791245 290.12075979371537 L 450.02147109660075 288.1199269675518 L 446.01947378528905 287.11951055447 L 442.01747647397735 285.11867772830647 L 439.0159784904936 284.1182613152247 L 435.0139811791819 283.1178449021429 L 433.01298252352603 283.1178449021429 L 430.01148454004226 282.1174284890611 L 426.00948722873056 281.11701207597935 L 423.0079892452468 280.11659566289757 L 420.006491261763 278.115762836734 L 416.0044939504513 277.1153464236522 L 413.00299596696755 275.1145135974887 L 410.0014979834838 274.11409718440694 L 407 273.11368077132516 L 403.9985020165162 272.1132643582434 L 401.9975033608604 271.1128479451616 L 398.9960053773766 270.1124315320798 L 396.99500672172076 269.11201511899804 L 394.9940080660649 268.11159870591626 L 392.99300941040906 267.1111822928345" fill="none" stroke="black" stroke-width="2" /><path d="M 695.1438064144421 359.14949229635795 L 695.1438064144421 359.14949229635795 L 695.1438064144421 359.14949229635795 L 695.1438064144421 364.15157436176685 L 699.1458037257538 372.1549056664211 L 703.1478010370655 379.1578205579935 L 708.1502976762051 385.16031903648417 L 712.1522949875168 391.16281751497485 L 713.1527943153447 394.1640667542202 L 714.1532936431727 395.16448316730197 L 714.1532936431727 396.16489958038375 L 714.1532936431727 397.1653159934655 L 714.1532936431727 399.16614881962903 L 714.1532936431727 401.1669816457926 L 713.1527943153447 401.1669816457926 L 712.1522949875168 401.1669816457926 L 708.1502976762051 401.1669816457926 L 701.1468023814097 400.1665652327108 L 690.1413097753025 395.16448316730197 L 676.1343191857115 389.1619846888113 L 662.1273285961206 383.1594862103206 L 650.1213366621855 377.15698773183 L 640.1163433839063 372.1549056664211 L 633.1128480891108 369.15365642717575 L 628.1103514499712 366.1524071879304 L 625.1088534664874 364.15157436176685 L 623.1078548108316 362.1507415356033 L 621.1068561551757 360.1499087094397 L 620.1063568273478 359.14949229635795" fill="none" stroke="black" stroke-width="2" /><path d="M 664.1283272517765 379.1578205579935 L 663.1278279239485 379.1578205579935 L 664.1283272517765 379.1578205579935 L 668.1303245630882 379.1578205579935 L 674.1333205300557 379.1578205579935 L 681.1368158248512 379.1578205579935 L 687.1398117918187 379.1578205579935 L 691.1418091031304 379.1578205579935 L 694.1433070866142 379.1578205579935 L 696.14430574227 379.1578205579935 L 697.144805070098 379.1578205579935 L 697.144805070098 380.15823697107527 L 696.14430574227 382.1590697972388 L 694.1433070866142 384.1599026234024 L 691.1418091031304 385.16031903648417 L 687.1398117918187 386.16073544956595 L 685.1388131361629 388.1615682757295 L 682.1373151526791 389.1619846888113 L 679.1358171691953 391.16281751497485 L 677.1348185135395 392.1632339280566 L 675.1338198578836 393.1636503411384 L 673.1328212022278 394.1640667542202 L 672.1323218743998 394.1640667542202 L 670.131323218744 394.1640667542202 L 670.131323218744 393.1636503411384 L 670.131323218744 390.16240110189307 L 670.131323218744 386.16073544956595 L 672.1323218743998 382.1590697972388 L 674.1333205300557 378.15740414491177 L 675.1338198578836 374.15573849258465 L 676.1343191857115 372.1549056664211 L 676.1343191857115 370.1540728402575 L 676.1343191857115 368.15324001409397 L 675.1338198578836 368.15324001409397 L 674.1333205300557 368.15324001409397 L 672.1323218743998 368.15324001409397 L 670.131323218744 369.15365642717575 L 668.1303245630882 371.1544892533393 L 666.1293259074323 374.15573849258465 L 665.1288265796044 376.1565713187482 L 665.1288265796044 378.15740414491177 L 665.1288265796044 378.15740414491177 L 665.1288265796044 377.15698773183 L 667.1298252352602 375.1561549056664 L 671.1318225465719 372.1549056664211 L 676.1343191857115 370.1540728402575 L 681.1368158248512 368.15324001409397 L 686.1393124639908 366.1524071879304 L 691.1418091031304 364.15157436176685 L 695.1438064144421 363.15115794868507 L 698.1453043979259 362.1507415356033 L 701.1468023814097 362.1507415356033 L 702.1473017092376 362.1507415356033 L 703.1478010370655 364.15157436176685 L 701.1468023814097 366.1524071879304 L 698.1453043979259 369.15365642717575 L 694.1433070866142 373.15532207950287 L 690.1413097753025 376.1565713187482 L 688.1403111196466 379.1578205579935 L 686.1393124639908 380.15823697107527 L 684.1383138083349 381.15865338415705 L 684.1383138083349 380.15823697107527 L 684.1383138083349 379.1578205579935 L 685.1388131361629 376.1565713187482 L 688.1403111196466 374.15573849258465 L 691.1418091031304 372.1549056664211 L 693.1428077587863 371.1544892533393 L 695.1438064144421 371.1544892533393 L 697.144805070098 371.1544892533393 L 698.1453043979259 371.1544892533393 L 698.1453043979259 372.1549056664211 L 699.1458037257538 374.15573849258465 L 699.1458037257538 377.15698773183 L 699.1458037257538 381.15865338415705 L 699.1458037257538 384.1599026234024 L 699.1458037257538 386.16073544956595 L 699.1458037257538 388.1615682757295" fill="none" stroke="black" stroke-width="2" /></svg>
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

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) # Input channels=1, Output channels=32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Input channels=32, Output channels=64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # Input channels=64, Output channels=128
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # Pooling layer
        self.fc1 = nn.Linear(128 * 3 * 3, 512) # Adjust input size based on your image size and layers
        self.fc2 = nn.Linear(512, 10) # Output size = number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.from_numpy(X_test).float())
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted.numpy() == y_test).mean()
        print(f'Accuracy: {accuracy:.4f}')
    return accuracy

def view_img(raster):
    grid = raster.reshape((28,28))
    plt.imshow(grid,cmap="gray")
    plt.show()

def get_pred(model, raster):
    model.eval()
    raster = torch.from_numpy(raster).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    with torch.no_grad():
        outputs = model(raster)
        _, predicted = torch.max(outputs, 1)
    predicted_label = predicted.item()
    return label_dict[predicted_label]

# Preprocess the data
X = np.array(X)
y = np.array(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Reshape X_train and X_test for CNN input (batch_size, channels, height, width)
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

# Initialize and train the model
model = SimpleCNN()
train_model(model, X_train, y_train, epochs=10, learning_rate=0.01)

# Save and reload the model
torch.save(model, "model_4_cnn.pt")
# model = torch.load("model_cnn.pt")

# Evaluate the model
evaluate_model(model, X_test, y_test)

# Test the model with some samples
with torch.no_grad():
    evaluate_model(model, X_test, y_test)
    for i in range(61, 9771, 299):
        print(f"Actual: {label_dict[y_train[i]]}, Pred: {get_pred(model, X_train[i])}")
        view_img(X_train[i])