"""Convert simple SVG outlines into pen-plotter style G-code.

This script traces SVG outlines only. It does not generate infill.
Supported elements:
- line
- polyline
- polygon
- rect
- path (M, m, L, l, H, h, V, v, C, c, Z, z)
"""

import argparse
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


PEN_UP_Z = 5
PEN_DOWN_Z = 0
PEN_Z_FEED_RATE = 300
DEFAULT_CURVE_SEGMENTS = 8
DEFAULT_MIN_MOVE = 1.0

COMMAND_RE = re.compile(r"[MmLlHhVvCcZz]")
NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+\.?|\.\d+)(?:[eE][-+]?\d+)?")
TRANSFORM_RE = re.compile(r"([A-Za-z]+)\(([^)]*)\)")


def parse_length(value):
    if value is None:
        return 0.0

    match = NUMBER_RE.match(value.strip())
    return float(match.group(0)) if match else 0.0


def parse_points(points_str):
    numbers = [float(value) for value in NUMBER_RE.findall(points_str)]
    if len(numbers) % 2 != 0:
        raise ValueError("SVG points attribute must contain x,y pairs.")

    points = []
    for index in range(0, len(numbers), 2):
        points.append((numbers[index], numbers[index + 1]))

    return points


def multiply_matrices(a, b):
    return [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2],
        ],
        [0.0, 0.0, 1.0],
    ]


def identity_matrix():
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def translate_matrix(tx, ty):
    return [
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0],
    ]


def scale_matrix(sx, sy):
    return [
        [sx, 0.0, 0.0],
        [0.0, sy, 0.0],
        [0.0, 0.0, 1.0],
    ]


def parse_transform(transform_text):
    matrix = identity_matrix()

    if not transform_text:
        return matrix

    for name, values_text in TRANSFORM_RE.findall(transform_text):
        values = [float(value) for value in NUMBER_RE.findall(values_text)]
        if name == "translate":
            tx = values[0] if values else 0.0
            ty = values[1] if len(values) > 1 else 0.0
            current = translate_matrix(tx, ty)
        elif name == "scale":
            sx = values[0] if values else 1.0
            sy = values[1] if len(values) > 1 else sx
            current = scale_matrix(sx, sy)
        else:
            continue

        matrix = multiply_matrices(matrix, current)

    return matrix


def apply_transform(point, matrix, scale):
    x, y = point
    transformed_x = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]
    transformed_y = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]
    return transformed_x * scale, transformed_y * scale


def g0(x, y):
    return f"G0 X{x:.3f} Y{y:.3f}"


def g1(x, y, feed_rate):
    return f"G1 X{x:.3f} Y{y:.3f} F{feed_rate}"


def distance(point_a, point_b):
    return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])


def simplify_points(points, min_move):
    if len(points) < 2 or min_move <= 0:
        return points

    simplified = [points[0]]

    for point in points[1:-1]:
        if distance(simplified[-1], point) >= min_move:
            simplified.append(point)

    if points[-1] != simplified[-1]:
        simplified.append(points[-1])

    return simplified


def add_outline_gcode(gcode, points, feed_rate, min_move):
    points = simplify_points(points, min_move)
    if len(points) < 2:
        return

    start_x, start_y = points[0]
    gcode.append(g0(start_x, start_y))
    gcode.append(f"G1 Z{PEN_DOWN_Z} F{PEN_Z_FEED_RATE}")
    first_x, first_y = points[1]
    gcode.append(g1(first_x, first_y, feed_rate))

    for x, y in points[2:]:
        gcode.append(f"G1 X{x:.3f} Y{y:.3f}")

    gcode.append(f"G0 Z{PEN_UP_Z}")


def rect_to_points(elem):
    x = parse_length(elem.get("x"))
    y = parse_length(elem.get("y"))
    width = parse_length(elem.get("width"))
    height = parse_length(elem.get("height"))

    return [[
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
        (x, y),
    ]]


def line_to_points(elem):
    return [[
        (parse_length(elem.get("x1")), parse_length(elem.get("y1"))),
        (parse_length(elem.get("x2")), parse_length(elem.get("y2"))),
    ]]


def polyline_to_points(elem, close_shape=False):
    points = parse_points(elem.get("points", ""))
    if close_shape and points:
        points.append(points[0])
    return [points]


def cubic_bezier_point(p0, p1, p2, p3, t):
    one_minus_t = 1.0 - t
    return (
        (one_minus_t ** 3) * p0[0]
        + 3 * (one_minus_t ** 2) * t * p1[0]
        + 3 * one_minus_t * (t ** 2) * p2[0]
        + (t ** 3) * p3[0],
        (one_minus_t ** 3) * p0[1]
        + 3 * (one_minus_t ** 2) * t * p1[1]
        + 3 * one_minus_t * (t ** 2) * p2[1]
        + (t ** 3) * p3[1],
    )


def tokenize_path(d_attr):
    token_re = re.compile(
        r"[MmLlHhVvCcZz]|[-+]?(?:\d+\.\d+|\d+\.?|\.\d+)(?:[eE][-+]?\d+)?"
    )
    return token_re.findall(d_attr)


def is_command(token):
    return len(token) == 1 and COMMAND_RE.fullmatch(token) is not None


def path_to_subpaths(d_attr, curve_segments):
    tokens = tokenize_path(d_attr)
    index = 0
    command = None
    current = (0.0, 0.0)
    start = None
    subpath = []
    subpaths = []

    def read_float():
        nonlocal index
        value = float(tokens[index])
        index += 1
        return value

    while index < len(tokens):
        token = tokens[index]
        if is_command(token):
            command = token
            index += 1
        elif command is None:
            raise ValueError("Path data must begin with a command.")

        if command in ("M", "m"):
            x = read_float()
            y = read_float()
            if command == "m":
                current = (current[0] + x, current[1] + y)
            else:
                current = (x, y)

            if subpath:
                subpaths.append(subpath)
            subpath = [current]
            start = current

            while index < len(tokens) and not is_command(tokens[index]):
                x = read_float()
                y = read_float()
                if command == "m":
                    current = (current[0] + x, current[1] + y)
                else:
                    current = (x, y)
                subpath.append(current)

            command = "l" if command == "m" else "L"

        elif command in ("L", "l"):
            while index < len(tokens) and not is_command(tokens[index]):
                x = read_float()
                y = read_float()
                if command == "l":
                    current = (current[0] + x, current[1] + y)
                else:
                    current = (x, y)
                subpath.append(current)

        elif command in ("H", "h"):
            while index < len(tokens) and not is_command(tokens[index]):
                x = read_float()
                if command == "h":
                    current = (current[0] + x, current[1])
                else:
                    current = (x, current[1])
                subpath.append(current)

        elif command in ("V", "v"):
            while index < len(tokens) and not is_command(tokens[index]):
                y = read_float()
                if command == "v":
                    current = (current[0], current[1] + y)
                else:
                    current = (current[0], y)
                subpath.append(current)

        elif command in ("C", "c"):
            while index < len(tokens) and not is_command(tokens[index]):
                x1 = read_float()
                y1 = read_float()
                x2 = read_float()
                y2 = read_float()
                x = read_float()
                y = read_float()

                if command == "c":
                    control1 = (current[0] + x1, current[1] + y1)
                    control2 = (current[0] + x2, current[1] + y2)
                    end = (current[0] + x, current[1] + y)
                else:
                    control1 = (x1, y1)
                    control2 = (x2, y2)
                    end = (x, y)

                for step in range(1, curve_segments + 1):
                    t = step / curve_segments
                    subpath.append(cubic_bezier_point(current, control1, control2, end, t))

                current = end

        elif command in ("Z", "z"):
            if subpath and start is not None and subpath[-1] != start:
                subpath.append(start)
            if subpath:
                subpaths.append(subpath)
            subpath = []
            start = None
            command = None

        else:
            raise ValueError(f"Unsupported SVG path command: {command}")

    if subpath:
        subpaths.append(subpath)

    return subpaths


def path_element_to_points(elem, curve_segments):
    d_attr = elem.get("d", "")
    if not d_attr.strip():
        return []
    return path_to_subpaths(d_attr, curve_segments)


def element_to_subpaths(elem, curve_segments):
    tag = elem.tag.split("}")[-1]

    if tag == "line":
        return line_to_points(elem)
    if tag == "polyline":
        return polyline_to_points(elem)
    if tag == "polygon":
        return polyline_to_points(elem, close_shape=True)
    if tag == "rect":
        return rect_to_points(elem)
    if tag == "path":
        return path_element_to_points(elem, curve_segments)

    return []


def walk_svg(elem, parent_matrix, scale, feed_rate, curve_segments, min_move, gcode):
    element_matrix = multiply_matrices(parent_matrix, parse_transform(elem.get("transform")))
    subpaths = element_to_subpaths(elem, curve_segments)

    for subpath in subpaths:
        transformed_points = [
            apply_transform(point, element_matrix, scale) for point in subpath
        ]
        add_outline_gcode(gcode, transformed_points, feed_rate, min_move)

    for child in elem:
        walk_svg(child, element_matrix, scale, feed_rate, curve_segments, min_move, gcode)


def svg_to_gcode(
    svg_file,
    gcode_file,
    scale=1.0,
    feed_rate=1000,
    curve_segments=DEFAULT_CURVE_SEGMENTS,
    min_move=DEFAULT_MIN_MOVE,
):
    tree = ET.parse(svg_file)
    root = tree.getroot()

    gcode = [
        "G21",
        "G90",
        f"G0 Z{PEN_UP_Z}",
    ]

    walk_svg(root, identity_matrix(), scale, feed_rate, curve_segments, min_move, gcode)

    gcode.append("M2")

    with open(gcode_file, "w", encoding="utf-8") as file:
        file.write("\n".join(gcode) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert simple SVG outlines into pen-plotter G-code without external libraries."
    )
    parser.add_argument("svg_file", help="Path to the input SVG file")
    parser.add_argument(
        "gcode_file",
        nargs="?",
        help="Path to the output G-code file. Defaults to the SVG name with a .gcode extension.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor applied after SVG transforms",
    )
    parser.add_argument(
        "--feed-rate",
        type=int,
        default=1000,
        help="Feed rate used for XY drawing moves",
    )
    parser.add_argument(
        "--curve-segments",
        type=int,
        default=DEFAULT_CURVE_SEGMENTS,
        help="Number of straight segments used to approximate each cubic curve",
    )
    parser.add_argument(
        "--min-move",
        type=float,
        default=DEFAULT_MIN_MOVE,
        help="Skip intermediate XY points that are closer than this distance",
    )

    args = parser.parse_args()
    output_file = args.gcode_file
    if output_file is None:
        output_file = str(Path(args.svg_file).with_suffix(".gcode"))

    svg_to_gcode(
        args.svg_file,
        output_file,
        scale=args.scale,
        feed_rate=args.feed_rate,
        curve_segments=max(1, args.curve_segments),
        min_move=max(0.0, args.min_move),
    )
    print(f"G-code written to {output_file}")


if __name__ == "__main__":
    main()
