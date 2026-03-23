"""Convert simple SVG outlines into pen-plotter style G-code.

This script only traces shape boundaries. It does not generate any infill.
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_length(value):
    if value is None:
        return 0.0

    cleaned = value.strip()
    numeric = []

    for char in cleaned:
        if char.isdigit() or char in ".-+":
            numeric.append(char)
        else:
            break

    return float("".join(numeric)) if numeric else 0.0


def parse_points(points_str):
    tokens = points_str.replace(",", " ").split()
    if len(tokens) % 2 != 0:
        raise ValueError("SVG points attribute must contain x,y pairs.")

    points = []
    for index in range(0, len(tokens), 2):
        x = float(tokens[index])
        y = float(tokens[index + 1])
        points.append((x, y))

    return points


def g0(x, y):
    return f"G0 X{x:.3f} Y{y:.3f}"


def g1(x, y, feed_rate):
    return f"G1 X{x:.3f} Y{y:.3f} F{feed_rate}"


def add_polyline_gcode(gcode, points, scale, feed_rate, close_shape=False):
    if not points:
        return

    start_x = points[0][0] * scale
    start_y = points[0][1] * scale

    gcode.append(g0(start_x, start_y))
    gcode.append("G1 Z0 F300")  # Pen down at the start of the outline.

    for x, y in points[1:]:
        gcode.append(g1(x * scale, y * scale, feed_rate))

    if close_shape and len(points) > 1:
        gcode.append(g1(start_x, start_y, feed_rate))

    gcode.append("G0 Z5")  # Pen up before moving to the next outline.


def rect_to_points(elem):
    x = parse_length(elem.get("x"))
    y = parse_length(elem.get("y"))
    width = parse_length(elem.get("width"))
    height = parse_length(elem.get("height"))

    return [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    ]


def line_to_points(elem):
    return [
        (parse_length(elem.get("x1")), parse_length(elem.get("y1"))),
        (parse_length(elem.get("x2")), parse_length(elem.get("y2"))),
    ]


def svg_to_gcode(svg_file, gcode_file, scale=1.0, feed_rate=1000):
    tree = ET.parse(svg_file)
    root = tree.getroot()

    gcode = [
        "G21",
        "G90",
        "G0 Z5",
    ]

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        if tag == "line":
            add_polyline_gcode(gcode, line_to_points(elem), scale, feed_rate)
        elif tag == "polyline":
            add_polyline_gcode(
                gcode,
                parse_points(elem.get("points", "")),
                scale,
                feed_rate,
            )
        elif tag == "polygon":
            add_polyline_gcode(
                gcode,
                parse_points(elem.get("points", "")),
                scale,
                feed_rate,
                close_shape=True,
            )
        elif tag == "rect":
            add_polyline_gcode(
                gcode,
                rect_to_points(elem),
                scale,
                feed_rate,
                close_shape=True,
            )

    gcode.append("M2")

    with open(gcode_file, "w", encoding="utf-8") as file:
        file.write("\n".join(gcode))


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
        help="Scale factor applied to all coordinates",
    )
    parser.add_argument(
        "--feed-rate",
        type=int,
        default=1000,
        help="Feed rate used for G1 drawing moves",
    )

    args = parser.parse_args()
    output_file = args.gcode_file
    if output_file is None:
        output_file = str(Path(args.svg_file).with_suffix(".gcode"))

    svg_to_gcode(args.svg_file, output_file, args.scale, args.feed_rate)
    print(f"G-code written to {output_file}")


if __name__ == "__main__":
    main()
