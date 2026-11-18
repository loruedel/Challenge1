
import ezdxf
import matplotlib.pyplot as plt
import math

from ezdxf import path as ezdxf_path
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, polygonize


# ============================================================
# KONFIGURATION
# ============================================================

SUPPORTED_ENTITIES = {"LINE", "ARC", "CIRCLE"}

# Approximationstoleranz für ARC und CIRCLE bei der
# geometrischen Verarbeitung (Einheit: mm).
FLATTEN_DISTANCE = 0.01

# Minimale Fläche, um degenerierte Polygone zu ignorieren (mm²).
AREA_TOLERANCE = 1e-3


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def _read_dxf(file_path):
    """Read a DXF file and return its modelspace."""
    doc = ezdxf.readfile(file_path)
    return doc, doc.modelspace()


def _arc_sweep(start_angle, end_angle):
    """
    Calculate the counterclockwise sweep angle of a DXF ARC.
    DXF angles are given in degrees.
    """
    sweep = (end_angle - start_angle) % 360

    if math.isclose(sweep, 0.0, abs_tol=1e-10):
        sweep = 360.0

    return sweep


def _flatten_entity(entity, flatten_dist=FLATTEN_DISTANCE):
    """
    Convert LINE, ARC or CIRCLE into a sequence of 2D points.

    Curved entities are approximated using ezdxf.path.
    """
    if entity.dxftype() == "LINE":
        s = entity.dxf.start
        t = entity.dxf.end
        return [(float(s.x), float(s.y)),
                (float(t.x), float(t.y))]

    p = ezdxf_path.make_path(entity)

    points = [
        (float(v.x), float(v.y))
        for v in p.flattening(flatten_dist)
    ]

    # Explicitly close circles to create valid closed linework.
    if entity.dxftype() == "CIRCLE" and len(points) >= 2:
        if not math.isclose(
            math.dist(points[0], points[-1]),
            0.0,
            abs_tol=1e-9
        ):
            points.append(points[0])

    return points


def _load_cutting_linework(file_path, flatten_dist=FLATTEN_DISTANCE):
    """
    Load all cutting entities and convert them into Shapely LineStrings.

    POINT entities are deliberately ignored.
    """
    doc, msp = _read_dxf(file_path)

    lines = []

    for entity in msp:
        if entity.dxftype() not in SUPPORTED_ENTITIES:
            continue

        points = _flatten_entity(entity, flatten_dist)

        if len(points) >= 2:
            lines.append(LineString(points))

    return lines


# ============================================================
# 1. BOUNDING BOX
# ============================================================

def get_dxf_bounds(file_path):
    """
    Calculate the actual width and height of the DXF geometry.

    Supports LINE, ARC, CIRCLE and POINT.
    POINTs are included in the bounding box, but not in cutting
    length or hole calculations.
    """
    try:
        doc, msp = _read_dxf(file_path)

        all_points = []

        for entity in msp:
            etype = entity.dxftype()

            if etype == "LINE":
                s = entity.dxf.start
                t = entity.dxf.end

                all_points.extend([
                    (s.x, s.y),
                    (t.x, t.y)
                ])

            elif etype == "CIRCLE":
                c = entity.dxf.center
                r = entity.dxf.radius

                all_points.extend([
                    (c.x - r, c.y - r),
                    (c.x + r, c.y + r)
                ])

            elif etype == "ARC":
                c = entity.dxf.center
                r = entity.dxf.radius

                start = entity.dxf.start_angle
                sweep = _arc_sweep(
                    start,
                    entity.dxf.end_angle
                )

                angles = [start, start + sweep]

                # Include cardinal angles if they lie on the arc.
                for angle in [0, 90, 180, 270]:
                    delta = (angle - start) % 360

                    if delta <= sweep + 1e-10:
                        angles.append(angle)

                for angle in angles:
                    rad = math.radians(angle)

                    x = c.x + r * math.cos(rad)
                    y = c.y + r * math.sin(rad)

                    all_points.append((x, y))

            elif etype == "POINT":
                p = entity.dxf.location
                all_points.append((p.x, p.y))

        if not all_points:
            return None, None

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        return width, height

    except (OSError, ezdxf.DXFError) as ex:
        print(f"Error reading {file_path}: {ex}")
        return None, None


# ============================================================
# 2. CUTTING LENGTH
# ============================================================

def get_cutting_length(file_path):
    """
    Calculate the exact total cutting length of all LINE, ARC
    and CIRCLE entities.

    POINT entities are ignored.
    """
    try:
        doc, msp = _read_dxf(file_path)

        total_length = 0.0

        for entity in msp:
            etype = entity.dxftype()

            if etype == "LINE":
                s = entity.dxf.start
                t = entity.dxf.end

                total_length += math.dist(
                    (s.x, s.y),
                    (t.x, t.y)
                )

            elif etype == "CIRCLE":
                r = entity.dxf.radius

                total_length += 2 * math.pi * r

            elif etype == "ARC":
                r = entity.dxf.radius

                sweep = _arc_sweep(
                    entity.dxf.start_angle,
                    entity.dxf.end_angle
                )

                theta = math.radians(sweep)

                total_length += r * theta

            # POINT and other non-cutting entities are ignored.

        return total_length

    except (OSError, ezdxf.DXFError) as ex:
        print(f"Error calculating cutting length for {file_path}: {ex}")
        return None


# ============================================================
# 3. NUMBER OF HOLES
# ============================================================

def get_number_of_holes(
    file_path,
    area_tolerance=AREA_TOLERANCE,
    flatten_dist=FLATTEN_DISTANCE
):
    """
    Estimate the number of internal cutouts based on contour
    nesting depth.

    LINE, ARC and CIRCLE entities are converted into linework.
    Closed contours are reconstructed using polygonize().

    Odd nesting depth = hole.
    Even nesting depth = outer contour / material island.
    """
    try:
        lines = _load_cutting_linework(
            file_path,
            flatten_dist
        )

        if not lines:
            return 0

        # Merge linework and connect intersections.
        merged = unary_union(lines)

        # Reconstruct closed polygons from individual segments.
        faces = list(polygonize(merged))

        # Use the exterior ring of every face as a complete shape.
        shapes = []

        for face in faces:
            shape = Polygon(face.exterior)

            # Repair invalid polygons if possible.
            if not shape.is_valid:
                shape = shape.buffer(0)

            if shape.is_empty:
                continue

            if shape.area > area_tolerance:
                shapes.append(shape)

        hole_count = 0

        for i, shape in enumerate(shapes):

            # Count all larger contours that completely contain
            # the current contour.
            depth = sum(
                1
                for j, other in enumerate(shapes)
                if (
                    i != j
                    and other.area > shape.area + area_tolerance
                    and other.contains(shape)
                )
            )

            # Odd nesting depth means an internal cutout.
            if depth % 2 == 1:
                hole_count += 1

        return hole_count

    except (OSError, ezdxf.DXFError) as ex:
        print(f"Error calculating holes for {file_path}: {ex}")
        return None

    except Exception as ex:
        print(f"Geometry error in {file_path}: {ex}")
        return None


# ============================================================
# 4. LOAD DXF GEOMETRY
# ============================================================

def load_dxf_geometry(
    file_path,
    flatten_dist=FLATTEN_DISTANCE
):
    """
    Extract cutting geometry into Shapely objects.

    Returns:
        polygons: List of circle polygons.
        lines: List of LINE and ARC LineStrings.
    """
    doc, msp = _read_dxf(file_path)

    polygons = []
    lines = []

    for entity in msp:
        etype = entity.dxftype()

        if etype not in SUPPORTED_ENTITIES:
            continue

        points = _flatten_entity(entity, flatten_dist)

        if len(points) < 2:
            continue

        if etype == "CIRCLE":
            polygons.append(Polygon(points))

        else:
            lines.append(LineString(points))

    return polygons, lines


# ============================================================
# 5. PLOT DXF
# ============================================================

def plot_dxf(file_path):
    """
    Plot all cutting geometry using matplotlib.
    """
    polygons, lines = load_dxf_geometry(file_path)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_aspect("equal", adjustable="box")

    # Plot circles.
    for poly in polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="black", linewidth=0.8)

    # Plot lines and arcs.
    for line in lines:
        x, y = line.xy
        ax.plot(x, y, color="black", linewidth=0.8)

    ax.set_title(f"DXF Shape: {file_path}")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.grid(True)

    plt.tight_layout()
    plt.show()
