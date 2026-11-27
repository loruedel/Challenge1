import ezdxf
import matplotlib.pyplot as plt
import math
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, polygonize                         # <<< NEW
from ezdxf.path import make_path                                        # <<< NEW
import os                                                           # <<< NEW (imports; may also be moved to the top)
import re
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.collections import PatchCollection
from shapely.geometry.polygon import orient

SNAP = 3  # <<< NEW: Round coordinates for closed contours(0.001 mm)


def flatten(e, dist=0.001):                             # <<< NEW
    """LINE/ARC/CIRCLE as a point list (x, y), rounded to SNAP"""
    return [(round(v.x, SNAP), round(v.y, SNAP)) for v in make_path(e).flattening(dist)]


def get_dxf_bounds(path):
    """Compute bounding box (width, height) for a DXF file."""
    try:
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()

        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for e in msp:
            pts = []

            if e.dxftype() == "LINE":
                pts = [e.dxf.start, e.dxf.end]

            elif e.dxftype() == "CIRCLE":                # <<< CHANGED: CIRCLE and ARC seperated
                c = e.dxf.center
                r = e.dxf.radius
                pts = [(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)]

            elif e.dxftype() == "ARC":                   # <<< NEW: only the actual arc, not the whole circle
                pts = flatten(e)

            elif e.dxftype() == "LWPOLYLINE":
                pts = [(float(p[0]), float(p[1])) for p in e.get_points() if len(p) >= 2]

            elif e.dxftype() == "POLYLINE":
                pts = [(float(v.dxf.x), float(v.dxf.y)) for v in e.vertices if hasattr(v, "dxf")]

            # <<< CHANGED: POINT removed (Markers are not part of the geometry)

            if not pts:
                continue
            
            # ensure each point has only x,y
            xs, ys = zip(*[(p[0], p[1]) for p in pts if len(p) >= 2])
            min_x, max_x = min(min_x, *xs), max(max_x, *xs)
            min_y, max_y = min(min_y, *ys), max(max_y, *ys)

        if min_x == float('inf') or min_y == float('inf'):
            return None, None

        width = max_x - min_x
        height = max_y - min_y
        return width, height

    except Exception as ex:
        print(f"Error reading {path}: {ex}")
        return None, None


def get_unconnected_lines(path, tol=0.01):               # <<< NEW: Returns unconnected LINEs
    """LINEs whose two endpoints touch no endpoint of any other LINE/ARC (within tol)."""
    msp = ezdxf.readfile(path).modelspace()
    ents = [e for e in msp if e.dxftype() in ("LINE", "ARC")]
    ends = [
        (e.dxf.start, e.dxf.end) if e.dxftype() == "LINE" else (e.start_point, e.end_point)
        for e in ents
    ]
    result = []
    for i, e in enumerate(ents):
        if e.dxftype() != "LINE":
            continue
        a, b = ends[i]
        touching = any(
            j != i and min(math.dist((a.x, a.y), (q.x, q.y)),
                           math.dist((b.x, b.y), (q.x, q.y))) <= tol
            for j, pair in enumerate(ends) for q in pair
        )
        if not touching:
            result.append(LineString([(a.x, a.y), (b.x, b.y)]))
    return result


def get_cutting_length(path):
    """Compute the total cutting length of all entities in a DXF file."""
    try:
        doc = ezdxf.readfile(path)
    except (IOError, ezdxf.DXFError):                    # <<< CHANGED: IOError added
        print(f"DXF read error: {path}")
        return None

    msp = doc.modelspace()
    total_length = 0.0

    for e in msp:
        etype = None                                     # <<< NEW: otherwise may be undefined in except statement
        try:
            etype = e.dxftype()

            if etype == "LINE":
                s, t = e.dxf.start, e.dxf.end
                total_length += math.dist((s[0], s[1]), (t[0], t[1]))

            elif etype == "CIRCLE":
                total_length += 2 * math.pi * e.dxf.radius

            elif etype == "ARC":
                sweep = (e.dxf.end_angle - e.dxf.start_angle) % 360 # <<< CHANGED: Circle calculation corrected
                if sweep == 0:
                    sweep = 360
                theta = math.radians(sweep)
                total_length += e.dxf.radius * theta

            elif etype == "LWPOLYLINE":
                pts = [(float(p[0]), float(p[1])) for p in e.get_points() if len(p) >= 2]
                if len(pts) >= 2:
                    line = LineString(pts)
                    length = line.length
                    if e.closed:
                        length += math.dist(pts[-1], pts[0])
                    total_length += length

            elif etype == "POLYLINE":
                pts = [(v.dxf.x, v.dxf.y) for v in e.vertices if hasattr(v, "dxf")]
                if len(pts) >= 2:
                    line = LineString(pts)
                    length = line.length
                    if e.is_closed:
                        length += math.dist(pts[-1], pts[0])
                    total_length += length

        except Exception as ex:
            print(f"Error in {path}, entity {etype}: {ex}")

    unconnected = sum(line.length for line in get_unconnected_lines(path))       # <<< CHANGED: subtract the length of unconnected lines
    total_length -= unconnected

    return total_length

def bridge_gaps(lines, gap_tol=2.0, touch_tol=1e-6):    # <<< NEW
    """Verbindet freie Endpunkte mit dem gegenseitig nächsten freien Endpunkt (<= gap_tol)."""
    endpoints = []

    for i, ln in enumerate(lines):
        if ln.coords[0] == ln.coords[-1]:
            continue

        endpoints.append((i, tuple(ln.coords[0])))
        endpoints.append((i, tuple(ln.coords[-1])))

    free = []

    for a, (i, p) in enumerate(endpoints):
        pt = Point(p)

        # Check whether another line endpoint is already located there
        connected = any(
            j != i and math.dist(p, q) <= touch_tol
            for j, q in endpoints
        )

        if not connected:
            free.append((i, p))
    
    bridges = []
    for a, (i, p) in enumerate(free):
        cand = [(math.dist(p, q), b) for b, (j, q) in enumerate(free) if j != i]
        if not cand:
            continue
        d, b = min(cand)
        if d <= gap_tol and a < b:
            j, q = free[b]
            back = min((math.dist(q, r), c) for c, (k, r) in enumerate(free) if k != j)
            if back[1] == a:                             # mutually nearest neighbors
                bridges.append(LineString([p, q]))
    return lines + bridges


def get_faces(path, area_tolerance=1e-3):               # <<< NEW
    """
    Combines all LINEs/ARCs/CIRCLEs into surfaces (polygonize).
    Returns: A list of (Polygon, Depth). Depth 0 = material, 1 = hole,
    2 = part inside a hole, etc. Even depths = material, odd depths = scrap.
    """
    doc = ezdxf.readfile(path)
    lines = []
    for e in doc.modelspace():
        etype = e.dxftype()
        if etype == "LINE" and "CENTER" in e.dxf.linetype.upper():   # <<< NEW: marker LINEs are not part of the contour
            continue
        elif etype in ("LINE", "ARC", "CIRCLE"):
            pts = flatten(e)
        elif etype == "LWPOLYLINE":
            pts = [(round(float(p[0]), SNAP), round(float(p[1]), SNAP)) for p in e.get_points()]
            if e.closed:
                pts.append(pts[0])
        elif etype == "POLYLINE":
            pts = [(round(float(v.dxf.x), SNAP), round(float(v.dxf.y), SNAP))
                   for v in e.vertices if hasattr(v, "dxf")]
            if e.is_closed:
                pts.append(pts[0])
        else:
            continue
        if len(set(pts)) >= 2:
            lines.append(LineString(pts))

    faces = [f for f in polygonize(unary_union(bridge_gaps(lines))) if f.area > area_tolerance]   # <<< CHANGED: includes bridge_gaps()
    outlines = [Polygon(f.exterior) for f in faces]
    result = []
    for i, f in enumerate(faces):
        rp = f.representative_point()
        depth = sum(1 for j, o in enumerate(outlines) if j != i and o.contains(rp))
        result.append((f, depth))
    return result


def get_rectangularity_ratio(path):
    """
    Rectangularity Ratio = material surfaces (netto) / Bounding-Box-Area.
    Returns: float between 0..1 (None for errors).
    """
    try:
        width, height = get_dxf_bounds(path)
        if not width or not height:
            return None

        faces = get_faces(path)
        # f.area ist bereits die Fläche ohne die eigenen Interiors (Löcher)
        material_area = sum(f.area for f, depth in faces if depth % 2 == 0)

        return material_area / (width * height)

    except Exception as ex:
        print(f"Error reading {path}: {ex}")
        return None


def get_pierce_density(path):
    try:
        length = get_cutting_length(path)
        if not length:
            return None
        return (len(get_faces(path)) * 1000) / length

    except Exception as ex:
        print(f"Error reading {path}: {ex}")
        return None     


def get_circle_density(path, r_max=2.5):
    """Compute the cutting length of all CIRCLEs & ARCs with radius smaller than 2.5mm."""
    try:
        doc = ezdxf.readfile(path)
    except (IOError, ezdxf.DXFError):                    # <<< CHANGED: IOError added
        print(f"DXF read error: {path}")
        return None

    msp = doc.modelspace()
    circle_length = 0.0

    for e in msp:
        etype = None                                     # <<< NEW: otherwise may be undefined in except statement
        try:
            etype = e.dxftype()

            if (etype == "CIRCLE") and (e.dxf.radius < r_max):
                circle_length += 2 * math.pi * e.dxf.radius

            elif (etype == "ARC") and (e.dxf.radius < r_max):
                sweep = (e.dxf.end_angle - e.dxf.start_angle) % 360 # <<< CHANGED: Circle calculation corrected
                if sweep == 0:
                    sweep = 360
                theta = math.radians(sweep)
                circle_length += e.dxf.radius * theta

        except Exception as ex:
            print(f"Error in {path}, entity {etype}: {ex}")

    length = get_cutting_length(path)
    if not length:
        return None
    return (circle_length * 100) / length


def _count_ring_corners(ring, angle_thresh):
    """Corners of one closed ring: vertices where the direction changes by more than angle_thresh (degrees)."""
    c = np.asarray(ring.coords)[:-1, :2]                         # last point == first point
    keep = np.ones(len(c), dtype=bool)
    keep[1:] = np.any(np.diff(c, axis=0) != 0, axis=1)           # drop duplicates created by SNAP rounding
    c = c[keep]
    if len(c) > 1 and np.array_equal(c[0], c[-1]):
        c = c[:-1]
    if len(c) < 3:
        return 0

    b = np.roll(c, -1, axis=0) - c                               # outgoing segment at vertex i
    a = np.roll(b, 1, axis=0)                                    # incoming segment at vertex i
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = (a * b).sum(axis=1)
    turn = np.abs(np.degrees(np.arctan2(cross, dot)))
    return int((turn > angle_thresh).sum())


def get_number_of_corners(path, angle_thresh=30.0):
    """Number of corners (direction change > angle_thresh) on all cutting contours."""
    try:
        n = 0
        for f, depth in get_faces(path):
            if depth % 2:                                        # scrap faces share their ring with the parent's hole
                continue
            n += _count_ring_corners(f.exterior, angle_thresh)
            n += sum(_count_ring_corners(r, angle_thresh) for r in f.interiors)
        return n
    except Exception as ex:
        print(f"Error reading {path}: {ex}")
        return None


def get_corner_density(path, angle_thresh=30.0):
    """Corners per 1m cutting length."""
    n = get_number_of_corners(path, angle_thresh)
    length = get_cutting_length(path)
    if n is None or not length:
        return None
    return n * 1000 / length


def get_number_of_holes(path, area_tolerance=1e-3):
    """
    Estimate the number of holes (internal cutouts) in a DXF file.

    Args:
        path (str): DXF file path.
        area_tolerance (float): small threshold to ignore degenerate loops.

    Returns:
        int: Number of detected holes (None on read error).
    """
    try:
        faces = get_faces(path, area_tolerance)         # <<< CHANGED: complete logic replaced
        # Hole = Inside hole of a material surface (even depth).
        return sum(len(f.interiors) for f, depth in faces if depth % 2 == 0)

    except Exception as ex:
        print(f"Error reading {path}: {ex}")
        return None                                      # <<< CHANGED: replacing 0


def get_number_of_parts(path):                           # <<< NEW
    """Returns (sum of POINTs, sum of material surfaces)"""
    try:
        msp = ezdxf.readfile(path).modelspace()
        n_points = len(msp.query("POINT"))
        n_faces = sum(1 for f, depth in get_faces(path) if depth % 2 == 0)
        return n_points, n_faces
    except Exception as ex:
        print(f"Error reading {path}: {ex}")
        return None, None


def get_point_context(path):                             # <<< NEW: (Study POINT-Relevance)
    """For each POINT: Depth of the deepest surface it lays in (0 = material, 1 = scrap)."""
    msp = ezdxf.readfile(path).modelspace()
    faces = get_faces(path)
    out = []
    for p in msp.query("POINT"):
        pt = Point(p.dxf.location.x, p.dxf.location.y)
        depths = [d for f, d in faces if Polygon(f.exterior).contains(pt)]
        out.append(max(depths) if depths else None)
    return out

def get_line_context(path, tol=0.01):                    # <<< NEW: (Study LINE-Relevance)
    """For each unconnected LINE: Depth of the deepest surface which completely contains the it
    (0 = Material, 1 = Scrap, None = in keiner Fläche bzw. schneidet eine Kontur)."""
    faces = get_faces(path)
    out = []
    for line in get_unconnected_lines(path, tol):
        depths = [d for f, d in faces if Polygon(f.exterior).contains(line)]
        out.append(max(depths) if depths else None)
    return out


def load_dxf_geometry(file_path):
    """Extract 2D geometry from DXF file into Shapely objects."""
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    polygons = []
    lines = []
    points = []                                          # <<< NEW

    for e in msp:
        etype = e.dxftype()

        # Closed 2D polyline
        if etype == "LWPOLYLINE" and e.closed:
            points_ = [(p[0], p[1]) for p in e.get_points()]
            polygons.append(Polygon(points_))

        # Open 2D polyline
        elif etype == "LWPOLYLINE" and not e.closed:
            points_ = [(p[0], p[1]) for p in e.get_points()]
            lines.append(LineString(points_))

        # Old-style 3D polyline (2D projection)
        elif etype == "POLYLINE":
            points_ = [(v.dxf.x, v.dxf.y) for v in e.vertices]
            if e.is_closed:
                polygons.append(Polygon(points_))
            else:
                lines.append(LineString(points_))

        # Circle
        elif etype == "CIRCLE":
            c = e.dxf.center
            r = e.dxf.radius
            circle = Point(c[0], c[1]).buffer(r, resolution=64)
            polygons.append(circle)

        # Arc
        elif etype == "ARC":                             # <<< NEW
            lines.append(LineString(flatten(e, 0.01)))

        # Line
        elif etype == "LINE":
            s, t = e.dxf.start, e.dxf.end
            lines.append(LineString([(s[0], s[1]), (t[0], t[1])]))

        # Point
        elif etype == "POINT":                           # <<< NEW
            points.append((e.dxf.location.x, e.dxf.location.y))

    return polygons, lines, points                       # <<< CHANGED: third return value


def plot_dxf(file_path):
    """Plot the geometry using matplotlib."""
    polygons, lines, points = load_dxf_geometry(file_path)   # <<< CHANGED

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_aspect("equal", "box")

    # Plot polygons
    for poly in polygons:
        x, y = poly.exterior.xy
        plt.plot(x, y, color="black", linewidth=0.8)

    # Plot lines
    for line in lines:
        x, y = line.xy
        plt.plot(x, y, color="black", linewidth=0.8)

    # Plot points
    for px, py in points:                                # <<< NEW
        plt.plot(px, py, "rx")

    plt.title(f"DXF Shape: {file_path}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

def get_dxf_extent(path):                                           # <<< NEW
    """
    Same logic as get_dxf_bounds(), but returns the position too:
    (min_x, min_y, max_x, max_y), or None on error / empty file.
    Needed to move the part to the origin before placing it on the sheet.
    The width/height are identical to get_dxf_bounds(), so the sheet view
    matches the numbers in the order table (bounding_box_width/height).
    """
    try:
        msp = ezdxf.readfile(path).modelspace()
        xs_all, ys_all = [], []
        for e in msp:
            t = e.dxftype()
            pts = []
            if t == "LINE":
                pts = [e.dxf.start, e.dxf.end]
            elif t == "CIRCLE":
                c, r = e.dxf.center, e.dxf.radius
                pts = [(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)]
            elif t == "ARC":                      # only the actual arc, rounded to 0.001 like flatten()
                pts = [(round(v.x, 3), round(v.y, 3)) for v in make_path(e).flattening(0.001)]
            elif t == "LWPOLYLINE":
                pts = [(float(p[0]), float(p[1])) for p in e.get_points() if len(p) >= 2]
            elif t == "POLYLINE":
                pts = [(float(v.dxf.x), float(v.dxf.y)) for v in e.vertices if hasattr(v, "dxf")]
            for p in pts:
                xs_all.append(p[0])
                ys_all.append(p[1])
        if not xs_all:
            return None
        return min(xs_all), min(ys_all), max(xs_all), max(ys_all)
    except Exception as ex:
        print(f"Error reading {path}: {ex}")
        return None


def get_sheet_layout(width, height, sheet_size, gap=0.0, margin=0.0):   # <<< NEW
    """
    Places identical rectangles in rows/columns on the sheet. Both orientations are tried
    (part as drawn / rotated by 90 degrees), the one with more parts wins (tie = as drawn).
    This is the same 'grid' logic as in the order-time notebook (gap = margin = 0 there).

    Args:
        width, height: bounding box of one part (mm)
        sheet_size: (X, Y) of the sheet (mm)
        gap: free space between two parts (mm)
        margin: free border around the sheet (mm)

    Returns:
        dict with rotated, cols, rows, count, cell (w, h) and positions [(x, y), ...]
        (lower left corner of each cell, filled row by row), or None if the part does not fit.
    """
    sx, sy = sheet_size
    best = None
    for rotated in (False, True):
        pw, ph = (height, width) if rotated else (width, height)
        cols = max(int((sx - 2 * margin + gap + 1e-9) // (pw + gap)), 0)
        rows = max(int((sy - 2 * margin + gap + 1e-9) // (ph + gap)), 0)
        count = cols * rows
        if best is None or count > best["count"]:
            best = {"rotated": rotated, "cols": cols, "rows": rows, "count": count, "cell": (pw, ph)}
    if best["count"] == 0:
        return None
    pw, ph = best["cell"]
    best["positions"] = [(margin + c * (pw + gap), margin + r * (ph + gap))
                         for r in range(best["rows"]) for c in range(best["cols"])]
    return best


def _face_path_data(poly, min_x, min_y):                            # <<< NEW
    """Shapely polygon (with holes) -> matplotlib vertices/codes, moved to the part origin."""
    poly = orient(poly, 1.0)    # outer ring counter-clockwise, holes clockwise -> holes are cut out
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        xy = np.asarray(ring.coords)[:, :2] - (min_x, min_y)
        verts.append(xy)
        codes += [Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY]
    return np.vstack(verts), np.array(codes)


def plot_sheet(file_path, sheet_size, quantity=None, sheet_number=1,
               gap=0.0, margin=0.0, show_cells=True, save_path=None):            # <<< NEW
    """
    Shows a sheet with the placed copies of one part, to see how production would run.

    Args:
        file_path: DXF file of the part.
        quantity: order quantity. With it, the number of sheets is calculated and the
                  selected sheet shows only the parts that are still needed (last sheet is
                  usually not full). Without it, one full sheet is shown.
        sheet_number: which sheet to show (1 = first; -1 = last). Keyword only.
        sheet_size: (X, Y) of the sheet in mm. Keyword only, e.g. sheet_size=(SHEET_X_MM, SHEET_Y_MM).
                    (Changing SHEET_SIZE in the notebook does NOT change this default - pass it explicitly.)
        gap / margin: free space between parts / at the sheet border (mm). Default 0,
                      like in the time calculation.
        show_cells: draw the bounding box of every placed part (dashed).
        save_path: optional, saves the picture (e.g. "sheet.png").

    Returns:
        dict with part size, rotation, parts_per_sheet, sheets_needed, parts_on_sheet, utilization,
        area_method_count (what the area formula would assume) - or None if nothing can be shown.

    Note: Placement is a simple grid of identical rectangles (not nested, as agreed).
    Only closed contours are drawn; open lines (e.g. engraving) are not shown.
    """
    if isinstance(quantity, (tuple, list)):
        raise TypeError("quantity must be a number. The sheet size goes in as "
                        "sheet_size=(X, Y), e.g. plot_sheet(path, quantity=9650, sheet_size=(3000, 1500))")
    if "get_faces" not in globals():
        raise NameError("plot_sheet needs get_faces() - please use the current helpers.py "
                        "(the version with get_faces / bridge_gaps) and restart the kernel.")
    extent = get_dxf_extent(file_path)
    if extent is None:
        print(f"No geometry found in {file_path}")
        return None
    min_x, min_y, max_x, max_y = extent
    w, h = max_x - min_x, max_y - min_y
    if w <= 0 or h <= 0:
        print(f"Invalid bounding box in {file_path}: {w} x {h}")
        return None

    layout = get_sheet_layout(w, h, sheet_size, gap, margin)
    if layout is None:
        print(f"{os.path.basename(file_path)}: part {w:.1f} x {h:.1f} mm does not fit on the "
              f"{sheet_size[0]} x {sheet_size[1]} mm sheet.")
        return None
    per_sheet = layout["count"]

    # --- how many sheets, how many parts on the selected sheet ---
    if quantity is None:
        sheets_needed, on_sheet, sheet_number = 1, per_sheet, 1
    else:
        quantity = int(quantity)
        if quantity <= 0:
            raise ValueError("quantity must be > 0")
        sheets_needed = math.ceil(quantity / per_sheet)
        if sheet_number < 0:                                        # -1 = last sheet
            sheet_number = sheets_needed + 1 + sheet_number
        if not 1 <= sheet_number <= sheets_needed:
            raise ValueError(f"sheet_number must be between 1 and {sheets_needed}")
        on_sheet = per_sheet if sheet_number < sheets_needed else quantity - (sheets_needed - 1) * per_sheet

    # --- geometry of one part (material faces incl. their holes), moved to the origin ---
    faces = [f for f, depth in get_faces(file_path) if depth % 2 == 0]
    if not faces:
        print(f"No closed contour found in {file_path}")
        return None
    base = [_face_path_data(f, min_x, min_y) for f in faces]
    material_area = sum(f.area for f in faces)

    patches, cells = [], []
    pw, ph = layout["cell"]
    for ox, oy in layout["positions"][:on_sheet]:
        for verts, codes in base:
            v = np.column_stack((h - verts[:, 1], verts[:, 0])) if layout["rotated"] else verts
            patches.append(PathPatch(Path(v + (ox, oy), codes)))
        cells.append(Rectangle((ox, oy), pw, ph))

    # --- plot ---
    sx, sy = sheet_size
    fig, ax = plt.subplots(figsize=(12, 12 * sy / sx + 1.2))
    ax.add_patch(Rectangle((0, 0), sx, sy, facecolor="#f4f4f4", edgecolor="black", linewidth=1.5))
    if show_cells:
        ax.add_collection(PatchCollection(cells, facecolor="none", edgecolor="#999999",
                                          linewidth=0.4, linestyle="--"))
    ax.add_collection(PatchCollection(patches, facecolor="#9ecae1", edgecolor="black", linewidth=0.5))
    pad = 0.02 * sx
    ax.set_xlim(-pad, sx + pad)
    ax.set_ylim(-pad, sy + pad)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")

    utilization = on_sheet * material_area / (sx * sy)
    area_count = int((sx * sy) // (w * h))
    title = (f"{os.path.basename(file_path)} - sheet {sheet_number}/{sheets_needed} "
             f"({sx} x {sy} mm)\n"
             f"{on_sheet} of {per_sheet} places used, "
             f"{'rotated by 90 degrees' if layout['rotated'] else 'not rotated'}, "
             f"part {w:.1f} x {h:.1f} mm, material utilisation {utilization:.0%}")
    if area_count > per_sheet:
        title += (f"\nNote: the area formula would assume {area_count} parts per sheet, "
                  f"geometrically only {per_sheet} fit")
    ax.set_title(title, fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return {
        "file": os.path.basename(file_path),
        "part_size": (w, h),
        "rotated": layout["rotated"],
        "parts_per_sheet": per_sheet,
        "sheets_needed": sheets_needed,
        "sheet_number": sheet_number,
        "parts_on_sheet": on_sheet,
        "utilization": utilization,
        "area_method_count": area_count,
    }


def _natural_key(text):                                             # <<< NEW
    """Sort 'Geo (2).DXF' before 'Geo (10).DXF'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


def choose_and_plot_sheet(folder, orders=None, **plot_kwargs):      # <<< NEW
    """
    The user picks a file (or an order) and the sheet with the placed parts is shown.

    Args:
        folder: folder with the DXF files, e.g. "10991360".
        orders: optional DataFrame (e.g. `merged`) with the columns file_name and order_quantity
                (customer_name / customer_branch are shown if present). Then the user picks an
                ORDER, so the number of sheets comes from the order quantity.
                Without it the user picks a DXF file and one full sheet is shown.
        **plot_kwargs: passed to plot_sheet (sheet_size, gap, margin, show_cells, save_path),
                       e.g. sheet_size=(SHEET_X_MM, SHEET_Y_MM).

    In Jupyter a dropdown with a "Show sheet" button is shown (needs ipywidgets, which comes
    with `jupyter`). Without ipywidgets a numbered text menu is used.
    """
    if isinstance(orders, str):
        raise TypeError(f'orders must be the DataFrame itself, not the text "{orders}". '
                        f"Use: choose_and_plot_sheet(folder, orders={orders})  (without quotes)")
    if orders is not None:
        missing = [c for c in ("file_name", "order_quantity") if c not in orders.columns]
        if missing:
            raise KeyError(f"orders is missing the column(s): {missing}")
    entries = []                                                    # (label, path, quantity)
    if orders is None:
        for f in sorted(os.listdir(folder), key=_natural_key):
            if f.lower().endswith(".dxf"):
                entries.append((f, os.path.join(folder, f), None))
    else:
        skipped = 0
        for _, row in orders.iterrows():
            path = os.path.join(folder, str(row["file_name"]))
            if not os.path.isfile(path):
                skipped += 1
                continue
            qty = row["order_quantity"]
            qty = None if qty != qty else int(qty)                  # NaN -> None
            parts = [str(row[c]) for c in ("customer_name", "customer_branch") if c in row.index]
            parts += [str(row["file_name"]), f"qty {qty}"]
            entries.append((" | ".join(parts), path, qty))
        if skipped:
            print(f"{skipped} orders skipped: DXF file not found in '{folder}'")
    if not entries:
        print("Nothing to choose from.")
        return None

    def show(index, sheet_number):
        _, path, qty = entries[index]
        try:
            plot_sheet(path, quantity=qty, sheet_number=sheet_number, **plot_kwargs)
        except ValueError as ex:
            print(ex)

    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError:                                             # text menu fallback
        for i, (label, _, _) in enumerate(entries):
            print(f"{i + 1:4d}  {label}")
        index = int(input("Number: ")) - 1
        sheet_number = int(input("Sheet number [1]: ") or 1)
        show(index, sheet_number)
        return None

    dropdown = widgets.Dropdown(options=[(label, i) for i, (label, _, _) in enumerate(entries)],
                                description="Order:" if orders is not None else "File:",
                                layout=widgets.Layout(width="95%"))
    sheet_input = widgets.BoundedIntText(value=1, min=-10**6, max=10**6, description="Sheet no.:")
    button = widgets.Button(description="Show sheet", button_style="primary")
    out = widgets.Output()

    def on_click(_):
        with out:
            clear_output(wait=True)
            show(dropdown.value, sheet_input.value)

    button.on_click(on_click)
    display(widgets.VBox([dropdown, widgets.HBox([sheet_input, button]), out]))
    return None