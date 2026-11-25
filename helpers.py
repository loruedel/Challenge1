import ezdxf
import matplotlib.pyplot as plt
import math
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, polygonize                         # <<< NEW
from ezdxf.path import make_path                                        # <<< NEW

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


def get_rectangularity(path):
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