import ezdxf
import matplotlib.pyplot as plt
import math
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, polygonize          # <<< NEU
from ezdxf.path import make_path                         # <<< NEU

SNAP = 3  # <<< NEU: Koordinaten auf 0.001 mm runden, damit sich Konturen schließen


def _flatten(e, dist=0.001):                             # <<< NEU
    """LINE/ARC/CIRCLE als Punktliste (x, y), auf SNAP gerundet."""
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

            elif e.dxftype() == "CIRCLE":                # <<< GEÄNDERT: CIRCLE und ARC getrennt
                c = e.dxf.center
                r = e.dxf.radius
                pts = [(c[0] - r, c[1] - r), (c[0] + r, c[1] + r)]

            elif e.dxftype() == "ARC":                   # <<< NEU: nur der echte Bogen, nicht der Vollkreis
                pts = _flatten(e)

            elif e.dxftype() == "LWPOLYLINE":
                pts = [(float(p[0]), float(p[1])) for p in e.get_points() if len(p) >= 2]

            elif e.dxftype() == "POLYLINE":
                pts = [(float(v.dxf.x), float(v.dxf.y)) for v in e.vertices if hasattr(v, "dxf")]

            # <<< GEÄNDERT: POINT-Zweig entfernt (Marker sind keine Geometrie)

            if not pts:
                continue
            # HIER FEHLT EIN KOMMENTAR
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


def get_cutting_length(path):
    """Compute the total cutting length of all entities in a DXF file."""
    try:
        doc = ezdxf.readfile(path)
    except (IOError, ezdxf.DXFError):                    # <<< GEÄNDERT: IOError ergänzt
        print(f"DXF read error: {path}")
        return None

    msp = doc.modelspace()
    total_length = 0.0

    for e in msp:
        etype = None                                     # <<< NEU: sonst evtl. undefiniert im except
        try:
            etype = e.dxftype()

            if etype == "LINE":
                s, t = e.dxf.start, e.dxf.end
                total_length += math.dist((s[0], s[1]), (t[0], t[1]))

            elif etype == "CIRCLE":
                total_length += 2 * math.pi * e.dxf.radius

            elif etype == "ARC":
                sweep = (e.dxf.end_angle - e.dxf.start_angle) % 360 # <<< GEÄNDERT: Kreisberechnung korrigiert
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

    return total_length


def get_marker_line_length(path, tol=0.01):         # <<< NEU: Berechnet die Länge unverbundener Linien
    """Länge aller LINEs, deren Mittelpunkt auf einem POINT liegt (vermutete Markierungen)."""
    try:
        msp = ezdxf.readfile(path).modelspace()
        lines = list(msp.query("LINE"))
        points = list(msp.query("POINT"))
        length, matched = 0.0, 0
        for p in points:
            px, py = p.dxf.location.x, p.dxf.location.y
            for e in lines:
                s, t = e.dxf.start, e.dxf.end
                if math.dist(((s.x + t.x) / 2, (s.y + t.y) / 2), (px, py)) < tol:
                    length += math.dist((s.x, s.y), (t.x, t.y))
                    matched += 1
                    break
        return length, matched, len(points)
    except (OSError, ezdxf.DXFError):
        return None, 0, 0


def get_cutting_lengths(path):                      # <<< NEU: Gibt Gesamtlänge & Länge zusammenhängender Linien zurück
    """Gibt (Länge inkl. Markierungslinien, Länge ohne Markierungslinien) zurück."""
    total = get_cutting_length(path)
    if total is None:
        return None, None
    marker, _, _ = get_marker_line_length(path)
    return total, total - marker


def _get_faces(path, area_tolerance=1e-3):               # <<< NEU
    """
    Verbindet alle Linien/Bögen/Kreise zu Flächen (polygonize).
    Rückgabe: Liste von (Polygon, Tiefe). Tiefe 0 = Material, 1 = Loch,
    2 = Teil im Loch usw. Gerade Tiefe = Material, ungerade = Aussparung.
    """
    doc = ezdxf.readfile(path)
    lines = []
    for e in doc.modelspace():
        etype = e.dxftype()
        if etype in ("LINE", "ARC", "CIRCLE"):
            pts = _flatten(e)
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

    faces = [f for f in polygonize(unary_union(lines)) if f.area > area_tolerance]
    outlines = [Polygon(f.exterior) for f in faces]
    result = []
    for i, f in enumerate(faces):
        rp = f.representative_point()
        depth = sum(1 for j, o in enumerate(outlines) if j != i and o.contains(rp))
        result.append((f, depth))
    return result


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
        faces = _get_faces(path, area_tolerance)         # <<< GEÄNDERT: komplette Logik ersetzt
        # Loch = Innenring einer Materialfläche (gerade Tiefe).
        return sum(len(f.interiors) for f, depth in faces if depth % 2 == 0)

    except Exception as ex:
        print(f"Error reading {path}: {ex}")
        return None                                      # <<< GEÄNDERT: statt 0


def get_number_of_parts(path):                           # <<< NEU
    """Gibt (Anzahl POINTs, Anzahl geschlossener Materialflächen) zurück."""
    try:
        msp = ezdxf.readfile(path).modelspace()
        n_points = len(msp.query("POINT"))
        n_faces = sum(1 for f, depth in _get_faces(path) if depth % 2 == 0)
        return n_points, n_faces
    except Exception as ex:
        print(f"Error reading {path}: {ex}")
        return None, None


def get_point_context(path):                             # <<< NEU (Diagnose der Punkt-Relevanz)
    """Pro POINT: Tiefe der innersten Fläche, in der er liegt (0 = Material, 1 = in Aussparung)."""
    msp = ezdxf.readfile(path).modelspace()
    faces = _get_faces(path)
    out = []
    for p in msp.query("POINT"):
        pt = Point(p.dxf.location.x, p.dxf.location.y)
        depths = [d for f, d in faces if Polygon(f.exterior).contains(pt)]
        out.append(max(depths) if depths else None)
    return out


def load_dxf_geometry(file_path):
    """Extract 2D geometry from DXF file into Shapely objects."""
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    polygons = []
    lines = []
    points = []                                          # <<< NEU

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
        elif etype == "ARC":                             # <<< NEU
            lines.append(LineString(_flatten(e, 0.01)))

        # Line
        elif etype == "LINE":
            s, t = e.dxf.start, e.dxf.end
            lines.append(LineString([(s[0], s[1]), (t[0], t[1])]))

        # Point
        elif etype == "POINT":                           # <<< NEU
            points.append((e.dxf.location.x, e.dxf.location.y))

    return polygons, lines, points                       # <<< GEÄNDERT: dritter Rückgabewert


def plot_dxf(file_path):
    """Plot the geometry using matplotlib."""
    polygons, lines, points = load_dxf_geometry(file_path)   # <<< GEÄNDERT

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
    for px, py in points:                                # <<< NEU
        plt.plot(px, py, "rx")

    plt.title(f"DXF Shape: {file_path}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()