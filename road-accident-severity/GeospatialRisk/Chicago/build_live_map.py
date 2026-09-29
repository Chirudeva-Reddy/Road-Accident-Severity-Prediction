"""
Bake the DBSCAN vs. Getis-Ord Gi* outputs into app/live_comparison_map.html.

The page is a single static file (no map library, no tile server): geometry is
simplified and pre-projected to SVG path strings here, so the browser only
parses ~100 KB of JSON. Re-run after esri_pipeline.py:

    python -m GeospatialRisk.Chicago.build_live_map
"""

from __future__ import annotations

import json
import math
import re
import pandas as pd
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from . import paths

PAGE = paths.CHICAGO_DIR / "app" / "live_comparison_map.html"
DOCS = MODULE_DIR.parents[2] / "docs" / "index.html"

WIDTH = 600
TOLERANCE = 0.0006  # degrees (~50 m); invisible at this map scale


def build_projection(bounds):
    """Equirectangular lon/lat -> SVG px, WIDTH wide; returns (project, height)."""
    min_lon, min_lat, max_lon, max_lat = bounds
    kx = math.cos(math.radians((min_lat + max_lat) / 2))
    scale = WIDTH / ((max_lon - min_lon) * kx)
    height = (max_lat - min_lat) * scale

    def project(lon, lat):
        return round((lon - min_lon) * kx * scale, 1), round((max_lat - lat) * scale, 1)

    return project, round(height)


def to_path(geom, project) -> str:
    """(Multi)Polygon -> simplified SVG path `d` string, holes included."""
    geom = geom.simplify(TOLERANCE, preserve_topology=True)
    polys = getattr(geom, "geoms", [geom])
    parts = []
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            pts = [project(x, y) for x, y in ring.coords]
            parts.append("M" + "L".join(f"{x} {y}" for x, y in pts) + "Z")
    return "".join(parts)


def main():
    areas_gj = json.loads((paths.COMMUNITY_AREAS).read_text())
    iso_gj = json.loads((paths.ESRI_ISOCHRONES).read_text())
    risk = pd.read_csv(paths.COMMUNITY_RISK).set_index("community_area_number")
    hot = pd.read_csv(paths.ESRI_HOTSPOTS_CSV)

    area_shapes = [(f["properties"], shape(f["geometry"])) for f in areas_gj["features"]]
    project, height = build_projection(unary_union([g for _, g in area_shapes]).bounds)

    areas = []
    for props, geom in area_shapes:
        n = int(float(props["area_numbe"]))
        r = risk.loc[n] if n in risk.index else None
        areas.append({
            "n": n,
            "name": props["community"].title(),
            "d": to_path(geom, project),
            "risk": None if r is None else round(float(r["risk_index"]), 4),
        })

    iso = []
    for brk in (12, 8, 5):
        rings = [shape(f["geometry"]) for f in iso_gj["features"] if f["properties"]["ToBreak"] == brk]
        iso.append({"b": brk, "d": to_path(unary_union(rings), project)})

    centers = {}
    for f in iso_gj["features"]:
        p = f["properties"]
        centers[p["Name_1"]] = project(p["longitude"], p["latitude"])

    # a 250 m DBSCAN radius percolates through the street grid; flag clusters that swallowed a whole community area
    area_totals = set(risk["total_crashes"].astype(int))
    hot["vrank"] =hot["total_crashes"].rank(ascending=False, method="min").astype(int)
    hot["grank"] = hot["gi_z_score"].rank(ascending=False, method="min").astype(int)
    clusters = []
    for row in hot.itertuples():
        pt = Point(row.centroid_lon, row.centroid_lat)
        area = next((p["community"].title() for p, g in area_shapes if g.contains(pt)), "Outside city limits")
        x, y = project(row.centroid_lon, row.centroid_lat)
        clusters.append({
            "id": int(row.cluster_id), "x": x, "y": y, "area": area,
            "n": int(row.total_crashes), "fatal": int(row.sum_fatal), "incap": int(row.sum_incapacitating),
            "ws": round(row.weighted_score, 4), "z": row.gi_z_score, "p": row.gi_p_value,
            "cat": row.gi_category, "ems": row.ems_reachability_status,
            "vrank": int(row.vrank), "grank": int(row.grank),
            "whole": int(row.total_crashes) in area_totals,
        })

    data = {
        "w": WIDTH, "h": height, "areas": areas, "iso": iso,
        "centers": [{"name": k, "x": v[0], "y": v[1]} for k, v in centers.items()],
        "clusters": clusters,
    }
    blob = json.dumps(data, separators=(",", ":"))
    html = PAGE.read_text()
    html, n = re.subn(
        r'(<script id="data" type="application/json">).*?(</script>)',
        lambda m: m.group(1) + blob + m.group(2), html, flags=re.S,
    )
    assert n == 1, "data <script> tag not found in page"
    PAGE.write_text(html)
    # GitHub Pages copy needs a real document shell (no doctype = quirks mode)
    DOCS.parent.mkdir(exist_ok=True)
    DOCS.write_text('<!doctype html>\n<html lang="en">\n' + html + "</html>\n")
    print(f"Wrote {PAGE} and {DOCS} ({PAGE.stat().st_size / 1024:.0f} KB, data {len(blob) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
