"""Extract named arterial roads around five Hyderabad localities from the OSM extract for the web mock seed.

Run (isolated, reads untrusted OSM data only as data):
  .venv/bin/python -I web/scripts/extract_roads.py datasets/extracted/geo/hyderabad.osm.pbf web/src/lib/mock/roads.json
"""

from __future__ import annotations

import json
import sys

AREAS = {
    "Ameerpet": (78.436, 17.428, 78.458, 17.442),
    "Punjagutta": (78.444, 17.418, 78.462, 17.432),
    "Khairatabad": (78.455, 17.402, 78.472, 17.420),
    "Mehdipatnam": (78.425, 17.385, 78.452, 17.402),
    "LB Nagar": (78.535, 17.338, 78.565, 17.362),
}
KEEP = {"trunk", "primary", "secondary", "tertiary"}


def main(pbf: str, out: str) -> None:
    from pyrosm import OSM
    from shapely.geometry import box

    result = []
    seen: set[int] = set()
    for area, bb in AREAS.items():
        osm = OSM(pbf, bounding_box=list(bb))
        gdf = osm.get_network(network_type="driving")
        if gdf is None or gdf.empty:
            continue
        gdf = gdf[gdf["highway"].isin(KEEP) & gdf["name"].notna()]
        clip = box(*bb)
        for _, r in gdf.iterrows():
            wid = int(r["id"])
            if wid in seen:
                continue
            geom = r.geometry.intersection(clip)
            if geom.is_empty:
                continue
            lines = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
            for ln in lines:
                if ln.geom_type != "LineString" or ln.length < 0.0012:  # ~130 m
                    continue
                seen.add(wid)
                lanes = r.get("lanes")
                try:
                    lanes_i = int(str(lanes).split(";")[0]) if lanes is not None and str(lanes) != "nan" else None
                except ValueError:
                    lanes_i = None
                result.append({
                    "way_id": wid,
                    "name": str(r["name"]),
                    "highway": str(r["highway"]),
                    "lanes": lanes_i,
                    "area": area,
                    "coords": [[round(x, 5), round(y, 5)] for x, y in ln.coords],
                })
    result.sort(key=lambda w: (w["area"], w["name"], w["way_id"]))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"source": "OpenStreetMap contributors (ODbL), hyderabad.osm.pbf extract",
                   "ways": result}, f, ensure_ascii=False, separators=(",", ":"))
    by_area: dict[str, int] = {}
    for w in result:
        by_area[w["area"]] = by_area.get(w["area"], 0) + 1
    print(len(result), by_area)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
