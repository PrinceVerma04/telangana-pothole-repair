#!/usr/bin/env bash
# Hyderabad basemap tiles (OPTIONAL - DECISIONS D7).
#
# The web app uses the online OpenFreeMap "liberty" style by default
# (https://tiles.openfreemap.org/styles/liberty) and overlays our own GeoJSON/MVT layers
# (road segments, potholes, gis/layers/*). This script only matters for an OFFLINE demo:
# it cuts a Protomaps PMTiles basemap for the Hyderabad bbox into app_data/tiles/hyd.pmtiles,
# which the PWA serves as /tiles/hyd.pmtiles (PLAN WP9/WP10, acceptance: file < 400 MB).
#
# Requirements: the `pmtiles` CLI (go-pmtiles, single static binary, no sudo):
#   https://github.com/protomaps/go-pmtiles/releases  -> put `pmtiles` on PATH or set PMTILES_BIN.
#   Download it yourself and verify the release checksum; this script never downloads or runs
#   third-party binaries on its own.
#
# Usage:
#   gis/build_tiles.sh                       # z0-15, bbox 78.1,17.1,78.8,17.7
#   BUILD_DATE=20261101 MAXZOOM=14 gis/build_tiles.sh
#   SOURCE=/path/to/planet.pmtiles gis/build_tiles.sh   # extract from a local planet file
#
# `pmtiles extract` fetches only the tiles inside the bbox with HTTP range requests, so the
# full planet (~120 GB) is never downloaded. z15 for this bbox is ~100-250 MB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BBOX="${BBOX:-78.1,17.1,78.8,17.7}"
MAXZOOM="${MAXZOOM:-15}"
BUILD_DATE="${BUILD_DATE:-20261101}"   # daily builds: https://maps.protomaps.com/builds/
SOURCE="${SOURCE:-https://build.protomaps.com/${BUILD_DATE}.pmtiles}"
OUT="${OUT:-$ROOT/app_data/tiles/hyd.pmtiles}"
PMTILES_BIN="${PMTILES_BIN:-$(command -v pmtiles || true)}"

if [[ -z "$PMTILES_BIN" || ! -x "$PMTILES_BIN" ]]; then
  cat >&2 <<EOF
pmtiles CLI not found - skipping offline tiles (the web app falls back to the online
OpenFreeMap style, DECISIONS D7). To build them:
  1. download go-pmtiles for linux x86_64 from https://github.com/protomaps/go-pmtiles/releases
  2. verify its checksum, unpack, and set PMTILES_BIN=/path/to/pmtiles
  3. re-run: gis/build_tiles.sh
Command that will run:
  pmtiles extract "$SOURCE" "$OUT" --bbox=$BBOX --maxzoom=$MAXZOOM
EOF
  exit 0
fi

mkdir -p "$(dirname "$OUT")"
echo "extracting $SOURCE -> $OUT (bbox $BBOX, z0-$MAXZOOM)"
"$PMTILES_BIN" extract "$SOURCE" "$OUT" --bbox="$BBOX" --maxzoom="$MAXZOOM"
"$PMTILES_BIN" show "$OUT" | head -20
SIZE=$(stat -c %s "$OUT")
echo "size: $((SIZE / 1024 / 1024)) MB"
if (( SIZE > 400 * 1024 * 1024 )); then
  echo "WARNING: tiles exceed 400 MB; re-run with MAXZOOM=14" >&2
  exit 1
fi
